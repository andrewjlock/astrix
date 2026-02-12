# pyright: standard
# pyright: reportArgumentType = false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import contextily as cx
import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.spatial.transform import Rotation

from astrix.functs import ned_rotation
from astrix.spatial.location import Path, Point
from astrix.spatial.ray import Ray
from astrix.time import Time, TimeGroup, time_linspace

_DEFAULT_COLOR_CYCLE = [
    "#8dd3c7",
    "#fb8072",
    "#b3de69",
    "#bebada",
    "#fdb462",
    "#ffffb3",
    "#fccde5",
    "#80b1d3",
]


def color_from_int(i: int) -> str:
    return _DEFAULT_COLOR_CYCLE[i % len(_DEFAULT_COLOR_CYCLE)]


def _geodet2ecef_grid(
    llat: NDArray, llon: NDArray, alt: float = 0
) -> tuple[NDArray, NDArray, NDArray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    geodet = np.stack([llat.ravel(), llon.ravel(), np.full(llat.size, alt)], axis=1)
    ecef = np.array(transformer.transform(geodet[:, 1], geodet[:, 0], geodet[:, 2])).T
    # ecef = geodet2ecef(geodet)
    xyz = ecef.reshape(*llat.shape, 3)
    X = xyz[:, :, 0]
    Y = xyz[:, :, 1]
    Z = xyz[:, :, 2]

    return X, Y, Z


@dataclass
class ConnectingLines:
    points_1: Point
    points_2: Point

    @classmethod
    def from_paths(cls, path1: Path, path2: Path, n=10) -> ConnectingLines:
        tg = TimeGroup([path1.time, path2.time])
        time = time_linspace(*tg.overlap_bounds, n)
        points_1 = path1.interp(time)
        points_2 = path2.interp(time)
        return cls(points_1, points_2)

    def truncate(self, start: Time, end: Time) -> ConnectingLines:
        mask = (self.points_1.time.unix >= start.unix) & (
            self.points_1.time.unix <= end.unix
        )
        return ConnectingLines(self.points_1[mask], self.points_2[mask])


def split_lon_interval(lon_min: float, lon_max: float):
    """
    Split a longitude interval into continuous segments in [-180, 180].
    Handles any wrap, including multi-wrap and snapped bounds.
    """

    # Normalize to [-180, 180)
    def norm(lon):
        return ((lon + 180) % 360) - 180

    lon_min_n = norm(lon_min)
    lon_max_n = norm(lon_max)

    # Case 1: no wrap
    if lon_min_n <= lon_max_n:
        return [(lon_min_n, lon_max_n)]

    # Case 2: wrap across 180°
    # Example: 165 → -125 becomes [(165, 180), (-180, -125)]
    return [
        (lon_min_n, 180.0),
        (-180.0, lon_max_n),
    ]


@dataclass
class PlotData:
    name: str
    type: str  # 'point', 'path', 'frame'
    actor: pv.Actor
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]
    data: dict = field(default_factory=dict)


class Plot3D:
    p: pv.Plotter
    data: dict[str, PlotData]
    text_actors: dict[str, pv.Actor]
    aspect_ratio: float

    def __init__(self, size: int = 900, aspect_ratio: float = 1.0, aa="ms"):
        self.aspect_ratio = aspect_ratio
        self.p = pv.Plotter()
        self.p.window_size = [
            int(size * (aspect_ratio**0.5)),
            int(size / (aspect_ratio**0.5)),
        ]
        self.p.set_background("black")  # pyright: ignore
        if aa == "ms":
            self.p.enable_anti_aliasing("msaa", multi_samples=16)
        elif aa == "fx":
            self.p.enable_anti_aliasing("fxaa")
            self.p.enable_depth_peeling(occlusion_ratio=0.0, number_of_peels=50)
        elif aa == "ss":
            self.p.enable_anti_aliasing("ssaa")
        else:
            self.p.enable_anti_aliasing(None)

        self.p.add_key_event("s", self.save)

        self.data = {}
        self.text_actors = {}

    def set_view(
        self,
        cent: Point,
        heading: float = 180,
        pitch: float = -45,
        zoom: float = 1.0,
        parrallel: bool = False,
    ):
        ecef_cent = cent.ecef[0]
        rot_ned = ned_rotation(cent.geodet)
        rot_cam = Rotation.from_euler("ZY", [heading, pitch], degrees=True)
        rot = rot_ned * rot_cam
        cam_dir = rot.apply(np.array([1, 0, 0]))
        cam_pos = ecef_cent + cam_dir * 5e5 / zoom  # 100 km

        self.p.set_position(cam_pos, render=False)
        self.p.set_focus(ecef_cent, render=False)
        if pitch == 90 or pitch == -90:
            # looking straight up or down, set_viewup() has no effect
            self.p.set_viewup(
                rot_ned.apply(np.array([1.0, 0, 0])), reset=False, render=False
            )
        else:
            self.p.set_viewup(
                rot_ned.apply(np.array([0.0, 0, -1])), reset=False, render=False
            )
        # self.p.camera.zoom(1)
        if parrallel:
            self.p.enable_parallel_projection()  # pyright: ignore

    def add_texture(
        self, lat_bounds: Sequence[float], lon_bounds: Sequence[float], alpha=0.6
    ):
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds

        # ------------------------------------------------------------
        # 0) Split longitude interval into continuous segments
        # ------------------------------------------------------------
        lon_segments = split_lon_interval(lon_min, lon_max)

        to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        actors = []

        for seg_lon_min, seg_lon_max in lon_segments:
            # --------------------------------------------------------
            # 1) Fetch basemap tiles for this segment
            # --------------------------------------------------------
            img, ext = cx.bounds2img(
                seg_lon_min,
                lat_min,
                seg_lon_max,
                lat_max,
                zoom="auto",
                source=cx.providers.Esri.WorldImagery,
                ll=True,
            )

            xmin_img, xmax_img, ymin_img, ymax_img = ext

            # --------------------------------------------------------
            # 2) Convert requested bounds (this segment) to Web Mercator
            # --------------------------------------------------------
            x0, y0 = to_merc.transform(seg_lon_min, lat_min)
            x1, y1 = to_merc.transform(seg_lon_max, lat_max)

            xmin_req, xmax_req = min(x0, x1), max(x0, x1)
            ymin_req, ymax_req = min(y0, y1), max(y0, y1)

            # --------------------------------------------------------
            # 3) Crop tile image to requested bounds
            # --------------------------------------------------------
            src = np.moveaxis(img, 2, 0)  # HWC → CHW
            dst = np.zeros_like(src)

            in_transform = from_bounds(
                xmin_img, ymin_img, xmax_img, ymax_img, img.shape[1], img.shape[0]
            )
            out_transform = from_bounds(
                xmin_req, ymin_req, xmax_req, ymax_req, img.shape[1], img.shape[0]
            )

            reproject(
                source=src,
                destination=dst,
                src_transform=in_transform,
                src_crs=CRS.from_epsg(3857),
                dst_transform=out_transform,
                dst_crs=CRS.from_epsg(3857),
                resampling=Resampling.bilinear,
            )

            img_seg = np.moveaxis(dst, 0, 2)  # CHW → HWC

            # --------------------------------------------------------
            # 4) Adaptive mesh resolution for this segment
            # --------------------------------------------------------
            H, W = img_seg.shape[:2]
            k = 16  # tweakable density factor
            n_lon = max(4, W // k)
            n_lat = max(4, H // k)

            lats = np.linspace(lat_min, lat_max, n_lat)
            lons = np.linspace(seg_lon_min, seg_lon_max, n_lon)
            llon, llat = np.meshgrid(lons, lats, indexing="ij")

            xm, ym = to_merc.transform(llon, llat)

            # --------------------------------------------------------
            # 5) UV texture coordinates for this segment
            # --------------------------------------------------------
            u = (xm - xmin_req) / (xmax_req - xmin_req)
            v = (ym - ymin_req) / (ymax_req - ymin_req)

            # --------------------------------------------------------
            # 6) Build PyVista grid and apply texture
            # --------------------------------------------------------
            X, Y, Z = _geodet2ecef_grid(llat, llon)
            grid = pv.StructuredGrid(X, Y, Z)

            tcoords = np.c_[u.ravel(order="F"), v.ravel(order="F")].astype(np.float32)
            grid.active_texture_coordinates = tcoords

            tex = pv.Texture(img_seg)
            tex.repeat = False

            act = self.p.add_mesh(
                grid,
                texture=tex,
                lighting=True,
                smooth_shading=True,
                show_edges=False,
                opacity=alpha,
                render=False,
            )
            actors.append(act)

        # ------------------------------------------------------------
        # 7) Store all segment actors under one logical texture entry
        # ------------------------------------------------------------
        self.data["texture"] = PlotData(
            "texture",
            "texture",
            actors,  # list of actors now
            lat_bounds=(lat_min, lat_max),
            lon_bounds=(lon_min, lon_max),
        )

    def add_grid(
        self,
        lat_bounds: Sequence[float],
        lon_bounds: Sequence[float],
        spacing: float = 5.0,
    ):
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds

        # ------------------------------------------------------------
        # 0) Snap bounds to spacing
        # ------------------------------------------------------------
        def snap_min(v):
            return np.floor(v / spacing) * spacing

        def snap_max(v):
            return np.ceil(v / spacing) * spacing

        lat_min_s = snap_min(lat_min)
        lat_max_s = snap_max(lat_max)
        lon_min_s = snap_min(lon_min)
        lon_max_s = snap_max(lon_max)

        # ------------------------------------------------------------
        # 1) Unwrap longitudes into a continuous range
        #    (so we can cross 180° without splitting)
        # ------------------------------------------------------------
        lon0, lon1 = lon_min_s, lon_max_s
        if lon1 < lon0:
            lon1 += 360.0  # unwrap across 180°

        # ------------------------------------------------------------
        # 2) Build latitude and longitude arrays in unwrapped space
        # ------------------------------------------------------------
        n_lat = int(round((lat_max_s - lat_min_s) / spacing)) + 1
        lats = np.linspace(lat_min_s, lat_max_s, n_lat)

        n_lon = int(round((lon1 - lon0) / spacing)) + 1
        lons_unwrapped = np.linspace(lon0, lon1, n_lon)

        # Wrap longitudes back into [-180, 180] for ECEF / labels
        lons = ((lons_unwrapped + 180.0) % 360.0) - 180.0

        # ------------------------------------------------------------
        # 3) Meshgrid with consistent indexing
        # ------------------------------------------------------------
        LLon, LLat = np.meshgrid(lons, lats, indexing="ij")

        # ------------------------------------------------------------
        # 4) Convert to ECEF and build grid
        # ------------------------------------------------------------
        X, Y, Z = _geodet2ecef_grid(LLat, LLon, alt=10)
        sg = pv.StructuredGrid(X, Y, Z)
        edges = sg.extract_all_edges()

        self.p.add_mesh(
            edges,
            color="grey",
            line_width=1,
            opacity=0.5,
            render=False,
        )

        # ------------------------------------------------------------
        # 5) Labels
        # ------------------------------------------------------------
        def fmt_lon(deg):
            s = "E" if deg >= 0 else "W"
            return f"{abs(deg):.1f}°{s}"

        def fmt_lat(deg):
            s = "N" if deg >= 0 else "S"
            return f"{abs(deg):.1f}°{s}"

        # Longitude labels (first latitude row)
        lon_label_points = np.c_[X[:, 0], Y[:, 0], Z[:, 0]]
        lon_labels = [fmt_lon(lon) for lon in lons]

        # Latitude labels (last longitude column)
        lat_label_points = np.c_[X[-1, :], Y[-1, :], Z[-1, :]]
        lat_labels = [fmt_lat(lat) for lat in lats]

        self.p.add_point_labels(
            lon_label_points,
            lon_labels,
            font_size=10,
            text_color="grey",
            show_points=False,
            shape_opacity=0.0,
            always_visible=False,
        )
        self.p.add_point_labels(
            lat_label_points,
            lat_labels,
            font_size=10,
            text_color="grey",
            show_points=False,
            shape_opacity=0.0,
            always_visible=False,
        )

    def _create_path_data(self, path: Path) -> pv.PolyData:
        curve = pv.lines_from_points(path.ecef)
        return curve

    def add_path(
        self,
        name: str,
        path: Path,
        path_max: Path | None = None,
        line_width: float = 2.0,
        color: str | int | None = None,
        alpha: float = 1.0,
    ):
        if isinstance(color, int):
            color = color_from_int(color)
        elif color is None:
            color = _DEFAULT_COLOR_CYCLE[len(self.data) % len(_DEFAULT_COLOR_CYCLE)]
        curve = self._create_path_data(path)
        act = self.p.add_mesh(
            curve,
            color=color,
            line_width=line_width,
            name=name,
            opacity=alpha,
            lighting=False,
            # render_lines_as_tubes=True,
            smooth_shading=True,
            render=False,
        )
        # Use maximum path for bounds (so bounds isn't changed during animation)
        if path_max is None:
            path_max = path
        geodet_max = path_max.points.geodet
        self.data[name] = PlotData(
            name=name,
            type="path",
            actor=act,
            lat_bounds=(np.min(geodet_max[:, 0]), np.max(geodet_max[:, 0])),
            lon_bounds=(np.min(geodet_max[:, 1]), np.max(geodet_max[:, 1])),
        )

    def update_path(self, name: str, path: Path):
        if name not in self.data:
            raise ValueError(f"Path '{name}' not found in plot data.")
        curve = self._create_path_data(path)
        act = self.data[name].actor
        act.mapper.SetInputData(curve)
        act.mapper.Update()

    def clear_path(self, name: str):
        if name not in self.data:
            raise ValueError(f"Path '{name}' not found in plot data.")
        act = self.data[name].actor
        act.mapper.SetInputData(pv.PolyData())  # empty
        act.mapper.Update()

    def add_ground_track(
        self,
        name: str,
        path: Path,
        dt: float = 10.0,
        line_width: float = 1.0,
        color: str | int | None = "white",
        alpha: float = 0.6,
    ):
        geodet = path.points.geodet
        geodet[:, 2] = 0.0  # set altitude to 0
        path_gt = Path(Point.from_geodet(geodet, time=path.time))

        if isinstance(color, int):
            color = color_from_int(color)
        elif color is None:
            color = _DEFAULT_COLOR_CYCLE[len(self.data) % len(_DEFAULT_COLOR_CYCLE)]

        # Get times of vertical path joins
        # We want this to start at first time and be spaced strictly at dt intervals,
        # with a final time at the end
        n_bars = int(np.floor((path.time.unix[-1] - path.time.unix[0]) / dt))
        bar_t = np.concatenate(
            [path.time.unix[0] + np.arange(n_bars + 1) * dt, [path.time.unix[-1]]]
        )
        bar_time = Time(bar_t)
        pt_path = path.interp(bar_time)
        pt_path_gt = path_gt.interp(bar_time)

        N = len(bar_t)
        points = np.vstack((pt_path.ecef, pt_path_gt.ecef))
        cells = np.c_[np.full(N, 2), np.arange(N), np.arange(N) + N].ravel()
        vert_lines = pv.PolyData(points, lines=cells)
        ground_lines = pv.lines_from_points(path_gt.ecef)
        lines = vert_lines + ground_lines

        act = self.p.add_mesh(
            lines,
            color=color,
            line_width=line_width,
            name=name,
            opacity=alpha,
            lighting=False,
            render=False,
        )
        self.data[name] = PlotData(
            name=name,
            type="ground_track",
            actor=act,
            lat_bounds=(np.min(geodet[:, 0]), np.max(geodet[:, 0])),
            lon_bounds=(np.min(geodet[:, 1]), np.max(geodet[:, 1])),
            data={
                "dt": dt,
                "color": color,
                "line_width": line_width,
                "alpha": alpha,
            },
        )

    def update_ground_track(self, name: str, path: Path):
        if name not in self.data:
            raise ValueError(f"Ground track '{name}' not found in plot data.")
        dt = self.data[name].data.get("dt", 10.0)

        geodet = path.points.geodet
        geodet[:, 2] = 0.0  # set altitude to 0
        path_gt = Path(Point.from_geodet(geodet, time=path.time))

        # Get times of vertical path joins
        n_bars = int(np.floor((path.time.unix[-1] - path.time.unix[0]) / dt))
        bar_t = np.concatenate(
            [path.time.unix[0] + np.arange(n_bars + 1) * dt, [path.time.unix[-1]]]
        )
        bar_time = Time(bar_t)
        pt_path = path.interp(bar_time)
        pt_path_gt = path_gt.interp(bar_time)

        N = len(bar_t)
        points = np.vstack((pt_path.ecef, pt_path_gt.ecef))
        cells = np.c_[np.full(N, 2), np.arange(N), np.arange(N) + N].ravel()
        vert_lines = pv.PolyData(points, lines=cells)
        ground_lines = pv.lines_from_points(path_gt.ecef)
        lines = vert_lines + ground_lines

        act = self.data[name].actor
        act.mapper.SetInputData(lines)
        act.mapper.Update()

    def clear_ground_track(self, name: str):
        if name not in self.data:
            raise ValueError(f"Ground track '{name}' not found in plot data.")
        act = self.data[name].actor
        act.mapper.SetInputData(pv.PolyData())  # empty
        act.mapper.Update()

    def add_point(
        self,
        name: str,
        point: Point,
        size: float = 2.0,
        color: str | int | None = None,
        alpha: float = 1.0,
    ):
        if isinstance(color, int):
            color = color_from_int(color)
        elif color is None:
            color = _DEFAULT_COLOR_CYCLE[len(self.data) % len(_DEFAULT_COLOR_CYCLE)]

        act = self.p.add_points(
            point.ecef,
            color=color,
            name=name,
            opacity=alpha,
            lighting=False,
            smooth_shading=True,
            render=False,
            render_points_as_spheres=True,
            point_size=size,
        )

        geodet = point.geodet[0]
        self.data[name] = PlotData(
            name=name,
            type="point",
            actor=act,
            lat_bounds=(geodet[0], geodet[0]),
            lon_bounds=(geodet[1], geodet[1]),
            data={"size": size},
        )

    def add_box(
        self,
        name: str,
        lat_bounds: Sequence[float],
        lon_bounds: Sequence[float],
        fill: bool = True,
        fill_color: str | int | None = None,
        line_color: str | int | None = None,
        alpha: float = 0.3,
        line_width: float = 2.0,
        resolution: int = 40,
        alt: float = 50.0,
    ):
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds

        # ------------------------------------------------------------
        # 1) Resolve colors (same logic as add_path)
        # ------------------------------------------------------------
        if isinstance(fill_color, int):
            fill_color = color_from_int(fill_color)
        elif fill_color is None:
            fill_color = _DEFAULT_COLOR_CYCLE[
                len(self.data) % len(_DEFAULT_COLOR_CYCLE)
            ]

        if isinstance(line_color, int):
            line_color = color_from_int(line_color)
        elif line_color is None:
            line_color = fill_color  # default: match fill

        # ------------------------------------------------------------
        # 2) Unwrap longitudes so the box can cross 180° safely
        # ------------------------------------------------------------
        lon0, lon1 = lon_min, lon_max
        if lon1 < lon0:
            lon1 += 360.0

        # Build unwrapped lon array
        lons_unwrapped = np.linspace(lon0, lon1, resolution)

        # Wrap back into [-180, 180]
        lons = ((lons_unwrapped + 180) % 360) - 180

        # ------------------------------------------------------------
        # 3) Build a lat/lon grid for the box surface
        # ------------------------------------------------------------
        lats = np.linspace(lat_min, lat_max, resolution)
        LLon, LLat = np.meshgrid(lons, lats, indexing="ij")

        # ------------------------------------------------------------
        # 4) Convert to ECEF with altitude offset
        # ------------------------------------------------------------
        X, Y, Z = _geodet2ecef_grid(LLat, LLon, alt=alt)

        # ------------------------------------------------------------
        # 5) Build a structured grid (curved surface)
        # ------------------------------------------------------------
        surface_actor = None
        if fill:
            grid = pv.StructuredGrid(X, Y, Z)
            surface_actor = self.p.add_mesh(
                grid,
                color=fill_color,
                opacity=alpha,
                name=name + "_fill",
                smooth_shading=True,
                lighting=False,
                render=False,
            )

        # ------------------------------------------------------------
        # 6) Build outline (4 edges)
        # ------------------------------------------------------------
        # Extract the four edges from the structured grid
        # Bottom: lat_min
        bottom = np.c_[X[:, 0], Y[:, 0], Z[:, 0]]
        # Top: lat_max
        top = np.c_[X[:, -1], Y[:, -1], Z[:, -1]]
        # Left: lon_min
        left = np.c_[X[0, :], Y[0, :], Z[0, :]]
        # Right: lon_max
        right = np.c_[X[-1, :], Y[-1, :], Z[-1, :]]

        # Build a closed polyline
        outline_pts = np.vstack([bottom, right, top[::-1], left[::-1]])

        outline_poly = pv.lines_from_points(outline_pts, close=True)

        outline_actor = self.p.add_mesh(
            outline_poly,
            color=line_color,
            line_width=line_width,
            name=name + "_outline",
            opacity=1.0,
            lighting=False,
            smooth_shading=True,
            render=False,
        )

        # ------------------------------------------------------------
        # 7) Store bounds exactly like add_path
        # ------------------------------------------------------------
        self.data[name] = PlotData(
            name=name,
            type="box",
            actor=(surface_actor, outline_actor),
            lat_bounds=(lat_min, lat_max),
            lon_bounds=(lon_min, lon_max),
            data={
                "fill": fill,
                "fill_color": fill_color,
                "line_color": line_color,
                "alpha": alpha,
                "line_width": line_width,
                "resolution": resolution,
                "alt": alt,
            },
        )

    def add_circle(
        self,
        name: str,
        center: tuple[float, float],  # (lat, lon)
        radius_km: float,
        fill: bool = True,
        fill_color: str | int | None = None,
        line_color: str | int | None = None,
        alpha: float = 0.3,
        line_width: float = 2.0,
        resolution: int = 120,  # number of points around circle
        alt: float = 150.0,  # altitude offset in meters
    ):
        lat_c, lon_c = center

        # ------------------------------------------------------------
        # 1) Resolve colors (same logic as add_path)
        # ------------------------------------------------------------
        if isinstance(fill_color, int):
            fill_color = color_from_int(fill_color)
        elif fill_color is None:
            fill_color = _DEFAULT_COLOR_CYCLE[
                len(self.data) % len(_DEFAULT_COLOR_CYCLE)
            ]

        if isinstance(line_color, int):
            line_color = color_from_int(line_color)
        elif line_color is None:
            line_color = fill_color

        # ------------------------------------------------------------
        # 2) Build geodesic circle in lat/lon
        # ------------------------------------------------------------
        # Earth radius (WGS84 mean)
        R_earth = 6371.0  # km
        ang_radius = radius_km / R_earth  # radians

        # Sample angles around circle
        theta = np.linspace(0, 2 * np.pi, resolution)

        # Convert center to radians
        lat0 = np.radians(lat_c)
        lon0 = np.radians(lon_c)

        # Compute circle boundary using spherical law of cosines
        lat_boundary = np.arcsin(
            np.sin(lat0) * np.cos(ang_radius)
            + np.cos(lat0) * np.sin(ang_radius) * np.cos(theta)
        )

        lon_boundary = lon0 + np.arctan2(
            np.sin(theta) * np.sin(ang_radius) * np.cos(lat0),
            np.cos(ang_radius) - np.sin(lat0) * np.sin(lat_boundary),
        )

        # Convert back to degrees
        lat_boundary = np.degrees(lat_boundary)
        lon_boundary = np.degrees(lon_boundary)

        # Wrap longitudes to [-180, 180]
        lon_boundary = ((lon_boundary + 180) % 360) - 180

        # ------------------------------------------------------------
        # 3) Convert boundary to ECEF
        # ------------------------------------------------------------
        Xb, Yb, Zb = _geodet2ecef_grid(
            lat_boundary.reshape(-1, 1), lon_boundary.reshape(-1, 1), alt=alt
        )

        boundary_pts = np.c_[Xb.ravel(), Yb.ravel(), Zb.ravel()]

        # ------------------------------------------------------------
        # 4) Build outline polyline
        # ------------------------------------------------------------
        outline_poly = pv.lines_from_points(boundary_pts, close=True)

        outline_actor = self.p.add_mesh(
            outline_poly,
            color=line_color,
            line_width=line_width,
            name=name + "_outline",
            opacity=1.0,
            lighting=False,
            smooth_shading=True,
            render=False,
        )

        # ------------------------------------------------------------
        # 5) Build filled surface (optional)
        # ------------------------------------------------------------
        surface_actor = None
        if fill:
            # Triangulate the circle interior
            poly = pv.PolyData(boundary_pts)
            poly["scalars"] = np.arange(len(boundary_pts))  # dummy
            filled = poly.delaunay_2d()

            surface_actor = self.p.add_mesh(
                filled,
                color=fill_color,
                opacity=alpha,
                name=name + "_fill",
                smooth_shading=True,
                lighting=False,
                render=False,
            )

        # ------------------------------------------------------------
        # 6) Store bounds (lat/lon min/max)
        # ------------------------------------------------------------
        self.data[name] = PlotData(
            name=name,
            type="circle",
            actor=(surface_actor, outline_actor),
            lat_bounds=(lat_boundary.min(), lat_boundary.max()),
            lon_bounds=(lon_boundary.min(), lon_boundary.max()),
            data={
                "center": center,
                "radius_km": radius_km,
                "fill": fill,
                "fill_color": fill_color,
                "line_color": line_color,
                "alpha": alpha,
                "line_width": line_width,
                "resolution": resolution,
                "alt": alt,
            },
        )

    def _create_point_data(self, point: Point, size: int):
        sphere = pv.Sphere(radius=size, center=point.ecef[0])
        return sphere

    def update_point(self, name: str, point: Point):
        if name not in self.data:
            raise ValueError(f"Point '{name}' not found in plot data.")
        size = self.data[name].data.get("size", 2.0)
        sphere = self._create_point_data(point, size)
        act = self.data[name].actor
        act.mapper.SetInputData(sphere)
        act.mapper.Update()

    def clear_point(self, name: str):
        if name not in self.data:
            raise ValueError(f"Point '{name}' not found in plot data.")
        act = self.data[name].actor
        act.mapper.SetInputData(pv.PolyData())  # empty
        act.mapper.Update()

    def _create_ray_data(self, ray: Ray, length: float | NDArray) -> pv.PolyData:
        ray = ray.to_ecef()
        origins = ray.origin_rel
        directions = ray.unit_rel
        endpoints = origins + directions * np.array(length).reshape(-1, 1)
        points = np.vstack((origins, endpoints))
        n_rays = origins.shape[0]
        cells = np.c_[
            np.full(n_rays, 2), np.arange(n_rays), np.arange(n_rays) + n_rays
        ].ravel()
        lines = pv.PolyData(points, lines=cells)
        return lines

    def add_ray(
        self,
        name: str,
        ray: Ray,
        length: float | NDArray = 1e5,
        color: str | int = "grey",
        alpha: float = 0.5,
        line_width: float = 1.0,
    ):
        if isinstance(color, int):
            color = color_from_int(color)

        lines = self._create_ray_data(ray, length)
        origins_geodet = ray.to_ecef().origin_points.geodet
        act = self.p.add_mesh(
            lines,
            color=color,
            opacity=alpha,
            lighting=False,
            render=False,
            line_width=line_width,
        )
        self.data[name] = PlotData(
            name=name,
            type="ray",
            actor=act,
            lat_bounds=(
                np.min(origins_geodet[:, 0]),
                np.max(origins_geodet[:, 0]),
            ),
            lon_bounds=(
                np.min(origins_geodet[:, 1]),
                np.max(origins_geodet[:, 1]),
            ),
            data={"color": color, "alpha": alpha},
        )

    def update_ray(self, name: str, ray: Ray | None, length: float | NDArray = 1e5):
        act = self.data[name].actor
        if ray is None:
            act.mapper.SetInputData(pv.PolyData())  # empty
            return
        if name not in self.data:
            raise ValueError(f"Ray '{name}' not found in plot data.")
        lines = self._create_ray_data(ray, length)
        act.mapper.SetInputData(lines)
        act.mapper.Update()

    def add_labelled_point(
        self,
        name: str,
        text: str,
        position: Point,
        font_size: int = 14,
        text_color: str | int = "lightgrey",
        marker_color: str | int = "red",
        marker_size: float = 6.0,
        show_points: bool = True,
        bold: bool = False,
        always_visible: bool = True,
    ):
        if isinstance(text_color, int):
            text_color = color_from_int(text_color)
        if isinstance(marker_color, int):
            marker_color = color_from_int(marker_color)

        pt = pv.PolyData(position.ecef)

        act = self.p.add_point_labels(
            pt,
            [text],
            font_size=font_size,
            text_color=text_color,
            background_color="black",
            point_color=marker_color,
            reset_camera=False,
            show_points=show_points,
            shape_opacity=0.4,
            always_visible=always_visible,
            name=name,
            bold=bold,
            point_size=marker_size,
            shape_color="grey",
            margin=3,
        )
        self.text_actors[name] = act
        self.data[name] = PlotData(
            name=name,
            type="labelled_point",
            actor=act,
            lat_bounds=(position.geodet[0, 0], position.geodet[0, 0]),
            lon_bounds=(position.geodet[0, 1], position.geodet[0, 1]),
            data={"pt": pt},
        )

    def update_labelled_point_pos(self, name: str, position: Point):
        if name not in self.data:
            raise ValueError(f"Labelled point '{name}' not found in plot data.")
        pt = self.data[name].data["pt"]
        pt.points[0] = position.ecef
        pt.Modified()
        # act.mapper.SetInputData(pv.PolyData(position.ecef))
        # act.mapper.Update()

    def add_2d_text(
        self,
        name: str,
        text: str,
        pos: tuple[float, float] = (50, 50),
        font_size: int = 12,
        from_tl=True,
    ):
        if from_tl:
            # convert from bottom-left to top-left
            size = self.p.window_size
            pos = (pos[0], size[1] - pos[1])
        act = self.p.add_text(
            text,
            position=pos,
            font_size=font_size,
            color="white",
            shadow=False,
            font="courier",
            name=name,
        )
        self.text_actors[name] = act

    def update_2d_text(self, name: str, text: str):
        if name not in self.text_actors:
            raise ValueError(f"2D text '{name}' not found in text actors.")
        act = self.text_actors[name]
        act.SetInput(text)

    def add_legend(self, labels: list[tuple[str, str]], pos_x: float = 0.7):
        """Add a legend to the plot

        Args:
            labels: List of (data_name, label) tuples
            font_size: Font size for the legend text
        """
        legend_entries = []
        for data_name, label in labels:
            if data_name not in self.data:
                raise ValueError(f"Data '{data_name}' not found in plot data.")
            color = self.data[data_name].actor.GetProperty().GetColor()
            legend_entries.append([label, color])
        txts = []
        y0 = self.p.window_size[1] - 50
        x0 = int(pos_x * self.p.window_size[0])
        line_h = 30
        for i, (label, rgb) in enumerate(legend_entries):
            t = self.p.add_text(
                "- " + label.split("-")[-1],
                position=(x0, y0 - i * line_h),
                font_size=10,
                color=rgb,
                font="courier",
            )
            # t.SetInput("■  " + label.split()[-1])  # content update later via SetInput
            txts.append(t)

    def calc_bounds(
        self, buffer=0.3
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        lat_mins = [d.lat_bounds[0] for d in self.data.values()]
        lat_maxs = [d.lat_bounds[1] for d in self.data.values()]
        lon_mins = [d.lon_bounds[0] for d in self.data.values()]
        lon_maxs = [d.lon_bounds[1] for d in self.data.values()]

        return (min(lat_mins) - buffer, max(lat_maxs) + buffer), (
            min(lon_mins) - buffer,
            max(lon_maxs) + buffer,
        )

    def autocomplete(
        self, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    ):
        if bounds is not None:
            lat_bounds, lon_bounds = bounds
        else:
            lat_bounds, lon_bounds = self.calc_bounds(buffer=0.4)
        self.add_texture(lat_bounds, lon_bounds)
        if bounds is None:
            lat_bounds, lon_bounds = self.calc_bounds(buffer=0.0)
        self.add_grid(lat_bounds, lon_bounds)

    def render(self):
        self.p.render()

    def save(self, filepath: str = "./plot3d_screenshot.png", scale: float = 1.0):
        old_size = self.p.window_size
        self.p.window_size = (
            int(np.round(old_size[0] * scale)),
            int(np.round(old_size[1] * scale)),
        )
        self.p.screenshot(filepath)
        self.p.window_size = old_size

    def show(self):
        self.p.show()

    def start_animation(self, filepath: str = "./animation.mp4", fps: int = 30):
        self.p.open_movie(filepath, framerate=fps)  # uses imageio/ffmpeg under the hood

    def frame(self):
        self.p.write_frame()

    def close(self):
        self.p.close()
