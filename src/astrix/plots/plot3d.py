# pyright: standard
# pyright: reportArgumentType = false

from __future__ import annotations
import pyvista as pv
import numpy as np
from typing import Sequence
from dataclasses import dataclass
from numpy.typing import NDArray
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import contextily as cx
from scipy.spatial.transform import Rotation

from astrix.functs import geodet2ecef, ned_rotation
from astrix.spatial.location import Path, Point
from astrix.spatial.frame import Frame
from astrix.time import Time, TimeGroup, time_linspace

_DEFAULT_COLOR_CYCLE = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
]


def _geodet2ecef_grid(
    llat: NDArray, llon: NDArray, alt: float = 0
) -> tuple[NDArray, NDArray, NDArray]:
    geodet = np.stack([llat.ravel(), llon.ravel(), np.full(llat.size, alt)], axis=1)
    ecef = geodet2ecef(geodet)
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
        mask = (self.points_1.time.unix >= start.unix) & (self.points_1.time.unix <= end.unix) 
        return ConnectingLines(self.points_1[mask], self.points_2[mask])


@dataclass
class PlotData:
    name: str
    type: str  # 'point', 'path', 'frame'
    actor: pv.Actor
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]


class Plot3D:
    p: pv.Plotter
    data: dict[str, PlotData]
    text_actors: dict[str, pv.Actor]

    def __init__(self):
        self.p = pv.Plotter()
        self.p.set_background("black")  # pyright: ignore
        self.data = {}
        self.text_actors = {}

    def set_view(
        self,
        cent: Point,
        heading_deg: float = 180,
        pitch_deg: float = -45,
    ):
        ecef_cent = cent.ecef[0]
        rot_ned = ned_rotation(cent.geodet)
        rot_cam = Rotation.from_euler("ZY", [heading_deg, pitch_deg], degrees=True)
        rot = rot_ned * rot_cam
        cam_dir = rot.apply(np.array([1, 0, 0]))
        cam_pos = ecef_cent + cam_dir * 2e5  # 100 km

        self.p.enable_parallel_projection() # pyright: ignore
        self.p.set_focus(ecef_cent)
        self.p.set_position(cam_pos)
        if pitch_deg == 90 or pitch_deg == -90:
            # looking straight up or down, set_viewup() has no effect
            self.p.set_viewup(rot_ned.apply(np.array([1.0, 0, 0])))
        else:
            self.p.set_viewup(rot_ned.apply(np.array([0.0, 0, -1])))
        self.p.camera.zoom(1.5)

    def add_texture(self, lat_bounds: Sequence[float], lon_bounds: Sequence[float], alpha=0.6):
        img, ext = cx.bounds2img(
            lon_bounds[0],
            lat_bounds[0],
            lon_bounds[1],
            lat_bounds[1],
            zoom="auto",
            source=cx.providers.Esri.WorldImagery,  # pyright: ignore
            ll=True,
        )
        img = np.flip(img, axis=0)  # flip y axis

        dst_crs = CRS.from_string("EPSG:4326")  # WGS84
        transform_dst = from_bounds(
            lon_bounds[0],
            lat_bounds[0],
            lon_bounds[1],
            lat_bounds[1],
            img.shape[1],
            img.shape[0],
        )
        transform_src = from_bounds(
            ext[0], ext[2], ext[1], ext[3], img.shape[1], img.shape[0]
        )
        dst = np.empty_like(img)
        for i, band in enumerate(img.transpose(2, 0, 1)):  # loop over bands
            dst_band = np.empty_like(band)
            reproject(
                band,
                dst_band,
                src_transform=transform_src,
                src_crs=CRS.from_string("EPSG:3857"),  # Web Mercator
                dst_transform=transform_dst,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            dst[:, :, i] = dst_band

        # 3) Build a WGS-84 curved patch and drape the reprojected texture
        n_lat, n_lon = 400, 400
        lats = np.linspace(lat_bounds[0], lat_bounds[1], n_lat)
        lons = np.linspace(lon_bounds[0], lon_bounds[1], n_lon)
        llon, llat = np.meshgrid(lons, lats, indexing="xy")
        X, Y, Z = _geodet2ecef_grid(llat, llon)
        grid = pv.StructuredGrid(X, Y, Z)  # creates a surface grid

        # Plate Carrée texture coords match lon/lat linearly
        u = (llon - lon_bounds[0]) / (lon_bounds[1] - lon_bounds[0])
        v = 1.0 - (llat - lat_bounds[0]) / (lat_bounds[1] - lat_bounds[0])
        tcoords = np.c_[u.ravel(order="F"), v.ravel(order="F")].astype(np.float32)
        grid.active_texture_coordinates = tcoords

        tex = pv.Texture(dst)  # pyright: ignore
        tex.repeat = False

        self.p.add_mesh(
            grid,
            texture=tex,
            lighting=True,
            smooth_shading=True,
            show_edges=False,
            opacity=alpha,
        )

    def add_grid(self, lat_bounds: Sequence[float], lon_bounds: Sequence[float]):
        # Get grid lines to nearest 0.5 degree
        lat_min = np.floor(lat_bounds[0] * 2) / 2
        lat_max = np.ceil(lat_bounds[1] * 2) / 2
        lon_min = np.floor(lon_bounds[0] * 2) / 2
        lon_max = np.ceil(lon_bounds[1] * 2) / 2
        n_lat = int((lat_max - lat_min) * 2) + 1
        n_lon = int((lon_max - lon_min) * 2) + 1
        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)

        LLon, LLat = np.meshgrid(lons, lats, indexing="xy")
        X, Y, Z = _geodet2ecef_grid(LLat, LLon, alt=1)
        sg = pv.StructuredGrid(X, Y, Z)  # creates a surface grid
        edges = sg.extract_all_edges()  # just the lines between nodes
        self.p.add_mesh(edges, color="grey", line_width=1, opacity=0.5)

        def fmt_lon(deg):
            s = "E" if deg >= 0 else "W"
            return f"{abs(deg):.1f}°{s}"

        def fmt_lat(deg):
            s = "N" if deg >= 0 else "S"
            return f"{abs(deg):.1f}°{s}"

        lon_label_points = np.c_[X[0, ::1], Y[0, ::1], Z[0, ::1]]
        lon_labels = [fmt_lon(lon) for lon in lons[::1]]
        lat_label_points = np.c_[X[::1, -1], Y[::1, -1], Z[::1, -1]]
        lat_labels = [fmt_lat(lat) for lat in lats[::1]]

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

    def add_path(
        self, name: str, path: Path, line_width: float = 2.0, color: str | None = None, alpha: float = 1.0
    ):
        if color is None:
            color = _DEFAULT_COLOR_CYCLE[len(self.data) % len(_DEFAULT_COLOR_CYCLE)]
        curve = pv.lines_from_points(path.ecef)
        act = self.p.add_mesh(curve, color=color, line_width=line_width, name=name, opacity=1)
        geodet = path.points.geodet
        self.data[name] = PlotData(
            name=name,
            type="path",
            actor=act,
            lat_bounds=(np.min(geodet[:, 0]), np.max(geodet[:, 0])),
            lon_bounds=(np.min(geodet[:, 1]), np.max(geodet[:, 1])),
        )

    # def add_multi_lines(self, name: str, Path, line_width: float = 1.0):

    def add_2d_text(
        self,
        name: str,
        test: str,
        position: tuple[float, float] = (10, 10),
        font_size: int = 12,
        color: str = "white",
    ):
        act = self.p.add_text(
            test,
            position=position,
            font_size=font_size,
            color=color,
            shadow=True,
            font="arial",
            name=name,
        )
        self.text_actors[name] = act

    def calc_bounds(
        self, buffer=0.5
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        lat_mins = [d.lat_bounds[0] for d in self.data.values()]
        lat_maxs = [d.lat_bounds[1] for d in self.data.values()]
        lon_mins = [d.lon_bounds[0] for d in self.data.values()]
        lon_maxs = [d.lon_bounds[1] for d in self.data.values()]

        return (min(lat_mins) - buffer, max(lat_maxs) + buffer), (
            min(lon_mins) - buffer,
            max(lon_maxs) + buffer,
        )

    def autocomplete(self):
        lat_bounds, lon_bounds = self.calc_bounds()
        self.add_texture(lat_bounds, lon_bounds)
        self.add_grid(lat_bounds, lon_bounds)

    def render(self):
        self.p.render()

    def show(self):
        self.p.show()


if __name__ == "__main__":
    # --- region (example) ---
    lat_bounds = (-29.2, -20.4)
    lon_bounds = (148.1, 157.3)

    plot = Plot3D()
    plot.add_texture(lat_bounds, lon_bounds)
    plot.add_grid(lat_bounds, lon_bounds)
    cent_pt = Point.from_geodet([-24.5, 152.7, 0])
    plot.set_view(cent_pt, heading_deg=180, pitch_deg=45)
    plot.show()
