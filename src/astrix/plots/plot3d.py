# pyright: standard
# pyright: reportArgumentType = false

from __future__ import annotations
import pyvista as pv
import numpy as np
from typing import Sequence
from dataclasses import dataclass, field
from numpy.typing import NDArray
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import contextily as cx
from scipy.spatial.transform import Rotation
from pyproj import Transformer

from astrix.functs import geodet2ecef, ned_rotation
from astrix.spatial.location import Path, Point
from astrix.spatial.frame import Frame
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

    def __init__(self):
        self.p = pv.Plotter()
        self.p.window_size = [1200, 600]
        self.p.set_background("black")  # pyright: ignore
        self.p.disable_anti_aliasing()
        self.p.enable_anti_aliasing("fxaa")
        # self.p.enable_anti_aliasing("msaa", multi_samples=8)
        # self.p.enable_anti_aliasing(None)
        self.p.enable_depth_peeling(occlusion_ratio=0.0, number_of_peels=50)
        self.data = {}
        self.text_actors = {}

    def set_view(
        self,
        cent: Point,
        heading_deg: float = 180,
        pitch_deg: float = -45,
        zoom: float = 2.5,
    ):
        ecef_cent = cent.ecef[0]
        rot_ned = ned_rotation(cent.geodet)
        rot_cam = Rotation.from_euler("ZY", [heading_deg, pitch_deg], degrees=True)
        rot = rot_ned * rot_cam
        cam_dir = rot.apply(np.array([1, 0, 0]))
        cam_pos = ecef_cent + cam_dir * 2e5  # 100 km

        self.p.enable_parallel_projection()  # pyright: ignore
        self.p.set_focus(ecef_cent)
        self.p.set_position(cam_pos)
        if pitch_deg == 90 or pitch_deg == -90:
            # looking straight up or down, set_viewup() has no effect
            self.p.set_viewup(rot_ned.apply(np.array([1.0, 0, 0])))
        else:
            self.p.set_viewup(rot_ned.apply(np.array([0.0, 0, -1])))
        self.p.camera.zoom(zoom)

    def add_texture(
        self, lat_bounds: Sequence[float], lon_bounds: Sequence[float], alpha=0.6
    ):
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

        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        l_bounds = transformer.transform_bounds(
            ext[0], ext[2], ext[1], ext[3], densify_pts=21
        )

        # 3) Build a WGS-84 curved patch and drape the reprojected texture
        n_lat, n_lon = 20, 20
        lats = np.linspace(l_bounds[1], l_bounds[3], n_lat)
        lons = np.linspace(l_bounds[0], l_bounds[2], n_lon)
        llon, llat = np.meshgrid(lons, lats, indexing="ij")

        to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xm, ym = to_merc.transform(llon, llat)  # same shapes as llon/llat

        xmin, xmax, ymin, ymax = ext
        u = (xm - xmin) / (xmax - xmin)
        v = 1.0 - (ym - ymin) / (ymax - ymin)  # invert vertical

        X, Y, Z = _geodet2ecef_grid(llat, llon)
        grid = pv.StructuredGrid(X, Y, Z)  # creates a surface grid

        tcoords = np.c_[u.ravel(order="F"), v.ravel(order="F")].astype(np.float32)
        grid.active_texture_coordinates = tcoords

        tex = pv.Texture(img)  # pyright: ignore
        tex.repeat = False

        act = self.p.add_mesh(
            grid,
            texture=tex,
            lighting=True,
            smooth_shading=True,
            show_edges=False,
            opacity=alpha,
        )

        self.data["texture"] = PlotData(
            "texture",
            "texture",
            act,
            (l_bounds[1], l_bounds[3]),
            (l_bounds[0], l_bounds[2]),
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
        X, Y, Z = _geodet2ecef_grid(LLat, LLon, alt=10)
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
            render_lines_as_tubes=True,
            smooth_shading=True,
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

    def add_ground_track(
        self,
        name: str,
        path: Path,
        vert_dt: float = 10.0,
        line_width: float = 1.0,
        color: str = "white",
        alpha: float = 0.6,
    ):
        geodet = path.points.geodet
        geodet[:, 2] = 0.0  # set altitude to 0
        path_gt = Path(Point.from_geodet(geodet, time=path.time))

        # Get times of vertical path joins
        # We want this to start at first time and be spaced strictly at vert_dt intervals,
        # with a final time at the end
        n_bars = int(np.floor((path.time.unix[-1] - path.time.unix[0]) / vert_dt))
        bar_t = np.concatenate(
            [path.time.unix[0] + np.arange(n_bars) * vert_dt, [path.time.unix[-1]]]
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
        )
        self.data[name] = PlotData(
            name=name,
            type="ground_track",
            actor=act,
            lat_bounds=(np.min(geodet[:, 0]), np.max(geodet[:, 0])),
            lon_bounds=(np.min(geodet[:, 1]), np.max(geodet[:, 1])),
        )

    def _create_point_data(self, point: Point, size: int):
        sphere = pv.Sphere(radius=size, center=point.ecef[0])
        return sphere

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

        sphere = self._create_point_data(point, size * 1000)
        act = self.p.add_mesh(
            sphere,
            color=color,
            name=name,
            opacity=alpha,
            lighting=False,
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

    def update_point(self, name: str, point: Point):
        if name not in self.data:
            raise ValueError(f"Point '{name}' not found in plot data.")
        size = self.data[name].data.get("size", 5.0)
        sphere = self._create_point_data(point, size * 1000)
        act = self.data[name].actor
        act.mapper.SetInputData(sphere)
        act.mapper.Update()

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

    def add_labelled_point(
        self,
        name: str,
        text: str,
        position: Point,
        font_size: int = 12,
        color: str | int = "lightgrey",
    ):
        act = self.p.add_point_labels(
            position.ecef,
            [text],
            font_size=font_size,
            text_color=color,
            show_points=True,
            shape_opacity=0.0,
            always_visible=True,
            name=name,
            bold=False,
            point_size=6,
            shape_color="grey",
            margin=2,
            
        )
        self.text_actors[name] = act

    def add_legend(self, labels: list[tuple[str, str]]):
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
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            legend_entries.append([label, color_hex])
        self.p.add_legend(
            legend_entries, bcolor="black", size=(0.15, 0.15), face="rectangle"
        )

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

    def autocomplete(
        self, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    ):
        if bounds is not None:
            lat_bounds, lon_bounds = bounds
        else:
            lat_bounds, lon_bounds = self.calc_bounds()
        self.add_texture(lat_bounds, lon_bounds)
        lat_bounds, lon_bounds = self.calc_bounds()
        self.add_grid(lat_bounds, lon_bounds)

    def render(self):
        self.p.render()

    def show(self):
        self.p.show()

    def start_animation(self, filepath: str = "./animation.mp4", fps: int = 10):
        self.p.open_movie(filepath, framerate=fps)  # uses imageio/ffmpeg under the hood

    def frame(self):
        self.p.write_frame()

    def close(self):
        self.p.close()


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
