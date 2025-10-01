import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

from astrix._backend_utils import ArrayLike
from astrix.functs import ensure_1d
from astrix.spatial.location import Path, Point
from astrix.spatial.ray import Ray

class CartPlot:
    def __init__(self, width: float = 12.0):

        self.fig = plt.figure()
        m = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        m.set_box_aspect(1)
        self.fig.set_figwidth(width)
        m.set_box_aspect(0.6)
        m.set_aspect(1)

        gl = m.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            linestyle="--",
            zorder=0,
        )
        m.add_feature(cartopy.feature.BORDERS)
        m.add_feature(cartopy.feature.COASTLINE)
        m.add_feature(cartopy.feature.STATES)
        m.add_feature(cartopy.feature.LAND)
        m.add_feature(cartopy.feature.LAKES)
        m.add_feature(cartopy.feature.RIVERS)
        m.add_feature(cartopy.feature.OCEAN)
        m.add_feature(cartopy.feature.STATES)
        m.coastlines()

        self.m = m
        self.gl = gl

    def add_path(self, path: Path, color: str="r", style: str="-", label:str | None=None):
        """Add a path to the map.
        Args:
            path (Path): The path to add.
            color (str, optional): The color of the path. Defaults to "r".
            style (str, optional): The style of the path. Defaults to "-".
        """
        pts_geo = path.points.geodet
        self.m.plot(pts_geo[:, 1], pts_geo[:, 0], style, color=color, linewidth=1)
        if label is not None:
            mid_idx = len(pts_geo) // 2
            self.m.text(pts_geo[mid_idx, 1] + 0.005, pts_geo[mid_idx, 0] + 0.005, label, size=12)

    def add_point(self, point: Point, color: str="r", marker: str="o", ms: float=6., label: str | None =None):
        pts_geo = point.geodet
        self.m.plot(pts_geo[:,1], pts_geo[:,0], marker, color=color, ms=ms)
        if label is not None:
            self.m.text(point.geodet[1] + 0.05, point.geodet[0, 0], label, size=12)

    def add_vectors(self, points1: Point, points2: Point, color: str="grey"):
        for pts_geo1, pts_geo2 in zip(points1.geodet, points2.geodet):
            self.m.plot(
                [pts_geo1[1], pts_geo2[1]],
                [pts_geo1[0], pts_geo2[0]],
                linewidth=0.5,
                color=color,
            )

    def add_rays(self, rays: Ray, length: ArrayLike = 2e5, color: str="g"):
        length = ensure_1d(length)
        if len(length) not in (1, len(rays)):
            raise ValueError("Length must be singular or the same length as rays to plot")
        ray_ecef = rays.to_ecef()
        pts_start = Point(ray_ecef.origin_rel)
        pts_end = Point(ray_ecef.origin_rel + ray_ecef.unit_rel * length.reshape(-1, 1))
        self.add_vectors(pts_start, pts_end, color=color)



    def show(self):
        plt.show()
