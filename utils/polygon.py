import numpy as np
import cupy as cp
from typing import Tuple
import scipy.ndimage as ndi

# ─── Utility Functions ─────────────────────────────────────────────────────────

def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """
    2D point-in-polygon test (ray-casting algorithm).
    poly: array of shape (n,2)
    """
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

# ─── Shape Classes ─────────────────────────────────────────────────────────────

class Pyramid:
    def __init__(self, apex: np.ndarray, base: np.ndarray):
        """
        apex: (3,) float
        base: (m,3) float, m>=3, ordered CCW as seen from outside
        """
        self.apex = apex
        self.base = base
        m = base.shape[0]
        self.faces = []
        # triangulate base (fan around base[0])
        for i in range(1, m - 1):
            self.faces.append((base[0], base[i], base[i + 1]))
        # lateral faces
        for i in range(m):
            v1 = base[i]
            v2 = base[(i + 1) % m]
            self.faces.append((apex, v1, v2))

# ─── Transforms ────────────────────────────────────────────────────────────────

def random_rotation_matrix() -> np.ndarray:
    u1, u2, u3 = np.random.rand(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1)       * np.sin(2 * np.pi * u3),
        np.sqrt(u1)       * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def apply_transform(pts: np.ndarray,
                    R: np.ndarray,
                    T: np.ndarray,
                    scale: float) -> np.ndarray:
    """Scale about origin, then rotate, then translate."""
    return (R @ (pts.T * scale)).T + T

# ─── Fast GPU Rasterization ───────────────────────────────────────────────────

def place_pyramid(volume: np.ndarray,
                  apex: np.ndarray,
                  base: np.ndarray,
                  scale_range: Tuple[float, float],
                  center: np.ndarray,
                  cluster_radius: float):
    unit = np.vstack([base, apex[np.newaxis, :]])
    # transform
    s = np.random.uniform(*scale_range)
    R = random_rotation_matrix()
    T = center + np.random.normal(scale=cluster_radius, size=3)
    pts3d = apply_transform(unit, R, T, s)
    base3d, apex3d = pts3d[:-1], pts3d[-1]
    shp = Pyramid(apex3d, base3d)
    # normals & offsets
    normals = np.stack([
        np.cross(v2 - v1, v3 - v1) * (1 if np.dot(np.cross(v2 - v1, v3 - v1), pts3d.mean(axis=0) - v1) >= 0 else -1)
        for v1, v2, v3 in shp.faces
    ])
    ds = -np.einsum('ij,ij->i', normals, np.array([v1 for v1, _, _ in shp.faces]))
    # bbox clamp
    mins = np.floor(pts3d.min(axis=0)).astype(int)
    maxs = np.ceil(pts3d.max(axis=0)).astype(int)
    D, H, W = volume.shape
    x0, x1 = max(0, mins[0]), min(W - 1, maxs[0])
    y0, y1 = max(0, mins[1]), min(H - 1, maxs[1])
    z0, z1 = max(0, mins[2]), min(D - 1, maxs[2])
    # GPU grid
    gz = cp.arange(z0, z1 + 1)
    gy = cp.arange(y0, y1 + 1)
    gx = cp.arange(x0, x1 + 1)
    Z, Y, X = cp.meshgrid(gz, gy, gx, indexing='ij')
    coords = cp.stack((X, Y, Z), axis=-1)
    cn = cp.asarray(normals)[:, None, None, None, :]
    cds = cp.asarray(ds)[:, None, None, None]
    lhs = (cn * coords[None, ...]).sum(axis=-1)
    mask = cp.all(lhs + cds >= 0, axis=0)
    mask_cpu = cp.asnumpy(mask)
    volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1][mask_cpu] = 1

# ─── Hexagon Placement in Free Space ───────────────────────────────────────────

def place_hexagon_in_free(volume: np.ndarray,
                           size_range: Tuple[int, int] = (5, 15),
                           height_range: Tuple[int, int] = (5, 15)):
    """
    Find the largest empty pocket via distance transform, carve it out,
    and place one hexagonal prism there for guaranteed visibility.
    """
    free = (volume == 0).astype(np.uint8)
    dist = ndi.distance_transform_edt(free)
    idx = np.unravel_index(np.argmax(dist), dist.shape)
    zc, yc, xc = idx
    max_r = dist[idx]
    if max_r < size_range[0]:
        return
    size = min(np.random.randint(size_range[0], size_range[1] + 1), int(max_r))
    height = min(np.random.randint(height_range[0], height_range[1] + 1), int(max_r))
    D, H, W = volume.shape
    y0, y1 = max(0, int(yc - size)), min(H - 1, int(yc + size))
    x0, x1 = max(0, int(xc - size)), min(W - 1, int(xc + size))
    z0, z1 = zc, min(D - 1, zc + height)
    volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 0
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    verts = np.vstack([xc + size * np.cos(angles), yc + size * np.sin(angles)]).T
    for z in range(z0, z1 + 1):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x - xc) ** 2 + (y - yc) ** 2 <= size ** 2:
                    if point_in_polygon(x + 0.5, y + 0.5, verts):
                        volume[z, y, x] = 1

# ─── Canonical Pyramids ────────────────────────────────────────────────────────

unit_rect_base = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
unit_rect_apex = np.array([0, 0, 1], float)

h = np.sqrt(3) / 2
unit_tri_base = np.array([[-1, 0, 0], [1, 0, 0], [0, 2 * h, 0]], float)
unit_tri_apex = np.array([0, 0, 1], float)

# ─── Scene Generator ─────────────────────────────────────────────────────────

class PolygonSceneGenerator:
    def generate_dataset(self,
                         volume: np.ndarray,
                         cluster_radius: float = 30.0
                         ) -> np.ndarray:
        scales = ((180, 260), (180, 260))
        D, H, W = volume.shape
        center = np.array([W/2, H/2, D/2], float)
        place_pyramid(volume, unit_rect_apex, unit_rect_base,
                      scales[0], center, cluster_radius)
        place_pyramid(volume, unit_tri_apex, unit_tri_base,
                      scales[1], center, cluster_radius)
        place_hexagon_in_free(volume)
        return volume
