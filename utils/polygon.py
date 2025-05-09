import numpy as np
import cupy as cp
from typing import Tuple
from skimage.draw import polygon
from transformations import choose_random_rotate
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
    # 1) choose random scale
    s = np.random.uniform(*scale_range)

    # 2) choose a random principal‐axis rotation
    angle = np.random.uniform(0, 2 * np.pi)
    R, axis_name = choose_random_rotate(angle)
    # 3) random translation around center
    T = center + np.random.normal(scale=cluster_radius, size=3)

    # 4) apply scale, rotate, translate
    pts3d = apply_transform(unit, R, T, s)
    base3d, apex3d = pts3d[:-1], pts3d[-1]
    shp = Pyramid(apex3d, base3d)

    # 5) compute face normals and offsets
    normals = np.stack([
        np.cross(v2 - v1, v3 - v1) *
        (1 if np.dot(np.cross(v2 - v1, v3 - v1),
                     pts3d.mean(axis=0) - v1) >= 0 else -1)
        for v1, v2, v3 in shp.faces
    ])
    ds = -np.einsum('ij,ij->i', normals, np.array([v1 for v1, _, _ in shp.faces]))

    # 6) compute axis-aligned bbox
    mins = np.floor(pts3d.min(axis=0)).astype(int)
    maxs = np.ceil(pts3d.max(axis=0)).astype(int)
    D, H, W = volume.shape
    x0, x1 = max(0, mins[0]), min(W - 1, maxs[0])
    y0, y1 = max(0, mins[1]), min(H - 1, maxs[1])
    z0, z1 = max(0, mins[2]), min(D - 1, maxs[2])

    # 7) rasterize on GPU
    gz = cp.arange(z0, z1 + 1)
    gy = cp.arange(y0, y1 + 1)
    gx = cp.arange(x0, x1 + 1)
    Z, Y, X = cp.meshgrid(gz, gy, gx, indexing='ij')
    coords = cp.stack((X, Y, Z), axis=-1)

    cn  = cp.asarray(normals)[:, None, None, None, :]
    cds = cp.asarray(ds)[:, None, None, None]
    lhs = (cn * coords[None, ...]).sum(axis=-1)
    mask = cp.all(lhs + cds >= 0, axis=0)

    # 8) write back to CPU volume
    mask_cpu = cp.asnumpy(mask)
    volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1][mask_cpu] = 1
    return angle, axis_name

# ─── Hexagon Placement in Free Space ───────────────────────────────────────────

def place_hexagon_in_free(volume: np.ndarray,
                          size_range:   Tuple[int, int] = (80, 130),
                          height_range: Tuple[int, int] = (80, 130),
                          padding:      int            = 0):
    """
    Carve the largest empty pocket, then place a fully-visible, randomly-rotated
    vertical hexagonal prism there, with extra `padding` added to its radius.
    """
    # 1) Largest empty pocket center & radius
    free = (volume == 0).astype(np.uint8)
    dist = ndi.distance_transform_edt(free)
    zc_o, yc_o, xc_o = np.unravel_index(np.argmax(dist), dist.shape)
    max_r = dist[zc_o, yc_o, xc_o]
    if max_r < size_range[0]:
        print("Hexagon placement failed (pocket too small)")
        return

    # 2) choose size & height (clamped by pocket)
    size   = min(np.random.randint(*size_range),   int(max_r))
    height = min(np.random.randint(*height_range), int(max_r))
    size_eff = size + padding

    D, H, W = volume.shape
    # 3) clamp the center so shape fits entirely in [0..W-1]×[0..H-1]×[0..D-1]
    xc = int(np.clip(xc_o, size_eff,      W - 1 - size_eff))
    yc = int(np.clip(yc_o, size_eff,      H - 1 - size_eff))
    zc = int(np.clip(zc_o, 0,             D - 1 - height))

    # 4) Build & rotate hexagon vertices in XY-plane
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    verts = np.column_stack([
        xc + size_eff * np.cos(angles),
        yc + size_eff * np.sin(angles)
    ])  # (6,2)

    theta = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[ c, -s],[ s,  c]])
    centered = verts - np.array([xc, yc])
    verts_rot = (centered @ R2.T) + np.array([xc, yc])

    # 5) full (unclamped) 2D bounding box of rotated hexagon
    x_min = int(np.floor (verts_rot[:,0].min()))
    x_max = int(np.ceil  (verts_rot[:,0].max()))
    y_min = int(np.floor (verts_rot[:,1].min()))
    y_max = int(np.ceil  (verts_rot[:,1].max()))
    mask_w = x_max - x_min + 1
    mask_h = y_max - y_min + 1

    # 6) rasterize full hexagon into 2D mask
    local = verts_rot - np.array([x_min, y_min])
    rr, cc = polygon(local[:,1], local[:,0],
                     shape=(mask_h, mask_w))
    mask2d = np.zeros((mask_h, mask_w), dtype=bool)
    mask2d[rr, cc] = True

    # 7) extrude along Z
    z0, z1 = zc, zc + height
    depth = z1 - z0 + 1
    mask3d = np.broadcast_to(mask2d, (depth, mask_h, mask_w))

    # 8) clamp that bbox to volume bounds
    x0 = max(0, x_min); x1 = min(W - 1, x_max)
    y0 = max(0, y_min); y1 = min(H - 1, y_max)
    sx = x0 - x_min; sy = y0 - y_min
    ex = sx + (x1 - x0); ey = sy + (y1 - y0)

    # 9) carve out & fill only the overlap
    volume[z0:z1+1, y0:y1+1, x0:x1+1] = 0
    submask = mask3d[:, sy:ey+1, sx:ex+1]
    volume[z0:z1+1, y0:y1+1, x0:x1+1][submask] = 1

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
        scales = ((170, 250), (170, 250))
        D, H, W = volume.shape
        center = np.array([W/2, H/2, D/2], float)
        angle1, axis_name_1 = place_pyramid(volume, unit_rect_apex, unit_rect_base,
                      scales[0], center, cluster_radius)
        print("Pyramid 1 placed")
        angle2, axis_name_2 = place_pyramid(volume, unit_tri_apex, unit_tri_base,
                      scales[1], center, cluster_radius)
        print("Pyramid 2 placed")
        place_hexagon_in_free(volume)
        print("Hexagon placed")
        return volume, {
            "pyramid1": {
                "angle": angle1,
                "axis": axis_name_1
            },
            "pyramid2": {
                "angle": angle2,
                "axis": axis_name_2
            }
        }
