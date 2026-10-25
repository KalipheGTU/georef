"""Géoréférencement + génération de tuiles XYZ, sans GDAL (numpy + Pillow)."""
import math, os, sqlite3
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000
R = 6378137.0
ORIGIN = math.pi * R
MIN_PTS = {"affine": 3, "poly2": 6, "tps": 4}

def ll2m(lon, lat):
    return lon * ORIGIN / 180, math.log(math.tan((90 + lat) * math.pi / 360)) * R

def m2ll(x, y):
    return x / ORIGIN * 180, math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)

class Model:
    """Transformation 2D -> 2D ajustée par moindres carrés (affine, poly2) ou thin plate spline."""
    def __init__(self, src, dst, method):
        self.method = method
        src, dst = np.asarray(src, float), np.asarray(dst, float)
        self.mu = src.mean(0)
        self.sc = max(float(src.std(0).max()), 1e-9)
        s = (src - self.mu) / self.sc
        self.dmu = dst.mean(0)
        self.dsc = max(float(dst.std(0).max()), 1e-9)
        d = (dst - self.dmu) / self.dsc
        if method == "tps":
            n = len(s)
            A = np.zeros((n + 3, n + 3))
            A[:n, :n] = self._k(np.linalg.norm(s[:, None] - s[None], axis=2))
            A[:n, n] = 1; A[:n, n + 1:] = s
            A[n, :n] = 1; A[n + 1:, :n] = s.T
            b = np.zeros((n + 3, 2)); b[:n] = d
            self.s = s
            self.w = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            self.w = np.linalg.lstsq(self._design(s), d, rcond=None)[0]

    @staticmethod
    def _k(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r == 0, 0, r * r * np.log(r * r))

    def _design(self, s):
        x, y = s[:, 0], s[:, 1]
        cols = [np.ones_like(x), x, y]
        if self.method == "poly2": cols += [x * x, x * y, y * y]
        return np.stack(cols, 1)

    def __call__(self, pts):
        s = (np.asarray(pts, float) - self.mu) / self.sc
        if self.method == "tps":
            n = len(self.s)
            K = self._k(np.linalg.norm(s[:, None] - self.s[None], axis=2))
            out = K @ self.w[:n] + self.w[n] + s @ self.w[n + 1:]
        else:
            out = self._design(s) @ self.w
        return out * self.dsc + self.dmu

def rmse(gcps, method):
    """Erreur résiduelle (en pixels de l'image source) par point."""
    px = [(g["px"], g["py"]) for g in gcps]
    mm = [ll2m(g["lon"], g["lat"]) for g in gcps]
    inv = Model(mm, px, method)
    err = np.linalg.norm(inv(mm) - np.array(px), axis=1)
    return err.tolist()

def make_tiles(img_path, gcps, method, out_dir, progress=lambda f: None, max_tiles=30000):
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    px = [(g["px"], g["py"]) for g in gcps]
    mm = [ll2m(g["lon"], g["lat"]) for g in gcps]
    fwd, inv = Model(px, mm, method), Model(mm, px, method)

    t = np.linspace(0, 1, 30)
    edge = np.concatenate([np.stack([t * w, 0 * t], 1), np.stack([t * w, 0 * t + h], 1),
                           np.stack([0 * t, t * h], 1), np.stack([0 * t + w, t * h], 1)])
    e = fwd(edge)
    minx, miny, maxx, maxy = e[:, 0].min(), e[:, 1].min(), e[:, 0].max(), e[:, 1].max()

    mpp = max((maxx - minx) / w, (maxy - miny) / h)
    zmax = min(20, max(0, math.ceil(math.log2(2 * ORIGIN / 256 / mpp))))

    def rng(z):
        size = 2 * ORIGIN / 2 ** z
        return (int((minx + ORIGIN) // size), int((maxx + ORIGIN) // size),
                int((ORIGIN - maxy) // size), int((ORIGIN - miny) // size))
    def count(a, b):
        n = 0
        for z in range(a, b + 1):
            x0, x1, y0, y1 = rng(z); n += (x1 - x0 + 1) * (y1 - y0 + 1)
        return n
    while zmax > 0 and count(max(0, zmax - 6), zmax) > max_tiles: zmax -= 1
    zmin = max(0, zmax - 6)
    total, done = count(zmin, zmax), 0

    os.makedirs(out_dir, exist_ok=True)
    for z in range(zmax, zmin - 1, -1):
        size = 2 * ORIGIN / 2 ** z
        res = size / 256
        scale = max(1.0, res / mpp)
        if scale > 1.5:
            src = im.resize((max(2, int(w / scale)), max(2, int(h / scale))), Image.LANCZOS)
        else:
            src, scale = im, 1.0
        arr = np.asarray(src)
        sh, sw = arr.shape[:2]
        x0, x1, y0, y1 = rng(z)
        step = (np.arange(256) + .5) * res
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                done += 1
                gx = -ORIGIN + tx * size + step
                gy = ORIGIN - ty * size - step
                X, Y = np.meshgrid(gx, gy)
                p = inv(np.stack([X.ravel(), Y.ravel()], 1)) / scale
                u, v = p[:, 0] - .5, p[:, 1] - .5
                ok = (u >= 0) & (u <= sw - 1) & (v >= 0) & (v <= sh - 1)
                if not ok.any(): continue
                u, v = np.clip(u, 0, sw - 1.001), np.clip(v, 0, sh - 1.001)
                iu, iv = u.astype(int), v.astype(int)
                fu, fv = (u - iu)[:, None], (v - iv)[:, None]
                c = (arr[iv, iu] * (1 - fu) * (1 - fv) + arr[iv, iu + 1] * fu * (1 - fv) +
                     arr[iv + 1, iu] * (1 - fu) * fv + arr[iv + 1, iu + 1] * fu * fv)
                rgba = np.dstack([c.reshape(256, 256, 3).astype(np.uint8),
                                  (ok.reshape(256, 256) * 255).astype(np.uint8)])
                d = os.path.join(out_dir, str(z), str(tx)); os.makedirs(d, exist_ok=True)
                Image.fromarray(rgba, "RGBA").save(os.path.join(d, f"{ty}.png"), optimize=False)
            progress(done / total)

    west, south = m2ll(minx, miny); east, north = m2ll(maxx, maxy)
    return [west, south, east, north], zmin, zmax

def to_mbtiles(tiles_dir, path, bounds, zmin, zmax, name):
    if os.path.exists(path): os.remove(path)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE metadata (name text, value text)")
    db.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob)")
    db.execute("CREATE UNIQUE INDEX t ON tiles (zoom_level, tile_column, tile_row)")
    meta = {"name": name, "format": "png", "type": "overlay", "version": "1",
            "bounds": ",".join(f"{b:.6f}" for b in bounds), "minzoom": zmin, "maxzoom": zmax}
    db.executemany("INSERT INTO metadata VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    for z in os.listdir(tiles_dir):
        for x in os.listdir(os.path.join(tiles_dir, z)):
            for f in os.listdir(os.path.join(tiles_dir, z, x)):
                y = int(f[:-4])
                with open(os.path.join(tiles_dir, z, x, f), "rb") as fh:
                    db.execute("INSERT INTO tiles VALUES (?,?,?,?)", (int(z), int(x), 2 ** int(z) - 1 - y, fh.read()))
    db.commit(); db.close()
