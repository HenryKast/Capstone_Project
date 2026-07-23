"""Process team mascot PNGs: remove flat background, light pixelation, fit height."""
from __future__ import annotations

from collections import deque
from pathlib import Path

from PIL import Image

MASCOT_DIR = Path(__file__).resolve().parents[2] / "data" / "assets" / "team_mascots"


def _color_dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _is_near_white_or_gray(r: int, g: int, b: int, a: int) -> bool:
    if a < 8:
        return True
    if r >= 245 and g >= 245 and b >= 245:
        return True
    if r >= 210 and g >= 210 and b >= 210 and abs(r - g) <= 12 and abs(g - b) <= 12:
        return True
    if r >= 190 and g >= 190 and b >= 190 and abs(r - g) <= 8 and abs(g - b) <= 8:
        return True
    return False


def cut_background(img: Image.Image, *, tolerance: int = 55) -> Image.Image:
    """Flood-fill from corners: remove white/gray OR solid flat backdrop color."""
    img = img.convert("RGBA")
    w, h = img.size
    px = img.load()
    corners = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w // 2, 0),
        (0, h // 2),
        (w - 1, h // 2),
        (w // 2, h - 1),
    ]
    samples = [px[x, y][:3] for x, y in corners if 0 <= x < w and 0 <= y < h]
    backdrop = max(set(samples), key=samples.count) if samples else (255, 255, 255)

    def is_bg(r: int, g: int, b: int, a: int) -> bool:
        if _is_near_white_or_gray(r, g, b, a):
            return True
        return _color_dist((r, g, b), backdrop) <= tolerance

    q: deque[tuple[int, int]] = deque(corners)
    seen: set[tuple[int, int]] = set()
    while q:
        x, y = q.popleft()
        if (x, y) in seen or x < 0 or y < 0 or x >= w or y >= h:
            continue
        r, g, b, a = px[x, y]
        if not is_bg(r, g, b, a):
            continue
        seen.add((x, y))
        px[x, y] = (0, 0, 0, 0)
        q.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
    return img


def pixelate(img: Image.Image, factor: float = 0.30) -> Image.Image:
    w, h = img.size
    sw, sh = max(8, int(w * factor)), max(8, int(h * factor))
    small = img.resize((sw, sh), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


def fit_height(img: Image.Image, height: int = 400) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    nw = max(1, int(w * (height / h)))
    return img.resize((nw, height), Image.Resampling.NEAREST)


def process_mascot_file(
    src: Path,
    dest: Path,
    *,
    height: int = 400,
    factor: float = 0.45,
    tolerance: int = 55,
) -> Path:
    MASCOT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    img = cut_background(img, tolerance=tolerance)
    img = pixelate(img, factor)
    img = fit_height(img, height)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return dest
