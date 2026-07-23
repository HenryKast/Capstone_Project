from pathlib import Path
import sys
sys.path.insert(0, r"C:\Users\kasth\School\Capstone Project\src")
from rookie_ppr.mascot_assets import process_mascot_file, MASCOT_DIR
from PIL import Image

raw_dir = Path(r"C:\Users\kasth\.cursor\projects\c-Users-kasth-School-Capstone-Project\assets")

process_mascot_file(
    raw_dir / "mascot_bal_raw.png",
    MASCOT_DIR / "bal.png",
    height=400, factor=0.45, tolerance=55,
)
process_mascot_file(
    raw_dir / "mascot_kc_raw.png",
    MASCOT_DIR / "kc.png",
    height=400, factor=0.45, tolerance=70,
)

for tid in ("bal", "kc"):
    p = MASCOT_DIR / f"{tid}.png"
    im = Image.open(p).convert("RGBA")
    corners = [im.getpixel(c) for c in [(0,0),(im.width-1,0),(0,im.height-1),(im.width-1,im.height-1)]]
    print(f"{tid}: size={im.size} bytes={p.stat().st_size} corners={corners}")
