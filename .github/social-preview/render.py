"""Render social-preview.html to a GitHub-spec 1280x640 PNG.

Upload the result under Settings -> General -> Social preview. It is not used
at build or run time, so Pillow is deliberately not a project dependency:

    pip install pillow
    python .github/social-preview/render.py

Chrome's headless --screenshot captures the whole window, and --window-size
sets the window rather than the viewport: on this machine the viewport comes
out 18px narrower and 96px shorter. So render at window 1298x736 (viewport
exactly 1280x640) and crop the dead margin off the right and bottom.
"""
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
TARGET = (1280, 640)          # GitHub's recommended social preview size
WINDOW = (1298, 736)          # calibrated so the viewport lands on TARGET

here = Path(__file__).parent
html = here / "social-preview.html"
raw = here / "_raw.png"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "social-preview.png"

subprocess.run(
    [
        CHROME,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={WINDOW[0]},{WINDOW[1]}",
        f"--screenshot={raw}",
        html.as_uri(),
    ],
    check=True,
    capture_output=True,
)

img = Image.open(raw).convert("RGB").crop((0, 0, *TARGET))
img.save(out, "PNG", optimize=True)
raw.unlink()

data = out.read_bytes()
w, h = struct.unpack(">II", data[16:24])
print(f"{out.name}: {w}x{h}, {len(data) / 1024:.0f} KB")
if (w, h) != TARGET:
    sys.exit(f"expected {TARGET}, got {(w, h)}")
if len(data) > 1_000_000:
    sys.exit("over GitHub's 1 MB social-preview limit")
