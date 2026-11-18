"""Inline the exported results (data.js), the StatsBomb logo and the fonts into one self-contained page that
makes no requests to other sites. Writes docs/index.html (served by GitHub Pages).
Fonts: Barlow and Barlow Condensed (SIL Open Font License 1.1, see explorer/fonts/OFL.txt).
Usage: python explorer/build.py   (after python explorer/export.py)"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONTS = [("Barlow", 400, "barlow-latin-400-normal"), ("Barlow", 500, "barlow-latin-500-normal"),
         ("Barlow", 600, "barlow-latin-600-normal"), ("Barlow Condensed", 500, "barlow-condensed-latin-500-normal"),
         ("Barlow Condensed", 600, "barlow-condensed-latin-600-normal"),
         ("Barlow Condensed", 700, "barlow-condensed-latin-700-normal")]


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


font_css = "\n".join(
    f"@font-face {{ font-family: '{fam}'; font-style: normal; font-weight: {w}; font-display: swap; "
    f"src: url(data:font/woff2;base64,{b64(HERE / 'fonts' / (name + '.woff2'))}) format('woff2'); }}"
    for fam, w, name in FONTS)
html = (HERE / "template.html").read_text()
html = html.replace("/*__FONTS_CSS__*/", font_css)
html = html.replace("/*__LOGO_B64__*/", b64(HERE / "statsbomb_logo_small.png"))
html = html.replace("/*__DATA_JS__*/", (HERE / "data.js").read_text())
assert "/*__" not in html, "unfilled placeholder"
assert "googleapis" not in html and "gstatic" not in html, "page must not load third-party resources"
out = HERE.parent / "docs" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
(out.parent / ".nojekyll").touch()  # serve the page as-is on GitHub Pages
print("wrote", out, f"{len(html) / 1024:.0f} KB")
