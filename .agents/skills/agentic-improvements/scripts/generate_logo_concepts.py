"""Regenerate logo variants from the base meredith.svg template.

Usage:
    uv run scripts/generate_logo_concepts.py --variant favicon
    uv run scripts/generate_logo_concepts.py --variant light
    uv run scripts/generate_logo_concepts.py --variant mono
    uv run scripts/generate_logo_concepts.py --variant small

Output is written to assets/meredith-{variant}.svg.

Requires: Python 3.13+
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent.parent.parent.parent / "assets"
SOURCE_FILE = ASSETS / "meredith.svg"


def load_svg() -> str:
    return SOURCE_FILE.read_text(encoding="utf-8")


def save_variant(content: str, name: str) -> Path:
    dest = ASSETS / f"meredith-{name}.svg"
    dest.write_text(content, encoding="utf-8")
    return dest


def make_favicon(svg: str) -> str:
    # Crop to M only, remove animation, shrink viewBox
    svg = re.sub(r'viewBox="[^"]*"', 'viewBox="-120 -140 240 240"', svg)
    svg = re.sub(r"<animate[^>]*/>", "", svg)
    # Remove wordmark, thrust trail, intake lines, heat diamonds
    svg = re.sub(r"<!-- Wordmark -->.*?</text>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!-- Thrust trail.*?-->(.*?)</g>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!-- Intake streamlines.*?-->(.*?)</g>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!-- Heat diamonds.*?-->(.*?)</g>", "", svg, flags=re.DOTALL)
    return svg


def make_light(svg: str) -> str:
    bg_light = (
        '<linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#ffffff"/>'
        '<stop offset="100%" stop-color="#f0f0f5"/>'
        "</linearGradient>"
    )
    svg = re.sub(r'<linearGradient id="bgGrad".*?</linearGradient>', bg_light, svg, flags=re.DOTALL)
    # Reduce glow opacity
    svg = svg.replace('opacity="0.04"', 'opacity="0.025"', 1)
    return svg


def make_mono(svg: str) -> str:
    # Remove all gradients, use flat fills
    svg = re.sub(r'<linearGradient .*?</linearGradient>', "", svg, flags=re.DOTALL)
    svg = re.sub(r'fill="url\(#[^)]+\)"', 'fill="#333333"', svg)
    svg = svg.replace('stroke="#5a9be5"', 'stroke="#666666"')
    svg = svg.replace('stroke="#3a7bd5"', 'stroke="#666666"')
    svg = svg.replace('stroke="#e67e22"', 'stroke="#666666"')
    svg = svg.replace('stroke="#f39c12"', 'stroke="#666666"')
    svg = svg.replace('stroke="#ff4400"', 'stroke="#666666"')
    svg = svg.replace('stroke="#ff6600"', 'stroke="#666666"')
    svg = svg.replace('stroke="#ff3300"', 'stroke="#666666"')
    svg = svg.replace('stroke="#ff5500"', 'stroke="#666666"')
    svg = svg.replace('stroke="#cc6600"', 'stroke="#666666"')
    svg = svg.replace('stroke="#aa1100"', 'stroke="#666666"')
    svg = svg.replace('stroke="#880000"', 'stroke="#666666"')
    svg = svg.replace('fill="#122a4a"', 'fill="#444444"')
    svg = svg.replace('fill="#8a4400"', 'fill="#444444"')
    svg = svg.replace('fill="#660800"', 'fill="#444444"')
    svg = svg.replace('fill="#550600"', 'fill="#444444"')
    svg = svg.replace('fill="#88ccff"', 'fill="#666666"')
    svg = svg.replace('fill="#ffcc00"', 'fill="#555555"')
    svg = svg.replace('fill="#ffee66"', 'fill="#555555"')
    svg = svg.replace('fill="#ff4400"', 'fill="#666666"')
    svg = svg.replace('fill="#ff8800"', 'fill="#555555"')
    svg = svg.replace('fill="#ff6600"', 'fill="#555555"')
    svg = svg.replace('fill="#ff6b00"', 'fill="#444444"')
    svg = svg.replace('fill="#88ccff"', 'fill="#666666"')
    svg = svg.replace('stroke="#88ccff"', 'stroke="#666666"')
    svg = svg.replace('stroke="#ffcc00"', 'stroke="#555555"')
    svg = svg.replace('stroke="#ff6600"', 'stroke="#555555"')
    svg = svg.replace('fill="#ff8800"', 'fill="#555555"')
    # Remove color-specific fills in glow/filters
    svg = svg.replace('fill="#ffcc00"', 'fill="#444444"')
    svg = svg.replace('fill="#ffee66"', 'fill="#444444"')
    svg = svg.replace('fill="#ff6600"', 'fill="#555555"')
    svg = svg.replace('fill="#555555"', 'fill="#444444"')
    # Wordmark
    svg = svg.replace('fill="#ff8800"', 'fill="#333333"')
    return svg


def make_small(svg: str) -> str:
    svg = re.sub(r'viewBox="[^"]*"', 'viewBox="-100 -100 200 160"', svg)
    svg = re.sub(r"<animate[^>]*/>", "", svg)
    svg = re.sub(r"<!-- Wordmark -->.*?</text>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!-- Thrust trail.*?-->(.*?)</g>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!-- Intake streamlines.*?-->(.*?)</g>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!-- Heat diamonds.*?-->(.*?)</g>", "", svg, flags=re.DOTALL)
    return svg


VARIANTS = {
    "favicon": make_favicon,
    "light": make_light,
    "mono": make_mono,
    "small": make_small,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Meredith logo variants")
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    args = parser.parse_args()

    svg = load_svg()
    fn = VARIANTS[args.variant]
    result = fn(svg)
    path = save_variant(result, args.variant)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
