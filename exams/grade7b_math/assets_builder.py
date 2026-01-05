from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class AssetSpec:
    filename: str
    builder_name: str


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Prefer common Windows fonts. Fall back to PIL's default bitmap font.
    """
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_axes_grid(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    step: int,
    axis_color=(0, 0, 0),
    grid_color=(220, 220, 220),
):
    for x in range(left, right + 1, step):
        draw.line([(x, top), (x, bottom)], fill=grid_color, width=1)
    for y in range(top, bottom + 1, step):
        draw.line([(left, y), (right, y)], fill=grid_color, width=1)

    # axes (center)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    draw.line([(left, cy), (right, cy)], fill=axis_color, width=3)
    draw.line([(cx, top), (cx, bottom)], fill=axis_color, width=3)
    # arrows
    draw.polygon([(right, cy), (right - 18, cy - 8), (right - 18, cy + 8)], fill=axis_color)
    draw.polygon([(cx, top), (cx - 8, top + 18), (cx + 8, top + 18)], fill=axis_color)

    return cx, cy


def build_parallel_lines_png(out_path: Path) -> None:
    w, h = 1400, 620
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = _try_load_font(30)
    font_small = _try_load_font(24)

    # Two parallel lines
    y1, y2 = 170, 450
    d.line([(120, y1), (1280, y1)], fill=(0, 0, 0), width=5)
    d.line([(120, y2), (1280, y2)], fill=(0, 0, 0), width=5)

    # Transversal
    d.line([(320, 60), (1080, 560)], fill=(0, 0, 0), width=5)

    # labels
    d.text((1250, y1 - 55), "l1", fill=(0, 0, 0), font=font)
    d.text((1250, y2 - 55), "l2", fill=(0, 0, 0), font=font)
    d.text((1040, 520), "t", fill=(0, 0, 0), font=font)

    # angle markers (simple arcs)
    # Top intersection around (600, 170)
    cx1, cy1 = 610, y1
    box1 = [cx1 - 70, cy1 - 70, cx1 + 70, cy1 + 70]
    d.arc(box1, start=300, end=360, fill=(0, 0, 0), width=4)
    d.text((cx1 + 20, cy1 - 90), "∠1", fill=(0, 0, 0), font=font_small)

    # Bottom intersection around (820, 450)
    cx2, cy2 = 810, y2
    box2 = [cx2 - 70, cy2 - 70, cx2 + 70, cy2 + 70]
    d.arc(box2, start=120, end=180, fill=(0, 0, 0), width=4)
    d.text((cx2 - 130, cy2 + 20), "∠2", fill=(0, 0, 0), font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def build_coordinate_plane_png(out_path: Path) -> None:
    w, h = 1400, 900
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = _try_load_font(26)
    font_small = _try_load_font(22)

    left, top, right, bottom = 120, 80, 1280, 820
    step = 70
    cx, cy = _draw_axes_grid(d, left=left, top=top, right=right, bottom=bottom, step=step)

    # axis labels
    d.text((right - 25, cy + 12), "x", fill=(0, 0, 0), font=font)
    d.text((cx + 12, top - 10), "y", fill=(0, 0, 0), font=font)
    d.text((cx + 8, cy + 8), "O", fill=(0, 0, 0), font=font_small)

    # coordinate transform: 1 unit per grid step
    def to_px(x: int, y: int) -> tuple[int, int]:
        return (cx + x * step, cy - y * step)

    # Points used in questions
    A = (2, 3)
    B = (-1, 1)

    for name, (x, y), color in [("A", A, (0, 102, 204)), ("B", B, (220, 0, 0))]:
        px, py = to_px(x, y)
        d.ellipse([(px - 8, py - 8), (px + 8, py + 8)], fill=color, outline=(0, 0, 0))
        d.text((px + 10, py - 32), f"{name}({x},{y})", fill=(0, 0, 0), font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def build_isosceles_triangle_png(out_path: Path) -> None:
    w, h = 1200, 800
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = _try_load_font(30)
    font_small = _try_load_font(24)

    # Triangle points
    A = (600, 120)
    B = (250, 640)
    C = (950, 640)
    D = ((B[0] + C[0]) // 2, (B[1] + C[1]) // 2)

    # Sides
    d.line([A, B], fill=(0, 0, 0), width=5)
    d.line([A, C], fill=(0, 0, 0), width=5)
    d.line([B, C], fill=(0, 0, 0), width=5)

    # Median AD
    d.line([A, D], fill=(0, 0, 0), width=3)

    # Tick marks on AB and AC to indicate equal
    def tick(p1, p2, t=0.55):
        x = int(p1[0] + (p2[0] - p1[0]) * t)
        y = int(p1[1] + (p2[1] - p1[1]) * t)
        # small perpendicular tick
        d.line([(x - 18, y - 10), (x + 18, y + 10)], fill=(0, 0, 0), width=4)

    tick(A, B)
    tick(A, C)

    # Labels
    d.text((A[0] - 20, A[1] - 55), "A", fill=(0, 0, 0), font=font)
    d.text((B[0] - 30, B[1] + 10), "B", fill=(0, 0, 0), font=font)
    d.text((C[0] - 10, C[1] + 10), "C", fill=(0, 0, 0), font=font)
    d.text((D[0] - 18, D[1] + 10), "D", fill=(0, 0, 0), font=font_small)

    # angle label near B
    d.text((B[0] + 35, B[1] - 75), "40°", fill=(0, 0, 0), font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def build_reading_bar_chart_png(out_path: Path) -> None:
    w, h = 1500, 900
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = _try_load_font(26)
    font_small = _try_load_font(22)

    left, top, right, bottom = 140, 100, 1400, 780
    axis_color = (0, 0, 0)
    d.line([(left, bottom), (right, bottom)], fill=axis_color, width=4)
    d.line([(left, bottom), (left, top)], fill=axis_color, width=4)
    d.polygon([(right, bottom), (right - 18, bottom - 8), (right - 18, bottom + 8)], fill=axis_color)
    d.polygon([(left, top), (left - 8, top + 18), (left + 8, top + 18)], fill=axis_color)

    d.text((right - 25, bottom + 10), "天", fill=axis_color, font=font)
    d.text((left - 60, top - 10), "小时", fill=axis_color, font=font)

    days = ["周一", "周二", "周三", "周四", "周五"]
    values = [1.5, 2.0, 2.5, 1.0, 3.0]
    max_y = 3.5
    chart_w = right - left - 80
    chart_h = bottom - top - 40
    bar_w = int(chart_w / len(days) * 0.55)
    gap = int(chart_w / len(days) * 0.45)
    x0 = left + 60

    # y ticks
    for i in range(0, 8):
        yv = i * 0.5
        y = bottom - 20 - int(chart_h * (yv / max_y))
        d.line([(left - 8, y), (left + 8, y)], fill=axis_color, width=2)
        d.text((left - 70, y - 12), f"{yv:g}", fill=axis_color, font=font_small)
        d.line([(left, y), (right, y)], fill=(235, 235, 235), width=1)

    # bars
    for i, (day, v) in enumerate(zip(days, values)):
        x = x0 + i * (bar_w + gap)
        y_top = bottom - 20 - int(chart_h * (v / max_y))
        d.rectangle([(x, y_top), (x + bar_w, bottom - 20)], fill=(80, 140, 220), outline=(0, 0, 0), width=2)
        d.text((x + 6, y_top - 30), f"{v:g}", fill=axis_color, font=font_small)
        d.text((x + 6, bottom - 10), day, fill=axis_color, font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


ASSETS: list[AssetSpec] = [
    AssetSpec(filename="parallel_lines.png", builder_name="build_parallel_lines_png"),
    AssetSpec(filename="coordinate_plane.png", builder_name="build_coordinate_plane_png"),
    AssetSpec(filename="isosceles_triangle.png", builder_name="build_isosceles_triangle_png"),
    AssetSpec(filename="reading_bar_chart.png", builder_name="build_reading_bar_chart_png"),
]


def ensure_assets(assets_dir: Path, *, overwrite: bool = True) -> list[Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for spec in ASSETS:
        out = assets_dir / spec.filename
        if out.exists() and not overwrite:
            continue
        builder = globals().get(spec.builder_name)
        if not callable(builder):
            raise RuntimeError(f"Missing builder: {spec.builder_name}")
        builder(out)
        generated.append(out)
    return generated









