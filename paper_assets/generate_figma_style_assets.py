#!/usr/bin/env python3
"""Generate Figma-style (flat, grid-aligned) PNG figures using Pillow (no matplotlib)."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (250, 250, 251)
FG = (17, 24, 39)
MUTED = (107, 114, 128)
BORDER = (229, 231, 235)
BLUE = (37, 99, 235)
INDIGO = (79, 70, 229)
EMERALD = (5, 150, 105)
AMBER = (217, 119, 6)
WHITE = (255, 255, 255)
HEADER = (243, 244, 246)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int] = MUTED,
    width: int = 3,
) -> None:
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 10
    hx, hy = x2, y2
    a1 = ang + math.pi * 7 / 8
    a2 = ang - math.pi * 7 / 8
    p1 = (hx + head * math.cos(a1), hy + head * math.sin(a1))
    p2 = (hx + head * math.cos(a2), hy + head * math.sin(a2))
    draw.polygon([(hx, hy), p1, p2], fill=color)


def _center_text(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = FG,
) -> None:
    w, h = _text_size(draw, text, font)
    draw.text((cx - w // 2, cy - h // 2), text, font=font, fill=fill)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        tw, _ = _text_size(draw, trial, font)
        if tw <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def figure1_architecture(out: Path) -> None:
    W, H = 2200, 1100
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    title = _font(44)
    sub = _font(22)
    body = _font(20)
    small = _font(18)

    _center_text(d, W // 2, 70, "Adaptive RAG — conceptual architecture", title, FG)
    _center_text(
        d,
        W // 2,
        125,
        "Drift signals route queries before hybrid retrieval and compliance validation",
        sub,
        MUTED,
    )

    cards = [
        (120, 320, 620, 620, "Layer 1 — Drift detection", "Compare v0 vs v1 chunks\nsemantic similarity + diff hashing\nflag stale candidates", INDIGO),
        (790, 320, 1290, 620, "Layer 2 — LangGraph orchestration", "Drift agent → query adaptation\n→ compliance validation", BLUE),
        (1460, 320, 1960, 620, "Layer 3 — Hybrid retrieval", "Dense + semantic + BM25 rerank\nclause-heavy weighting", EMERALD),
    ]
    for x1, y1, x2, y2, t1, t2, accent in cards:
        _rounded_rect(d, (x1, y1, x2, y2), 22, WHITE, BORDER, 2)
        d.line((x1 + 18, y1 + 70, x2 - 18, y1 + 70), fill=accent, width=6)
        _center_text(d, (x1 + x2) // 2, y1 + 42, t1, _font(24))
        for i, line in enumerate(t2.split("\n")):
            _center_text(d, (x1 + x2) // 2, y1 + 120 + i * 34, line, small, MUTED)

    _arrow(d, 640, 470, 780, 470)
    _arrow(d, 1310, 470, 1450, 470)

    _rounded_rect(d, (420, 720, 1780, 980), 24, WHITE, BORDER, 2)
    d.line((420 + 18, 790, 1780 - 18, 790), fill=AMBER, width=6)
    _center_text(d, 1100, 760, "Outputs", _font(24))
    _center_text(d, 1100, 840, "Ranked passages + version-aware routing + compliance-checked answer", small, MUTED)

    _arrow(d, 1100, 620, 1100, 710)

    im.save(out, format="PNG")


def figure2_evaluation(out: Path) -> None:
    W, H = 2200, 950
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    title = _font(44)
    sub = _font(22)
    small = _font(18)

    _center_text(d, W // 2, 70, "Evaluation setting (versioned corpus)", title, FG)
    _center_text(
        d,
        W // 2,
        125,
        "All systems indexed on v0; amendments yield v1; same query set pre/post",
        sub,
        MUTED,
    )

    boxes = [
        (140, 260, 640, 520, "Corpus v0", "initial snapshot"),
        (860, 260, 1360, 520, "Amendments", "controlled updates"),
        (1580, 260, 2080, 520, "Corpus v1", "authoritative text"),
    ]
    for x1, y1, x2, y2, t1, t2 in boxes:
        _rounded_rect(d, (x1, y1, x2, y2), 22, WHITE, BORDER, 2)
        d.line((x1 + 18, y1 + 70, x2 - 18, y1 + 70), fill=BLUE, width=6)
        _center_text(d, (x1 + x2) // 2, y1 + 42, t1, _font(24))
        _center_text(d, (x1 + x2) // 2, y1 + 120, t2, small, MUTED)

    _arrow(d, 650, 390, 850, 390)
    _arrow(d, 1370, 390, 1570, 390)

    _rounded_rect(d, (220, 600, 2000, 820), 24, WHITE, BORDER, 2)
    d.line((238, 670, 1982, 670), fill=INDIGO, width=6)
    _center_text(d, 1110, 640, "Post-amendment runs", _font(24))
    _center_text(
        d,
        1110,
        730,
        "Static baselines: no automatic refresh signal  •  Adaptive pipeline: drift/adaptation consumes v1 updates",
        small,
        MUTED,
    )

    im.save(out, format="PNG")


def figure3_procedure(out: Path) -> None:
    W, H = 2200, 780
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    title = _font(44)
    foot = _font(18)

    _center_text(d, W // 2, 70, "Four-step evaluation procedure", title, FG)

    steps = [
        ("1", "Index\nv0 snapshot"),
        ("2", "Run queries\n(pre)"),
        ("3", "Inject\namendments"),
        ("4", "Re-run queries\n(post)"),
    ]
    x0 = 260
    w, h = 360, 220
    gap = 80
    for i, (n, txt) in enumerate(steps):
        x1 = x0 + i * (w + gap)
        y1 = 200
        x2, y2 = x1 + w, y1 + h
        _rounded_rect(d, (x1, y1, x2, y2), 26, WHITE, BORDER, 2)
        _center_text(d, (x1 + x2) // 2, y1 + 55, n, _font(34), BLUE)
        for j, line in enumerate(txt.split("\n")):
            _center_text(d, (x1 + x2) // 2, y1 + 120 + j * 34, line, _font(20))
        if i < len(steps) - 1:
            _arrow(d, x2 + 6, y1 + h // 2, x2 + gap - 6, y1 + h // 2)

    _center_text(
        d,
        W // 2,
        700,
        "Metrics: SCR / SHR / VRP / VA-RAcc / CAS with bootstrap 95% CIs (n=24)",
        foot,
        MUTED,
    )

    im.save(out, format="PNG")


def table1_image(out: Path) -> None:
    W, H = 2400, 1100
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    head_f = _font(22)
    cell_f = _font(20)
    tiny = _font(17)

    title = "Post-amendment summary (means over n=24 queries; headline SCR is micro-averaged over retrieved chunks)"
    _center_text(d, W // 2, 55, title, head_f, FG)

    headers = ["System", "SCR", "SHR", "VRP", "VA-RAcc", "CAS", "Notes"]
    rows = [
        ["FAISS RAG [1]", "100%", "60%", "0%", "56%", "57%", "all chunks v0"],
        ["MMA-RAG [3]", "100%", "62%", "0%", "58%", "59%", "text-only reimpl."],
        ["Azure hybrid [2]", "50%", "86%", "48%", "73%", "74%", "mixed v0/v1"],
        ["Adaptive (proposed)", "0%", "94%", "92%", "94%", "95%", "drift path → v1"],
    ]

    x0, y0 = 120, 120
    cw = [780, 200, 200, 200, 220, 200, 420]
    row_h = 92
    header_h = 86

    # header
    x = x0
    for i, htxt in enumerate(headers):
        _rounded_rect(d, (x, y0, x + cw[i], y0 + header_h), 12, HEADER, BORDER, 2)
        _center_text(d, x + cw[i] // 2, y0 + header_h // 2, htxt, _font(20))
        x += cw[i]

    for r, row in enumerate(rows):
        y = y0 + header_h + r * row_h
        x = x0
        fill = WHITE if r % 2 == 0 else (252, 252, 253)
        for i, cell in enumerate(row):
            _rounded_rect(d, (x, y, x + cw[i], y + row_h - 6), 10, fill, BORDER, 1)
            _center_text(d, x + cw[i] // 2, y + (row_h - 6) // 2, cell, cell_f)
            x += cw[i]

    foot = (
        "SHR = semantic hit rate  •  SCR = stale chunk rate  •  VRP = version-aware retrieval precision  •  "
        "VA-RAcc = version-sensitive answer accuracy  •  CAS = compliance alignment"
    )
    for i, line in enumerate(_wrap(d, foot, tiny, W - 240)):
        _center_text(d, W // 2, 980 + i * 28, line, tiny, MUTED)

    im.save(out, format="PNG")


def main() -> None:
    root = Path(__file__).resolve().parent
    figure1_architecture(root / "figure1_architecture.png")
    figure2_evaluation(root / "figure2_evaluation_setting.png")
    figure3_procedure(root / "figure3_procedure.png")
    table1_image(root / "table1_metrics.png")
    print("Wrote PNG assets to:", root)


if __name__ == "__main__":
    main()
