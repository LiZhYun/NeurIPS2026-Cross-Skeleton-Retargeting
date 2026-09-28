"""Lay the two halves out as one figure, and add the labels.

This step only arranges and labels. The pictures themselves are the ones the two drawing
steps produced; nothing here redraws or retouches them. The labels use the DejaVu Sans
font that comes with matplotlib.

Runs in the robot environment, in a few seconds:
    python -m robots.render.compose
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from robots.lafan1_to_six_robots.data import ROBOTS
from robots.render.paths import DEFAULT_OUT

INK = (32, 32, 36)
SOFT = (110, 112, 120)
RULE = (196, 198, 205)
BACKGROUND = (255, 255, 255)

NAMES = {
    "human": "Human source (capsule figure)",
    "unitree_g1": "Unitree G1",
    "booster_t1_29dof": "Booster T1",
    "fourier_n1": "Fourier N1",
    "stanford_toddy": "Stanford Toddy",
    "engineai_pm01": "EngineAI PM01",
    "pal_talos": "PAL TALOS",
}
ROW_LABELS = {
    "true_retarget": ("true retarget", (96, 99, 108)),
    "true_pair_model": ("model trained on true pairs", (46, 125, 91)),
    "averaging_objective": ("averaging objective", (181, 106, 42)),
    "unpaired_objective": ("unpaired objective", (181, 106, 42)),
}
ROW_ORDER = ["true_retarget", "true_pair_model", "averaging_objective",
             "unpaired_objective"]


def _font_file(bold=False):
    """A text font that is present wherever matplotlib is installed."""
    import matplotlib
    from matplotlib import font_manager

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    bundled = Path(matplotlib.get_data_path()) / "fonts/ttf" / name
    if bundled.exists():
        return str(bundled)
    return font_manager.findfont(font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"))


def font(size, bold=False):
    return ImageFont.truetype(_font_file(bold), size)


def text(draw, at, string, size=26, bold=False, fill=INK, anchor="la"):
    """Write one line of text at `at`."""
    draw.text(at, string, font=font(size, bold), fill=fill, anchor=anchor)


def swatch(draw, x, y, colour, w=16, h=16):
    """A small rounded square of colour, for a row's key."""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=4, fill=colour)


def wrap(draw, string, text_font, width):
    """Break a label into lines no wider than `width` pixels."""
    lines, line = [], ""
    for word in string.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=text_font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    return lines


def first_half(report, out_dir, tile_width, gap=26, label_height=42, gutter=370):
    """The human on its own row, then the six robots in three rows of two."""
    order = ["human"] + ROBOTS
    strips = {k: Image.open(out_dir / report["rows"][k]["still"]).convert("RGB")
              for k in order}
    source_w, source_h = strips["unitree_g1"].size
    tile_height = int(round(source_h * tile_width / source_w))
    width = gutter + 2 * tile_width + gap
    head = 118
    image = Image.new("RGB", (width, head + 4 * (label_height + tile_height) + 3 * gap),
                      BACKGROUND)
    draw = ImageDraw.Draw(image)

    text(draw, (0, 4), "A", 46, True)
    text(draw, (62, 12), "One dance, six robots", 38, True)
    text(draw, (62, 66),
         "LAFAN1 dance routine, 10 s;  five key poses, earliest faded;  "
         "true retarget on every robot", 25, False, SOFT)
    draw.line([(0, head - 16), (width, head - 16)], fill=RULE, width=2)

    def place(key, x, y):
        name = NAMES[key]
        swatch(draw, x + 2, y + 9,
               (140, 150, 166) if key == "human" else (120, 124, 134))
        text(draw, (x + 28, y + 4), name, 27, True)
        if key != "human":
            text(draw, (x + 28 + draw.textlength(name, font=font(27, True)), y + 6),
                 f"   ·  {report['rows'][key]['median_base_height_m']:.2f} m base height",
                 24, False, SOFT)
        image.paste(strips[key].resize((tile_width, tile_height), Image.LANCZOS),
                    (x, y + label_height))

    place("human", gutter + (tile_width + gap) // 2, head)
    for i, key in enumerate(ROBOTS):
        row, col = divmod(i, 2)
        place(key, gutter + col * (tile_width + gap),
              head + (row + 1) * (label_height + tile_height + gap))
    return image


def second_half(report, out_dir, tile_width, gap=26, gutter=370, column_head=54):
    """Two performers side by side, one row per model, with each row's fit error."""
    clips = [p["clip_id"] for p in report["performers"]]
    strips = {}
    for clip_id in clips:
        for row_key in ROW_ORDER:
            strips[(clip_id, row_key)] = Image.open(
                out_dir / report["tiles"][f"{clip_id}__{row_key}"]["still"]).convert("RGB")
    source_w, source_h = next(iter(strips.values())).size
    tile_height = int(round(source_h * tile_width / source_w))
    width = gutter + 2 * tile_width + gap
    head = 118
    image = Image.new("RGB", (width, head + column_head + 4 * tile_height + 3 * gap),
                      BACKGROUND)
    draw = ImageDraw.Draw(image)

    text(draw, (0, 4), "B", 46, True)
    text(draw, (62, 12), "Two performers, same routine, on Unitree G1", 38, True)
    text(draw, (62, 66),
         "same routine and 256-frame window for both performers;  "
         "same five key poses and camera in every row", 25, False, SOFT)
    draw.line([(0, head - 16), (width, head - 16)], fill=RULE, width=2)

    for col in range(2):
        x = gutter + col * (tile_width + gap)
        text(draw, (x + tile_width // 2, head + 6), f"Performer {'AB'[col]}", 30, True,
             INK, "ma")

    for row, row_key in enumerate(ROW_ORDER):
        y = head + column_head + row * (tile_height + gap)
        title, colour = ROW_LABELS[row_key]
        lines = wrap(draw, title, font(30, True), gutter - 60)
        fit = None
        if row_key != "true_retarget":
            first = report["tiles"][f"{clips[0]}__{row_key}"]["fit_error_m"]["rms_m"] * 100
            second = report["tiles"][f"{clips[1]}__{row_key}"]["fit_error_m"]["rms_m"] * 100
            fit = f"fit error {first:.1f} / {second:.1f} cm"
        block = len(lines) * 36 + (34 if fit else 0)
        at = y + tile_height // 2 - block // 2
        swatch(draw, 4, at + 9, colour, 18, 18)
        for line in lines:
            text(draw, (34, at), line, 30, True, INK)
            at += 36
        if fit:
            text(draw, (34, at + 4), fit, 25, False, SOFT)
        for col, clip_id in enumerate(clips):
            x = gutter + col * (tile_width + gap)
            image.paste(strips[(clip_id, row_key)].resize((tile_width, tile_height),
                                                          Image.LANCZOS), (x, y))
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--tile_width", type=int, default=1700)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    panels = out_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reports/six_robots.json") as f:
        six = json.load(f)
    with open(out_dir / "reports/two_performers.json") as f:
        two = json.load(f)

    top = first_half(six, out_dir, args.tile_width)
    bottom = second_half(two, out_dir, args.tile_width)
    top.save(panels / "six_robots.png")
    bottom.save(panels / "two_performers.png")

    margin, between = 46, 70
    width = max(top.width, bottom.width) + 2 * margin
    height = top.height + between + bottom.height + 2 * margin
    figure = Image.new("RGB", (width, height), BACKGROUND)
    figure.paste(top, ((width - top.width) // 2, margin))
    draw = ImageDraw.Draw(figure)
    rule_y = margin + top.height + between // 2
    draw.line([(margin, rule_y), (width - margin, rule_y)], fill=RULE, width=3)
    figure.paste(bottom, ((width - bottom.width) // 2, margin + top.height + between))
    figure.save(panels / "robot_figure.png")
    figure.save(panels / "robot_figure.pdf", "PDF", resolution=300.0, quality=95,
                subsampling=0)
    figure.resize((figure.width // 3, figure.height // 3), Image.LANCZOS).save(
        panels / "robot_figure_preview.png")
    print(f"figure {figure.size} written to {panels / 'robot_figure.png'}")


if __name__ == "__main__":
    main()
