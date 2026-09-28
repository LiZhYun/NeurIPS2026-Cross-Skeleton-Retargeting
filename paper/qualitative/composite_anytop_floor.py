"""Add a soft floor band under a rendered skeleton image.

The renderer writes skeletons on a transparent background. This puts a light floor
strip beneath the figure so the pose reads as standing on the ground.

    python paper/qualitative/composite_anytop_floor.py <image.png>
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


def mix_rgb(c0, c1, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(c0[i] * (1.0 - t) + c1[i] * t)) for i in range(3))


def thumbnail_component_mask(alpha: Image.Image) -> Image.Image:
    w, h = alpha.size
    pix = alpha.load()
    visited: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, int, int, int, list[tuple[int, int]]]] = []
    seed_x = int(round(w * 0.34))
    seed_y = int(round(h * 0.38))
    threshold = 4
    for sy in range(seed_y):
        for sx in range(seed_x):
            if (sx, sy) in visited or pix[sx, sy] <= threshold:
                continue
            q = deque([(sx, sy)])
            visited.add((sx, sy))
            pts: list[tuple[int, int]] = []
            minx = maxx = sx
            miny = maxy = sy
            while q:
                x, y = q.popleft()
                pts.append((x, y))
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx == x and ny == y:
                            continue
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if (nx, ny) in visited or pix[nx, ny] <= threshold:
                            continue
                        visited.add((nx, ny))
                        q.append((nx, ny))
            if maxx <= w * 0.38 and maxy <= h * 0.42 and len(pts) > 24:
                candidates.append((len(pts), minx, miny, maxx, maxy, pts))
    mask = Image.new("L", (w, h), 0)
    if not candidates:
        draw = ImageDraw.Draw(mask)
        draw.rectangle((0, 0, int(round(w * 0.24)), int(round(h * 0.27))), fill=255)
        return mask
    _, _, _, _, _, pts = max(candidates, key=lambda item: item[0])
    mpix = mask.load()
    for x, y in pts:
        mpix[x, y] = 255
    return mask


def split_thumbnail_and_skeleton(fg: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Keep the upper-left mesh thumbnail fixed and lower only the skeleton strip."""
    w, h = fg.size
    alpha = fg.getchannel("A")
    thumb_mask = thumbnail_component_mask(alpha)
    thumb_alpha = ImageChops.multiply(alpha, thumb_mask)

    skel_alpha = ImageChops.subtract(alpha, thumb_alpha)
    skel_box = skel_alpha.getbbox()
    if skel_box is None:
        return fg, Image.new("RGBA", (w, h), (0, 0, 0, 0))

    desired_shift = int(round(h * 0.145))
    max_shift = max(0, h - 5 - skel_box[3])
    shift = min(desired_shift, max_shift)

    thumbnail = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    thumbnail.putalpha(thumb_alpha)
    thumbnail = Image.composite(fg, thumbnail, thumb_alpha)

    skeleton = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    skeleton.putalpha(skel_alpha)
    skeleton = Image.composite(fg, skeleton, skel_alpha)
    lowered = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    lowered.alpha_composite(skeleton, (0, shift))
    return thumbnail, lowered


def composite(render_path: Path) -> None:
    fg = Image.open(render_path).convert("RGBA")
    w, h = fg.size
    bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(bg, "RGBA")

    top = int(round(h * 0.465))
    bottom = h
    rows = 6
    tile = max(70, int(round(w / 7.2)))
    slant = int(round(w * 0.13))
    x_start = -tile * 2
    x_end = w + tile * 3

    def x_at(x_base: float, y: float) -> float:
        depth = (bottom - y) / max(bottom - top, 1)
        return x_base + slant * depth

    base_a = (205, 212, 218)
    base_b = (235, 237, 239)
    white = (255, 255, 255)
    for r in range(rows):
        y0 = top + (bottom - top) * r / rows
        y1 = top + (bottom - top) * (r + 1) / rows
        fade = 0.06
        c = 0
        x = x_start
        while x < x_end:
            base = base_a if (r + c) % 2 == 0 else base_b
            color = mix_rgb(base, white, fade)
            poly = [
                (x_at(x, y0), y0),
                (x_at(x + tile, y0), y0),
                (x_at(x + tile, y1), y1),
                (x_at(x, y1), y1),
            ]
            draw.polygon(poly, fill=(*color, 248))
            x += tile
            c += 1

    fade_overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    fade_draw = ImageDraw.Draw(fade_overlay, "RGBA")
    fade_height = max(1, int(round((bottom - top) * 0.34)))
    fade_start = bottom - fade_height
    for i in range(fade_height):
        alpha = int(round(165 * (i / fade_height) ** 1.65))
        y = fade_start + i
        fade_draw.rectangle((0, y, w, y + 1), fill=(255, 255, 255, alpha))
    bg = Image.alpha_composite(bg, fade_overlay)
    thumbnail, skeleton = split_thumbnail_and_skeleton(fg)
    bg = Image.alpha_composite(bg, thumbnail)
    Image.alpha_composite(bg, skeleton).save(render_path)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: composite_anytop_floor.py RENDER_PATH")
    composite(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
