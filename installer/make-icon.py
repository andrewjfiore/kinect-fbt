#!/usr/bin/env python3
"""Generate the Marionette app icon: a puppet on strings, cyan on deep navy.

Renders a high-res master, then exports the multi-size Windows .ico and the
Linux .png. Run when the icon changes:  python3 installer/make-icon.py
Requires Pillow (pip install Pillow).
"""
import os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_WIN = os.path.join(ROOT, "windows")
OUT_LNX = os.path.join(ROOT, "linux")
os.makedirs(OUT_WIN, exist_ok=True)
os.makedirs(OUT_LNX, exist_ok=True)

S = 1024
SS = 4  # supersample factor


def px(v):
    return int(round(v * SS))


W = S * SS
img = Image.new("RGBA", (W, W), (0, 0, 0, 0))

# Rounded-square background with a subtle vertical gradient (deep navy).
bg = Image.new("RGBA", (W, W), (0, 0, 0, 0))
bd = ImageDraw.Draw(bg)
top, bot = (13, 20, 38), (5, 7, 13)
for y in range(W):
    t = y / W
    c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
    bd.line([(0, y), (W, y)], fill=c + (255,))
mask = Image.new("L", (W, W), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, W - 1], radius=px(224), fill=255)
img.paste(bg, (0, 0), mask)
d = ImageDraw.Draw(img)

CY = (34, 211, 238, 255)   # --accent
CY2 = (94, 234, 212, 255)  # --accent-2
STRING = (94, 234, 212, 120)
cx = S / 2

# Control bar (the puppeteer's cross) near the top.
bar_y = 250
d.rounded_rectangle([px(cx - 220), px(bar_y - 16), px(cx + 220), px(bar_y + 16)],
                    radius=px(16), fill=CY2)
d.rounded_rectangle([px(cx - 16), px(bar_y - 90), px(cx + 16), px(bar_y + 16)],
                    radius=px(16), fill=CY2)

head = (cx, 470)
head_r = 66
shoulder_l, shoulder_r = (cx - 150, 590), (cx + 150, 590)
hand_l, hand_r = (cx - 235, 760), (cx + 235, 760)
hip = (cx, 720)
foot_l, foot_r = (cx - 120, 900), (cx + 120, 900)


def line(a, b, col, w):
    d.line([px(a[0]), px(a[1]), px(b[0]), px(b[1])], fill=col, width=px(w))


for a, b in [((cx - 220, bar_y), head), ((cx - 220, bar_y), hand_l),
             ((cx + 220, bar_y), head), ((cx + 220, bar_y), hand_r)]:
    line(a, b, STRING, 5)
for a, b in [(head, hip), (shoulder_l, shoulder_r), (shoulder_l, hand_l),
             (shoulder_r, hand_r), (hip, foot_l), (hip, foot_r)]:
    line(a, b, CY, 26)
for p, r in [(shoulder_l, 20), (shoulder_r, 20), (hip, 22), (hand_l, 22),
             (hand_r, 22), (foot_l, 22), (foot_r, 22)]:
    d.ellipse([px(p[0] - r), px(p[1] - r), px(p[0] + r), px(p[1] + r)], fill=CY)
d.ellipse([px(head[0] - head_r), px(head[1] - head_r),
           px(head[0] + head_r), px(head[1] + head_r)], fill=CY)
d.ellipse([px(head[0] - head_r + 14), px(head[1] - head_r + 14),
           px(head[0] + head_r - 14), px(head[1] + head_r - 14)], fill=(150, 244, 255, 255))

master = img.resize((S, S), Image.LANCZOS)
master.save(os.path.join(OUT_LNX, "marionette.png"))
master.save(os.path.join(OUT_WIN, "marionette.ico"),
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote icon assets to", OUT_WIN, "and", OUT_LNX)
