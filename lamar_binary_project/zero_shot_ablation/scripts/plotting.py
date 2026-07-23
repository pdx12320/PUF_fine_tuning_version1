#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def line_plot(
    series, path, title, x_label, y_label, diagonal=False,
    x_max=1.0, y_max=1.0,
):
    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 170, 1700, 100, 1040
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, bottom, left, top), fill="black", width=3)
    x_format = "{:.3f}" if x_max <= 0.02 else "{:.1f}"
    y_format = "{:.3f}" if y_max <= 0.02 else "{:.1f}"
    for fraction in np.linspace(0, 1, 6):
        x_tick = fraction * x_max
        y_tick = fraction * y_max
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, bottom, x, bottom + 10), fill="black", width=2)
        draw.text(
            (x - 32, bottom + 18), x_format.format(x_tick),
            fill="black", font=font(28),
        )
        draw.line((left - 10, y, left, y), fill="black", width=2)
        draw.text(
            (left - 95, y - 18), y_format.format(y_tick),
            fill="black", font=font(28),
        )
    if diagonal:
        draw.line((left, bottom, right, top), fill="#777777", width=3)
    for index, (name, x_values, y_values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = [
            (
                left + np.clip(float(x) / x_max, 0, 1) * (right - left),
                bottom - np.clip(float(y) / y_max, 0, 1) * (bottom - top),
            )
            for x, y in zip(x_values, y_values)
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_y = 145 + index * 48
        draw.line((1240, legend_y, 1300, legend_y), fill=color, width=7)
        draw.text((1320, legend_y - 18), name, fill="black", font=font(29))
    draw.text((width // 2, 35), title, fill="black", anchor="mm", font=font(42, True))
    draw.text((width // 2, 1155), x_label, fill="black", anchor="mm", font=font(34))
    label = Image.new("RGBA", (500, 60), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((250, 30), y_label, fill="black", anchor="mm", font=font(34))
    label = label.rotate(90, expand=True)
    image.paste(label, (15, 360), label)
    image.save(path, dpi=(180, 180))


def scatter_plot(frame, path, title):
    width, height = 1800, 1200
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, right, top, bottom = 170, 1700, 100, 1040
    x_values = frame.PC1.to_numpy()
    y_values = frame.PC2.to_numpy()
    x_low, x_high = np.quantile(x_values, [.01, .99])
    y_low, y_high = np.quantile(y_values, [.01, .99])
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, bottom, left, top), fill="black", width=3)
    order = np.arange(len(frame))
    if len(order) > 20000:
        order = np.random.default_rng(42).choice(order, 20000, replace=False)
    for index in order:
        x = np.clip((x_values[index] - x_low) / max(x_high - x_low, 1e-8), 0, 1)
        y = np.clip((y_values[index] - y_low) / max(y_high - y_low, 1e-8), 0, 1)
        px = left + x * (right - left)
        py = bottom - y * (bottom - top)
        color = (228, 87, 86, 150) if int(frame.iloc[index].label) else (76, 120, 168, 45)
        radius = 5 if int(frame.iloc[index].label) else 2
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color)
    draw.text((width // 2, 35), title, fill="black", anchor="mm", font=font(42, True))
    draw.text((width // 2, 1155), "PC1", fill="black", anchor="mm", font=font(34))
    draw.text((50, 570), "PC2", fill="black", anchor="mm", font=font(34))
    draw.ellipse((1250, 140, 1262, 152), fill=(228, 87, 86, 220))
    draw.text((1280, 130), "computational positive", fill="black", font=font(29))
    draw.ellipse((1250, 190, 1258, 198), fill=(76, 120, 168, 180))
    draw.text((1280, 178), "strict computational negative", fill="black", font=font(29))
    image.convert("RGB").save(path, dpi=(180, 180))


def heatmap(frame, path, title):
    components = list(dict.fromkeys(frame.component))
    features = list(dict.fromkeys(frame.feature))
    matrix = np.zeros((len(components), len(features)))
    for _, row in frame.iterrows():
        matrix[components.index(row.component), features.index(row.feature)] = row.pearson
    cell_w, cell_h = 230, 55
    left, top = 220, 150
    width = left + cell_w * len(features) + 170
    height = top + cell_h * len(components) + 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 45), title, fill="black", anchor="mm", font=font(38, True))
    for column, feature in enumerate(features):
        draw.text((left + column * cell_w + cell_w / 2, 105), feature, fill="black", anchor="mm", font=font(24))
    for row, component in enumerate(components):
        draw.text((left - 20, top + row * cell_h + cell_h / 2), component, fill="black", anchor="rm", font=font(23))
        for column in range(len(features)):
            value = float(matrix[row, column])
            if value >= 0:
                shade = int(255 - 170 * min(value, 1))
                color = (255, shade, shade)
            else:
                shade = int(255 - 170 * min(-value, 1))
                color = (shade, shade, 255)
            x0, y0 = left + column * cell_w, top + row * cell_h
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color, outline="white")
            draw.text((x0 + cell_w / 2, y0 + cell_h / 2), f"{value:+.2f}", fill="black", anchor="mm", font=font(22))
    image.save(path, dpi=(180, 180))
