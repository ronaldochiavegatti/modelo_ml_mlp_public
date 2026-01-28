#!/usr/bin/env python3
import argparse
import os
import sys


def parse_layer_sizes(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("layer_sizes="):
                raw = line.split("=", 1)[1].strip()
                if not raw:
                    break
                sizes = [int(x) for x in raw.split()]
                if sizes:
                    return sizes
    raise ValueError("layer_sizes not found in .net file")


def layer_label(index, total):
    if index == 0:
        return "Input"
    if index == total - 1:
        return "Output"
    if total == 3:
        return "Hidden"
    return f"Hidden {index}"


def clamp(value, low, high):
    return max(low, min(high, value))


def build_svg(layer_sizes, title, output_path):
    width = 1200
    height = 700
    top = 140
    box_height = 420
    bottom_text_y = 640

    num_layers = len(layer_sizes)
    box_width = 220
    gap = 120

    total = num_layers * box_width + (num_layers - 1) * gap
    if total > width - 40:
        gap = 60
        total = num_layers * box_width + (num_layers - 1) * gap
    if total > width - 40:
        box_width = max(160, int((width - 40 - (num_layers - 1) * gap) / num_layers))
        total = num_layers * box_width + (num_layers - 1) * gap

    left = int((width - total) / 2)
    box_top = top
    box_bottom = top + box_height

    nodes_by_layer = []
    for size in layer_sizes:
        display_limit = 10 if size <= 10 else 8
        draw_count = min(size, display_limit)
        inner_top = box_top + 70
        inner_bottom = box_bottom - 60
        if draw_count == 1:
            positions = [int((inner_top + inner_bottom) / 2)]
        else:
            step = (inner_bottom - inner_top) / float(draw_count - 1)
            positions = [int(inner_top + step * i) for i in range(draw_count)]
        nodes_by_layer.append({"count": size, "draw": draw_count, "ys": positions})

    with open(output_path, "w", encoding="utf-8") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        out.write("  <defs>\n")
        out.write('    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">\n')
        out.write('      <stop offset="0%" stop-color="#f8fafc" />\n')
        out.write('      <stop offset="100%" stop-color="#eef2ff" />\n')
        out.write("    </linearGradient>\n")
        out.write("    <style>\n")
        out.write(
            "      .title { font: 600 22px Helvetica, Arial, sans-serif; fill: #0f172a; }\n"
        )
        out.write(
            "      .subtitle { font: 12px Helvetica, Arial, sans-serif; fill: #475569; }\n"
        )
        out.write(
            "      .layer-box { fill: #ffffff; stroke: #cbd5e1; stroke-width: 2; }\n"
        )
        out.write(
            "      .layer-title { font: 600 16px Helvetica, Arial, sans-serif; fill: #0f172a; }\n"
        )
        out.write(
            "      .layer-info { font: 12px Helvetica, Arial, sans-serif; fill: #475569; }\n"
        )
        out.write("      .node { stroke: #0f172a; stroke-width: 1; }\n")
        out.write("      .node-input { fill: #1d4ed8; }\n")
        out.write("      .node-hidden { fill: #2563eb; }\n")
        out.write("      .node-output { fill: #16a34a; }\n")
        out.write(
            "      .connection { stroke: #94a3b8; stroke-width: 1; opacity: 0.35; }\n"
        )
        out.write(
            "      .ellipsis { font: 18px Helvetica, Arial, sans-serif; fill: #64748b; }\n"
        )
        out.write("    </style>\n")
        out.write("  </defs>\n\n")

        out.write('  <rect width="100%" height="100%" fill="url(#bg)" />\n')
        out.write(
            f'  <text x="{width // 2}" y="40" text-anchor="middle" class="title">{title}</text>\n'
        )
        out.write(
            f'  <text x="{width // 2}" y="65" text-anchor="middle" class="subtitle">'
            f"layer_sizes={' '.join(str(s) for s in layer_sizes)}</text>\n\n"
        )

        # Connections (sampled for readability).
        out.write('  <g id="connections">\n')
        for idx in range(num_layers - 1):
            left_layer = nodes_by_layer[idx]
            right_layer = nodes_by_layer[idx + 1]
            left_x = left + idx * (box_width + gap) + box_width // 2
            right_x = left + (idx + 1) * (box_width + gap) + box_width // 2
            left_nodes = left_layer["ys"][:4]
            right_nodes = right_layer["ys"][:4]
            for y1 in left_nodes:
                for y2 in right_nodes:
                    out.write(
                        f'    <line x1="{left_x}" y1="{y1}" x2="{right_x}" y2="{y2}" '
                        f'class="connection" />\n'
                    )
        out.write("  </g>\n\n")

        # Layers
        for idx, size in enumerate(layer_sizes):
            x = left + idx * (box_width + gap)
            center_x = x + box_width // 2
            label = layer_label(idx, num_layers)
            bias_note = " (incl bias)" if idx < num_layers - 1 and size > 1 else ""
            node_class = "node-hidden"
            if idx == 0:
                node_class = "node-input"
            elif idx == num_layers - 1:
                node_class = "node-output"

            out.write(f'  <g id="layer-{idx}">\n')
            out.write(
                f'    <rect x="{x}" y="{box_top}" width="{box_width}" '
                f'height="{box_height}" rx="16" class="layer-box" />\n'
            )
            out.write(
                f'    <text x="{center_x}" y="{box_top + 30}" '
                f'text-anchor="middle" class="layer-title">{label}</text>\n'
            )
            out.write(
                f'    <text x="{center_x}" y="{box_top + 52}" '
                f'text-anchor="middle" class="layer-info">{size} neurons{bias_note}</text>\n'
            )

            nodes = nodes_by_layer[idx]
            for y in nodes["ys"]:
                out.write(
                    f'    <circle cx="{center_x}" cy="{y}" r="10" class="node {node_class}" />\n'
                )
            if size > nodes["draw"]:
                out.write(
                    f'    <text x="{center_x}" y="{box_bottom - 20}" '
                    f'text-anchor="middle" class="ellipsis">...</text>\n'
                )
            out.write("  </g>\n\n")

        out.write(
            f'  <text x="{width // 2}" y="{bottom_text_y}" text-anchor="middle" class="subtitle">'
            f"Fully connected layers (weights omitted)</text>\n"
        )
        out.write("</svg>\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simple SVG diagram from a FANN .net file."
    )
    parser.add_argument("--input", required=True, help="Path to FANN .net file")
    parser.add_argument(
        "--output", help="Output SVG path (default: input basename + .svg)"
    )
    parser.add_argument("--title", default="MLP Architecture", help="SVG title")
    args = parser.parse_args()

    try:
        sizes = parse_layer_sizes(args.input)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(args.input)
        output_path = base + ".svg"

    print(f"Layer sizes: {' '.join(str(s) for s in sizes)}")
    build_svg(sizes, args.title, output_path)
    print(f"Saved SVG to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
