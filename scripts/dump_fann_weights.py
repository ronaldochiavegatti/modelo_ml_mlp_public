#!/usr/bin/env python3
"""Dump the weights inside a FANN .net file into CSV tables.

Each fully connected layer (input->hidden, hidden->output, etc.) gets a CSV where
every row is a target neuron and every column is a source neuron from the previous
layer. The CSVs live under `<output_dir>/<model stem>/` so you can open them in a
spreadsheet viewer to inspect the actual numbers that underlie the SVG.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_layer_sizes(text: str) -> List[int]:
    match = re.search(r"^layer_sizes=([0-9 ]+)", text, re.MULTILINE)
    if not match:
        raise ValueError("layer_sizes line not found in .net file")
    return [int(token) for token in match.group(1).split()]


def parse_neurons(text: str) -> List[Tuple[int, int, float]]:
    match = re.search(
        r"neurons \(num_inputs, activation_function, activation_steepness\)=(.*?)connections \(connected_to_neuron, weight\)=",
        text,
        re.S,
    )
    if not match:
        raise ValueError("neurons block missing")
    block = match.group(1)
    triples: List[Tuple[int, int, float]] = []
    for item in re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", block):
        triples.append((int(item.group(1)), int(item.group(2)), float(item.group(3))))
    return triples


def parse_connections(text: str) -> List[Tuple[int, float]]:
    match = re.search(r"connections \(connected_to_neuron, weight\)=(.*)", text, re.S)
    if not match:
        raise ValueError("connections block missing")
    block = match.group(1)
    return [(int(item.group(1)), float(item.group(2))) for item in re.finditer(r"\(([^,]+),\s*([^)]+)\)", block)]


def build_mapping(layer_sizes: List[int]) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    offsets: List[int] = []
    current = 0
    for size in layer_sizes:
        offsets.append(current)
        current += size
    mapping: Dict[int, Tuple[int, int]] = {}
    for layer_idx, (offset, size) in enumerate(zip(offsets, layer_sizes)):
        for local_idx in range(size):
            mapping[offset + local_idx] = (layer_idx, local_idx)
    return offsets, mapping


def collect_weights(
    neurons: List[Tuple[int, int, float]],
    connections: List[Tuple[int, float]],
    layer_sizes: List[int],
    offsets: List[int],
    mapping: Dict[int, Tuple[int, int]],
):
    rows_by_layer: Dict[int, List[Dict[str, object]]] = {}
    conn_cursor = 0
    total_neurons = sum(layer_sizes)
    if len(neurons) != total_neurons:
        raise ValueError(
            f"Expected {total_neurons} neuron entries but found {len(neurons)}"
        )

    for global_idx, (num_inputs, _, _) in enumerate(neurons):
        if num_inputs == 0:
            continue
        target_layer, target_local = mapping[global_idx]
        if target_layer == 0:
            conn_cursor += num_inputs
            continue
        prev_layer = target_layer - 1
        layer_size = layer_sizes[prev_layer]
        if conn_cursor + num_inputs > len(connections):
            raise ValueError("connection block ended unexpectedly")
        window = connections[conn_cursor : conn_cursor + num_inputs]
        conn_cursor += num_inputs
        weights = [None] * layer_size
        extras: List[Tuple[int, float]] = []
        for source_global, weight in window:
            source_info = mapping.get(source_global)
            if source_info and source_info[0] == prev_layer:
                weights[source_info[1]] = weight
            else:
                extras.append((source_global, weight))
        rows_by_layer.setdefault(target_layer, []).append(
            {
                "target_global": global_idx,
                "target_local": target_local,
                "weights": weights,
                "extras": extras,
            }
        )
    if conn_cursor != len(connections):
        print("Warning: not all connections were consumed (some may belong to unused neurons)")
    return rows_by_layer


def write_layer_csv(
    output_dir: Path,
    model_stem: str,
    layer_idx: int,
    prev_layer: int,
    rows: List[Dict[str, object]],
    offsets: List[int],
    layer_sizes: List[int],
) -> Path:
    source_offset = offsets[prev_layer]
    source_globals = [source_offset + i for i in range(layer_sizes[prev_layer])]
    filename = f"{model_stem}-layer{prev_layer}-to-layer{layer_idx}.csv"
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["target_global", "target_layer", "target_local"] + [
            f"src_global_{idx}" for idx in source_globals
        ]
        writer.writerow(header)
        for entry in rows:
            packed = [
                entry["target_global"],
                layer_idx,
                entry["target_local"],
            ]
            for weight in entry["weights"]:
                packed.append("" if weight is None else repr(weight))
            writer.writerow(packed)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump a trained FANN .net file into per-layer CSV weight tables."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to the .net file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Base directory for the exported CSVs (default: weights/)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model file '{args.model}' does not exist")
    text = args.model.read_text(encoding="utf-8", errors="ignore")
    layer_sizes = parse_layer_sizes(text)
    neurons = parse_neurons(text)
    connections = parse_connections(text)
    offsets, mapping = build_mapping(layer_sizes)
    rows_by_layer = collect_weights(neurons, connections, layer_sizes, offsets, mapping)

    model_stem = args.model.stem
    exported_paths: List[Path] = []
    for layer_idx, rows in sorted(rows_by_layer.items()):
        prev_layer = layer_idx - 1
        if prev_layer < 0:
            continue
        target_path = write_layer_csv(
            args.output_dir,
            model_stem,
            layer_idx,
            prev_layer,
            rows,
            offsets,
            layer_sizes,
        )
        exported_paths.append(target_path)
        print(
            f"Layer {prev_layer}→{layer_idx}: {len(rows)} targets × {layer_sizes[prev_layer]} sources written to {target_path}"
        )
    if not exported_paths:
        print("No weights found (the model may not have any fully connected layers)")
        return 1
    print(f"All CSVs live under {args.output_dir / model_stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
