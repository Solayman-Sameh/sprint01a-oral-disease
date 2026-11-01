"""Read an immutable Kaggle ZIP and inventory images; never execute notebook code.

Requires Pillow and NumPy. Writes manifests, exact-duplicate records, summaries,
and contact sheets. Perceptual-hash matches are review candidates, not removals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ORIGINAL_PREFIXES = {
    "Calculus": "Calculus/Calculus/",
    "Caries": "Data caries/Data caries/caries orignal data set/done/",
    "Gingivitis": "Gingivitis/Gingivitis/",
    "Hypodontia": "hypodontia/hypodontia/",
    "Mouth Ulcer": "Mouth Ulcer/Mouth Ulcer/ulcer original dataset/ulcer original dataset/",
    "Tooth Discoloration": "Tooth Discoloration/Tooth Discoloration /tooth discoloration original dataset/tooth discoloration original dataset/",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
N = 32
DCT = np.cos(np.pi * (2 * np.arange(N)[None, :] + 1) * np.arange(8)[:, None] / (2 * N))


def phash(im: Image.Image) -> str:
    gray = np.asarray(im.convert("L").resize((N, N), Image.Resampling.LANCZOS), dtype=np.float32)
    coeff = DCT @ gray @ DCT.T
    cutoff = np.median(coeff.flatten()[1:])
    bits = coeff.flatten() > cutoff
    bits[0] = False
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, nonimages, failures = [], [], []
    file_groups, pixel_groups = defaultdict(list), defaultdict(list)
    with ZipFile(args.archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for index, member in enumerate(members):
            name = member.filename
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                nonimages.append(name)
                continue
            class_name = next((c for c, prefix in ORIGINAL_PREFIXES.items() if name.startswith(prefix)), "")
            subset = "original_source_folder" if class_name else "other_or_augmented"
            if not class_name:
                class_name = {"Data caries": "Caries", "hypodontia": "Hypodontia"}.get(name.split("/")[0], name.split("/")[0])
            content = z.read(member)
            file_hash = hashlib.sha256(content).hexdigest()
            row = {"path": name, "class": class_name, "subset": subset, "bytes": len(content), "sha256": file_hash,
                   "width": "", "height": "", "mode": "", "format": "", "pixel_sha256": "", "phash": "", "decode_error": ""}
            try:
                with Image.open(io.BytesIO(content)) as im:
                    im.load()
                    row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    pixel_hash = hashlib.sha256(str(rgb.size).encode() + b":" + rgb.tobytes()).hexdigest()
                    row["pixel_sha256"] = pixel_hash
                    if subset == "original_source_folder":
                        row["phash"] = phash(rgb)
                    pixel_groups[pixel_hash].append(len(rows))
            except Exception as exc:
                row["decode_error"] = str(exc)
                failures.append({"path": name, "error": str(exc)})
            file_groups[file_hash].append(len(rows))
            rows.append(row)
            if (index + 1) % 1000 == 0:
                print(f"Read {index + 1} archive files", flush=True)

        def duplicate_records(groups):
            return [{"hash": digest, "paths": [rows[i]["path"] for i in indexes],
                     "classes": sorted({rows[i]["class"] for i in indexes}),
                     "original_paths": [rows[i]["path"] for i in indexes if rows[i]["subset"] == "original_source_folder"]}
                    for digest, indexes in groups.items() if len(indexes) > 1]

        exact = duplicate_records(file_groups)
        pixels = duplicate_records(pixel_groups)
        original = [r for r in rows if r["subset"] == "original_source_folder"]
        valid_original = [r for r in original if not r["decode_error"]]
        original_file_groups = defaultdict(list)
        original_pixel_groups = defaultdict(list)
        for row in valid_original:
            original_file_groups[row["sha256"]].append(row)
            original_pixel_groups[row["pixel_sha256"]].append(row)

        # Compare one representative of each decoded-pixel group, not duplicate copies.
        representatives = [group[0] for group in original_pixel_groups.values()]
        near = []
        for i, left in enumerate(representatives):
            value = int(left["phash"], 16)
            for right in representatives[i + 1:]:
                distance = (value ^ int(right["phash"], 16)).bit_count()
                if distance <= 4:
                    near.append({"left": left["path"], "right": right["path"], "distance": distance,
                                 "same_label": left["class"] == right["class"]})

        def group_stats(groups):
            duplicates = [v for v in groups.values() if len(v) > 1]
            return {"groups": len(duplicates), "redundant_copies": sum(len(v) - 1 for v in duplicates),
                    "files_in_groups": sum(map(len, duplicates)), "unique_values": len(groups),
                    "cross_label_groups": sum(len({r['class'] for r in v}) > 1 for v in duplicates)}

        by_class = []
        for class_name in ORIGINAL_PREFIXES:
            group = [r for r in original if r["class"] == class_name]
            valid = [r for r in group if not r["decode_error"]]
            by_class.append({"class": class_name, "source_folder_images": len(group), "readable": len(valid),
                             "unique_file_sha256": len({r["sha256"] for r in valid}),
                             "unique_decoded_pixels": len({r["pixel_sha256"] for r in valid}),
                             "min_width": min((r["width"] for r in valid), default=0),
                             "max_width": max((r["width"] for r in valid), default=0),
                             "min_height": min((r["height"] for r in valid), default=0),
                             "max_height": max((r["height"] for r in valid), default=0),
                             "short_edge_below_100": sum(min(r["width"], r["height"]) < 100 for r in valid),
                             "short_edge_below_224": sum(min(r["width"], r["height"]) < 224 for r in valid)})
        sizes = Counter((r["width"], r["height"]) for r in valid_original)
        summary = {"dataset_ref": "salmansajid05/oral-diseases", "dataset_version": 3, "audit_date": "2026-10-27",
                   "archive": args.archive.name, "archive_bytes": args.archive.stat().st_size, "archive_sha256": sha(args.archive),
                   "archive_file_count": len(members), "image_count": len(rows), "nonimage_count": len(nonimages),
                   "nonimage_paths": nonimages, "image_decode_failures": failures,
                   "original_source_folder_images": len(original), "other_or_augmented_images": len(rows) - len(original),
                   "by_original_class": by_class, "original_file_duplicates": group_stats(original_file_groups),
                   "original_pixel_duplicates": group_stats(original_pixel_groups),
                   "all_image_file_duplicate_groups": len(exact), "all_image_pixel_duplicate_groups": len(pixels),
                   "all_image_file_redundant_copies": sum(len(g["paths"]) - 1 for g in exact),
                   "all_image_pixel_redundant_copies": sum(len(g["paths"]) - 1 for g in pixels),
                   "original_modes": dict(Counter(r["mode"] for r in valid_original)),
                   "most_common_original_dimensions": [{"width": s[0], "height": s[1], "count": n} for s, n in sizes.most_common(12)],
                   "phash_review_candidate_pairs_distance_le_4": len(near),
                   "phash_cross_label_candidate_pairs": sum(not p["same_label"] for p in near),
                   "limitations": ["Folder names do not establish patient identity or original acquisition provenance.",
                                   "No manual clinical label validation was performed.",
                                   "Perceptual hashes are review candidates, not confirmed duplicates or automatic exclusions.",
                                   "RGB pixel hash includes oriented dimensions and exact decoded pixels; it does not group crops or color changes."]}
        fields = list(rows[0])
        write_csv(args.output / "image_manifest.csv", rows, fields)
        write_csv(args.output / "class_inventory.csv", by_class, list(by_class[0]))
        write_csv(args.output / "near_duplicate_candidates.csv", near, ["left", "right", "distance", "same_label"])
        (args.output / "exact_duplicate_groups.json").write_text(json.dumps({"file_sha256": exact, "decoded_rgb_sha256": pixels}, indent=2), encoding="utf-8")
        (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
            small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
        except OSError:
            font = small = ImageFont.load_default()
        tile_w, tile_h, columns = 245, 190, 4
        canvas = Image.new("RGB", (tile_w * columns + 30, 55 + 6 * (tile_h + 36)), "#f4f6f8")
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 14), "Original-folder examples | 4 distinct decoded images per class", font=font, fill="#172335")
        for y, class_name in enumerate(ORIGINAL_PREFIXES):
            choices = [r for r in representatives if r["class"] == class_name]
            # Uniform positions in a stable list, not examples selected for model performance.
            picked = [choices[int(i * (len(choices) - 1) / (columns - 1))] for i in range(columns)]
            top = 55 + y * (tile_h + 36)
            draw.text((15, top), class_name, font=font, fill="#172335")
            for x, row in enumerate(picked):
                with Image.open(io.BytesIO(z.read(row["path"]))) as im:
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    thumb = ImageOps.contain(rgb, (tile_w - 10, tile_h - 22))
                    canvas.paste(thumb, (15 + x * tile_w, top + 25))
                draw.text((15 + x * tile_w, top + tile_h + 6), f"{row['width']} x {row['height']} px", font=small, fill="#42536a")
        canvas.save(args.output / "class_examples.jpg", quality=90)

        review = [g for g in pixels if len(g["original_paths"]) >= 2][:6]
        dup_canvas = Image.new("RGB", (1000, 50 + len(review) * 205), "#f4f6f8")
        dd = ImageDraw.Draw(dup_canvas)
        dd.text((15, 14), "Confirmed identical decoded pixels under different original-folder filenames", font=font, fill="#172335")
        for y, group in enumerate(review):
            for x, name in enumerate(group["original_paths"][:2]):
                with Image.open(io.BytesIO(z.read(name))) as im:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (460, 155))
                    dup_canvas.paste(thumb, (15 + 500 * x, 50 + y * 205))
                label = name if len(name) <= 65 else "..." + name[-62:]
                dd.text((15 + x * 500, 50 + y * 205 + 160), label, font=small, fill="#42536a")
        dup_canvas.save(args.output / "confirmed_duplicate_examples.jpg", quality=90)
        assert sum(c["source_folder_images"] for c in by_class) == len(original)
        assert len(rows) + len(nonimages) == len(members)
        display_summary = {k: v for k, v in summary.items() if k != "nonimage_paths"}
        print(json.dumps(display_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
