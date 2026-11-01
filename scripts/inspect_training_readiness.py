"""Read the preserved ZIP, measure image quality, and review rotated/flipped matches."""

from pathlib import Path
from zipfile import ZipFile
import hashlib
import io
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research" / "deep_inspection_2026-10-29"
CLASSES = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth Ulcer", "Tooth Discoloration"]
EXPECTED = "0f17eb5efff8ceba5ecb00b2667cebaa6d014c0e1ae44d160ecc30a9de64a367"
N = 32
DCT = np.cos(np.pi * (2 * np.arange(N)[None, :] + 1) * np.arange(8)[:, None] / (2 * N))


def phash(gray):
    coefficients = (DCT @ gray @ DCT.T).ravel()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = PROJECT / "data/raw/oral-diseases-v3.zip"
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == EXPECTED
    source = pd.read_csv(PROJECT / "research/kaggle_starter_check/source_manifest.csv", keep_default_na=False)
    originals = source.sort_values("path").drop_duplicates("pixel_sha256").reset_index(drop=True)
    eligible = pd.read_csv(PROJECT / "research/oral_preparation/eligible_proposal_candidates.csv")
    all_families = pd.read_csv(PROJECT / "research/oral_preparation/all_source_families.csv")
    families = all_families.drop_duplicates("pixel_sha256").set_index("pixel_sha256")["family_id"].to_dict()
    labels = source.groupby("pixel_sha256")["class"].agg(lambda values: " | ".join(sorted(set(values)))).to_dict()
    variants = ["identity", "rotate90", "rotate180", "rotate270", "mirror", "mirror_rotate90", "mirror_rotate180", "mirror_rotate270"]
    hash_values = {name: [] for name in variants}
    records = []
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    with ZipFile(archive) as z:
        def rgb(path):
            with Image.open(io.BytesIO(z.read(path))) as im:
                return ImageOps.exif_transpose(im).convert("RGB")

        for i, row in originals.iterrows():
            im = rgb(row["path"])
            actual = hashlib.sha256(str(im.size).encode() + b":" + im.tobytes()).hexdigest()
            assert actual == row["pixel_sha256"], row["path"]
            gray = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
            for rotation in range(4):
                hash_values[variants[rotation]].append(phash(np.rot90(gray, rotation)))
                hash_values[variants[rotation + 4]].append(phash(np.rot90(np.fliplr(gray), rotation)))
            preview = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32)
            laplacian = (preview[1:-1, :-2] + preview[1:-1, 2:] + preview[:-2, 1:-1] + preview[2:, 1:-1] - 4 * preview[1:-1, 1:-1])
            channel = np.asarray(im.resize((128, 128)), dtype=np.float32)
            records.append({"path": row["path"], "pixel_sha256": actual, "width": im.width, "height": im.height,
                            "laplacian_variance_128": float(laplacian.var()),
                            "dark_pixel_fraction": float((channel.max(axis=2) < 10).mean()),
                            "white_pixel_fraction": float((channel.min(axis=2) > 245).mean()),
                            "mean_channel_difference": float(np.abs(channel[:, :, 0] - channel[:, :, 1]).mean() + np.abs(channel[:, :, 1] - channel[:, :, 2]).mean())})
            if (i + 1) % 1000 == 0:
                print(f"Read and verified {i+1}/{len(originals)} distinct source images", flush=True)
        quality = pd.DataFrame(records)
        quality.to_csv(OUT / "quality_measurements.csv", index=False)
        original_values = np.array(hash_values["identity"], dtype=np.uint64)
        edges = {}
        for variant in variants:
            values = np.array(hash_values[variant], dtype=np.uint64)
            for start in range(0, len(values), 256):
                distances = np.bitwise_count(values[start:start+256, None] ^ original_values[None, :])
                lefts, rights = np.where(distances <= 4)
                for offset, right in zip(lefts, rights):
                    left = start + int(offset)
                    right = int(right)
                    if left >= right:
                        continue
                    key = (left, right)
                    distance = int(distances[offset, right])
                    if key not in edges or distance < edges[key][0]:
                        edges[key] = (distance, variant)
            print("Compared", variant, flush=True)
        edge_rows = []
        for (left, right), (distance, variant) in sorted(edges.items()):
            a, b = originals.iloc[left], originals.iloc[right]
            edge_rows.append({"left": a["path"], "right": b["path"], "left_hash": a["pixel_sha256"], "right_hash": b["pixel_sha256"],
                              "left_labels": labels[a["pixel_sha256"]], "right_labels": labels[b["pixel_sha256"]],
                              "distance": distance, "transform": variant,
                              "outside_current_family": families[a["pixel_sha256"]] != families[b["pixel_sha256"]]})
        edge_frame = pd.DataFrame(edge_rows)
        edge_frame.to_csv(OUT / "rotation_mirror_similarity_pairs.csv", index=False)
        additions = edge_frame.loc[edge_frame["outside_current_family"]].copy()
        additions.to_csv(OUT / "new_family_link_candidates.csv", index=False)
        proposal = pd.read_csv(PROJECT / "research/oral_preparation/proposed_split_manifest.csv")
        splits = proposal.set_index("pixel_sha256")["split"].to_dict()
        additions["left_split"] = additions["left_hash"].map(splits)
        additions["right_split"] = additions["right_hash"].map(splits)
        crossing = additions.loc[additions["left_split"].notna() & additions["right_split"].notna() & additions["left_split"].ne(additions["right_split"])].copy()
        crossing.to_csv(OUT / "new_links_crossing_proposed_split.csv", index=False)

        def sheet(frame, filename):
            canvas = Image.new("RGB", (1200, 5 * 215), "#eef1f6")
            draw = ImageDraw.Draw(canvas)
            for slot, (_, row) in enumerate(frame.head(20).iterrows()):
                x, y = (slot % 4) * 300, (slot // 4) * 215
                im = rgb(row["path"])
                tile = ImageOps.contain(im, (286, 167))
                canvas.paste(tile, (x + 7 + (286-tile.width)//2, y + 4 + (167-tile.height)//2))
                draw.text((x+7,y+174), f"{slot+1:02d} | {im.width}x{im.height} | {Path(row['path']).name[:20]}", font=font, fill="#142138")
                draw.text((x+7,y+193), f"family {row['family_id'][-9:]}", font=font, fill="#142138")
            canvas.save(OUT / filename)

        sample_records = []
        for label in CLASSES:
            frame = eligible.loc[eligible["class"].eq(label)].merge(quality[["pixel_sha256","laplacian_variance_128"]], on="pixel_sha256", validate="one_to_one")
            low = frame.sort_values(["short_edge","path"]).head(3)
            wide = frame.sort_values(["aspect_ratio","path"], ascending=[False,True]).head(3)
            diffuse = frame.sort_values(["laplacian_variance_128","path"]).head(3)
            independent = frame.drop_duplicates("family_id").sample(n=min(20,frame["family_id"].nunique()),random_state=20261029)
            sample = pd.concat([low,wide,diffuse,independent]).drop_duplicates("path").head(20)
            sample_records.extend(sample.to_dict("records"))
            sheet(sample, label.lower().replace(" ","_")+"_review.jpg")
        pd.DataFrame(sample_records).to_csv(OUT / "visual_review_samples.csv", index=False)

        # Prioritize potential cross-partition matches and vary the source class.
        review_pool = crossing.sort_values(["distance","left"])
        review_pool["left_class"] = review_pool["left_labels"].str.split(" | ",regex=False).str[0]
        sample_pairs = review_pool.groupby("left_class",group_keys=False).head(3).head(12)
        canvas = Image.new("RGB", (1000,max(1,len(sample_pairs))*250),"#eef1f6")
        draw = ImageDraw.Draw(canvas)
        for slot, (_, row) in enumerate(sample_pairs.iterrows()):
            for side, x in [("left",0),("right",500)]:
                im=rgb(row[side]); tile=ImageOps.contain(im,(480,200))
                canvas.paste(tile,(x+10+(480-tile.width)//2,slot*250))
                label=f"{row[side+'_labels']} | {row[side+'_split']}"
                draw.text((x+10,slot*250+204),label[:60],font=font,fill="#142138")
                draw.text((x+10,slot*250+225),f"{row['transform']} | d={row['distance']} | {Path(row[side]).name[:20]}",font=font,fill="#142138")
        canvas.save(OUT / "new_cross_partition_matches.jpg")
        sample_pairs.to_csv(OUT / "new_cross_partition_preview.csv",index=False)
        summary={"archive_sha256":EXPECTED,"distinct_source_images_read_and_hash_verified":len(originals),
                 "visual_sample_images":len(sample_records),"all_dihedral_phash_pairs":len(edge_frame),
                 "additional_family_link_candidates":len(additions),"additional_links_crossing_proposed_split":len(crossing),
                 "new_candidate_transforms":additions["transform"].value_counts().to_dict(),
                 "warning":"Perceptual matches require visual review. These are candidates, not validated duplicate or patient labels."}
        (OUT / "inspection_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
