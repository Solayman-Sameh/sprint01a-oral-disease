"""Real checkpoint and preprocessing checks; no network or training required."""

import ast
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import unittest

from PIL import Image, ImageOps
import torch
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oral_app.inference import CLASSES, ImageInputError, load_model, predict, prepare_image
from oral_app.app import build_results_html, classify_image, clear_image, switch_language
from oral_app.translations import friendly_label, text


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.mapping = load_model()
        cls.example_files = sorted((ROOT / "oral_app/assets/examples").glob("*.jpg"))

    def test_model_is_cached_and_uses_evaluation_mode(self):
        self.assertIs(load_model()[0], self.model)
        self.assertFalse(self.model.training)
        self.assertEqual(tuple(self.mapping.values()), CLASSES)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 23_520_326)

    def test_transform_matches_the_actual_training_source(self):
        # Execute only the two original transform definitions, without importing
        # training libraries or starting its distributed runner.
        source = ast.parse((ROOT / "models/fine-tuned/train_kaggle.py").read_text())
        definitions = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in ("PadSquare", "image_transform")]
        namespace = {"ImageOps": ImageOps, "models": models, "transforms": transforms}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "training_transform", "exec"), namespace)
        reference, _ = namespace["image_transform"](
            "resnet50", {"resnet50": {"weights_enum": "ResNet50_Weights.IMAGENET1K_V2"}}, 224, False)
        images = [Image.new("L", (57, 93), 101), Image.new("RGBA", (93, 57), (29, 60, 140, 0)),
                  Image.new("RGB", (127, 128), (117, 33, 208))]
        with Image.open(self.example_files[0]) as real:
            images.append(real.copy())
        for image in images:
            with self.subTest(mode=image.mode, size=image.size):
                expected = reference(ImageOps.exif_transpose(image).convert("RGB"))
                torch.testing.assert_close(prepare_image(image), expected, atol=0, rtol=0)

    def test_exif_phone_rotation_is_applied(self):
        image = Image.new("RGB", (81, 123), "red")
        image.paste("blue", (0, 0, 30, 70))
        exif = image.getexif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        buffer.seek(0)
        with Image.open(buffer) as phone_image:
            upright = ImageOps.exif_transpose(phone_image)
            self.assertEqual(upright.size, (123, 81))
            torch.testing.assert_close(prepare_image(phone_image), prepare_image(upright), atol=0, rtol=0)

    def test_real_examples_and_non_rgb_have_valid_repeatable_scores(self):
        latencies = []
        for path in self.example_files:
            with Image.open(path) as image:
                started = time.perf_counter()
                results = predict(self.model, image, self.mapping)
                latencies.append(time.perf_counter() - started)
                repeated = predict(self.model, image, self.mapping)
                self.assertEqual(results, repeated)
                self.assertEqual({row["class"] for row in results}, set(CLASSES))
                self.assertAlmostEqual(sum(row["probability"] for row in results), 1.0, places=6)
                self.assertTrue(all(0 <= row["probability"] <= 1 for row in results))
                self.assertEqual(results, sorted(results, key=lambda row: row["probability"], reverse=True))
        for image in [Image.new("L", (73, 123), 90), Image.new("RGBA", (123, 73), (0, 200, 20, 128))]:
            self.assertEqual(len(predict(self.model, image, self.mapping)), 6)
        print(f"\nLocal CPU end-to-end prediction: median={sorted(latencies)[len(latencies)//2]:.3f}s, max={max(latencies):.3f}s")

    def test_oversized_or_missing_input_is_handled(self):
        with self.assertRaisesRegex(ImageInputError, "image_too_large"):
            prepare_image(Image.new("RGB", (4473, 1)))
        with self.assertRaisesRegex(ImageInputError, "invalid_image"):
            prepare_image("not an image")
        for lang in ("en", "ar"):
            state, html = classify_image(None, lang)
            self.assertIsNone(state)
            self.assertIn(text("no_image", lang), html)

    def test_language_switch_translates_existing_results_and_preserves_input(self):
        with Image.open(self.example_files[0]) as image:
            results = predict(self.model, image, self.mapping)
        for lang in ("en", "ar"):
            update = switch_language(lang, results)
            self.assertEqual(update[0], lang)
            self.assertNotIn("value", update[3])  # Image label changes, uploaded pixels survive.
            self.assertIn(text(results[0]["class"], lang), update[12])
            self.assertEqual(update[12].count('role="meter"'), 6)
            self.assertIn(f'dir="{"rtl" if lang == "ar" else "ltr"}"', update[12])
        self.assertIn("rtl-mode", switch_language("ar", results)[1])
        self.assertIn("ltr-mode", switch_language("en", results)[1])
        self.assertIsNone(clear_image("ar")[0])
        self.assertIsNone(clear_image("ar")[1])

    def test_score_threshold_boundaries(self):
        for score, key in [(1.0, "very_likely"), (.70, "very_likely"), (.699, "likely"),
                           (.40, "likely"), (.15, "possible"), (.05, "unlikely"), (0, "very_unlikely")]:
            for lang in ("en", "ar"):
                self.assertEqual(friendly_label(score, lang), text(key, lang))

    def test_examples_match_the_frozen_training_split(self):
        import csv
        import hashlib
        manifest_path = ROOT / "research/oral_preparation/proposed_split_manifest.csv"
        examples = json.loads((ROOT / "oral_app/assets/examples/manifest.json").read_text())
        self.assertEqual(hashlib.sha256(manifest_path.read_text().encode()).hexdigest(), examples["manifest_sha256"])
        with manifest_path.open(encoding="utf-8-sig") as stream:
            rows = {row["path"]: row for row in csv.DictReader(stream)}
        self.assertEqual(len(examples["examples"]), 6)
        for item in examples["examples"]:
            row = rows[item["source_path"]]
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["class"], item["class"])
            self.assertEqual(hashlib.sha256((ROOT / "oral_app/assets/examples" / item["file"]).read_bytes()).hexdigest(), item["file_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
