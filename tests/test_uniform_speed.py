from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UniformSpeedTests(unittest.TestCase):
    def test_sharp_sources_use_smaller_generation_bucket(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("source_min_dimension >= 360", source)
        self.assertIn("source_face_detail >= 120.0", source)
        self.assertIn(
            "uniform_generation_dimension = 1024 if fast_portrait_bucket else 1216",
            source,
        )

    def test_uniform_route_uses_only_person_and_garment_visuals(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        route = source[source.index("def qwen_vl_image_edit_uniform_swap("):]
        references = route[
            route.index("qwen_garment_references = ["):
            route.index("arm_generation_instruction = (")
        ]
        self.assertIn("uniform_validation_reference", references)
        self.assertNotIn("selected_bg_reference", references)

    def test_exact_prompt_embeddings_are_cached_with_bounded_retention(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('prompt_cache_dir = CACHE_DIR / "prompt_embeddings"', source)
        self.assertIn("cache_hasher.update(temp_path.read_bytes())", source)
        self.assertIn("for stale_entry in cached_entries[12:]", source)

    def test_confident_hair_accessories_receive_face_free_visual_reference(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("hair_accessory_reference.png", source)
        self.assertIn("face-free crown/temple detail", source)
        self.assertIn("qwen_garment_references.append(accessory_reference)", source)


if __name__ == "__main__":
    unittest.main()
