from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UniformSpeedTests(unittest.TestCase):
    def test_generation_bucket_selection(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_UNIFORM_GENERATION_DIMENSION", source)
        self.assertIn('"turbo" in str(generation_mode).lower()', source)
        self.assertIn("uniform_generation_dimension = 1024", source)

    def test_uniform_route_uses_clean_person_plus_one_garment_visual(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        route = source[source.index("def qwen_vl_image_edit_uniform_swap("):]
        references = route[
            route.index("qwen_garment_references = "):
            route.index("arm_generation_instruction = (")
        ]
        self.assertIn(
            "qwen_garment_references = [uniform_validation_reference]",
            references,
        )
        self.assertIn("direct_uniform_seed", route)
        self.assertNotIn("selected_bg_reference", references)

    def test_exact_prompt_embeddings_are_cached_with_bounded_retention(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('prompt_cache_dir = CACHE_DIR / "prompt_embeddings"', source)
        self.assertIn("cache_hasher.update(temp_path.read_bytes())", source)
        self.assertIn("for stale_entry in cached_entries[64:]", source)

    def test_hair_accessories_do_not_confuse_garment_conditioning(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("Never attach this head-shaped crop to Qwen", source)
        self.assertIn("confuse garment conditioning", source)


if __name__ == "__main__":
    unittest.main()
