"""Hair-extent measurement and long-hair prompt tests.

Job 30399633 uploaded a girl with two waist-length plaits. The conditioning
Qwen received had both plaits intact, but the pipeline reported

    measured_hair_end_ratio  0.212   ("ends around the neck-to-shoulder line")

against a ground truth of at least 0.73 face-heights below the chin, and the
live prompt carried no length or tied-section fact at all. Qwen returned a
jaw-length bob and the plaits were lost.

The cause was that extent was read off the *protection* mask, whose tied-hair
band is deliberately capped at y2 + face_h * .24 so dark outerwear is never
copied into the guide. Measurement now uses a wider, measurement-only mask.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COMPOSITE = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
BIREFNET = (ROOT / "pipelines/birefnet_service.py").read_text(encoding="utf-8")


class ExtentMaskTests(unittest.TestCase):
    """The measurement mask must be separate from the protection mask."""

    def test_extent_is_measured_on_its_own_mask(self):
        self.assertIn("extent_hair_mask", COMPOSITE)
        self.assertIn("hair_y, hair_x = np.where(extent_hair_mask)", COMPOSITE)

    def test_protection_mask_keeps_its_conservative_cap(self):
        # Widening what gets *copied* would drag a dark vest into the guide.
        self.assertIn("full_tail_band = (yy >= int(y2 - face_h * .03)) "
                      "& (yy <= int(y2 + face_h * .24))", COMPOSITE)

    def test_extent_band_reaches_past_the_protection_cap(self):
        self.assertIn("yy <= int(y2 + face_h * 1.60)", COMPOSITE)

    def test_extent_requires_connectivity_to_the_crown(self):
        # A dark collar must not be counted as a plait, so components still
        # have to connect to parsed hair - but via a dilated crown, because a
        # braid is routinely occluded between the crown and where it reappears.
        start = COMPOSITE.index("extent_hair_mask = measured_hair_mask")
        block = COMPOSITE[start:start + 2600]
        self.assertIn("component_mask & crown_reach", block)

    def test_connectivity_bridges_an_occlusion_gap(self):
        # On job 30399633's passport crop SCHP labels hair only to 0.24
        # face-heights ABOVE the chin while the plaits resume ~100 rows lower.
        # A strict touch test linked zero components and discarded both
        # plaits (20k and 15k pixels reaching 0.71 face-heights).
        start = COMPOSITE.index("extent_hair_mask = measured_hair_mask")
        block = COMPOSITE[start:start + 2600]
        self.assertIn("crown_reach = cv2.dilate(", block)
        self.assertIn("face_h * .95", block)

    def test_connectivity_dilation_draws_no_pixels(self):
        # crown_reach is a connectivity test only; it must never be OR-ed
        # into the mask that gets measured or copied.
        start = COMPOSITE.index("crown_reach = cv2.dilate(")
        block = COMPOSITE[start:start + 1200]
        self.assertNotIn("linked |= crown_reach", block)
        self.assertNotIn("extent_candidates |= crown_reach", block)

    def test_extent_never_shrinks_the_reported_hair(self):
        start = COMPOSITE.index("extent_hair_mask = measured_hair_mask")
        block = COMPOSITE[start:start + 3200]
        self.assertIn(
            "if np.count_nonzero(linked) >= np.count_nonzero(measured_hair_mask):",
            block,
        )

    def test_measurement_failure_cannot_break_the_composite(self):
        start = COMPOSITE.index("extent_hair_mask = measured_hair_mask")
        block = COMPOSITE[start:start + 3400]
        self.assertIn("except Exception:", block)

    def test_long_hair_bucket_still_exists(self):
        self.assertIn('"extends visibly below the shoulder line"', COMPOSITE)


class LongHairPromptTests(unittest.TestCase):
    """Measured length and tied count must reach the prompt that is sent."""

    def test_measured_length_is_stated_in_the_live_prompt(self):
        # Length is now phrased as a fact among the style facts rather than a
        # standalone imperative: the forceful wording produced correct length
        # but loose straight hair in place of the source's plaits, because
        # satisfying a length order is easier than reproducing a weave.
        self.assertIn("at its existing length, keeping the same strand ", PIPELINE)
        self.assertIn("weave or plaiting", PIPELINE)
        self.assertNotIn("do not shorten, cut or bob it.", PIPELINE)

    def test_tied_section_count_is_stated(self):
        self.assertIn("tied section(s); ", PIPELINE)

    def test_hair_facts_are_appended_to_the_sent_requirement(self):
        # compact_hair_requirement is the string the live compact prompt
        # interpolates; hair_shape_instruction is built but never sent.
        self.assertIn("compact_hair_requirement +=", PIPELINE)

    def test_shortening_terms_are_in_the_negative(self):
        for term in ("bob", "shortened hair", "lost braid", "lost plait"):
            self.assertIn(term, PIPELINE)

    def test_shortening_terms_are_pinned_against_budget_trimming(self):
        # The negative runs ~2.5x over the 512-token budget, so late terms are
        # dropped by position. These must survive in the retained tail.
        idx = PIPELINE.index('label="Uniform negative prompt"')
        tail = PIPELINE[idx - 1200:idx]
        self.assertIn("lost braid", tail)
        self.assertIn("bob", tail)

    def test_long_hair_terms_are_conditional_not_unconditional(self):
        # Banning "bob" on a genuinely short-haired upload would fight the
        # source, so the terms are gated on the measurement.
        self.assertIn("measured_tied_count >= 1 or measured_hair_extent.lower() in {",
                      PIPELINE)


if __name__ == "__main__":
    unittest.main()


class PlaitStructureTests(unittest.TestCase):
    """Length alone was not enough; structure must be named too.

    Runs braidfix2 and bothfix delivered the correct hair length but replaced
    the source's plaited braids with loose straight hair and dropped the blue
    bead ties. The VL analyser reports hair_texture "Straight" for this
    upload, so no braid fact reached Qwen from analysis.
    """

    def test_weave_and_ties_are_named_alongside_length(self):
        self.assertIn("weave or plaiting", PIPELINE)
        self.assertIn("tie, band and bead", PIPELINE)

    def test_tied_sections_ask_for_the_plait_construction(self):
        self.assertIn("if the strands are interwoven", PIPELINE)
        self.assertIn("keep its elastics, bands and beads", PIPELINE)

    def test_negative_names_the_observed_failure(self):
        for term in ("unbraided hair", "unplaited hair", "straightened hair",
                     "missing beads"):
            self.assertIn(term, PIPELINE)

    def test_bilateral_tied_hair_is_not_downgraded_by_fragmented_schp(self):
        # SCHP can miss both plaits even when the focused VL pass and visible
        # accessories independently confirm them.  Bilateral/count evidence
        # must keep the tied arrangement active instead of appending the
        # contradictory "no tied-back hairstyle" fallback.
        start = PIPELINE.index("preserve_tied_back_hair = bool(")
        block = PIPELINE[start:start + 650]
        self.assertIn('"bilateral"', block)
        self.assertIn("measured_tied_count > 0", block)
        self.assertIn("tied_accessory_corroborated", block)

    def test_impossible_one_sided_endpoint_is_not_sent_to_qwen(self):
        self.assertIn("Ignoring contradictory bilateral", PIPELINE)
        self.assertIn("_endpoint_disagreement > .60", PIPELINE)
        self.assertIn("measured_left_end_ratio = None", PIPELINE)
        self.assertIn("measured_right_end_ratio = None", PIPELINE)

    def test_vl_accessory_count_cannot_cap_visible_objects(self):
        # The focused pass counted two large bows but missed two lower beaded
        # ties. Image pixels, not that lossy integer, must remain authoritative.
        self.assertIn("not a numeric ", PIPELINE)
        self.assertIn("do not drop a second attachment", PIPELINE)
        retained = PIPELINE[
            PIPELINE.index("retained_hair_generation_lock = ("):
            PIPELINE.index("repair_regions =", PIPELINE.index("retained_hair_generation_lock = ("))
        ]
        self.assertNotIn("Keep exactly {analyzed_accessory_count}", retained)

    def test_focused_vl_distinguishes_plaits_and_counts_end_ties(self):
        analyzer = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("never call a visibly plaited section merely", analyzer)
        self.assertIn("beaded elastic at the lower end", analyzer)

    def test_tied_hair_recovery_reaches_the_measured_endpoint(self):
        self.assertIn("_measured_recovery_end", PIPELINE)
        self.assertIn("_tied_recovery_confirmed", PIPELINE)
        self.assertIn("_measured_recovery_end + .42", PIPELINE)
        self.assertIn("_measured_recovery_end + .14", PIPELINE)

    def test_truncated_long_tail_uses_visual_path_not_short_numeric_guess(self):
        self.assertIn("measured_hair_ratio_value <= .30", PIPELINE)
        self.assertIn("_measured_recovery_end = .82", PIPELINE)
        retained = PIPELINE[
            PIPELINE.index("retained_hair_generation_lock = ("):
            PIPELINE.index("repair_regions =", PIPELINE.index("retained_hair_generation_lock = ("))
        ]
        self.assertIn("+ occluded_tail_lock", retained)
        self.assertIn("literal source strand construction", PIPELINE)
        self.assertNotIn(
            'if not any(term in source_hair_style for term in ("braid", "plait"))',
            PIPELINE,
        )


class BowMattePreservationTests(unittest.TestCase):
    def test_source_palette_cannot_delete_attached_raw_subject_components(self):
        self.assertIn("def _connected_raw_subject_core(", BIREFNET)
        palette_at = BIREFNET.index("learned_scene_leak = (")
        palette_block = BIREFNET[palette_at:palette_at + 700]
        self.assertIn("& ~raw_subject_core", palette_block)

    def test_complete_attached_bow_component_is_preserved(self):
        from pipelines.birefnet_service import _connected_raw_subject_core

        raw = np.zeros((120, 120), dtype=np.uint8)
        labels = np.zeros((120, 120), dtype=np.uint8)
        labels[38:92, 43:78] = 13
        raw[38:92, 43:78] = 255
        # A large ribbon loop is outside the parser but connected to the head.
        raw[45:65, 15:44] = 230
        # Detached scenery must remain excluded.
        raw[4:12, 4:12] = 255

        attached = _connected_raw_subject_core(raw, labels)
        self.assertTrue(attached[52, 20])
        self.assertFalse(attached[7, 7])
