"""Prompt token-budget regression tests (no model loading).

Job 5a1bbeaa shipped a 700-token positive prompt and a 1266-token negative
against the encoder's 512-token budget. ``encode_prompt`` truncates silently,
so the dropped tail carried the neck/pose/sleeve contract and the neck-related
negatives; the decode tripled the exposed skin below the chin. These tests
pin the trimmer's behaviour and the presence of the neck lock in the live
prompt, using a stub tokenizer so no model is loaded.
"""
import ast
from pathlib import Path
import unittest
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")


class _StubLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        pass


def _load_trimmer():
    """Exec only the trimmer helpers, with a whitespace-token stub tokenizer."""
    tree = ast.parse(SOURCE)
    wanted = {
        "_split_prompt_units", "fit_prompt_to_token_budget", "count_prompt_tokens",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == len(wanted), f"missing helpers: {wanted - {n.name for n in nodes}}"
    import re as _re
    log = _StubLogger()
    ns = {
        "re": _re, "logger": log,
        "Tuple": Tuple, "List": List, "Dict": Dict, "Any": Any, "Optional": Optional,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "trimmer", "exec"), ns)
    # One "token" per whitespace word keeps the test independent of the real
    # tokenizer, which needs the model directory on disk.
    ns["count_prompt_tokens"] = lambda text: len(text.split())
    return ns, log


class PromptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.log = _load_trimmer()
        self.fit = self.ns["fit_prompt_to_token_budget"]
        self.count = self.ns["count_prompt_tokens"]

    def test_under_budget_prompt_is_untouched(self):
        text = "Keep the face. Keep the hair. Keep the neck."
        self.assertEqual(self.fit(text, 512), text)
        self.assertEqual(self.log.warnings, [])

    def test_over_budget_prompt_is_trimmed_to_fit(self):
        text = " ".join(f"Sentence number {i} here." for i in range(100))
        self.assertGreater(self.count(text), 50)
        fitted = self.fit(text, 50)
        self.assertLessEqual(self.count(fitted), 50)
        self.assertTrue(self.log.warnings, "a trim must be logged, never silent")

    def test_keep_tail_survives_trimming(self):
        tail = "NECK LOCK: keep the jaw-to-collar gap."
        text = " ".join(f"Filler sentence {i}." for i in range(100))
        fitted = self.fit(text, 40, keep_tail=tail)
        self.assertIn("NECK LOCK", fitted)
        self.assertLessEqual(self.count(fitted), 40)

    def test_tail_longer_than_budget_still_returns_the_tail(self):
        tail = " ".join(f"word{i}" for i in range(30))
        text = " ".join(f"Filler sentence {i}." for i in range(50))
        fitted = self.fit(text, 10, keep_tail=tail)
        self.assertEqual(fitted.strip(), tail)

    def test_comma_list_splits_on_commas_not_sentences(self):
        text = ", ".join(f"term{i}" for i in range(100))
        fitted = self.fit(text, 20)
        self.assertLessEqual(self.count(fitted), 20)
        self.assertIn("term0", fitted)

    def test_duplicate_units_are_dropped_first(self):
        # The negative list repeats terms verbatim; dedupe must buy room
        # before any distinct term is sacrificed.
        text = ", ".join(["long neck"] * 20 + ["orange skin", "detached collar"])
        fitted = self.fit(text, 8)
        self.assertEqual(fitted.count("long neck"), 1)
        self.assertIn("orange skin", fitted)
        self.assertIn("detached collar", fitted)

    def test_zero_budget_is_a_no_op(self):
        self.assertEqual(self.fit("anything here", 0), "anything here")

    def test_packing_uses_the_whole_budget(self):
        # Summing per-unit token costs overcounts, because tokens merge across
        # the joiner. A first cut of the trimmer read 1767 tokens for job
        # 5a1bbeaa's 1266-token negative and stopped 157 tokens short of the
        # budget, discarding neck terms there was room for. The packer must
        # measure the assembled string and land near the budget.
        text = ", ".join(f"term{i}" for i in range(400))
        fitted = self.fit(text, 200)
        self.assertLessEqual(self.count(fitted), 200)
        self.assertGreaterEqual(
            self.count(fitted), 195,
            "packer left usable budget unused",
        )

    def test_pinned_tail_terms_outrank_position(self):
        # Terms late in an over-long negative list are dropped by position
        # alone; the neck guards must be pinned, not left to chance.
        text = ", ".join(f"filler{i}" for i in range(400)) + ", cut-and-paste neck"
        fitted = self.fit(text, 60, keep_tail="long neck, cut-and-paste neck, mismatched skin tone")
        self.assertIn("cut-and-paste neck", fitted)
        self.assertIn("mismatched skin tone", fitted)
        self.assertLessEqual(self.count(fitted), 60)


class LivePromptWiringTests(unittest.TestCase):
    """The defect was a correct prompt that never reached the encoder."""

    def test_neck_lock_is_built_and_used(self):
        self.assertIn("neck_collar_lock = (", SOURCE)
        # Built once, referenced in the live prompt and in the retained tail.
        self.assertGreaterEqual(SOURCE.count("{neck_collar_lock}"), 2)

    def test_both_prompts_are_budgeted_before_encoding(self):
        budget_at = SOURCE.index("generated_prompt = fit_prompt_to_token_budget")
        negative_at = SOURCE.index("generation_negative_prompt = fit_prompt_to_token_budget")
        encode_at = SOURCE.index("max_sequence_length=uniform_prompt_token_budget")
        self.assertLess(budget_at, encode_at, "positive prompt trimmed after encoding")
        self.assertLess(negative_at, encode_at, "negative prompt trimmed after encoding")

    def test_dead_duplicate_prompt_block_is_gone(self):
        # The removed block assembled a full prompt that every later branch
        # overwrote unconditionally.
        self.assertNotIn("ONE PERSON ONLY. OUTPUT CONTRACT", SOURCE)


if __name__ == "__main__":
    unittest.main()


class SourceGarmentRemovalTests(unittest.TestCase):
    """The clause that removes the old uniform must always be encoded.

    Job 30399633 shipped at 506/512 tokens with "Remove every source-clothing
    pixel" trimmed away: the girl's black/white checked pinafore and her blue
    school badge survived into the delivered plaid shirt. The generic
    negatives ("source clothing retained", "original clothing visible") were
    present and did not prevent it.
    """

    def test_source_garment_clause_is_pinned_to_the_retained_tail(self):
        idx = SOURCE.index('label="Positive uniform prompt"')
        tail = SOURCE[idx - 1800:idx]
        self.assertIn("SOURCE GARMENT:", tail)
        self.assertIn("None of image 1's garment may remain visible.", tail)

    def test_pinned_clause_names_the_observed_leftovers(self):
        idx = SOURCE.index('label="Positive uniform prompt"')
        tail = SOURCE[idx - 1800:idx]
        for item in ("pinafore", "badge", "crest", "emblem"):
            self.assertIn(item, tail)

    def test_source_badge_terms_are_in_the_pinned_negative(self):
        idx = SOURCE.index('label="Uniform negative prompt"')
        tail = SOURCE[idx - 1600:idx]
        for term in ("source badge", "pinafore", "garment over garment"):
            self.assertIn(term, tail)
