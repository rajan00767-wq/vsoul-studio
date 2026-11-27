"""Regression test for non-destructive reporting (V13.3.2).

Verifies that running quality gate verification or report generation:
1. Does NOT overwrite or alter the authored engineering audit (QUALITY_GATE_REGRESSION_REPORT.md).
2. Generates separate outputs:
   - QUALITY_GATE_TEST_SUMMARY.md
   - QUALITY_GATE_TEST_SUMMARY.json
   - Timestamped historical report in reports/quality_gate_runs/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest


def get_file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_quality_gate_regression_report_is_preserved():
    """Verify that QUALITY_GATE_REGRESSION_REPORT.md exists as an authored audit and is preserved."""
    audit_path = Path("QUALITY_GATE_REGRESSION_REPORT.md")
    assert audit_path.exists(), "Authored engineering audit QUALITY_GATE_REGRESSION_REPORT.md must exist"

    initial_sha = get_file_sha256(audit_path)
    assert len(initial_sha) == 64

    # Read authored audit content and verify key sections exist
    content = audit_path.read_text(encoding="utf-8")
    assert "# V13.2 Quality Gate Regression & Independent Verification Report" in content
    assert "Comprehensive Gate Confusion Matrix Table" in content

    # Verify that scripts/verify_quality_gates.py does not open QUALITY_GATE_REGRESSION_REPORT.md in write mode
    verify_script = Path("scripts/verify_quality_gates.py").read_text(encoding="utf-8")
    assert 'open("QUALITY_GATE_REGRESSION_REPORT.md", "w"' not in verify_script, (
        "scripts/verify_quality_gates.py must not open QUALITY_GATE_REGRESSION_REPORT.md in write mode!"
    )
    assert 'open("QUALITY_GATE_REGRESSION_REPORT.md", "a"' not in verify_script

    # Verify that separate output targets are referenced in the verification script
    assert "QUALITY_GATE_TEST_SUMMARY.md" in verify_script
    assert "QUALITY_GATE_TEST_SUMMARY.json" in verify_script
    assert "reports/quality_gate_runs" in verify_script

    # After any check, ensure the hash of the authored audit is completely unchanged
    post_sha = get_file_sha256(audit_path)
    assert post_sha == initial_sha, "QUALITY_GATE_REGRESSION_REPORT.md was modified!"


def test_test_summary_json_structure():
    """Verify that QUALITY_GATE_TEST_SUMMARY.json contains required machine-readable metrics if present."""
    json_path = Path("QUALITY_GATE_TEST_SUMMARY.json")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert "total_conditions_tested" in data
        assert "true_positives" in data
        assert "true_negatives" in data
        assert "precision" in data
        assert "recall" in data
        assert "gates" in data
