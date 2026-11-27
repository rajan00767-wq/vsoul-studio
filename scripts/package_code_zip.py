import os
import zipfile
from pathlib import Path

def main():
    zip_path = Path("vsoul_studio_uniform_swap_code.zip")
    temp_zip = Path("vsoul_studio_uniform_swap_code_temp.zip")

    if temp_zip.exists():
        temp_zip.unlink()

    exclude_dirs = {
        ".git", ".venv", "__pycache__", "outputs", "models", "test_dataset",
        "overnight_test_results", "backups", "scratch", ".qwen", ".pytest_cache",
        "weights", ".cache", "checkpoints", "uploads", "logs"
    }
    exclude_exts = {
        ".pyc", ".pyo", ".zip", ".tar", ".gz", ".log", ".err",
        ".safetensors", ".onnx", ".pth", ".pt", ".bin", ".ckpt", ".h5"
    }

    root = Path(".")
    included_files = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = p.parts
        if any(ex in parts for ex in exclude_dirs):
            continue
        if p.suffix.lower() in exclude_exts:
            continue
        if p.parent == root and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        rel_path = p.relative_to(root)
        included_files.append(rel_path)

    included_files.sort()
    print(f"Total files selected for zip: {len(included_files)}")

    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in included_files:
            zf.write(f, str(f).replace(os.sep, "/"))

    if zip_path.exists():
        zip_path.unlink()
    temp_zip.rename(zip_path)

    print(f"Successfully generated {zip_path.name}: {zip_path.stat().st_size:,} bytes")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        print(f"Verification: {len(names)} files archived.")
        check_list = [
            "pipelines/uniform_swap/orchestrator.py",
            "pipelines/uniform_swap/compositor.py",
            "pipelines/uniform_swap/quality_validator.py",
            "pipelines/uniform_swap/identity_extractor.py",
            "pipelines/uniform_swap/conditioning_builder.py",
            "pipelines/uniform_swap/image_normalizer.py",
            "pipelines/uniform_swap/uniform_generator.py",
            "pipelines/uniform_swap/cache_manager.py",
            "pipelines/uniform_swap/pose_analyzer.py",
            "pipelines/uniform_swap/reference_analyzer.py",
            "pipelines/uniform_swap/fidelity_validator.py",
            "pipelines/uniform_swap/recovery_controller.py",
            "pipelines/uniform_swap/edge_decontamination.py",
            "scripts/verify_quality_gates.py",
            "scripts/test_v13_compositor.py",
            "scripts/test_v13_multi_background.py",
            "scripts/benchmark_fresh_inference.py",
            "scripts/run_unseen_regression.py",
            "scripts/run_production_acceptance.py",
            "scripts/build_comparison_sheets.py",
            "tests/test_pose_analyzer.py",
            "tests/test_studio_quality_gates.py",
            "tests/test_passport_and_multibackground.py",
            "tests/test_force_regenerate_policy.py",
            "tests/test_v13_3_components.py",
            "tests/test_v13_3_1_gate_fidelity.py",
            "tests/test_v13_3_1_recovery.py",
            "tests/test_v13_3_1_fidelity_universal.py",
            "tests/test_v13_3_2_edge_decontamination.py",
            "tests/test_v13_3_2_bow_and_seam.py",
            "tests/test_non_destructive_reporting.py",
            "scripts/build_v13_3_2_comparison_sheets.py",
            "V13_3_2_HANDOVER_STATUS.md",
            "V13_3_2_PROGRESS.md",
            "V13_3_2_FRESH_INFERENCE_REPORT.md",
            "V13_3_1_IMPLEMENTATION_PLAN.md",
            "UNIFORM_FIDELITY_REPORT.md",
            "RECOVERY_EFFECTIVENESS_REPORT.md",
            "STUDIO_POSE_VERIFICATION_REPORT.md",
            "PERFORMANCE_PROFILING_REPORT.md",
            "PRODUCTION_READINESS_CHECKLIST.md",
            "FINAL_REPORT.md",
            "TEST_RESULTS.csv",
            "FIX_HISTORY.md",
            "COMPOSITOR_DEBUG_REPORT.md",
            "QUALITY_GATE_REGRESSION_REPORT.md",
            "PERFORMANCE_REPORT.md",
            "REMAINING_ISSUES.md",
            "STATIC_PROMPT_AUDIT.md",
            "CACHE_PROVENANCE_REPORT.md",
            "FRESH_INFERENCE_BENCHMARK.csv",
            "UNSEEN_REGRESSION_REPORT.md",
            "main.py",
            "requirements.txt"
        ]
        for item in check_list:
            status = "INCLUDED" if item in names else "MISSING"
            print(f"  [+] {item}: {status}")

if __name__ == "__main__":
    main()
