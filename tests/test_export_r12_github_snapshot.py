from pathlib import Path


def test_export_script_is_copy_only_and_excludes_generated_output() -> None:
    script = Path("scripts/export_r12_github_snapshot.ps1").read_text(encoding="utf-8")

    assert "Copy-Item" in script
    assert "Remove-Item" not in script
    assert "output/" not in script.lower()
    assert "Destination must be empty" in script
    assert "xh202615/data.py" in script
    assert "xh202615/r12_dataa_augmented_split.py" in script
    assert "dataa-augmented-internal-runbook" not in script
    assert "r12_training_bootstrap.py" in script
    assert "r12_bootstrap_training.py" in script
