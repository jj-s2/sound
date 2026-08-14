from pathlib import Path


def test_export_script_is_copy_only_and_excludes_generated_output() -> None:
    script = Path("scripts/export_r12_github_snapshot.ps1").read_text(encoding="utf-8")

    assert "Copy-Item" in script
    assert "Remove-Item" not in script
    assert "output/" not in script.lower()
    assert "Destination must be empty" in script
