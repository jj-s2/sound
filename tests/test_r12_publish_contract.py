from pathlib import Path


def test_publish_contract_excludes_outputs_and_internal_test() -> None:
    text = Path("docs/r12/r12-train-and-publish.md").read_text(encoding="utf-8").lower()

    assert "internal test" in text
    assert "not committed" in text
    assert "output/" in text
    assert "copy" in text
