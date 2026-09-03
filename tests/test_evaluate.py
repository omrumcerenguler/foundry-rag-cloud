from pathlib import Path

import evaluate
from config import Settings


def test_disabled_mode_returns_structured_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        evaluate,
        "get_rag_service",
        lambda _settings: (_ for _ in ()).throw(ValueError("disabled")),
    )
    result = evaluate.evaluate_mode("AZURE_CLOUD", Settings(), tmp_path / "result.json")
    assert result["mode"] == "AZURE_CLOUD"
    assert result["error"] == "disabled"
    assert (tmp_path / "result.json").exists()
