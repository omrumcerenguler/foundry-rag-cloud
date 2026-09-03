from types import SimpleNamespace

import main


def test_chat_keyboard_interrupt_is_user_friendly(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        main,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda: SimpleNamespace(command="chat")),
    )
    monkeypatch.setattr(main, "Settings", SimpleNamespace(from_env=lambda: object()))
    monkeypatch.setattr(main, "get_rag_service", lambda _settings: object())
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    assert main.main() == 130
    assert "cancelled" in capsys.readouterr().out.lower()


def test_cli_flag_aliases_are_supported() -> None:
    parser = main.build_parser()
    assert parser.parse_args(["--ingest"]).ingest_flag is True
    assert parser.parse_args(["--query", "question"]).query_flag == "question"
    assert parser.parse_args(["--health"]).health_flag is True
