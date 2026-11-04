from ollama_on_mac.chat import HOST, MODEL, ask, connect, stream_chat


class _Message:
    def __init__(self, content: str = "", thinking: str = "") -> None:
        self.content = content
        self.thinking = thinking


class _Response:
    def __init__(self, content: str = "", thinking: str = "") -> None:
        self.message = _Message(content, thinking)


class _Client:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def chat(self, **kwargs: object) -> _Response | list[_Response]:
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return [
                _Response(thinking="hmm"),
                _Response(content="hello"),
            ]
        return _Response(content="hello")


def test_connect_skips_http_proxy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, host: str | None = None, **kwargs: object) -> None:
            captured["host"] = host
            captured["trust_env"] = kwargs.get("trust_env")

    monkeypatch.setattr("ollama_on_mac.chat.Client", _FakeClient)
    connect("http://127.0.0.1:11434")
    assert captured == {"host": "http://127.0.0.1:11434", "trust_env": False}


def test_ask_uses_gpt_oss(monkeypatch) -> None:
    fake = _Client()
    monkeypatch.setattr("ollama_on_mac.chat.connect", lambda host=HOST: fake)

    assert ask("hi") == "hello"
    assert fake.kwargs is not None
    assert fake.kwargs["model"] == MODEL == "gpt-oss:20b"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert fake.kwargs["stream"] is False
    assert fake.kwargs["think"] is True


def test_stream_chat_yields_thinking_and_content(monkeypatch) -> None:
    fake = _Client()
    monkeypatch.setattr("ollama_on_mac.chat.connect", lambda host=HOST: fake)

    chunks = list(stream_chat([{"role": "user", "content": "hi"}], think=True))

    assert [(chunk.thinking, chunk.content) for chunk in chunks] == [
        ("hmm", ""),
        ("", "hello"),
    ]
    assert fake.kwargs is not None
    assert fake.kwargs["model"] == "gpt-oss:20b"
    assert fake.kwargs["think"] is True
