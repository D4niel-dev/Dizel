from dizel_core import tools


def _external(name: str):
    return tools.Tool(name, "read a file", {"path": "string"}, lambda path: f"read {path}")


def test_catalog_includes_external_and_executes():
    tools.set_external({"fs__read": _external("fs__read")})
    try:
        assert "fs__read" in tools.catalog()
        assert tools.execute({"name": "fs__read", "args": {"path": "x"}}) == "read x"
        assert "fs__read" in tools.prompt_block()
    finally:
        tools.set_external({})


def test_builtin_wins_name_collision():
    tools.set_external({"read_file": _external("read_file")})
    try:
        assert tools.catalog()["read_file"].description != "read a file"
    finally:
        tools.set_external({})


def test_cleared_external_disappears():
    tools.set_external({"x__y": _external("x__y")})
    tools.set_external({})
    assert "x__y" not in tools.catalog()
