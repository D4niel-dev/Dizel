"""A tiny stdio MCP server used by the manager tests."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("dizel-test")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool()
def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


if __name__ == "__main__":
    server.run()
