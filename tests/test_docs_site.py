from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = {
    "index.html",
    "model.html",
    "releases.html",
    "404.html",
    "guide/index.html",
    "guide/install.html",
    "guide/launch.html",
}


@pytest.fixture(scope="session", autouse=True)
def built():
    subprocess.run(
        [sys.executable, str(DOCS / "tools" / "build_site.py")],
        cwd=ROOT,
        check=True,
    )


def read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_all_pages_exist():
    for name in PAGES:
        assert (DOCS / name).is_file(), name


def test_no_unrendered_tokens():
    for name in PAGES:
        text = read(name)
        assert "{{" not in text, name


def test_theme_init_precedes_stylesheet_and_covers_both_themes():
    text = read("index.html")
    assert text.index("dizel.site.theme") < text.index("styles/site.css")
    css = read("styles/site.css")
    assert '[data-theme="dark"]' in css
    for token in (
        "--bg",
        "--surface",
        "--ink",
        "--accent",
        "--on-accent",
        "--code-bg",
        "--shot-mat",
    ):
        assert token in css


def test_nav_footer_and_theme_toggle_everywhere():
    for name in PAGES:
        text = read(name)
        assert text.count('class="s-nav"') == 1, name
        assert text.count('class="s-foot"') == 1, name
        assert "data-theme-toggle" in text, name


def test_no_review_shell_or_old_landing():
    for name in PAGES:
        text = read(name)
        for forbidden in (
            "review-bar",
            "notes-panel",
            "data-view-tab",
            "landing.css",
            "landing.js",
        ):
            assert forbidden not in text, f"{name}: {forbidden}"


def test_home_keeps_real_evidence():
    text = read("index.html")
    for value in (
        "454.78M",
        "435.3M",
        "293.7M",
        "1.1236",
        "3.4479",
        "10,500",
        "Not published",
    ):
        assert value in text, value


def test_model_card_has_no_invented_benchmark_table():
    text = read("model.html")
    assert "Downstream benchmarks" in text
    assert "Not published" in text
    assert "1.1236" in text and "3.4479" in text


def test_docs_have_real_commands_and_no_dead_controls():
    install = read("guide/install.html")
    launch = read("guide/launch.html")
    for command in (
        "git clone https://github.com/D4niel-dev/Dizel.git",
        "pip install -r requirements.txt",
    ):
        assert command in install, command
    for command in ("npm run dev", "python -m inference.tui.main"):
        assert command in launch, command
    for name in ("guide/index.html", "guide/install.html", "guide/launch.html"):
        text = read(name)
        assert "Was this page helpful?" not in text, name
        assert 'type="search"' not in text, name
        assert "Report a docs issue" in text, name


def test_demo_is_untouched():
    demo = read("demo.html")
    assert "styles/demo.css" in demo and "js/demo.js" in demo
    assert "styles/site.css" not in demo and "js/site.js" not in demo


def test_sitemap_and_robots():
    sitemap = (DOCS / "sitemap.xml").read_text(encoding="utf-8")
    assert "https://d4niel-dev.github.io/Dizel/" in sitemap
    robots = (DOCS / "robots.txt").read_text(encoding="utf-8")
    assert "Sitemap:" in robots


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ("href", "src"):
            if key in values:
                self.values.append(values[key])


def _ids(text):
    return set(re.findall(r'\sid="([^"]+)"', text))


def test_internal_links_and_assets_resolve():
    for name in PAGES:
        text = read(name)
        page_dir = (DOCS / name).parent
        parser = _Links()
        parser.feed(text)
        for value in parser.values:
            if not value or value.startswith(("http://", "https://", "mailto:", "data:", "tel:")):
                continue
            if value.startswith("#"):
                anchor = unquote(value[1:])
                assert anchor in _ids(text), f"{name}: missing anchor {value}"
                continue
            parts = urlsplit(value)
            target = (page_dir / unquote(parts.path)).resolve()
            if parts.path.endswith("/"):
                target = target / "index.html"
            assert target.is_file(), f"{name}: missing target {value}"
            if parts.fragment:
                target_text = target.read_text(encoding="utf-8")
                assert unquote(parts.fragment) in _ids(target_text), (
                    f"{name}: missing fragment {value}"
                )


def test_404_is_noindex_and_not_in_sitemap():
    not_found = read("404.html")
    assert 'name="robots" content="noindex"' in not_found
    sitemap = (DOCS / "sitemap.xml").read_text(encoding="utf-8")
    assert "404.html" not in sitemap


def test_theme_toggle_is_hidden_without_js():
    css = read("styles/site.css")
    assert ".theme-toggle { display: none; }" in css
    assert ".js .theme-toggle" in css
