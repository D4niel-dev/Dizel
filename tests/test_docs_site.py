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
    "guide/capabilities.html",
    "guide/modes.html",
    "guide/providers.html",
    "guide/sessions.html",
    "guide/plugins.html",
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


def test_scoped_resets_do_not_outrank_component_spacing():
    css = read("styles/site.css")
    assert ".site :where(p)" in css
    assert ".site :where(a)" in css
    assert ".site p {" not in css
    assert ".site a {" not in css


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
        "3.3158",
        "16,500",
        "36.0%",
        "Not published",
    ):
        assert value in text, value


def test_model_card_has_no_invented_benchmark_table():
    text = read("model.html")
    assert "Downstream benchmarks" in text
    assert "Not published" in text
    assert "1.1236" in text and "3.3158" in text


def test_model_card_install_link_opens_the_install_guide():
    text = read("model.html")
    assert '<a href="guide/install.html"><small>Next</small>Install Dizel →</a>' in text


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


def test_mode_guide_documents_current_controls():
    assert (DOCS / "guide/modes.html").is_file()
    text = read("guide/modes.html")
    for value in (
        "Fast",
        "Planning",
        "Coding",
        "Thinking",
        "Ultrafast",
        "X-High",
        "Max",
        "/verbosity detailed",
    ):
        assert value in text, value
    assert "/tier" not in text


def test_provider_guide_documents_supported_setup_paths():
    assert (DOCS / "guide/providers.html").is_file()
    text = read("guide/providers.html")
    for value in ("Ctrl+,", "/provider", "/model", "ollama serve", "DIZEL_CONFIG_DIR"):
        assert value in text, value
    assert "ProviderRegistry" not in text


def test_session_guide_distinguishes_saved_sessions_and_context_resets():
    assert (DOCS / "guide/sessions.html").is_file()
    text = read("guide/sessions.html")
    for value in ("/session", "/reset", "/clear", "/context", "DIZEL.md", "AGENTS.md"):
        assert value in text, value
    assert "12,000" in text and "6,000" in text


def test_plugin_guide_separates_direct_calls_from_model_tools():
    assert (DOCS / "guide/plugins.html").is_file()
    text = read("guide/plugins.html")
    for value in ("Settings → MCP", "/mcp", "/plugins run", "/plugins prompt", "/tools tools on"):
        assert value in text, value
    assert "/tools web_search on" not in text
    assert "do not inherit" in text


def test_task_guides_are_in_both_docs_navigations():
    text = read("guide/index.html")
    for name in ("modes.html", "providers.html", "sessions.html", "plugins.html"):
        assert text.count(f'href="{name}"') >= 2, name


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
    assert ".theme-toggle { display: none;" in css
    assert ".js .theme-toggle" in css


def test_hover_motion_stays_inside_its_element():
    css = read("styles/site.css")
    arrow = re.search(r"\.s-arrow\{([^}]*)\}", css)
    assert arrow and "display:inline-block" in arrow.group(1), (
        "inline spans ignore transforms — the arrow hover never moved"
    )
    row = re.search(r"\.s-links-list a\{([^}]*)\}", css)
    assert row and "padding:17px 6px 17px 0" in row.group(1), (
        "the row needs room for the arrow's 4px slide"
    )
    underline = re.search(r"\.s-nav-links a::after\{([^}]*)\}", css)
    assert underline and "bottom:0" in underline.group(1), (
        "the nav underline must not paint over the header rule"
    )
    token = re.search(r"\.s-token:hover\{([^}]*)\}", css)
    assert token and "z-index:1" in token.group(1)

    demo = read("styles/demo.css")
    pill = re.search(r"\.action-pill:hover \{([^}]*)\}", demo)
    assert pill and "z-index: 1" in pill.group(1), (
        "a lifted pill must paint above its neighbours"
    )


def test_edition_badge_never_wraps():
    css = read("styles/site.css")
    edition = re.search(r"\.s-edition\{([^}]*)\}", css)
    assert edition and "white-space:nowrap" in edition.group(1)
    assert css.count(".s-edition{display:none}") >= 2, (
        "the edition badge should drop out at the tablet breakpoint, not wrap"
    )


def test_site_text_never_uses_pixel_sizes_below_eleven():
    css = read("styles/site.css")
    sizes = [float(value) for value in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)px", css)]
    assert sizes
    assert min(sizes) >= 11, min(sizes)


def test_guide_source_sections_are_explicitly_balanced():
    for path in sorted((DOCS / "_site" / "pages").glob("guide-*.html")):
        source = path.read_text(encoding="utf-8")
        assert source.count("<section") == source.count("</section>"), path.name


def test_release_body_and_downloads_match_the_current_app_version():
    from dizel_core.version import __version__

    text = read("releases.html")
    main = text.split('<main id="main"', 1)[1].split("</main>", 1)[0]
    assert f"v{__version__} is live." in main
    assert f"Dizel-TUI-{__version__}.zip" in main
    assert main.count(f"releases/tag/v{__version__}") == 2
    assert "v2.0.3" not in main


def test_docs_chrome_and_demo_match_the_current_app_version():
    from dizel_core.version import __version__

    for name in PAGES:
        text = read(name)
        assert "v2.0.1" not in text, name
        assert f"v{__version__}" in text, name
    demo_js = (DOCS / "js" / "demo.js").read_text(encoding="utf-8")
    assert f"{__version__} (desktop shell)" in demo_js


def test_capabilities_page_is_generated_from_the_manifest():
    from dizel_core import capabilities

    text = read("guide/capabilities.html")
    rows = capabilities.load()
    both = sum(1 for row in rows if row.tui and row.gui)
    assert f"{both} of {len(rows)} capabilities" in text
    for row in rows:
        assert f'data-capability-row="{row.id}"' in text, row.id
        for surface in ("gui", "tui"):
            entry = row.entry(surface)
            if entry:
                assert entry.label in text, (row.id, surface)
    gaps = sum(
        1 for row in rows for surface in ("gui", "tui") if row.entry(surface) is None
    )
    assert text.count('class="s-cap-gap"') == gaps
