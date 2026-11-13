#!/usr/bin/env python3
"""Assemble the static Dizel docs site from layouts, partials, and page fragments."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

DOCS = Path(__file__).resolve().parents[1]
SITE = DOCS / "_site"
SITE_URL = "https://d4niel-dev.github.io/Dizel/"
REPO = "https://github.com/D4niel-dev/Dizel"

PAGE_MAP: Dict[str, Tuple[str, str]] = {
    "index.html": ("index", "public"),
    "model.html": ("model", "public"),
    "releases.html": ("releases", "public"),
    "404.html": ("404", "public"),
    "guide/index.html": ("guide-index", "docs"),
    "guide/install.html": ("guide-install", "docs"),
    "guide/launch.html": ("guide-launch", "docs"),
}

META_RE = re.compile(r"<!--meta\s*(\{.*?\})\s*-->", re.S)
PLACEHOLDER_RE = re.compile(r"\{\{[a-z_]+\}\}")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_page(path: Path) -> Tuple[Dict[str, Any], str]:
    text = read(path)
    match = META_RE.search(text)
    if not match:
        raise SystemExit(f"missing <!--meta ...--> in {path}")
    try:
        meta = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise SystemExit(f"bad page metadata in {path}: {error}") from error
    return meta, text[match.end() :].strip()


def canonical_for(output: str) -> str:
    if output == "index.html":
        return SITE_URL
    if output.endswith("/index.html"):
        return SITE_URL + output[: -len("index.html")]
    return SITE_URL + output


def mark_nav(nav: str, active: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        attribute = ' aria-current="page"' if name and name == active else ""
        return f'data-nav="{name}"{attribute}'

    return re.sub(r'data-nav="([^"]+)"', replace, nav)


def mark_sidebar(sidebar: str, active: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name and name == active:
            return f'data-docs="{name}" class="is-active" aria-current="page"'
        return f'data-docs="{name}"'

    return re.sub(r'data-docs="([^"]+)"', replace, sidebar)


def toc_links(content: str) -> str:
    rows = []
    for match in re.finditer(r'<h2[^>]*\bid="([^"]+)"[^>]*>(.*?)</h2>', content, re.S):
        anchor = match.group(1)
        label = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        rows.append(f'<a href="#{anchor}">{html.escape(label)}</a>')
    return "\n".join(rows)


def fill(text: str, values: Dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def build() -> int:
    layout_public = read(SITE / "layouts" / "public.html")
    layout_docs = read(SITE / "layouts" / "docs.html")
    nav = read(SITE / "partials" / "nav.html")
    footer = read(SITE / "partials" / "footer.html")
    sidebar = read(SITE / "partials" / "docs-sidebar.html")
    toc = read(SITE / "partials" / "docs-toc.html")
    theme_init = read(SITE / "partials" / "theme-init.html")

    urls = []
    for output, (source, layout) in PAGE_MAP.items():
        page_path = SITE / "pages" / f"{source}.html"
        meta, content = parse_page(page_path)
        base = "../" if "/" in output else ""
        canonical = meta.get("canonical") or canonical_for(output)
        edit_url = f"{REPO}/blob/main/docs/_site/pages/{source}.html"
        robots = (
            '<meta name="robots" content="noindex">'
            if meta.get("robots") == "noindex"
            else ""
        )

        nav_html = mark_nav(nav, str(meta.get("nav", ""))).replace("{{base}}", base)
        footer_html = footer.replace("{{base}}", base)
        sidebar_html = mark_sidebar(
            sidebar, str(meta.get("docs_active", ""))
        ).replace("{{base}}", base)
        toc_html = fill(
            toc,
            {
                "toc_links": toc_links(content),
                "edit_url": edit_url,
                "base": base,
            },
        )

        if layout == "docs":
            body = fill(
                layout_docs,
                {
                    "content": content,
                    "docs_sidebar": sidebar_html,
                    "docs_toc": toc_html,
                    "base": base,
                },
            )
        else:
            body = content

        page = fill(
            layout_public,
            {
                "title": html.escape(str(meta.get("title", "Dizel")), quote=True),
                "description": html.escape(
                    str(meta.get("description", "")), quote=True
                ),
                "canonical": html.escape(canonical, quote=True),
                "robots": robots,
                "theme_init": theme_init,
                "nav": nav_html,
                "footer": footer_html,
                "body": body,
                "base": base,
            },
        )

        remaining = PLACEHOLDER_RE.search(page)
        if remaining:
            raise SystemExit(
                f"unrendered placeholder {remaining.group(0)} in {output} "
                f"(from {page_path.name})"
            )

        target = DOCS / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8")
        if output != "404.html":
            urls.append(canonical)

    sitemap = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    sitemap.extend(f"  <url><loc>{html.escape(url)}</loc></url>" for url in urls)
    sitemap.append("</urlset>")
    (DOCS / "sitemap.xml").write_text("\n".join(sitemap) + "\n", encoding="utf-8")
    (DOCS / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n"
        f"Sitemap: {SITE_URL}sitemap.xml\n",
        encoding="utf-8",
    )

    print(f"built {len(PAGE_MAP)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(build())
