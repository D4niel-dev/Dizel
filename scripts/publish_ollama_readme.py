#!/usr/bin/env python3
"""Publish the README/model card for a model on ollama.com.

`ollama push` uploads weights and manifests only. The summary and README shown
on the model page are stored by the website and updated with an authenticated
form POST, which is what this script does.

Authentication uses the signed-in browser session cookie. Create it from
DevTools -> Application -> Cookies -> https://ollama.com and save either the
full `Cookie` header or just the session token value to a local file:

    ollama/.cookie

That file is gitignored. The cookie grants account access; treat it like a
password and delete it when you are done.

Usage:
    python scripts/publish_ollama_readme.py --dry-run
    python scripts/publish_ollama_readme.py --cookie-file ollama/.cookie
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_MODEL = "D4niel-dev/Dizel-1.5"
DEFAULT_README = pathlib.Path(__file__).resolve().parent.parent / "ollama" / "README.md"
README_MARKER = "from-scratch causal language model"


def resolve_cookie(cookie_file: str | None) -> str:
    cookie = (os.environ.get("OLLAMA_COM_COOKIE") or "").strip()
    if not cookie and cookie_file:
        path = pathlib.Path(cookie_file)
        if not path.is_file():
            raise SystemExit(f"Cookie file not found: {path}")
        cookie = path.read_text(encoding="utf-8").strip()

    if not cookie:
        raise SystemExit(
            "Missing session cookie. Pass --cookie-file or set OLLAMA_COM_COOKIE."
        )

    # Accept either a full Cookie header or a bare session token.
    if cookie.startswith(("__Secure-", "__Host-")) or ";" in cookie:
        return cookie
    return f"__Secure-next-auth.session-token={cookie}"


def post_form(url: str, fields: dict[str, str], cookie: str) -> str:
    body = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    request.add_header("Cookie", cookie)
    request.add_header("User-Agent", "dizel-publish-readme/1.0")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace") if error.fp else ""
        raise SystemExit(f"POST {url} failed: HTTP {error.code}\n{detail[:2000]}")


def fetch_public_page(url: str) -> str:
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "dizel-publish-readme/1.0")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an ollama.com model card.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="owner/model")
    parser.add_argument("--readme", default=str(DEFAULT_README), help="Markdown file")
    parser.add_argument("--cookie-file", help="File holding the ollama.com session cookie")
    parser.add_argument("--summary", help="Optional short summary (max 255 chars)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without posting")
    args = parser.parse_args()

    model = args.model.strip().strip("/")
    readme_path = pathlib.Path(args.readme)
    if not readme_path.is_file():
        raise SystemExit(f"README not found: {readme_path}")

    readme = readme_path.read_text(encoding="utf-8").strip()
    if not readme:
        raise SystemExit(f"README is empty: {readme_path}")
    if args.summary and len(args.summary) > 255:
        raise SystemExit(f"Summary is {len(args.summary)} chars; ollama.com allows 255.")
    if README_MARKER not in readme:
        raise SystemExit(f"README does not contain the verification marker: {README_MARKER!r}")

    url = f"https://ollama.com/{model}"
    if args.dry_run:
        print(f"Target: {url}")
        print(f"README: {readme_path} ({len(readme)} chars)")
        if args.summary:
            print(f"Summary: {len(args.summary)} chars")
        print("Dry run complete; nothing was posted.")
        return 0

    cookie = resolve_cookie(args.cookie_file)

    if args.summary:
        post_form(url, {"summary": args.summary}, cookie)
        print(f"Updated summary ({len(args.summary)} chars)")

    post_form(url, {"readme": readme}, cookie)
    print(f"Updated readme ({len(readme)} chars)")

    page = fetch_public_page(url)
    if "No readme" in page or README_MARKER not in page:
        raise SystemExit(
            "POST returned success, but the public page does not show the README yet. "
            "Re-check the page; it may be cached."
        )

    print(f"Verified: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
