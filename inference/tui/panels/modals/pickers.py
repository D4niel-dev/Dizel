"""Concrete pickers for models, providers, modes, checkpoints and exports."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from inference.tui.constants import (
    MODE_DETAILS,
    MODES,
    TIER_DETAILS,
    TIERS,
)
from inference.tui.panels.modals.common import (
    PickerItem,
    PickerModal,
    TextPromptModal,
)
from inference.tui.runtime_actions import (
    checkpoint_paths,
    key_state,
    provider_catalog,
    provider_meta,
)


class ModelPickerModal(PickerModal):
    """Local model variants, or the live model list of the active API provider."""

    def __init__(self) -> None:
        super().__init__(
            "Model",
            [],
            subtitle="Loading…",
            placeholder="Search models, or type an exact id…",
            footer="↑↓ move · enter select · type an id for a custom model · esc close",
            empty_text="No models returned — type a model id",
            allow_custom=True,
        )

    def _after_mount(self) -> None:
        manager = self.app.chat_bridge.manager
        self.set_items([], subtitle=f"Loading models from {self.app.active_provider}…")
        self._manager = manager
        self.run_worker(self._fetch, thread=True, exclusive=True)

    def _fetch(self) -> None:
        manager = self._manager
        try:
            models = manager._api_provider.list_models(key=manager._api_key)
            payload = [
                (str(model.id), str(model.name), int(model.context_length or 0))
                for model in models
            ]
            error = ""
        except Exception as exc:
            payload = []
            error = str(exc)

        try:
            self.app.call_from_thread(self._loaded, payload, error)
        except Exception:
            pass

    def _loaded(self, payload: list[tuple[str, str, int]], error: str) -> None:
        if not self.is_mounted:
            return
        active = self.app.active_model
        seen: set[str] = set()
        items: list[PickerItem] = []
        for model_id, name, context in payload:
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            detail = name if name and name != model_id else "API model"
            if context:
                detail += f" · {context:,} ctx"
            items.append(
                PickerItem(
                    model_id,
                    model_id,
                    detail,
                    trailing="current" if model_id == active else "",
                    active=model_id == active,
                )
            )

        provider = self.app.active_provider
        if error:
            self.set_items(
                items,
                subtitle=f"{provider} · could not list models: {error}",
                empty_text="Could not list models — type a model id",
            )
        else:
            self.set_items(items, subtitle=f"{provider} · {len(items)} models")


class ProviderPickerModal(PickerModal):
    """Switch between local inference and the registered API providers."""

    def __init__(self) -> None:
        super().__init__(
            "Provider",
            [],
            subtitle="Where Dizel runs",
            placeholder="Search providers…",
            footer="↑↓ move · enter switch · esc close",
            empty_text="No providers",
        )

    def _after_mount(self) -> None:
        app = self.app
        items: list[PickerItem] = []
        for meta in provider_catalog():
            slug = meta["slug"]
            active = slug == app.active_provider
            badges = []
            if active:
                badges.append("current")
            state = key_state(app, slug)
            if state:
                badges.append(state)
            items.append(
                PickerItem(
                    slug,
                    meta["name"],
                    meta["description"],
                    trailing=" · ".join(badges),
                    active=active,
                )
            )
        self.set_items(items, subtitle=f"Active · {app.active_provider}")


class ModePickerModal(PickerModal):
    """Response profiles that change sampling and the system prompt."""

    def __init__(self) -> None:
        super().__init__(
            "Mode",
            [],
            subtitle="Sampling and prompt profile for the next reply",
            placeholder="Search modes…",
            footer="↑↓ move · enter apply · esc close",
        )

    def _after_mount(self) -> None:
        current = self.app.active_mode
        items = [
            PickerItem(
                mode,
                mode,
                MODE_DETAILS[mode],
                trailing="current" if mode == current else "",
                active=mode == current,
            )
            for mode in MODES
        ]
        self.set_items(items, subtitle=f"Active · {current}")


class TierPickerModal(PickerModal):
    """The model family: Flash or Pro. Modes sit underneath the tier."""

    def __init__(self) -> None:
        super().__init__(
            "Tier",
            [],
            subtitle="Model family for the next reply",
            placeholder="Search tiers…",
            footer="↑↓ move · enter apply · esc close",
        )

    def _after_mount(self) -> None:
        current = self.app.active_tier
        items = [
            PickerItem(
                tier,
                tier,
                TIER_DETAILS[tier],
                trailing="current" if tier == current else "",
                active=tier == current,
            )
            for tier in TIERS
        ]
        self.set_items(items, subtitle=f"Active · {current}")


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


class CheckpointPickerModal(PickerModal):
    """Checkpoints discovered in checkpoints/ plus free-form path entry."""

    def __init__(self) -> None:
        super().__init__(
            "Load checkpoint",
            [],
            subtitle="Files on disk",
            placeholder="Search checkpoints, or type a path…",
            footer="↑↓ move · enter load · type a path for any file · esc close",
            empty_text="No checkpoints in checkpoints/ — type a path",
            allow_custom=True,
        )

    def _after_mount(self) -> None:
        try:
            from model.naming import model_display_name
        except Exception:
            def model_display_name(path: str) -> str:
                return Path(path).stem

        current = getattr(self.app, "_checkpoint", "")
        items: list[PickerItem] = []
        seen: set[str] = set()

        for path in checkpoint_paths():
            seen.add(str(path.resolve()))
            try:
                stat = path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %H:%M")
                detail = f"{path.name} · {_human_size(stat.st_size)} · {modified}"
            except OSError:
                detail = path.name
            items.append(
                PickerItem(
                    str(path.resolve()),
                    model_display_name(str(path)),
                    detail,
                    trailing="current" if str(path.resolve()) == current else "",
                    active=str(path.resolve()) == current,
                )
            )

        if current and os.path.isfile(current) and str(Path(current).resolve()) not in seen:
            items.insert(
                0,
                PickerItem(
                    str(Path(current).resolve()),
                    model_display_name(current),
                    f"{Path(current).name} · saved in settings",
                    trailing="current",
                    active=True,
                ),
            )

        count = len(items)
        self.set_items(items, subtitle=f"{count} checkpoint{'s' if count != 1 else ''} in checkpoints/")


class ExportFormatModal(PickerModal):
    """Choose the transcript format for /export."""

    def __init__(self, message_count: int) -> None:
        super().__init__(
            "Export session",
            [
                PickerItem(
                    "md",
                    "Markdown",
                    "Readable transcript with [ROLE] headers · .md",
                    trailing="recommended",
                    active=True,
                ),
                PickerItem(
                    "json",
                    "JSON",
                    "Full message objects, for scripts and tooling · .json",
                ),
                PickerItem(
                    "txt",
                    "Plain text",
                    "Role headers and message bodies · .txt",
                ),
            ],
            subtitle=f"{message_count} messages · written to the current directory",
            placeholder="Filter formats…",
            footer="↑↓ move · enter export · esc close",
        )


class ApiKeyModal(TextPromptModal):
    """Prompt for a provider key when a switch needs one."""

    _PLACEHOLDERS = {
        "openai": "sk-…",
        "anthropic": "sk-ant-…",
        "google": "AIza…",
        "groq": "gsk_…",
        "mistral": "…",
        "xai": "xai-…",
        "cohere": "…",
    }

    def __init__(self, slug: str) -> None:
        meta = provider_meta(slug) or {"name": slug, "slug": slug}
        super().__init__(
            f"API key · {meta['name']}",
            "Key",
            placeholder=self._PLACEHOLDERS.get(slug, "Paste the API key"),
            password=True,
            help_text="Stored encrypted in settings.json (machine-derived key).",
            submit_label="Connect",
            validate=lambda key: None if len(key) >= 8 else "That key looks too short.",
        )
