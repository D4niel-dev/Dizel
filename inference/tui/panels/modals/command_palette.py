"""Ctrl+K command palette, now a proper modal screen."""

from inference.tui.commands.registry import registry
from inference.tui.panels.modals.common import PickerItem, PickerModal


class CommandPaletteModal(PickerModal):
    def __init__(self) -> None:
        commands = sorted(
            registry.list_all(),
            key=lambda command: (command.name == "exit", command.category, command.name),
        )
        items = [
            PickerItem(
                command.name,
                f"/{command.name}",
                command.help_text,
                trailing=command.category,
                keywords=" ".join(
                    [command.usage, " ".join(command.aliases), " ".join(command.examples)]
                ),
                value=command.insert_text,
            )
            for command in commands
        ]
        super().__init__(
            "Commands",
            items,
            subtitle=f"{len(commands)} commands · aliases and examples are searchable",
            placeholder="Search commands…",
            footer="↑↓ move · enter insert into prompt · esc close",
            empty_text="No matching commands",
        )
