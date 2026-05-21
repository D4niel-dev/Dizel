"""
core/protocol/registry.py — Tool registration and discovery.

Central registry where tools declare their capabilities.
Agents query the registry to find available tools.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """Describes a registered tool's capabilities."""
    name: str
    description: str
    handler: Callable
    input_schema: Dict[str, str] = field(default_factory=dict)
    output_schema: Dict[str, str] = field(default_factory=dict)
    agents: List[str] = field(default_factory=lambda: ["*"])  # who can use it
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "agents": self.agents,
            "tags": self.tags,
            "enabled": self.enabled,
        }


class ToolRegistry:
    """Central registry for tool discovery and access control."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, handler: Callable,
                 description: str = "",
                 input_schema: Optional[Dict[str, str]] = None,
                 output_schema: Optional[Dict[str, str]] = None,
                 agents: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None) -> ToolSpec:
        """Register a tool. Overwrites if name already exists."""
        spec = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            agents=agents or ["*"],
            tags=tags or [],
        )
        self._tools[name] = spec
        return spec

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_tools(self, agent: Optional[str] = None,
                   tag: Optional[str] = None,
                   enabled_only: bool = True) -> List[ToolSpec]:
        """List tools, optionally filtered by agent access or tag."""
        results = list(self._tools.values())

        if enabled_only:
            results = [t for t in results if t.enabled]

        if agent:
            results = [t for t in results if "*" in t.agents or agent in t.agents]

        if tag:
            results = [t for t in results if tag in t.tags]

        return results

    def list_names(self, agent: Optional[str] = None) -> List[str]:
        """Quick list of available tool names."""
        return [t.name for t in self.list_tools(agent=agent)]

    def enable(self, name: str):
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str):
        if name in self._tools:
            self._tools[name].enabled = False

    def count(self) -> int:
        return len(self._tools)

    def describe_for_prompt(self, agent: Optional[str] = None) -> str:
        """Generate a prompt-friendly description of available tools."""
        tools = self.list_tools(agent=agent)
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for t in tools:
            params = ", ".join(f"{k}: {v}" for k, v in t.input_schema.items())
            lines.append(f"  - {t.name}: {t.description}")
            if params:
                lines.append(f"    Parameters: {params}")
        return "\n".join(lines)
