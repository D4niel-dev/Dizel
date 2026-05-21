"""
core/protocol/dispatcher.py — Consistent tool invocation across agents.

Provides a unified invoke() interface that handles message creation,
tool lookup, execution, error wrapping, and logging.
"""

from typing import Any, Dict, Optional

from .schema import ToolMessage, ToolStatus
from .registry import ToolRegistry


class ToolDispatcher:
    """Dispatches tool invocations through the registry with error handling."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._log: list = []

    def invoke(self, task_id: str, agent: str,
               tool_name: str, params: Dict[str, Any]) -> ToolMessage:
        """
        Invoke a tool by name. Returns a ToolMessage with the result.
        Handles errors gracefully — never raises.
        """
        # Create the request message
        request = ToolMessage.invoke(
            task_id=task_id,
            source=agent,
            target=tool_name,
            payload=params,
        )

        # Lookup tool
        spec = self._registry.get(tool_name)
        if not spec:
            response = ToolMessage.error_response(request, f"Tool '{tool_name}' not found")
            self._record(request, response)
            return response

        if not spec.enabled:
            response = ToolMessage.error_response(request, f"Tool '{tool_name}' is disabled")
            self._record(request, response)
            return response

        # Check agent access
        if "*" not in spec.agents and agent not in spec.agents:
            response = ToolMessage.error_response(
                request, f"Agent '{agent}' has no access to tool '{tool_name}'"
            )
            self._record(request, response)
            return response

        # Execute
        request.status = ToolStatus.RUNNING
        try:
            result = spec.handler(**params)
            if isinstance(result, dict):
                payload = result
            else:
                payload = {"result": result}
            response = ToolMessage.respond(request, payload)
        except Exception as e:
            response = ToolMessage.error_response(request, f"{type(e).__name__}: {e}")

        self._record(request, response)
        return response

    def invoke_if_available(self, task_id: str, agent: str,
                            tool_name: str, params: Dict[str, Any],
                            fallback: Any = None) -> Any:
        """Try to invoke a tool; return fallback value if unavailable."""
        spec = self._registry.get(tool_name)
        if not spec or not spec.enabled:
            return fallback
        response = self.invoke(task_id, agent, tool_name, params)
        if response.is_error:
            return fallback
        return response.payload

    @property
    def log(self) -> list:
        return list(self._log)

    def clear_log(self):
        self._log.clear()

    def _record(self, request: ToolMessage, response: ToolMessage):
        self._log.append({
            "request": request.to_dict(),
            "response": response.to_dict(),
        })
        # Bound log size
        if len(self._log) > 1000:
            self._log = self._log[-500:]
