"""A provider-by-provider usage window with estimated costs."""

from __future__ import annotations

from . import provider_pricing


def snapshot(usage) -> dict:
    """Every provider bucket, with a labelled cost estimate per row."""
    rows = usage.providers_snapshot()
    for row in rows:
        row["estimated_cost"] = provider_pricing.estimate_cost(
            row["provider"], row["used"]
        )
        row["pricing"] = provider_pricing.rate_label(row["provider"])
    return {"providers": rows, "window_hours": int(usage.RESET_WINDOW // 3600)}


def _reset_label(seconds: int) -> str:
    minutes = max(0, seconds) // 60
    if minutes <= 0:
        return "now"
    return f"{minutes // 60}h {minutes % 60:02d}m"


def report_text(report: dict) -> str:
    hours = report.get("window_hours", 12)
    lines = [
        f"Provider usage — {hours}-hour window",
        "",
        f"{'provider':<12}{'used':>10}{'cap':>10}{'%':>5}{'reset':>10}{'~cost':>11}",
        "-" * 58,
    ]
    for row in report.get("providers", []):
        cost = row.get("estimated_cost")
        cost_label = "—" if cost is None else f"${float(cost):.2f}"
        marker = "*" if row.get("active") else " "
        lines.append(
            f"{row['provider'] + marker:<12}"
            f"{float(row['used']):>10.0f}"
            f"{float(row['capacity']):>10.0f}"
            f"{int(row['percent']):>4}%"
            f"{_reset_label(int(row.get('resets_in_seconds', 0))):>10}"
            f"{cost_label:>11}"
        )
    lines.extend(
        [
            "",
            "* active provider",
            "Costs are list-price estimates, not billing. Local models are free.",
            "Only the tracked window is counted; headless and CLI runs are excluded.",
        ]
    )
    return "\n".join(lines)
