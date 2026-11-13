"""
CSV + human-readable logging for DSKD-lite runs.

Writes <log_dir>/<run_name>_kd.csv (step,ce,kd,total,active,aligned,alpha,temp,lr)
and a matching <log_dir>/<run_name>_kd.log with one human line per step.
"""
import os


class KDMetricsLogger:
    """Per-step KD metrics logger (CSV + human log)."""

    def __init__(self, log_dir: str = "logs", run_name: str = "xkd"):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{run_name}_kd.csv")
        self.log_path = os.path.join(log_dir, f"{run_name}_kd.log")
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self._closed = False
        self.csv_file.write("step,ce,kd,total,active,aligned,alpha,temp,lr\n")
        self.csv_file.flush()

    # -- formatting helpers --------------------------------------------------

    def _fmt(self, value, spec):
        if isinstance(value, str):
            return value
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt_exp(value) -> str:
        """Scientific notation without a leading zero in the exponent (1.2e-4)."""
        s = f"{value:.1e}"
        return s.replace("e-0", "e-")

    # -- per-step logging ----------------------------------------------------

    def log_step(self, step, ce, kd, total, active, aligned, alpha, temp, lr):
        self.csv_file.write(
            f"{step},{self._fmt(ce, '.6f')},{self._fmt(kd, '.6f')},"
            f"{self._fmt(total, '.6f')},{1 if active else 0},{aligned},"
            f"{self._fmt(alpha, '.2f')},{self._fmt(temp, '.1f')},"
            f"{self._fmt_exp(lr)}\n"
        )
        self.csv_file.flush()

        if active:
            line = (
                f"[KD] step={step} ACTIVE | CE={self._fmt(ce, '.4f')} "
                f"KD={self._fmt(kd, '.4f')} total={self._fmt(total, '.4f')} | "
                f"aligned={aligned} tokens | \u03b1={self._fmt(alpha, '.2f')} "
                f"T={self._fmt(temp, '.1f')} lr={self._fmt_exp(lr)}"
            )
        else:
            line = (
                f"[KD] step={step} DISABLED reason=no_alignment | "
                f"CE={self._fmt(ce, '.4f')} total={self._fmt(total, '.4f')} | "
                f"\u03b1={self._fmt(alpha, '.2f')} T={self._fmt(temp, '.1f')} "
                f"lr={self._fmt_exp(lr)}"
            )
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def log_eval(self, step, val_metrics, mode):
        if isinstance(val_metrics, dict):
            ce = val_metrics.get("ce_loss", val_metrics.get("ce"))
            kd = val_metrics.get("kd_loss", val_metrics.get("kd"))
            total = val_metrics.get("loss", val_metrics.get("total"))
            active = bool(val_metrics.get("active", False))
        else:
            ce = kd = total = None
            active = False
        ce_s = "?" if ce is None else self._fmt(ce, ".4f")
        kd_s = "?" if kd is None else self._fmt(kd, ".4f")
        tot_s = "?" if total is None else self._fmt(total, ".4f")
        state = "ACTIVE" if active else "DISABLED"
        line = f"[EVAL] step={step} mode={mode} {state} | CE={ce_s} KD={kd_s} total={tot_s}"
        self.log_file.write(line + "\n")
        self.log_file.flush()
        print(line)

    def header(self, mode, kd_weight, temperature, extra=""):
        expected = "ACTIVE" if mode == "align" else "DISABLED"
        banner = (
            "=" * 70
            + "\n"
            + f"[DSKD-lite] arm={mode} | kl_weight={kd_weight} | temperature={temperature} | {extra}"
            + "\n"
            + f"KD expected: {expected} (aligned response positions)"
            + "\n"
            + "=" * 70
        )
        print(banner)
        self.log_file.write(banner + "\n")
        self.log_file.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.csv_file.flush()
        self.log_file.flush()
        self.csv_file.close()
        self.log_file.close()