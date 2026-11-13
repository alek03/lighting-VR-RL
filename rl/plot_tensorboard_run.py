"""Export TensorBoard scalar plots from one training run as PNG files."""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth_values(values, smoothing):
    """Apply exponential moving-average smoothing."""
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in the range [0, 1)")
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append((1.0 - smoothing) * value + smoothing * smoothed[-1])
    return smoothed


def export_scalar_plots(log_dir, output_dir, smoothing=None):
    log_dir = Path(log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise ValueError("TensorBoard run directory does not exist: %s" % log_dir)
    if not any(log_dir.glob("events.out.tfevents.*")):
        raise ValueError("no TensorBoard event files found in: %s" % log_dir)

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        raise ValueError("TensorBoard run contains no scalar metrics: %s" % log_dir)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, tag in enumerate(tags, start=1):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        steps = [event.step for event in events]
        values = [event.value for event in events]
        filename_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-") or "metric"
        path = output_dir / ("%03d_%s.png" % (index, filename_tag))

        fig, ax = plt.subplots(figsize=(9, 5))
        if smoothing is None:
            ax.plot(steps, values)
            title = tag
        else:
            ax.plot(steps, values, alpha=0.25, label="Raw")
            ax.plot(steps, smooth_values(values, smoothing), label="Smoothed")
            ax.legend()
            title = "%s (smoothing=%.3g)" % (tag, smoothing)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print("Saved %d scalar plots to %s" % (len(tags), output_dir))


def main():
    parser = argparse.ArgumentParser(description="Get Tensorboard plots for a specific run")
    parser.add_argument("--log_dir", type=Path, required=True,
                        help="TensorBoard log directory containing event files")
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="directory for PNG plots (default: <run_dir>/plots)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        help="optional exponential moving-average factor in [0, 1); 0 leaves values unchanged",
    )
    args = parser.parse_args()
    output_dir = args.out_dir or args.log_dir / "plots"
    try:
        export_scalar_plots(args.log_dir, output_dir, smoothing=args.smoothing)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
