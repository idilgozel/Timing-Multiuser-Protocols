"""Project full-run walltime from a partial train.log, to confirm the 48h fit before fanning out.

Parses the metrics lines (which carry a [ISO8601Z] timestamp, episode=, global_step=, and
avg_steps_window=) and projects time-to-finish for --target episodes. Because steps/episode GROWS
across the curriculum (the 60->120->200->300 horizons), a single early-stage rate UNDER-estimates
total walltime. So we report TWO projections:

  * realistic  : remaining episodes run at the CURRENT observed steps/episode.
  * conservative: remaining episodes run at the FINAL-stage cap (--final-cap) steps/episode
                  (pessimistic; a good policy solves well under the cap).

sec/step is measured over the last --window metric lines (steady-state, and avoids resume gaps).
Run it on a contiguous segment of the log.

Usage:
    python scripts/project_walltime.py --log qamel/outputs/runs/<run>/train.log --target 70000
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime

LINE = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\].*?episode=(?P<ep>\d+).*?"
    r"global_step=(?P<gs>\d+).*?avg_steps_window=(?P<sw>[\d.]+)"
)


def _parse(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                rows.append((
                    datetime.fromisoformat(m["ts"].replace("Z", "+00:00")),
                    int(m["ep"]), int(m["gs"]), float(m["sw"]),
                ))
    return rows


def _hours(seconds: float) -> float:
    return seconds / 3600.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--target", type=int, default=70000)
    ap.add_argument("--final-cap", type=int, default=300)
    ap.add_argument("--window", type=int, default=60, help="trailing metric lines for the rate estimate")
    ap.add_argument("--wall-hours", type=float, default=48.0)
    args = ap.parse_args()

    rows = _parse(args.log)
    if len(rows) < 2:
        raise SystemExit(f"need >=2 metric lines, found {len(rows)} in {args.log}")

    t_first, ep_first, gs_first, _ = rows[0]
    t_last, ep_last, gs_last, sw_last = rows[-1]

    win = rows[-args.window:] if len(rows) > args.window else rows
    wt0, wep0, wgs0, _ = win[0]
    wt1, wep1, wgs1, _ = win[-1]
    w_secs = (wt1 - wt0).total_seconds()
    d_step = max(wgs1 - wgs0, 1)
    d_ep = max(wep1 - wep0, 1)
    sec_per_step = w_secs / d_step
    steps_per_ep_now = (wgs1 - wgs0) / d_ep  # observed recent steps/episode

    elapsed_so_far = (t_last - t_first).total_seconds()
    remaining_ep = max(args.target - ep_last, 0)

    realistic_remaining = remaining_ep * steps_per_ep_now * sec_per_step
    conservative_remaining = remaining_ep * args.final_cap * sec_per_step
    realistic_total = _hours(elapsed_so_far + realistic_remaining)
    conservative_total = _hours(elapsed_so_far + conservative_remaining)

    def verdict(h):
        return "FITS" if h < args.wall_hours else "OVER WALL"

    print(f"log: {args.log}")
    print(f"parsed {len(rows)} metric lines; episode {ep_first} -> {ep_last} of target {args.target}")
    print(f"elapsed so far: {_hours(elapsed_so_far):.2f} h   (this segment)")
    print(f"recent window: {len(win)} lines, {_hours(w_secs):.2f} h")
    print(f"  sec/step        = {sec_per_step:.4f}")
    print(f"  steps/episode   = {steps_per_ep_now:.1f} (recent)   latest avg_steps_window={sw_last:.1f}")
    print(f"  remaining eps   = {remaining_ep}")
    print()
    print(f"PROJECTED FULL {args.target}-ep walltime vs {args.wall_hours:.0f}h wall:")
    print(f"  realistic   (current steps/ep) : {realistic_total:6.1f} h   [{verdict(realistic_total)}]")
    print(f"  conservative(final cap {args.final_cap:>3}) : {conservative_total:6.1f} h   [{verdict(conservative_total)}]")
    print()
    if conservative_total < args.wall_hours:
        print("=> Even the conservative bound fits in one wall window. Safe to fan out (walltime-wise).")
    elif realistic_total < args.wall_hours:
        print("=> Realistic fits but conservative does not: marginal. Re-run this on the 200/300 "
              "stages before fanning out; the resumable checkpoint covers a wall overrun.")
    else:
        print("=> Does NOT fit 48h even realistically. Resume across two submissions, or cut "
              "train_episodes / horizons before fanning out.")


if __name__ == "__main__":
    main()
