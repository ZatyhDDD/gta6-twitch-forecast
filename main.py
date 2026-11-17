"""GTA VI launch-day Twitch viewership forecaster (make.fun bounty).

Entry point: `python3 main.py`. Reads every `*.in` case file from
$MAKEFUN_INPUT_DIR (default /input) and writes one `<id>.out` per case to
$MAKEFUN_OUTPUT_DIR (default /output). Python 3.13, standard library only,
no network, fully deterministic (two runs are byte-identical).

Model (v2): size-aware shape transfer from comparable launches.
Each comparable launch contributes its launch-day shape: the ratios of its
hourly viewers (hours 2..24) to its hour-1 print. The forecast median for
hour h is the hour-1 print times exp(0.9 * shape_h), where shape_h is a
robust centre of the log-ratios across a library of comparable launches.

The library is chosen by the size of the target's hour-1 print (p1):
  * Mega launch (p1 >= 300,000): a Gaussian-kernel weighted median over all
    clean comparables, weight exp(-0.5 * ((ln p1_lib - ln p1) / 0.6) ** 2),
    so the biggest launches dominate. Spread: the same kernel-weighted
    standard deviation of the log-ratios per hour, 3-hour smoothed, x0.8.
  * Otherwise: the median over size-matched comparables (hour-1 print in
    [0.25x, 4x] of p1; if fewer than 5 match, the 8 biggest launches).
    Spread: the per-hour standard deviation of the log-ratios in that
    library, 3-hour smoothed, x0.75.
Spreads are clamped to [0.10, 0.90]. Quantiles use the standard normal
z-scores (-1.645, -0.674, 0, 0.674, 1.645) around the median.

Comparables are dropped when they are early-access contaminated (their
launch-day shape is polluted by the early-access window) or when they are
the GTA VI category's own pre-release series (not a launch day at all).

Validation (leave-one-out over the 28 clean comparables in the kit):
  mean pinball loss on the log scale, all targets: 0.797 (v1: 0.808)
  big targets only (hour-1 >= 100k, the GTA VI-like regime): 0.186 (v1: 0.208)
  mega targets only (hour-1 >= 300k): 0.118 (v1: 0.174)
  house baseline on the same sets: 1.137 / 0.253 / 0.235.
The v1 entry scored 0.216 on the server's practice phase (baseline 0.387);
v2's gain is concentrated in the mega-launch regime, which the practice
cases (hour-1 prints 23k-270k) do not cover, so the practice score is
expected to move little while the final-case score should improve.
"""

import json
import math
import os
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
MEGA_MIN = 300_000
KERNEL_SIGMA = 0.6
SHRINK = 0.9
MEGA_SPREAD_SCALE = 0.8
REST_SPREAD_SCALE = 0.75
SPREAD_MIN = 0.10
SPREAD_MAX = 0.90


def is_clean(comp):
    """A comparable is usable when its launch day is a real launch day."""
    if "early-access" in comp.get("unlock", ""):
        return False
    if "before release" in comp.get("name", ""):
        return False
    return True


def launch_day_hours(comp):
    """{hour: viewers} for the comparable's launch day (hours 1..24).

    The launch day starts at the hourly point whose `from` timestamp equals
    the comparable's `day_start`. Returns None when the day does not align
    or the data runs out before hour 24.
    """
    hourly = comp.get("hourly") or []
    start = comp.get("day_start")
    idx = next((i for i, p in enumerate(hourly) if p.get("from") == start), None)
    if idx is None or idx + 24 > len(hourly):
        return None
    return {h: max(0, hourly[idx + h - 1].get("viewers", 0)) for h in range(1, 25)}


def build_library(case):
    """Clean comparables with launch-day hours, sorted by name (deterministic)."""
    lib = []
    for comp in case.get("comparables") or []:
        if not is_clean(comp):
            continue
        day = launch_day_hours(comp)
        if day is None or day[1] < 1:
            continue
        lib.append((comp.get("name", ""), day))
    lib.sort(key=lambda t: t[0])
    return lib


def log_ratios(sub, target_name):
    """Per-hour lists of ln(viewers_h / viewers_1) over a library."""
    per_hour = {h: [] for h in range(2, 25)}
    for name, day in sub:
        if name == target_name:
            continue
        p1 = day[1]
        if p1 < 1:
            continue
        for h in range(2, 25):
            per_hour[h].append(math.log(max(day[h], 1) / p1))
    return per_hour


def medians_of(per_hour):
    return [statistics.median(per_hour[h]) if per_hour[h] else 0.0
            for h in range(2, 25)]


def weighted_median(vals, weights):
    pairs = sorted(zip(vals, weights))
    total = sum(weights)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2.0:
            return value
    return pairs[-1][0]


def weighted_medians(lib, target_name, p1):
    """Per-hour kernel-weighted median of ln ratios (mega-launch shape)."""
    lp = math.log(max(p1, 1))
    per_hour = {h: ([], []) for h in range(2, 25)}
    for name, day in lib:
        if name == target_name or day[1] < 1:
            continue
        w = math.exp(-0.5 * ((math.log(day[1]) - lp) / KERNEL_SIGMA) ** 2)
        for h in range(2, 25):
            per_hour[h][0].append(math.log(max(day[h], 1) / day[1]))
            per_hour[h][1].append(w)
    return [weighted_median(*per_hour[h]) if per_hour[h][0] else 0.0
            for h in range(2, 25)]


def smooth_spreads(raw):
    """3-hour moving average of per-hour spreads (tames small samples)."""
    sm = []
    for i in range(23):
        lo = max(0, i - 1)
        hi = min(23, i + 2)
        sm.append(sum(raw[lo:hi]) / (hi - lo))
    return [min(max(s, SPREAD_MIN), SPREAD_MAX) for s in sm]


def empirical_spreads(sub, target_name, scale):
    """Per-hour std of ln ratios in a library, smoothed and scaled."""
    per_hour = log_ratios(sub, target_name)
    raw = []
    for h in range(2, 25):
        vals = per_hour[h]
        raw.append(statistics.pstdev(vals) if len(vals) > 1 else 0.15)
    return smooth_spreads([s * scale for s in raw])


def weighted_spreads(lib, target_name, p1, scale):
    """Per-hour kernel-weighted std of ln ratios, smoothed and scaled."""
    lp = math.log(max(p1, 1))
    per_hour = {h: ([], []) for h in range(2, 25)}
    for name, day in lib:
        if name == target_name or day[1] < 1:
            continue
        w = math.exp(-0.5 * ((math.log(day[1]) - lp) / KERNEL_SIGMA) ** 2)
        for h in range(2, 25):
            per_hour[h][0].append(math.log(max(day[h], 1) / day[1]))
            per_hour[h][1].append(w)
    raw = []
    for h in range(2, 25):
        vals, ws = per_hour[h]
        if len(vals) < 2:
            raw.append(0.15)
            continue
        mean = sum(v * w for v, w in zip(vals, ws)) / sum(ws)
        var = sum(w * (v - mean) ** 2 for v, w in zip(vals, ws)) / sum(ws)
        raw.append(math.sqrt(var))
    return smooth_spreads([s * scale for s in raw])


def size_matched(lib, p1):
    """Comparables with hour-1 print in [0.25x, 4x] of the target's."""
    sized = [(n, d) for n, d in lib if 0.25 * p1 <= d[1] <= 4 * p1]
    if len(sized) < 5:
        sized = sorted(lib, key=lambda t: -t[1][1])[:8]
    return sized


def baseline_rows(p1):
    """House baseline: flat median, spread growing with the hour."""
    rows = []
    for h in range(2, 25):
        s = min(max(0.15 * math.sqrt(h - 1), SPREAD_MIN), SPREAD_MAX)
        rows.append({"hour": h, "q": quantiles(p1, 0.0, s)})
    return rows


def quantiles(p1, median_log, spread):
    """Five quantiles around q50 = p1 * exp(median_log)."""
    q50 = p1 * math.exp(median_log)
    out = []
    prev = 0.0
    for z in Z:
        q = q50 * math.exp(z * spread)
        q = float(Decimal(str(q)).quantize(Decimal("0.01"),
                                            rounding=ROUND_HALF_UP))
        q = min(max(q, 1.0), 1e9)
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    """Forecast hours 2..24 for one case. Always returns exactly 23 rows."""
    known = case.get("known") or []
    p1 = 1
    if isinstance(known, dict):
        p1 = max(1, known.get("viewers", 1))
    else:
        for row in known:
            if isinstance(row, dict) and row.get("hour") == 1:
                p1 = max(1, row.get("viewers", 1))
                break
        else:
            if known and isinstance(known[0], dict):
                p1 = max(1, known[0].get("viewers", 1))
    lib = build_library(case)
    if not lib:
        return baseline_rows(p1)
    target_name = case.get("game", {}).get("name", "")
    if p1 >= MEGA_MIN:
        meds = weighted_medians(lib, target_name, p1)
        spreads = weighted_spreads(lib, target_name, p1, MEGA_SPREAD_SCALE)
    else:
        sized = size_matched(lib, p1)
        meds = medians_of(log_ratios(sized, target_name))
        spreads = empirical_spreads(sized, target_name, REST_SPREAD_SCALE)
    rows = []
    for i, h in enumerate(range(2, 25)):
        rows.append({"hour": h,
                     "q": quantiles(p1, SHRINK * meds[i], spreads[i])})
    return rows


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(input_dir.glob("*.in")):
        case = json.loads(in_path.read_text())
        rows = forecast_case(case)
        out_path = output_dir / (in_path.stem + ".out")
        out_path.write_text(json.dumps({"hours": rows}))


if __name__ == "__main__":
    main()
