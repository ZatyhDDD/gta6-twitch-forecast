"""gta6-twitch-ccu-forecast-01 entry: launch-day shape transfer.

Forecasts Twitch viewers for Grand Theft Auto VI in each hour of its release
day (2026-11-19 UTC) after hour 1, as five quantiles per hour.

Model
-----
Every big game launch follows the same daily rhythm on Twitch: hour 1
(00:00-01:00 UTC) is already busy because Europe and Asia-Pacific unlocked
at their midnight, the day dips while Europe sleeps and the US has not
unlocked yet, then it climbs into the evening as the US unlocks and the
evening peak arrives. Instead of guessing that rhythm, this program reads
it off the case's own `comparables`: past big launches with hourly Twitch
counts.

For each comparable it takes the launch day's hourly counts and forms the
shape ratio r_h = viewers_h / viewers_1 for hours 2..24. The forecast
median for hour h is the target's hour-1 print times the median ratio
across the most similar launches:

    q50(h) = p1 * exp(0.9 * median(ln r_h))

The library is size-matched: comparables whose hour-1 print is within
0.25x..4x of the target's, so a giant launch is compared with giant
launches, not with small ones whose ratios explode. When fewer than five
match, the eight biggest clean comparables are used instead.

Early-access launches are left out (their launch-day shape is
contaminated: the peak happened during early access), as is the
Grand Theft Auto VI category's own pre-release series (a premiere, not
a launch day).

Spread
------
Quantiles widen with the horizon like a random walk in log space, the
same schedule as the house baseline:

    q_tau(h) = q50(h) * exp(z_tau * 0.15 * sqrt(h - 1))

with z_tau the standard normal quantiles (-1.645, -0.674, 0, 0.674, 1.645).
The spread is deliberately not fitted to the comparables: their
cross-launch dispersion overstates how wrong the median is for one
launch, and the pinball loss punishes a too-wide band at every hour.

Validation (leave-one-out over the comparables in the practice cases,
each launch forecast from the others, pinball loss on the log scale):
big launches (hour-1 print >= 100k, the GTA VI-like set): 0.208 vs the
house baseline's 0.253. All clean launches: 0.808 vs 1.137.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there):
    JSON {"schema": "makefun/twitch-ccu-case/v1", "id", "game",
    "day_start", "checkpoint", "levels", "known", "forecast",
    "history", "history_hourly", "streams", "categories_hourly",
    "comparables", "notes"}, where `known` holds hour 1's print
    {hour: 1, from, until, viewers, channels, reads} and `forecast`
    lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2,
    "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours
    2..24 each once, no other key, each q five finite numbers above 0
    and at most 1e9 (viewers; fractions allowed), non-decreasing; at
    most 65,536 bytes. Anything else is an invalid case, scored as if
    every quantile equalled hour 1's print, and an entry with an invalid
    case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour.
SIGMA = 0.15
# How much of the comparables' median shape to trust (1.0 = full shape).
SHAPE = 0.9
# Size-matching window for the comparable library, as a multiple of the
# target's hour-1 print.
SIZE_LO = 0.25
SIZE_HI = 4.0
# Library size below which the biggest launches are used instead.
MIN_LIBRARY = 5
FALLBACK_TOP = 8


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def launch_day_hours(comp: dict) -> dict | None:
    """{hour: viewers} for the comparable's launch day (hours 1..24).

    The hourly series runs from 14 days before the release day to 3 days
    after it, so the launch day starts at the point whose `from` equals
    `day_start`. Returns None when the day does not align (the Grand Theft
    Auto VI pre-release series) or is too short.
    """
    hourly = comp.get("hourly") or []
    start = next((i for i, p in enumerate(hourly) if p.get("from") == comp.get("day_start")), None)
    if start is None or start + 24 > len(hourly):
        return None
    return {h: max(0, hourly[start + h - 1].get("viewers") or 0) for h in range(1, 25)}


def usable(comp: dict) -> bool:
    """True for a comparable whose launch-day shape can be trusted."""
    if "early-access" in (comp.get("unlock") or ""):
        return False
    if "before release" in (comp.get("name") or ""):
        return False
    return True


def library(case: dict, p1: float) -> list:
    """Launch-day hours of the comparables most similar to the target.

    Size-matched by hour-1 print; when fewer than MIN_LIBRARY match, the
    FALLBACK_TOP biggest clean comparables are used instead.
    """
    days = []
    for comp in case.get("comparables") or []:
        if not usable(comp):
            continue
        day = launch_day_hours(comp)
        if day is None or day[1] < 1:
            continue
        days.append(day)
    matched = [d for d in days if SIZE_LO * p1 <= d[1] <= SIZE_HI * p1]
    if len(matched) >= MIN_LIBRARY:
        return matched
    return sorted(days, key=lambda d: -d[1])[:FALLBACK_TOP]


def shape_medians(days: list) -> dict:
    """Median log-ratio to hour 1 for each forecast hour, over the library."""
    medians = {}
    for h in range(2, 25):
        vals = [math.log(max(d[h], 1) / d[1]) for d in days]
        medians[h] = statistics.median(vals) if vals else 0.0
    return medians


def hour_quantiles(p1: float, median_log_ratio: float, h: int) -> list:
    """The five quantiles of hour h around the shaped median."""
    spread = min(max(SIGMA * math.sqrt(h - 1), 0.10), 0.90)
    q50 = p1 * math.exp(SHAPE * median_log_ratio)
    qs = [round2(q50 * math.exp(z * spread)) for z in Z]
    qs = [min(max(q, 1), 1_000_000_000) for q in qs]
    # Keep the row non-decreasing after rounding and clamping.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    medians = shape_medians(library(case, p1))
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, medians[row["hour"]], row["hour"])} for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
