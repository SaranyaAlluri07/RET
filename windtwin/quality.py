"""Data-quality flags per signal and per row (causal: a flag at t only uses data up to t).

Bits: 1 missing, 2 out of physical range, 4 flatline / stuck, 8 implausible jump.
Rows whose core inputs (wind, power, ambient) are flagged, or where >= 30 % of the monitored raw
sensors are flagged, get quality 'bad' -> alarm state 'unknown' (never an alarm)."""
import numpy as np
import pandas as pd

from . import config as C

MISSING, RANGE, FLATLINE, JUMP = 1, 2, 4, 8
BIT_NAMES = {MISSING: "missing", RANGE: "out of range", FLATLINE: "stuck/flatline", JUMP: "implausible jump"}
CORE = [C.WIND, C.POWER, C.AMBIENT]
BAD_SHARE = 0.30
HISTORY_STEPS = max(C.FLATLINE_STEPS.values()) + 2      # rows of history the streaming monitor keeps


def run_length(x):
    """Length of the current run of identical values (causal), NaN breaks runs."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.zeros(0, int)
    change = np.ones(n, bool)
    change[1:] = ~(x[1:] == x[:-1])          # NaN != NaN -> change
    gid = np.cumsum(change)
    start = np.flatnonzero(change)
    return np.arange(n) - start[gid - 1] + 1


def signal_flags(values, kind):
    """Flag bits for one raw signal."""
    v = np.asarray(values, float)
    f = np.zeros(len(v), np.int8)
    miss = ~np.isfinite(v)
    f[miss] |= MISSING
    lo, hi = C.RANGES.get(kind, (-np.inf, np.inf))
    with np.errstate(invalid="ignore"):
        f[(~miss) & ((v < lo) | (v > hi))] |= RANGE
    n_flat = C.FLATLINE_STEPS.get(kind)
    if n_flat:
        rl = run_length(v)
        stuck = (rl >= n_flat) & ~miss
        if kind == "wind":
            stuck &= v > 0.5                    # calm air reported as constant low wind is plausible
        f[stuck] |= FLATLINE
    jmax = C.MAX_JUMP.get(kind)
    if jmax:
        d = np.abs(np.diff(v, prepend=np.nan))
        with np.errstate(invalid="ignore"):
            f[d > jmax] |= JUMP
    return f


def monitored_raw():
    """Raw sensors the detector depends on (targets' sources + core inputs)."""
    s = set(CORE) | {C.ROTOR_RPM, C.NACELLE}
    for t in C.TARGETS:
        s.update(C.raw_sources(t))
    return sorted(s)


def assess(df):
    """Return (flags DataFrame[int8] per raw/derived signal, row quality array 'ok'|'degraded'|'bad')."""
    raw = [c for c in monitored_raw() if c in df]
    flags = {c: signal_flags(df[c].to_numpy(float), C.kind_of(c)) for c in raw}
    for c in monitored_raw():
        if c not in df:
            flags[c] = np.full(len(df), MISSING, np.int8)
    for key in C.DERIVED:
        src = C.raw_sources(key)
        f = np.zeros(len(df), np.int8)
        for s in src:
            f |= flags[s]
        flags[key] = f
    # derived ratio signals are NaN without load: that is not a quality problem
    for key in ("current_imbalance", "voltage_imbalance"):
        flags[key] = flags[key] & ~np.int8(MISSING)
    fl = pd.DataFrame(flags, index=df.index)
    core_bad = (fl[[c for c in CORE]] != 0).any(axis=1).to_numpy()
    share = (fl[raw] != 0).mean(axis=1).to_numpy() if raw else np.ones(len(df))
    any_bad = (fl[raw] != 0).any(axis=1).to_numpy() if raw else np.ones(len(df), bool)
    q = np.where(core_bad | (share >= BAD_SHARE), "bad", np.where(any_bad, "degraded", "ok"))
    return fl, q


def issues(flags_row):
    """Readable list of issues for one row (Series of flag bits)."""
    out = []
    for sig, bits in flags_row.items():
        if bits and sig not in C.DERIVED:
            names = [n for b, n in BIT_NAMES.items() if bits & b]
            out.append(f"{C.label_of(sig)}: {', '.join(names)}")
    return out


def summary(flags):
    """Share of rows flagged per signal and bit (for reports)."""
    return {sig: {n: round(float(((flags[sig] & b) != 0).mean()), 5) for b, n in BIT_NAMES.items()}
            for sig in flags.columns}
