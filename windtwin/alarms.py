"""Temporal decision layer: EWMA, CUSUM, persistence + hysteresis + cooldown state machine.

Per target: EWMA e_t = lam*z_t + (1-lam)*e_{t-1} -> one-sided evidence max(e,0), two-sided |e|;
CUSUM (one-sided, upper) S_t = max(0, S_{t-1} + z_t - k); two-sided targets also run the lower CUSUM;
optional bound S <= cusum_cap*h so the statistic recovers soon after a deviation ends.
Both statistics are divided by their 99.5th percentile on the chronological validation split (per
target), which absorbs each residual's autocorrelation: limits L / h are multiples of that level.
Missing z (bad quality / not judged in this operating state) freezes the statistics.
Component statistic = max over its targets; turbine statistic = max over components.

States: normal -> watch (statistic above watch fraction of the limit) -> warning (limit exceeded for
`persist` consecutive judged steps) -> fault (warning sustained for `fault_mult`*persist steps, and both
detectors agree when both are used). An alarm clears only after all statistics stay below `clear` *
limit for `cooldown` judged steps (hysteresis). 'unknown' = data quality bad; 'offline' = turbine
not producing and no latched alarm. Batch (vectorised statistics) and streaming give identical results."""
import math
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C

LEVELS = ("normal", "watch", "warning", "fault")


@dataclass
class AlarmParams:
    lam: float = 0.05
    L: float = 4.0
    k: float = 1.0
    h: float = 40.0
    persist: int = 6
    cooldown: int = 36
    clear: float = 0.5
    use_ewma: bool = True
    use_cusum: bool = True
    watch: float = 0.7
    fault_mult: int = 4
    cusum_cap: float = 0.0      # >0: CUSUM bounded at cusum_cap*h (recovers quickly after a deviation ends)
    combine: str = "or"         # 'or': EWMA or CUSUM exceedance counts as a hit; 'and': both required

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ statistics (batch)
def ewma(Z, lam, e0=None):
    """Hold-on-NaN EWMA per column. Z: (n x T)."""
    n, T = Z.shape
    out = np.empty((n, T))
    e = np.zeros(T) if e0 is None else np.array(e0, float)
    a, b = lam, 1.0 - lam
    for i in range(n):
        z = Z[i]
        ok = np.isfinite(z)
        e = np.where(ok, a * np.where(ok, z, 0.0) + b * e, e)
        out[i] = e
    return out


def cusum(Z, k, two_sided, s0=None, cap=None):
    """Hold-on-NaN CUSUM per column; two-sided columns return max(upper, lower)."""
    n, T = Z.shape
    up = np.zeros(T) if s0 is None else np.array(s0[0], float)
    lo = np.zeros(T) if s0 is None else np.array(s0[1], float)
    out = np.empty((n, T))
    for i in range(n):
        z = Z[i]
        ok = np.isfinite(z)
        zz = np.where(ok, z, 0.0)
        up = np.where(ok, np.maximum(0.0, up + zz - k), up)
        lo = np.where(ok & two_sided, np.maximum(0.0, lo - zz - k), lo)
        if cap is not None:
            up, lo = np.minimum(up, cap), np.minimum(lo, cap)
        out[i] = np.maximum(up, lo)
    return out


Q_SCALE = 0.995       # statistics are expressed in units of their validation 99.5th percentile


def ewma_scale(Z_val, lam, two_sided=None):
    """99.5th percentile of the EWMA evidence on the validation split (per target); this absorbs each
    target's autocorrelation. Floored at the iid value (2.576 * sqrt(lam / (2 - lam)))."""
    E = ewma(Z_val, lam)
    two = np.zeros(Z_val.shape[1], bool) if two_sided is None else np.asarray(two_sided, bool)
    iid = 2.576 * np.sqrt(lam / (2 - lam))
    s = np.full(Z_val.shape[1], iid)
    for j in range(Z_val.shape[1]):
        e = E[np.isfinite(Z_val[:, j]), j]
        if len(e) > 50:
            ev = np.abs(e) if two[j] else np.maximum(e, 0.0)
            s[j] = max(float(np.quantile(ev, Q_SCALE)), iid)
    return s


def cusum_scale(Z_val, k, two_sided):
    """99.5th percentile of the CUSUM statistic on the validation split (per target), floor 1."""
    S = cusum(Z_val, k, np.asarray(two_sided, bool))
    s = np.ones(Z_val.shape[1])
    for j in range(Z_val.shape[1]):
        v = S[np.isfinite(Z_val[:, j]), j]
        if len(v) > 50:
            s[j] = max(float(np.quantile(v, Q_SCALE)), 1.0)
    return s


def evidence(E, scale, two_sided):
    """Normalised one-/two-sided EWMA evidence (n x T); NaN where the target never had data."""
    x = E / scale
    return np.where(two_sided, np.abs(x), np.maximum(x, 0.0))


def component_max(X, comp_idx, valid=None):
    """(n x n_comp) max over each component's targets, NaN when none is valid."""
    out = np.full((X.shape[0], len(comp_idx)), np.nan)
    ok = ~np.isnan(X) if valid is None else (~np.isnan(X) & valid)
    Xm = np.where(ok, X, -np.inf)
    for c, idx in enumerate(comp_idx):
        if len(idx):
            m = Xm[:, idx].max(axis=1)
            out[:, c] = np.where(m == -np.inf, np.nan, m)
    return out


# ------------------------------------------------------------------ state machine (shared)
class _Machine:
    def __init__(self, p: AlarmParams):
        self.p = p
        self.level, self.run, self.calm, self.dev_start = 0, 0, 0, -1
        self.t = -1

    def step(self, e, s, bad, off, judged=True):
        """e, s: turbine-level EWMA evidence / CUSUM; judged: any target judged at this step.
        Unjudged steps (bad quality, stopped) freeze counters. Returns (level, state)."""
        p = self.p
        self.t += 1
        e_ok, s_ok = p.use_ewma and math.isfinite(e), p.use_cusum and math.isfinite(s)
        if bad:
            return self.level, "unknown"
        if not judged or not (e_ok or s_ok):
            if self.level >= 2:
                return self.level, LEVELS[self.level]
            return self.level, ("offline" if off else "unknown")
        if p.combine == "and" and p.use_ewma and p.use_cusum:
            hit = e_ok and s_ok and e > p.L and s > p.h
        else:
            hit = (e_ok and e > p.L) or (s_ok and s > p.h)
        near = (e_ok and e > p.watch * p.L) or (s_ok and s > p.watch * p.h)
        self.run = self.run + 1 if hit else 0
        if self.level < 2:
            if self.run >= p.persist:
                self.level, self.calm = 2, 0
            elif hit or near:
                self.level = 1
            else:
                self.level = 0
            if self.level >= 1 and self.dev_start < 0:
                self.dev_start = self.t
            if self.level == 0:
                self.dev_start = -1
        else:
            if self.level == 2 and self.run >= p.fault_mult * p.persist:
                both = (not (p.use_ewma and p.use_cusum)) or ((e_ok and e > p.L) and (s_ok and s > p.h))
                if both:
                    self.level = 3
            below = (not e_ok or e < p.clear * p.L) and (not s_ok or s < p.clear * p.h)
            self.calm = self.calm + 1 if below else 0
            if self.calm >= p.cooldown:
                self.level, self.run, self.calm, self.dev_start = 0, 0, 0, -1
        state = LEVELS[self.level]
        if off and self.level < 2:
            state = "offline"
        return self.level, state


def health(e, s, p: AlarmParams):
    r = 0.0
    if p.use_ewma and math.isfinite(e):
        r = max(r, e / p.L)
    if p.use_cusum and math.isfinite(s):
        r = max(r, s / p.h)
    return 100.0 * math.exp(-0.6931471805599453 * r * r)


def cap_of(p, cscale):
    """Per-target bound of the raw CUSUM (cusum_cap * h in normalised units), or None."""
    return p.cusum_cap * p.h * np.asarray(cscale, float) if p.cusum_cap and p.cusum_cap > 0 else None


# ------------------------------------------------------------------ batch
def run_batch(Z, bad, off, p: AlarmParams, scale, two_sided, comp_idx, cscale=None):
    """Z: signed z (n x T); scale / cscale: validation scales of the EWMA / CUSUM statistics.
    Returns dict of arrays (level, state, alarm, E, S, Ec, Sc, health, dev_start)."""
    lam = p.lam if p.use_ewma else 1.0
    cs = np.ones(Z.shape[1]) if cscale is None else np.asarray(cscale, float)
    Ev = evidence(ewma(Z, lam), scale, two_sided)
    Sv = cusum(Z, p.k, two_sided, cap=cap_of(p, cs)) / cs if p.use_cusum else np.zeros_like(Z)
    seen = np.maximum.accumulate(np.isfinite(Z), axis=0)       # target judged at least once
    Ec, Sc = component_max(Ev, comp_idx, seen), component_max(Sv, comp_idx, seen)
    return _decide(Ec, Sc, bad, off, p, np.isfinite(Z).any(axis=1))


def _turbine(Ec, Sc):
    with np.errstate(all="ignore"):
        e = np.where(np.all(np.isnan(Ec), axis=1), np.nan, np.nanmax(np.nan_to_num(Ec, nan=-np.inf), axis=1))
        s = np.where(np.all(np.isnan(Sc), axis=1), np.nan, np.nanmax(np.nan_to_num(Sc, nan=-np.inf), axis=1))
    return e, s


def _decide(Ec, Sc, bad, off, p, judged):
    n = len(Ec)
    E, S = _turbine(Ec, Sc)
    m = _Machine(p)
    level = np.zeros(n, np.int8)
    state = np.empty(n, dtype=object)
    dev = np.full(n, -1)
    hl = np.empty(n)
    El, Sl, bl, ol, jl = E.tolist(), S.tolist(), np.asarray(bad, bool).tolist(), np.asarray(off, bool).tolist(),         np.asarray(judged, bool).tolist()
    for i in range(n):
        level[i], state[i] = m.step(El[i], Sl[i], bl[i], ol[i], jl[i])
        dev[i] = m.dev_start
        hl[i] = health(El[i], Sl[i], p)
    return {"level": level, "state": state, "alarm": level >= 2, "E": E, "S": S, "Ec": Ec, "Sc": Sc,
            "health": hl, "dev_start": dev}


# ------------------------------------------------------------------ streaming
class AlarmStream:
    """Stateful step-by-step version of run_batch (identical results)."""

    def __init__(self, p: AlarmParams, scale, two_sided, comp_idx, cscale=None):
        self.p, self.scale, self.two = p, np.asarray(scale, float), np.asarray(two_sided, bool)
        self.comp_idx = comp_idx
        T = len(self.scale)
        self.cscale = np.ones(T) if cscale is None else np.asarray(cscale, float)
        self.cap = cap_of(p, self.cscale)
        self.e, self.up, self.lo = np.zeros(T), np.zeros(T), np.zeros(T)
        self.seen = np.zeros(T, bool)
        self.m = _Machine(p)

    def step(self, z, bad, off):
        p = self.p
        z = np.asarray(z, float)
        ok = np.isfinite(z)
        zz = np.where(ok, z, 0.0)
        lam = p.lam if p.use_ewma else 1.0
        self.e = np.where(ok, lam * zz + (1.0 - lam) * self.e, self.e)
        if p.use_cusum:
            self.up = np.where(ok, np.maximum(0.0, self.up + zz - p.k), self.up)
            self.lo = np.where(ok & self.two, np.maximum(0.0, self.lo - zz - p.k), self.lo)
            if self.cap is not None:
                self.up, self.lo = np.minimum(self.up, self.cap), np.minimum(self.lo, self.cap)
        self.seen |= ok
        ev = evidence(self.e[None, :], self.scale, self.two)
        sv = (np.maximum(self.up, self.lo) / self.cscale)[None, :]
        Ec = component_max(ev, self.comp_idx, self.seen[None, :])
        Sc = component_max(sv, self.comp_idx, self.seen[None, :])
        E, S = _turbine(Ec, Sc)
        e0, s0 = float(E[0]), float(S[0])
        level, state = self.m.step(e0, s0, bool(bad), bool(off), bool(ok.any()))
        return {"level": level, "state": state, "alarm": level >= 2, "E": e0, "S": s0,
                "Ec": Ec[0], "Sc": Sc[0], "health": health(e0, s0, p), "dev_start": self.m.dev_start}


def comp_index(targets):
    """List (per component in C.COMPONENTS order) of target column indices."""
    pos = {t: i for i, t in enumerate(targets)}
    return [np.array([pos[t] for t in C.SUBSYSTEMS[c] if t in pos], int) for c in C.COMPONENTS]


def episodes(alarm):
    """[(start, end_exclusive)] runs of True."""
    a = np.asarray(alarm, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], a, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
