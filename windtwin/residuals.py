"""Regime-aware residual normalization.

z = direction * (r - median_bin) / sigma_bin, with bins = power bins x ambient-temperature terciles.
median and sigma (1.4826 * MAD) come from the chronological validation split only, are shrunk toward
the global value for sparse bins (weight n / (n + N0)) and sigma has a physical floor per signal kind.
Two-sided targets keep their sign (direction 0); the alarm layer treats them symmetrically."""
import numpy as np

from . import config as C

POWER_EDGES = np.array([C.P_PRODUCTION, 0.2, 0.5, 0.8])     # 5 power bins (first = idle/stopped)
N0 = 100                                                   # shrinkage strength (rows)
Z_CLIP = 8.0


class RegimeNormalizer:
    def __init__(self, targets):
        self.targets = list(targets)
        self.amb_edges = np.array([10.0, 20.0])
        self.center, self.sigma = {}, {}          # target -> array[n_bins]
        self.glob = {}

    @property
    def n_bins(self):
        return (len(POWER_EDGES) + 1) * (len(self.amb_edges) + 1)

    def regime(self, power, ambient):
        pb = np.searchsorted(POWER_EDGES, np.nan_to_num(np.asarray(power, float), nan=0.0), side="right")
        ab = np.searchsorted(self.amb_edges, np.nan_to_num(np.asarray(ambient, float), nan=15.0), side="right")
        return pb * (len(self.amb_edges) + 1) + ab

    def fit(self, R, power, ambient, valid):
        """R: residual matrix (n x T) on validation rows; valid: bool matrix (n x T)."""
        amb = np.asarray(ambient, float)
        amb = amb[np.isfinite(amb)]
        if len(amb) > 30:
            self.amb_edges = np.quantile(amb, [1 / 3, 2 / 3])
        reg = self.regime(power, ambient)
        for j, t in enumerate(self.targets):
            r, ok = R[:, j], valid[:, j] & np.isfinite(R[:, j])
            floor = C.SIGMA_FLOOR.get(C.kind_of(t), 1e-6)
            if ok.sum() < 30:
                self.center[t], self.sigma[t] = np.zeros(self.n_bins), np.full(self.n_bins, np.inf)
                continue
            g_med = float(np.median(r[ok]))
            g_sig = max(float(1.4826 * np.median(np.abs(r[ok] - g_med))), floor)
            self.glob[t] = (g_med, g_sig)
            cen, sig = np.full(self.n_bins, g_med), np.full(self.n_bins, g_sig)
            for b in range(self.n_bins):
                m = ok & (reg == b)
                n = int(m.sum())
                if n < 10:
                    continue
                med = float(np.median(r[m]))
                s = float(1.4826 * np.median(np.abs(r[m] - med)))
                w = n / (n + N0)
                cen[b] = w * med + (1 - w) * g_med
                sig[b] = max(w * s + (1 - w) * g_sig, floor)
            self.center[t], self.sigma[t] = cen, sig
        return self

    def transform(self, R, power, ambient):
        """Signed z matrix (n x T): positive = bad direction for one-sided targets."""
        reg = self.regime(power, ambient)
        Z = np.full(R.shape, np.nan)
        for j, t in enumerate(self.targets):
            if t not in self.center:
                continue
            z = (R[:, j] - self.center[t][reg]) / self.sigma[t][reg]
            d = C.DIRECTION.get(t, 1)
            Z[:, j] = np.clip(z * (d if d != 0 else 1), -Z_CLIP, Z_CLIP)
        return Z

    def band(self, t, power, ambient, zq=1.645):
        """Expected-value offset and half-width of the normal band for target t (in signal units)."""
        reg = self.regime(power, ambient)
        return self.center[t][reg], zq * self.sigma[t][reg]


def two_sided_mask(targets):
    return np.array([C.DIRECTION.get(t, 1) == 0 for t in targets])
