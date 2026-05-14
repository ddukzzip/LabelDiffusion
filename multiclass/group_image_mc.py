"""
group_image_mc.py
==================
MulticlassGroupImageBuilder – 15-class group images for CICIDS2018.

For each class k (0=benign, 1-14=attack):
  ch1[f,b] = fraction of ANCHOR samples from class k in bin b
  ch2[f,b] = KDE density of anchor samples from class k
  ch3[f,b] = mean(labels_fine in bin b) / 14.0  (shared across all images)

Anchor selection: KNN consistency on labels_fine.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

LABEL_NAMES = {
    0: "Benign",
    1: "FTP-BruteForce",
    2: "SSH-Bruteforce",
    3: "DoS-GoldenEye",
    4: "DoS-Slowloris",
    5: "DoS-SlowHTTPTest",
    6: "DoS-Hulk",
    7: "DDoS-LOIC-HTTP",
    8: "DDoS-LOIC-UDP",
    9: "DDoS-HOIC",
    10: "BruteForce-Web",
    11: "BruteForce-XSS",
    12: "SQL-Injection",
    13: "Infiltration",
    14: "Bot",
}


class MulticlassGroupImageBuilder:
    """
    Parameters
    ----------
    n_bins      : histogram bin count per feature
    cons_thresh : KNN consistency threshold for anchor selection
    k           : number of KNN neighbors
    min_anchors : fallback if consistent anchors < min_anchors → use all class
    max_anchors : cap per class (capped by random sampling)
    """

    def __init__(self, n_bins: int = 33, cons_thresh: float = 0.7,
                 k: int = 20, min_anchors: int = 10, max_anchors: int = 200):
        self.n_bins      = n_bins
        self.cons_thresh = cons_thresh
        self.k           = k
        self.min_anchors = min_anchors
        self.max_anchors = max_anchors
        self.scaler      = StandardScaler()

    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray,
            labels_fine: np.ndarray,
            seed: int = 42) -> "MulticlassGroupImageBuilder":
        """
        features    : (N, F) float32
        labels_fine : (N,)   int  (0=benign, 1..14=attack)
        """
        self.features    = features.astype(np.float32)
        self.labels_fine = labels_fine.astype(int)
        self.N, self.F   = self.features.shape
        self.classes     = sorted(np.unique(labels_fine).tolist())
        self.rng         = np.random.default_rng(seed)

        feat_s = self.scaler.fit_transform(self.features)

        # ── KNN on fine labels ────────────────────────────────────────
        print(f"[MulticlassGroupImage] KNN (k={self.k}, N={self.N:,})…")
        nn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)
        nn.fit(feat_s)
        _, idx = nn.kneighbors(feat_s)
        idx = idx[:, 1:]                           # exclude self
        neighbor_lf = labels_fine[idx]             # (N, k)

        # Anchor selection per class
        self.anchors: dict = {}
        for cls in self.classes:
            cons_k = (neighbor_lf == cls).mean(axis=1)
            mask_k = (cons_k >= self.cons_thresh) & (labels_fine == cls)
            if mask_k.sum() < self.min_anchors:
                mask_k = (labels_fine == cls)      # fallback: all samples
            # Cap
            if mask_k.sum() > self.max_anchors:
                sel = self.rng.choice(np.where(mask_k)[0],
                                      self.max_anchors, replace=False)
                mask_k2 = np.zeros(self.N, dtype=bool)
                mask_k2[sel] = True
                mask_k = mask_k2
            self.anchors[cls] = mask_k

        for cls in self.classes:
            n_tot = (labels_fine == cls).sum()
            n_anc = self.anchors[cls].sum()
            name  = LABEL_NAMES.get(cls, f"cls{cls}")
            print(f"  anchor_{cls}({name})={n_anc:,}/{n_tot:,}")

        # ── Bin edges ─────────────────────────────────────────────────
        B = self.n_bins
        self.bin_edges = []
        for f in range(self.F):
            v = self.features[:, f]
            eps = max((v.max() - v.min()) * 1e-6, 1e-8)
            self.bin_edges.append(
                np.linspace(v.min() - eps, v.max() + eps, B + 1))

        # ── Shared ch3: mean(labels_fine) / 14.0 per bin ─────────────
        lf_norm = labels_fine.astype(np.float32) / 14.0
        self._ch3 = np.full((self.F, B), 0.5, dtype=np.float32)
        for f in range(self.F):
            b_all = np.clip(
                np.digitize(self.features[:, f], self.bin_edges[f][1:-1]),
                0, B - 1)
            cnt    = np.bincount(b_all, minlength=B).astype(np.float32)
            w_sum  = np.bincount(b_all, weights=lf_norm, minlength=B)
            with np.errstate(invalid="ignore", divide="ignore"):
                self._ch3[f] = np.where(cnt > 0, w_sum / cnt, 0.5).astype(np.float32)

        return self

    # ------------------------------------------------------------------
    def _build_one(self, cls: int) -> np.ndarray:
        F, B = self.F, self.n_bins
        img  = np.zeros((F, B, 3), dtype=np.float32)
        img[:, :, 2] = self._ch3                   # shared ch3

        anc  = self.features[self.anchors[cls]]
        Na   = len(anc)
        if Na == 0:
            print(f"  Warning: class {cls} has no anchors")
            return img

        for f in range(F):
            e  = self.bin_edges[f]
            v  = anc[:, f]
            bi = np.clip(np.digitize(v, e[1:-1]), 0, B - 1)
            ct = np.bincount(bi, minlength=B).astype(np.float32)
            img[f, :, 0] = ct / (Na + 1e-8)

            if v.std() > 1e-8 and len(v) > 1:
                try:
                    vk = (v if len(v) <= 5_000 else
                          v[self.rng.choice(len(v), 5_000, replace=False)])
                    kde = gaussian_kde(vk.astype(np.float64))
                    ctr = ((e[:-1] + e[1:]) / 2).astype(np.float64)
                    dn  = kde(ctr).astype(np.float32)
                    img[f, :, 1] = dn / (dn.max() + 1e-8)
                except Exception:
                    img[f, :, 1] = img[f, :, 0]
            else:
                img[f, :, 1] = img[f, :, 0]

        return img

    # ------------------------------------------------------------------
    def build(self) -> dict:
        """
        Returns dict: {class_id: (F, B, 3) ndarray}
        """
        print(f"[MulticlassGroupImage] Building {len(self.classes)} class images…")
        imgs = {}
        for cls in self.classes:
            name = LABEL_NAMES.get(cls, f"cls{cls}")
            imgs[cls] = self._build_one(cls)
        print(f"  Done. shape={list(imgs.values())[0].shape}")
        return imgs
