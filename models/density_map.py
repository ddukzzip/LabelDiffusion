import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde


class DensityMapBuilder:
    """
    2-Channel Density Map Builder — RAPIER Fig3 철학을 2D로 확장

    ch1[f, b] = feature x bin 히스토그램 (anchor 기반 분포)
    ch2[f, b] = be/ma density 합성 grayscale
                1.0 = be 영역 (white)
                0.0 = ma 영역 (gray/black)
                0.5 = 겹침 / 데이터 없음

    noisy detection 철학:
      given_label=be(0) 인데 ch2 낮음(ma 영역) → noisy
      given_label=ma(1) 인데 ch2 높음(be 영역) → noisy
      → 분포와 라벨 불일치 = noisy score
    """

    def __init__(self, n_bins=33, cons_thresh=0.7, k=20):
        self.n_bins = n_bins
        self.cons_thresh = cons_thresh
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, features, given_label):
        self.features = features.astype(np.float32)
        self.given_label = given_label.astype(int)
        self.N, self.F = self.features.shape

        feat_s = self.scaler.fit_transform(self.features)

        # KNN consistency로 anchor 선택 (기존 방식 유지)
        print(f"[DensityMapBuilder] KNN consistency (k={self.k}, N={self.N:,})...")
        nn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)
        nn.fit(feat_s)
        _, indices = nn.kneighbors(feat_s)
        indices = indices[:, 1:]
        neighbor_labels = self.given_label[indices]
        consistency = (neighbor_labels == self.given_label.reshape(-1, 1)).mean(axis=1)

        self.anchor_be = (consistency >= self.cons_thresh) & (self.given_label == 0)
        self.anchor_ma = (consistency >= self.cons_thresh) & (self.given_label == 1)
        print(f"  anchor be={self.anchor_be.sum():,}  ma={self.anchor_ma.sum():,}")

        # bin edges (전체 데이터 기준)
        self.bin_edges = []
        for f in range(self.F):
            vals = self.features[:, f]
            vmin, vmax = float(vals.min()), float(vals.max())
            eps = max((vmax - vmin) * 1e-6, 1e-8)
            self.bin_edges.append(
                np.linspace(vmin - eps, vmax + eps, self.n_bins + 1)
            )
        return self

    def _kde_density(self, vals, edges):
        """KDE density at bin centres, normalised to [0,1]."""
        B = self.n_bins
        if vals.std() < 1e-8 or len(vals) < 2:
            # fallback: histogram
            b_idx = np.clip(np.digitize(vals, edges[1:-1]), 0, B - 1)
            counts = np.bincount(b_idx, minlength=B).astype(np.float32)
            d = counts / (counts.max() + 1e-8)
            return d
        try:
            v = vals if len(vals) <= 5000 else \
                vals[np.random.default_rng(0).choice(len(vals), 5000, replace=False)]
            kde = gaussian_kde(v.astype(np.float64))
            centres = ((edges[:-1] + edges[1:]) / 2).astype(np.float64)
            d = kde(centres).astype(np.float32)
            return d / (d.max() + 1e-8)
        except Exception:
            b_idx = np.clip(np.digitize(vals, edges[1:-1]), 0, B - 1)
            counts = np.bincount(b_idx, minlength=B).astype(np.float32)
            return counts / (counts.max() + 1e-8)

    def build(self):
        """
        Returns single image of shape (F, n_bins, 2):
          img[:, :, 0] = ch1: anchor histogram
          img[:, :, 1] = ch2: be/ma density map (be=1.0, ma=0.0)
        """
        F, B = self.F, self.n_bins
        img = np.full((F, B, 2), 0.5, dtype=np.float32)

        feat_be = self.features[self.anchor_be]
        feat_ma = self.features[self.anchor_ma]
        N_be = len(feat_be)
        N_ma = len(feat_ma)

        print(f"[DensityMapBuilder] 이미지 생성 (be={N_be:,}, ma={N_ma:,})...")

        for f in range(F):
            edges = self.bin_edges[f]

            # ch1: 전체 anchor histogram (be+ma 합산, 분포 참조용)
            all_anchor = self.features[self.anchor_be | self.anchor_ma]
            if len(all_anchor) > 0:
                b_all = np.clip(
                    np.digitize(all_anchor[:, f], edges[1:-1]), 0, B - 1
                )
                counts = np.bincount(b_all, minlength=B).astype(np.float32)
                img[f, :, 0] = counts / (counts.max() + 1e-8)

            # ch2: be density (white=1.0) vs ma density (gray=0.0)
            d_be = self._kde_density(feat_be[:, f], edges) if N_be > 1 else np.zeros(B)
            d_ma = self._kde_density(feat_ma[:, f], edges) if N_ma > 1 else np.zeros(B)

            denom = d_be + d_ma
            # be/(be+ma): be만 있으면 1.0, ma만 있으면 0.0, 둘 다 없으면 0.5
            ch2 = np.where(denom > 1e-8, d_be / denom, 0.5).astype(np.float32)
            img[f, :, 1] = ch2

        print(f"  완료. shape={img.shape}  "
              f"ch2 mean={img[:,:,1].mean():.3f}  "
              f"ch2 std={img[:,:,1].std():.3f}")
        self.img = img
        return img
