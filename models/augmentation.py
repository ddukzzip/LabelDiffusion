import numpy as np


class DataAugmentor:
    """
    Two-stage data augmentation for network traffic features.

    Stage 1 – Gaussian noise
        For each sample, add noise ~ N(0, r * σ_f) per feature f,
        where r ~ Uniform(noise_ratio_min, noise_ratio_max)  [5 – 10 %].

    Stage 2 – Mixup (same-class only)
        For a fraction (mixup_ratio) of samples, sample pairs (a, b)
        from the same class and interpolate:
            x_new = λ · x_a + (1−λ) · x_b,   λ ~ Beta(alpha, alpha)
        Label is the shared class label (no soft labels needed).
    """

    def __init__(
        self,
        noise_ratio_min: float = 0.05,
        noise_ratio_max: float = 0.10,
        mixup_alpha: float = 0.2,
        mixup_ratio: float = 0.3,
        seed: int = 42,
    ):
        self.noise_ratio_min = noise_ratio_min
        self.noise_ratio_max = noise_ratio_max
        self.mixup_alpha = mixup_alpha
        self.mixup_ratio = mixup_ratio
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def augment(self, features: np.ndarray, labels: np.ndarray):
        """
        Returns (X_aug, y_aug) with the original samples included.
        """
        features = features.astype(np.float32)
        labels   = labels.astype(int)
        feat_std = features.std(axis=0, keepdims=True) + 1e-8   # (1, F)

        aug_feats  = [features]
        aug_labels = [labels]

        # ── Stage 1: Gaussian noise ───────────────────────────────────
        r = self.rng.uniform(
            self.noise_ratio_min,
            self.noise_ratio_max,
            size=features.shape,
        ).astype(np.float32)
        noise = self.rng.standard_normal(features.shape).astype(np.float32)
        X_noisy = features + noise * r * feat_std
        aug_feats.append(X_noisy)
        aug_labels.append(labels.copy())

        # ── Stage 2: Mixup within class ───────────────────────────────
        n_mixup = int(len(features) * self.mixup_ratio)
        mixed_f, mixed_l = [], []

        for cls in [0, 1]:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) < 2:
                continue

            n_cls = max(1, int(n_mixup * (labels == cls).mean()))
            idx_a = self.rng.choice(cls_idx, size=n_cls, replace=True)
            idx_b = self.rng.choice(cls_idx, size=n_cls, replace=True)

            lam = self.rng.beta(
                self.mixup_alpha, self.mixup_alpha, size=(n_cls, 1)
            ).astype(np.float32)

            X_mix = lam * features[idx_a] + (1.0 - lam) * features[idx_b]
            mixed_f.append(X_mix)
            mixed_l.append(np.full(n_cls, cls, dtype=int))

        if mixed_f:
            aug_feats.append(np.vstack(mixed_f))
            aug_labels.append(np.concatenate(mixed_l))

        X_out = np.vstack(aug_feats)
        y_out = np.concatenate(aug_labels)

        print(f"  [DataAugmentor] {len(features):,} → {len(X_out):,} samples  "
              f"(be={( y_out==0).sum():,}, ma={(y_out==1).sum():,})")
        return X_out, y_out
