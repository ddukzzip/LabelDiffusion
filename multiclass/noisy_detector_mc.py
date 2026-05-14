"""
noisy_detector_mc.py
====================
MulticlassNoisyDetector – similarity-based noisy label detector.

Detection:
  For each sample x and each class k, compute similarity:
    sim_k = sum_f(ch1_k[f,b_f] * ch2_k[f,b_f]) / norm_k

  predicted_class = argmax_k(sim_k)
  noisy_score = 1 - sim_given / (sum_k(sim_k) + 1e-8)

  High noisy_score → sample likely mislabeled.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


class MulticlassNoisyDetector:

    def __init__(self):
        self.imgs       = None   # {cls: (F, B, 3)}
        self.bin_edges  = None
        self.classes    = None

    # ------------------------------------------------------------------
    def fit(self, imgs: dict, bin_edges: list) -> "MulticlassNoisyDetector":
        """
        imgs       : {class_id: (F, B, 3)}
        bin_edges  : list of F arrays, each length B+1
        """
        self.imgs      = {k: v.astype(np.float32) for k, v in imgs.items()}
        self.bin_edges = bin_edges
        self.classes   = sorted(imgs.keys())

        F = list(imgs.values())[0].shape[0]
        B = list(imgs.values())[0].shape[1]
        self.F = F
        self.B = B

        # Normalisation per class: sum of ch1*ch2 over all (f,b)
        self._norms = {}
        for cls in self.classes:
            img = self.imgs[cls]
            tot = (img[:, :, 0] * img[:, :, 1]).sum()
            self._norms[cls] = max(float(tot), 1e-8)

        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _gather(channel: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
        """
        channel : (F, B)
        bin_idx : (N, F)
        Returns : (N, F)
        """
        N, F = bin_idx.shape
        f_idx = np.tile(np.arange(F), N)
        b_idx = bin_idx.reshape(-1)
        return channel[f_idx, b_idx].reshape(N, F)

    # ------------------------------------------------------------------
    def detect(self, features: np.ndarray,
               given_label: np.ndarray) -> np.ndarray:
        """
        features    : (N, F) float32
        given_label : (N,)   int  (0..14)
        Returns     : noisy_score (N,) float32 in [0, 1]
        """
        features    = features.astype(np.float32)
        given_label = given_label.astype(int)
        self._given_label = given_label.copy()
        N = len(features)

        # ── Bin indices ───────────────────────────────────────────────
        bin_idx = np.zeros((N, self.F), dtype=np.int32)
        for f in range(self.F):
            b = np.digitize(features[:, f], self.bin_edges[f][1:-1])
            bin_idx[:, f] = np.clip(b, 0, self.B - 1)

        # ── Similarity per class ──────────────────────────────────────
        sim_mat   = np.zeros((N, len(self.classes)), dtype=np.float32)
        class_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            img  = self.imgs[cls]
            ch1  = self._gather(img[:, :, 0], bin_idx)   # (N, F)
            ch2  = self._gather(img[:, :, 1], bin_idx)   # (N, F)
            sim  = (ch1 * ch2).sum(axis=1) / self._norms[cls]  # (N,)
            sim_mat[:, class_idx[cls]] = sim

        # ── Noisy score ───────────────────────────────────────────────
        sim_total = sim_mat.sum(axis=1)                   # (N,)
        sim_given = sim_mat[np.arange(N),
                            [class_idx[g] for g in given_label]]  # (N,)

        # One-vs-Best formula:
        #   noisy_score = sim_best_other / (sim_given + sim_best_other + 1e-8)
        # 0 → given class is the clear winner (clean)
        # 0.5 → uniform (ambiguous)
        # 1 → another class strongly dominates (noisy)
        sim_best_other = sim_mat.copy()
        sim_best_other[np.arange(N),
                       [class_idx[g] for g in given_label]] = -1.0
        sim_best_other = sim_best_other.max(axis=1)          # (N,)
        sim_best_other = np.maximum(sim_best_other, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            noisy_score = sim_best_other / (sim_given + sim_best_other + 1e-8)
        noisy_score = np.clip(noisy_score, 0.0, 1.0).astype(np.float32)

        self.noisy_score      = noisy_score
        self.sim_mat          = sim_mat
        self.predicted_class  = np.array(
            [self.classes[i] for i in sim_mat.argmax(axis=1)])

        print(f"  [MC-detect] noisy_score: "
              f"mean={noisy_score.mean():.4f}  "
              f"std={noisy_score.std():.4f}  "
              f"max={noisy_score.max():.4f}  "
              f"(>0.5: {(noisy_score > 0.5).mean()*100:.1f}%)")

        # Agreement stats
        agree = (self.predicted_class == given_label).mean()
        print(f"  predicted == given_label: {agree*100:.1f}%")

        return noisy_score

    # ------------------------------------------------------------------
    def clean_zscore(self, features: np.ndarray,
                     given_label: np.ndarray,
                     z_flip: float = 2.0,
                     z_remove: float = 1.5):
        """
        Z-score threshold cleaning.
        Flip: change given_label to predicted_class.
        Remove: remove ambiguous samples.

        Returns: X_clean, y_clean, info
        """
        score  = self.noisy_score
        mu, sg = score.mean(), score.std()
        ft     = mu + z_flip   * sg
        rt     = mu + z_remove * sg

        new_y  = given_label.copy().astype(int)
        flip_m = score > ft
        rm_m   = (score > rt) & ~flip_m

        # Flip to predicted class (not 1-y)
        new_y[flip_m] = self.predicted_class[flip_m].astype(int)

        keep = ~rm_m
        info = {
            "mu": float(mu), "sg": float(sg),
            "ft": float(ft), "rt": float(rt),
            "flip": int(flip_m.sum()), "remove": int(rm_m.sum()),
            "retain": int(keep.sum()),
        }
        print(f"  [Z-clean] mu={mu:.4f} sg={sg:.4f} "
              f"flip>{ft:.4f}={flip_m.sum():,}  "
              f"rm>{rt:.4f}={rm_m.sum():,}  "
              f"retain={keep.sum():,}")

        return features[keep], new_y[keep], info

    # ------------------------------------------------------------------
    def evaluate(self, true_label: np.ndarray) -> dict:
        """Evaluate detection quality against ground truth."""
        given = self._given_label
        noise_mask = (given != true_label.astype(int))

        if noise_mask.sum() == 0 or noise_mask.sum() == len(noise_mask):
            print("  [MC-eval] Trivial noise mask – skip AUC")
            return {}

        auc = roc_auc_score(noise_mask.astype(int), self.noisy_score)

        thresh    = np.percentile(self.noisy_score, 70)
        pred_flag = self.noisy_score > thresh
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(noise_mask, pred_flag, zero_division=0)
        rec  = recall_score(noise_mask, pred_flag, zero_division=0)
        f1   = f1_score(noise_mask, pred_flag, zero_division=0)

        print(f"  [MC-eval] Detection AUC={auc:.4f}  "
              f"P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        return {"auc": auc, "prec": prec, "rec": rec, "f1": f1}
