import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class DensityNoisyDetector:
    """
    RAPIER Fig3 철학 기반 Noisy Label Detector

    clean 전략: 클래스별 독립 percentile
      → be 샘플 중 상위 top_remove_pct% 제거
      → ma 샘플 중 상위 top_remove_pct% 제거
      → 클래스 균형 유지
    """

    def __init__(self, top_remove_pct=30.0, top_flip_pct=0.0):
        self.top_remove_pct = top_remove_pct
        self.top_flip_pct   = top_flip_pct

    def fit(self, img, bin_edges):
        self.img = img.astype(np.float32)
        self.bin_edges = bin_edges
        self.F, self.B, _ = img.shape
        return self

    def detect(self, features, given_label):
        features    = features.astype(np.float32)
        given_label = given_label.astype(int)
        self._given_label = given_label.copy()
        N, F = features.shape

        bin_indices = np.zeros((N, F), dtype=np.int32)
        for f in range(F):
            b = np.digitize(features[:, f], self.bin_edges[f][1:-1])
            bin_indices[:, f] = np.clip(b, 0, self.B - 1)

        ch2      = self.img[:, :, 1]
        f_idx    = np.tile(np.arange(F), N)
        b_idx    = bin_indices.reshape(-1)
        ch2_vals = ch2[f_idx, b_idx].reshape(N, F)

        ch1      = self.img[:, :, 0]
        ch1_vals = ch1[f_idx, b_idx].reshape(N, F)
        w = ch1_vals / (ch1_vals.sum(axis=1, keepdims=True) + 1e-8)

        mean_ch2 = (ch2_vals * w).sum(axis=1)

        noisy_score = np.where(
            given_label == 0,
            1.0 - mean_ch2,
            mean_ch2,
        ).astype(np.float32)

        self.noisy_score = noisy_score
        self.mean_ch2    = mean_ch2

        print(f"  [DensityDetector] noisy_score: "
              f"mean={noisy_score.mean():.4f}  "
              f"std={noisy_score.std():.4f}  "
              f"max={noisy_score.max():.4f}  "
              f"p70={np.percentile(noisy_score, 70):.4f}  "
              f"p90={np.percentile(noisy_score, 90):.4f}")
        return noisy_score

    def evaluate(self, true_label):
        given      = self._given_label
        noise_mask = (given != true_label.astype(int)).astype(int)
        if noise_mask.sum() == 0 or noise_mask.sum() == len(noise_mask):
            print("  [DensityDetector] Cannot evaluate (trivial mask)")
            return {}
        auc  = roc_auc_score(noise_mask, self.noisy_score)
        thresh    = np.percentile(self.noisy_score, 70)
        pred_flag = (self.noisy_score > thresh).astype(int)
        prec = precision_score(noise_mask, pred_flag, zero_division=0)
        rec  = recall_score(noise_mask, pred_flag, zero_division=0)
        f1   = f1_score(noise_mask, pred_flag, zero_division=0)
        print(f"  [DensityDetector] AUC={auc:.4f}  "
              f"P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        return {"auc": auc, "prec": prec, "rec": rec, "f1": f1}

    def clean(self, features, given_label):
        scores     = self.noisy_score
        new_labels = given_label.copy().astype(int)
        keep_mask  = np.ones(len(scores), dtype=bool)

        # 클래스별 독립 percentile → 균형 유지
        for cls in [0, 1]:
            cls_mask = given_label == cls
            cls_idx  = np.where(cls_mask)[0]
            if len(cls_idx) == 0:
                continue
            cls_scores = scores[cls_idx]

            if self.top_flip_pct > 0:
                flip_thresh = np.percentile(cls_scores, 100 - self.top_flip_pct)
                flip_in_cls = cls_scores >= flip_thresh
                new_labels[cls_idx[flip_in_cls]] = 1 - new_labels[cls_idx[flip_in_cls]]

            if self.top_remove_pct > 0:
                remove_thresh = np.percentile(cls_scores, 100 - self.top_remove_pct)
                remove_in_cls = cls_scores >= remove_thresh
                keep_mask[cls_idx[remove_in_cls]] = False

        retained   = keep_mask.sum()
        be_ret     = (new_labels[keep_mask] == 0).sum()
        ma_ret     = (new_labels[keep_mask] == 1).sum()
        print(f"  [DensityDetector] remove={( ~keep_mask).sum():,}  "
              f"→ retained={retained:,}  "
              f"(be={be_ret:,}, ma={ma_ret:,})")

        # 안전장치: 클래스 하나라도 비면 전체 반환
        if (new_labels[keep_mask] == 0).sum() == 0 or (new_labels[keep_mask] == 1).sum() == 0:
            print('  [DensityDetector] 클래스 불균형 → 전체 데이터 사용')
            return features, given_label.copy().astype(int)
        self._keep_mask = keep_mask.copy()
        return features[keep_mask], new_labels[keep_mask]
