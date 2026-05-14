"""
Enhanced GMM Filter
기존 GMM 강화 - 여러 신호 결합으로 안정화

기존 GMM 문제:
  XGBoost loss 하나만 사용
  → 소량에서 bimodal 불명확
  → seed마다 다른 결과

해결:
  신호 1: XGBoost loss (기존)
  신호 2: KNN inconsistency (1 - consistency)
  신호 3: density ratio distance
  → 세 신호 표준화 후 가중 합산
  → GMM에 더 명확한 bimodal 입력
  → 더 안정적인 clean/noisy 분리
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde
from xgboost import XGBClassifier


class EnhancedGMMFilter:
    """
    강화된 GMM 필터
    GMMFiltering과 동일한 인터페이스
    """

    def __init__(self,
                 n_estimators=100,
                 cons_thresh=0.7,
                 k=20,
                 w_loss=1.0,
                 w_knn=1.0,
                 w_density=0.5,
                 top_remove_pct=20.0,
                 seed=42):
        self.n_estimators   = n_estimators
        self.cons_thresh    = cons_thresh
        self.k              = k
        self.w_loss         = w_loss
        self.w_knn          = w_knn
        self.w_density      = w_density
        self.top_remove_pct = top_remove_pct
        self.seed           = seed
        self._keep_mask     = None

    def _xgb_loss(self, X, y_noisy):
        """신호 1: XGBoost loss"""
        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.seed,
            n_jobs=-1, verbosity=0)
        xgb.fit(X, y_noisy)
        prob = xgb.predict_proba(X)[:, 1]
        loss = -(y_noisy * np.log(prob + 1e-8) +
                 (1-y_noisy) * np.log(1-prob + 1e-8))
        return loss.astype(np.float32)

    def _knn_inconsistency(self, X, y_noisy):
        """신호 2: KNN inconsistency (1 - consistency)"""
        k = min(self.k, len(X)-1)
        knn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)
        knn.fit(X)
        _, idx = knn.kneighbors(X)
        idx = idx[:, 1:]
        consistency = np.array([
            (y_noisy[idx[i]] == y_noisy[i]).mean()
            for i in range(len(X))])
        return (1 - consistency).astype(np.float32)

    def _density_distance(self, X, y_noisy):
        """
        신호 3: per-feature KDE density ratio distance
        given label과 density ratio 불일치 정도
        """
        ratios = np.zeros(len(X), dtype=np.float32)
        cnt = 0
        be_idx = np.where(y_noisy==0)[0]
        ma_idx = np.where(y_noisy==1)[0]
        if len(be_idx) < 5 or len(ma_idx) < 5:
            return ratios
        for f in range(X.shape[1]):
            try:
                b = X[be_idx, f]; m = X[ma_idx, f]
                if b.std() < 1e-6 or m.std() < 1e-6:
                    continue
                kde_b = gaussian_kde(b)
                kde_m = gaussian_kde(m)
                bd = kde_b(X[:, f])
                md = kde_m(X[:, f])
                ratios += bd / (bd + md + 1e-8)
                cnt += 1
            except:
                continue
        if cnt > 0:
            ratios /= cnt
        # given label과 불일치 정도
        dist = np.abs(ratios - (1 - y_noisy.astype(float)))
        return dist.astype(np.float32)

    def _normalize(self, x):
        """0~1 정규화"""
        mn = x.min(); mx = x.max()
        if mx - mn < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def _gmm_split(self, score, cls_indices):
        """per-class GMM으로 clean/noisy 분리"""
        keep = np.ones(len(score), dtype=bool)
        for cls in [0, 1]:
            idx = cls_indices[cls]
            if len(idx) < 10:
                continue
            s = score[idx].reshape(-1, 1)
            try:
                gmm = GaussianMixture(
                    n_components=2,
                    random_state=self.seed,
                    n_init=3)
                gmm.fit(s)
                clean_comp = np.argmin(gmm.means_.flatten())
                probs = gmm.predict_proba(s)[:, clean_comp]
                keep[idx[probs < 0.5]] = False
            except:
                pass
        return keep

    def fit_clean(self, X, y_noisy, y_true=None):
        """
        반환: X_clean, y_clean, det_auc
        GMMFiltering과 동일한 인터페이스
        """
        cls_indices = {
            0: np.where(y_noisy==0)[0],
            1: np.where(y_noisy==1)[0]
        }

        # 신호 1: XGBoost loss
        print("  [EnhGMM] XGBoost loss 계산...")
        s1 = self._normalize(self._xgb_loss(X, y_noisy))

        # 신호 2: KNN inconsistency
        print("  [EnhGMM] KNN inconsistency 계산...")
        s2 = self._normalize(
            self._knn_inconsistency(X, y_noisy))

        # 신호 3: density distance
        print("  [EnhGMM] density distance 계산...")
        s3 = self._normalize(
            self._density_distance(X, y_noisy))

        # 가중 합산
        combined = (self.w_loss * s1 +
                    self.w_knn  * s2 +
                    self.w_density * s3)
        combined = self._normalize(combined)

        print(f"  [EnhGMM] signal stats: "
              f"loss={s1.mean():.3f}±{s1.std():.3f}  "
              f"knn={s2.mean():.3f}±{s2.std():.3f}  "
              f"den={s3.mean():.3f}±{s3.std():.3f}")

        # GMM으로 clean/noisy 분리
        keep = self._gmm_split(combined, cls_indices)

        new_labels = y_noisy.copy()

        # DetAUC
        det_auc = 0.0
        if y_true is not None:
            nm = (y_noisy != y_true).astype(int)
            if nm.sum() > 0 and nm.sum() < len(nm):
                try:
                    det_auc = roc_auc_score(nm, combined)
                except:
                    pass

        if (new_labels[keep]==0).sum() == 0 or \
           (new_labels[keep]==1).sum() == 0:
            print("  한쪽 클래스 없음 → 전체 반환")
            self._keep_mask = np.ones(len(X), dtype=bool)
            return X, y_noisy.copy(), det_auc

        self._keep_mask = keep
        print(f"  [EnhGMM] DetAUC={det_auc:.4f}  "
              f"remove={(~keep).sum():,}  "
              f"retained={keep.sum():,}  "
              f"(be={(new_labels[keep]==0).sum():,}  "
              f"ma={(new_labels[keep]==1).sum():,})")

        self.noisy_score = combined
        return X[keep], new_labels[keep], det_auc
