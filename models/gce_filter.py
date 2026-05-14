"""
GCE Loss Filter (Generalized Cross Entropy)
CoTeaching 교체용 최신 noisy label filter

Zhang & Sabuncu, NeurIPS 2018
"Generalized Cross Entropy Loss for Training Deep Neural
Networks with Noisy Labels"

핵심 아이디어:
  GCE loss = noise-robust loss function
  → noisy sample은 loss가 높게 유지됨
  → clean sample은 loss가 낮아짐
  → loss 기반으로 clean/noisy 분리

CoTeaching과 차이:
  CoTeaching: 두 모델 교차 학습
  GCE: 단일 모델 + robust loss
  → 더 단순하고 안정적
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


class GCEFilter:
    """
    GCE Loss 기반 clean sample 선택
    CoTeaching과 동일한 인터페이스

    GCE loss:
      L_q(f(x), y) = (1 - f_y(x)^q) / q
      q=0: cross entropy
      q=1: MAE (가장 robust)
      q=0.7: 권장값 (balance)

    구현:
      XGBoost로 pred prob 계산
      → GCE loss 계산
      → 하위 (100-noise_rate)% 선택
    """

    def __init__(self,
                 n_estimators=100,
                 q=0.7,
                 keep_pct=80.0,
                 cons_thresh=0.7,
                 k=20,
                 use_class_weight=False,
                 seed=42):
        self.n_estimators     = n_estimators
        self.q                = q
        self.keep_pct         = keep_pct
        self.cons_thresh      = cons_thresh
        self.k                = k
        self.use_class_weight = use_class_weight
        self.seed             = seed
        self._keep_mask       = None

    def _gce_loss(self, prob, y):
        """GCE loss per sample"""
        q = self.q
        # prob of true class
        p_y = np.where(y==1, prob, 1-prob)
        p_y = np.clip(p_y, 1e-7, 1.0)
        loss = (1 - p_y**q) / q
        return loss.astype(np.float32)

    def _knn_consistency(self, X, y_noisy):
        """KNN label consistency score"""
        k = min(self.k, len(X)-1)
        knn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)
        knn.fit(X)
        _, idx = knn.kneighbors(X)
        idx = idx[:, 1:]
        consistency = np.array([
            (y_noisy[idx[i]] == y_noisy[i]).mean()
            for i in range(len(X))])
        return consistency.astype(np.float32)

    def fit_clean(self, X, y_noisy, y_true=None):
        """
        반환: X_clean, y_clean, det_auc
        CoTeaching과 동일한 인터페이스
        """
        print("  [GCE] XGBoost 학습...")
        # class_weight: 불균형 데이터 보정
        spw = 1.0
        if self.use_class_weight:
            n_be = (y_noisy==0).sum()
            n_ma = (y_noisy==1).sum()
            if n_ma > 0 and n_be > 0:
                spw = float(n_be) / float(n_ma)
            print("  [GCE] scale_pos_weight={:.3f}".format(spw))
        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=spw,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=0)
        xgb.fit(X, y_noisy)
        prob = xgb.predict_proba(X)[:, 1]

        # GCE loss
        gce = self._gce_loss(prob, y_noisy)

        # KNN consistency (보조 신호)
        print("  [GCE] KNN consistency 계산...")
        cons = self._knn_consistency(X, y_noisy)
        incons = 1 - cons  # inconsistency

        # 결합 score (정규화)
        def norm(x):
            mn, mx = x.min(), x.max()
            return (x-mn)/(mx-mn+1e-8)

        score = norm(gce) + 0.5 * norm(incons)
        score = norm(score)

        # per-class top keep_pct% 선택
        keep = np.ones(len(X), dtype=bool)
        for cls in [0, 1]:
            idx = np.where(y_noisy==cls)[0]
            if len(idx) == 0:
                continue
            s = score[idx]
            thresh = np.percentile(s, self.keep_pct)
            keep[idx[s > thresh]] = False

        new_labels = y_noisy.copy()

        # DetAUC
        det_auc = 0.0
        if y_true is not None:
            nm = (y_noisy != y_true).astype(int)
            if nm.sum() > 0 and nm.sum() < len(nm):
                try:
                    det_auc = roc_auc_score(nm, score)
                except:
                    pass

        if (new_labels[keep]==0).sum()==0 or \
           (new_labels[keep]==1).sum()==0:
            print("  한쪽 클래스 없음 → 전체 반환")
            self._keep_mask = np.ones(len(X), dtype=bool)
            return X, y_noisy.copy(), det_auc

        self._keep_mask = keep
        print("  [GCE] DetAUC={:.4f}  remove={}  "
              "retained={}  (be={}  ma={})".format(
              det_auc, (~keep).sum(), keep.sum(),
              (new_labels[keep]==0).sum(),
              (new_labels[keep]==1).sum()))

        self.noisy_score = score
        return X[keep], new_labels[keep], det_auc
