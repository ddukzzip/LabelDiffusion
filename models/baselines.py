"""
비교 방법 구현:
  1. LossFiltering  : XGBoost loss 기반 noisy 제거
  2. CoTeaching     : 두 모델 교차 학습 (XGBoost 적용)
  3. GMMFiltering   : GMM으로 clean/noisy 분리
"""

import numpy as np
import xgboost as xgb
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42


def _make_xgb(n_estimators=200):
    params = dict(n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", random_state=SEED,
                  verbosity=0, use_label_encoder=False)
    try:
        return xgb.XGBClassifier(device="cuda", tree_method="hist", **params)
    except Exception:
        return xgb.XGBClassifier(tree_method="hist", **params)


# ── 1. Loss Filtering ────────────────────────────────────────────────
class LossFiltering:
    """
    XGBoost 예측 확신도 기반 noisy 제거.
    확신도 = |P(y=1) - 0.5| × 2  (0=불확실, 1=확실)
    확신도 낮은 상위 top_remove_pct% 제거
    """

    def __init__(self, top_remove_pct=30.0, n_estimators=100):
        self.top_remove_pct = top_remove_pct
        self.n_estimators   = n_estimators

    def fit_clean(self, X, y_noisy, y_true=None):
        clf = _make_xgb(self.n_estimators)
        clf.fit(X, y_noisy)

        prob = clf.predict_proba(X)[:, 1]
        confidence = np.abs(prob - 0.5) * 2   # [0,1]: 0=불확실

        # noisy_score = 1 - confidence (불확실할수록 높음)
        noisy_score = 1.0 - confidence

        # 클래스별 독립 percentile
        keep = np.ones(len(X), dtype=bool)
        new_labels = y_noisy.copy()
        for cls in [0, 1]:
            idx = np.where(y_noisy == cls)[0]
            if len(idx) == 0:
                continue
            s = noisy_score[idx]
            thresh = np.percentile(s, 100 - self.top_remove_pct)
            keep[idx[s >= thresh]] = False

        det_auc = 0.0
        if y_true is not None:
            noise_mask = (y_noisy != y_true).astype(int)
            if noise_mask.sum() > 0:
                det_auc = roc_auc_score(noise_mask, noisy_score)

        print(f"  [LossFilter] remove={(~keep).sum():,}  "
              f"retained={keep.sum():,}  "
              f"(be={(new_labels[keep]==0).sum():,}, "
              f"ma={(new_labels[keep]==1).sum():,})  "
              f"DetAUC={det_auc:.4f}")

        self._keep_mask = keep.copy()
        return X[keep], new_labels[keep], det_auc


# ── 2. Co-teaching ───────────────────────────────────────────────────
class CoTeaching:
    """
    XGBoost 기반 Co-teaching.
    두 모델이 서로 clean 샘플 선택해서 교차 학습.

    논문 원본(Han et al. 2018) 핵심:
      - 각 모델이 loss 작은 R(t)% 샘플을 상대방에게 제공
      - R(t) = 1 - min(t*tk, tk_max) * noise_rate  (점진적 증가)

    XGBoost 적용:
      - loss = |P(y=1|x) - y|  (예측 오차)
      - 여러 라운드 반복
    """

    def __init__(self, noise_rate=0.30, n_rounds=5,
                 keep_pct_start=70.0, keep_pct_end=85.0,
                 n_estimators=100):
        self.noise_rate      = noise_rate
        self.n_rounds        = n_rounds
        self.keep_pct_start  = keep_pct_start
        self.keep_pct_end    = keep_pct_end
        self.n_estimators    = n_estimators

    def fit_clean(self, X, y_noisy, y_true=None):
        N = len(X)
        rng = np.random.default_rng(SEED)

        # 두 subset으로 분할
        idx_all = np.arange(N)
        rng.shuffle(idx_all)
        half = N // 2
        idx1, idx2 = idx_all[:half], idx_all[half:]

        keep1 = np.ones(N, dtype=bool)
        keep2 = np.ones(N, dtype=bool)

        for r in range(self.n_rounds):
            # keep_pct 점진적 증가
            t = r / max(self.n_rounds - 1, 1)
            keep_pct = self.keep_pct_start + t * (self.keep_pct_end - self.keep_pct_start)

            # Model1: subset1의 clean으로 학습 → subset2 loss 계산
            s1_clean = idx1[keep1[idx1]]
            if len(np.unique(y_noisy[s1_clean])) < 2:
                continue
            clf1 = _make_xgb(self.n_estimators)
            clf1.fit(X[s1_clean], y_noisy[s1_clean])
            prob1 = clf1.predict_proba(X[idx2])[:, 1]
            loss1 = np.abs(prob1 - y_noisy[idx2].astype(float))

            # Model2: subset2의 clean으로 학습 → subset1 loss 계산
            s2_clean = idx2[keep2[idx2]]
            if len(np.unique(y_noisy[s2_clean])) < 2:
                continue
            clf2 = _make_xgb(self.n_estimators)
            clf2.fit(X[s2_clean], y_noisy[s2_clean])
            prob2 = clf2.predict_proba(X[idx1])[:, 1]
            loss2 = np.abs(prob2 - y_noisy[idx1].astype(float))

            # 각 subset에서 loss 낮은 keep_pct% 선택
            thresh1 = np.percentile(loss1, keep_pct)
            thresh2 = np.percentile(loss2, keep_pct)

            keep1[:] = False
            keep1[idx1[loss2 <= thresh2]] = True
            keep2[:] = False
            keep2[idx2[loss1 <= thresh1]] = True

            print(f"    [CoTeach r{r+1}] keep={keep1.sum()+keep2.sum():,}  "
                  f"keep_pct={keep_pct:.0f}%")

        # 최종 clean 샘플
        keep_final = keep1 | keep2
        # 클래스 균형 확인
        be_cnt = (y_noisy[keep_final] == 0).sum()
        ma_cnt = (y_noisy[keep_final] == 1).sum()
        if be_cnt == 0 or ma_cnt == 0:
            print("  [CoTeach] 클래스 불균형 → 전체 사용")
            keep_final = np.ones(N, dtype=bool)

        noisy_score = (~keep_final).astype(float)
        det_auc = 0.0
        if y_true is not None:
            noise_mask = (y_noisy != y_true).astype(int)
            if noise_mask.sum() > 0 and noisy_score.sum() > 0:
                det_auc = roc_auc_score(noise_mask, noisy_score)

        print(f"  [CoTeach] retained={keep_final.sum():,}  "
              f"(be={(y_noisy[keep_final]==0).sum():,}, "
              f"ma={(y_noisy[keep_final]==1).sum():,})  "
              f"DetAUC={det_auc:.4f}")

        self._keep_mask = keep_final.copy()
        return X[keep_final], y_noisy[keep_final], det_auc


# ── 3. GMM Filtering ─────────────────────────────────────────────────
class GMMFiltering:
    """
    DivideMix 단순화 버전.
    각 클래스 내에서 GMM(2성분)으로 clean/noisy 분리.
    loss 분포를 GMM으로 모델링:
      낮은 loss 성분 = clean
      높은 loss 성분 = noisy
    """

    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators

    def fit_clean(self, X, y_noisy, y_true=None):
        clf = _make_xgb(self.n_estimators)
        clf.fit(X, y_noisy)

        prob = clf.predict_proba(X)[:, 1]
        loss = np.abs(prob - y_noisy.astype(float))

        keep = np.ones(len(X), dtype=bool)
        for cls in [0, 1]:
            idx = np.where(y_noisy == cls)[0]
            if len(idx) < 10:
                continue
            l = loss[idx].reshape(-1, 1)
            try:
                gmm = GaussianMixture(n_components=2, random_state=SEED)
                gmm.fit(l)
                # 낮은 loss 성분 = clean
                means = gmm.means_.flatten()
                clean_comp = np.argmin(means)
                probs = gmm.predict_proba(l)[:, clean_comp]
                # clean 확률 0.5 이상 = keep
                keep[idx[probs < 0.5]] = False
            except Exception:
                pass

        noisy_score = loss.copy()
        det_auc = 0.0
        if y_true is not None:
            noise_mask = (y_noisy != y_true).astype(int)
            if noise_mask.sum() > 0:
                det_auc = roc_auc_score(noise_mask, noisy_score)

        new_labels = y_noisy.copy()
        self._keep_mask = keep.copy()
        print(f"  [GMMFilter] remove={(~keep).sum():,}  "
              f"retained={keep.sum():,}  "
              f"(be={(new_labels[keep]==0).sum():,}, "
              f"ma={(new_labels[keep]==1).sum():,})  "
              f"DetAUC={det_auc:.4f}")

        self._keep_mask = keep.copy()
        return X[keep], new_labels[keep], det_auc
