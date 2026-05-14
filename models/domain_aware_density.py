"""
Domain-Aware Density Map
자동 도메인 분리 + 도메인별 density map

문제:
  Mixed train에서 density map이 혼재
  → DoH benign + Gotham benign → bimodal
  → density map 희석

해결:
  Step 1: train 데이터 자동 클러스터링 (KMeans)
  Step 2: 클러스터별 독립 density map 구성
  Step 3: 테스트 샘플 → 가장 가까운 클러스터
          → 해당 density map으로 noisy score
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde


class DomainAwareDensityDetector:
    """
    자동 도메인 분리 기반 density noisy label 탐지

    사용법:
        detector = DomainAwareDensityDetector(n_domains=2)
        detector.fit(X, y_noisy)
        noisy_score = detector.noisy_score
        X_clean, y_clean = detector.clean(X, y_noisy)
    """

    def __init__(self,
                 n_domains=2,
                 n_bins=50,
                 cons_thresh=0.7,
                 k=20,
                 top_remove_pct=20.0,
                 top_flip_pct=10.0,
                 seed=42):
        self.n_domains      = n_domains
        self.n_bins         = n_bins
        self.cons_thresh    = cons_thresh
        self.k              = k
        self.top_remove_pct = top_remove_pct
        self.top_flip_pct   = top_flip_pct
        self.seed           = seed
        np.random.seed(seed)

    def _knn_anchor(self, X, y_noisy):
        k = min(self.k, len(X)-1)
        knn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)
        knn.fit(X)
        _, idx = knn.kneighbors(X)
        idx = idx[:, 1:]
        anchor = np.array([
            (y_noisy[idx[i]] == y_noisy[i]).mean()
            >= self.cons_thresh
            for i in range(len(X))])
        return anchor & (y_noisy==0), anchor & (y_noisy==1)

    def _build_density_score(self, X, anchor_be, anchor_ma, y_noisy):
        """per-feature KDE density ratio → noisy score"""
        if anchor_be.sum() < 3 or anchor_ma.sum() < 3:
            return np.zeros(len(X), dtype=np.float32)

        ratios = np.zeros(len(X), dtype=np.float32)
        cnt = 0
        for f in range(X.shape[1]):
            try:
                b = X[anchor_be, f]
                m = X[anchor_ma, f]
                if b.std() < 1e-6 or m.std() < 1e-6:
                    continue
                kbd = gaussian_kde(b)
                kmd = gaussian_kde(m)
                bd  = kbd(X[:, f])
                md  = kmd(X[:, f])
                ratios += bd / (bd + md + 1e-8)
                cnt += 1
            except:
                continue
        if cnt > 0:
            ratios /= cnt
        return np.abs(
            ratios - (1 - y_noisy.astype(float))
        ).astype(np.float32)

    def fit(self, X, y_noisy, y_true=None):
        """
        X: (N, F) feature matrix
        y_noisy: noisy labels
        """
        N = len(X)

        # Step 1: 자동 도메인 클러스터링
        print("  [DomainAware] 도메인 클러스터링...")
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        self.scaler_ = sc

        # 클러스터 수 자동 결정
        # 데이터가 적으면 클러스터 줄이기
        n_dom = min(self.n_domains, N // 20)
        n_dom = max(1, n_dom)

        if n_dom == 1:
            # 단일 도메인 → 기존 방법과 동일
            domain_labels = np.zeros(N, dtype=int)
        else:
            km = KMeans(n_clusters=n_dom,
                        random_state=self.seed,
                        n_init=10)
            domain_labels = km.fit_predict(X_scaled)
            self.kmeans_ = km

        domains, counts = np.unique(
            domain_labels, return_counts=True)
        print(f"  도메인 수: {n_dom}")
        for d, c in zip(domains, counts):
            be_c = ((domain_labels==d) & (y_noisy==0)).sum()
            ma_c = ((domain_labels==d) & (y_noisy==1)).sum()
            print(f"    domain {d}: {c:,}개 "
                  f"(be={be_c:,}, ma={ma_c:,})")

        # Step 2: 도메인별 density map + noisy score
        print("  [DomainAware] 도메인별 density score 계산...")
        combined_score = np.zeros(N, dtype=np.float32)
        domain_scores  = {}

        for d in domains:
            d_mask = (domain_labels == d)
            X_d    = X[d_mask]
            y_d    = y_noisy[d_mask]

            if (y_d==0).sum() < 5 or (y_d==1).sum() < 5:
                print(f"    domain {d}: 샘플 부족 → skip")
                continue

            # KNN anchor (도메인 내)
            anchor_be, anchor_ma = self._knn_anchor(X_d, y_d)

            # density score (도메인 내)
            score_d = self._build_density_score(
                X_d, anchor_be, anchor_ma, y_d)

            combined_score[d_mask] = score_d
            domain_scores[d] = {
                'anchor_be': anchor_be,
                'anchor_ma': anchor_ma,
                'X_d': X_d,
                'y_d': y_d,
                'score': score_d
            }

        self.domain_labels_ = domain_labels
        self.domain_scores_  = domain_scores
        self.noisy_score     = combined_score

        # DetAUC
        det_auc = 0.0
        if y_true is not None:
            nm = (y_noisy != y_true).astype(int)
            if nm.sum() > 0 and nm.sum() < N:
                try:
                    det_auc = roc_auc_score(nm, combined_score)
                except:
                    pass
        print(f"  [DomainAware] DetAUC={det_auc:.4f}  "
              f"score mean={combined_score.mean():.4f}")
        return self

    def predict_domain(self, X_test):
        """테스트 샘플 → 도메인 할당"""
        if not hasattr(self, 'kmeans_'):
            return np.zeros(len(X_test), dtype=int)
        X_scaled = self.scaler_.transform(X_test)
        return self.kmeans_.predict(X_scaled)

    def clean(self, X, y_noisy):
        scores     = self.noisy_score
        new_labels = y_noisy.copy().astype(int)
        keep_mask  = np.ones(len(scores), dtype=bool)

        for cls in [0, 1]:
            idx = np.where(y_noisy==cls)[0]
            if len(idx) == 0:
                continue
            s = scores[idx]
            if self.top_flip_pct > 0:
                ft = np.percentile(s, 100-self.top_flip_pct)
                new_labels[idx[s>=ft]] ^= 1
            if self.top_remove_pct > 0:
                rt = np.percentile(s, 100-self.top_remove_pct)
                keep_mask[idx[s>=rt]] = False

        if (new_labels[keep_mask]==0).sum()==0 or \
           (new_labels[keep_mask]==1).sum()==0:
            return X, y_noisy.copy().astype(int)

        self._keep_mask = keep_mask
        print(f"  [DomainAware] remove={(~keep_mask).sum():,}  "
              f"retained={keep_mask.sum():,}")
        return X[keep_mask], new_labels[keep_mask]
