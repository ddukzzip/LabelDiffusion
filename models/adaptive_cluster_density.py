"""
Adaptive Cluster Density Filter v2
KMeans 기반 sub-clustering (소량 데이터 대응)

설계:
  be/ma 각각 KMeans(k=2)
  → 다수 클러스터: B_main / M_main
    → density map 구성 + noisy 제거
  → 소수 클러스터: U cluster
    → zero-day 후보 → 그대로 통과

HDBSCAN 대신 KMeans 쓰는 이유:
  N=250~1000 소량에서 HDBSCAN outlier 50%+
  KMeans는 소량에서도 안정적
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, silhouette_score
from scipy.stats import gaussian_kde


class AdaptiveClusterDensity:

    def __init__(self,
                 max_k=3,
                 minor_thresh=0.15,
                 pure_thresh=0.80,
                 cons_thresh=0.7,
                 k_knn=15,
                 top_remove_pct=20.0,
                 top_flip_pct=10.0,
                 seed=42):
        """
        Parameters
        ----------
        max_k : int
            be/ma 각각 최대 클러스터 수
        minor_thresh : float
            전체 N 대비 이 비율 미만 클러스터 = U
        pure_thresh : float
            be/ma 순도 임계값
        """
        self.max_k          = max_k
        self.minor_thresh   = minor_thresh
        self.pure_thresh    = pure_thresh
        self.cons_thresh    = cons_thresh
        self.k_knn          = k_knn
        self.top_remove_pct = top_remove_pct
        self.top_flip_pct   = top_flip_pct
        self.seed           = seed
        self._keep_mask     = None

    def _best_k(self, X, max_k):
        """silhouette score 기반 최적 k 선택"""
        if len(X) < 10:
            return 1
        best_k   = 1
        best_sil = -1
        for k in range(2, min(max_k+1, len(X)//5+1)):
            try:
                km  = KMeans(n_clusters=k,
                             random_state=self.seed,
                             n_init=10)
                lab = km.fit_predict(X)
                sil = silhouette_score(X, lab)
                if sil > best_sil:
                    best_sil = sil
                    best_k   = k
            except:
                pass
        return best_k

    def _subcluster(self, X, y, cls, N_total):
        """
        단일 클래스(be or ma)를 sub-clustering
        반환: main_mask, minor_mask (U 후보)
        """
        idx = np.where(y == cls)[0]
        if len(idx) < 10:
            return idx, np.array([], dtype=int)

        X_cls = X[idx]
        k     = self._best_k(X_cls, self.max_k)

        if k == 1:
            return idx, np.array([], dtype=int)

        km   = KMeans(n_clusters=k,
                      random_state=self.seed,
                      n_init=10)
        labs = km.fit_predict(X_cls)

        main_idx  = []
        minor_idx = []
        thresh    = N_total * self.minor_thresh

        for c in range(k):
            c_idx = idx[labs == c]
            if len(c_idx) >= thresh:
                main_idx.extend(c_idx.tolist())
            else:
                minor_idx.extend(c_idx.tolist())
                print("  [ACD] U cluster: cls={} "
                      "sub={} n={} (zero-day 후보)".format(
                      cls, c, len(c_idx)))

        return (np.array(main_idx, dtype=int),
                np.array(minor_idx, dtype=int))

    def _knn_anchor(self, X, y):
        k   = min(self.k_knn, len(X)-1)
        knn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)
        knn.fit(X)
        _, idx = knn.kneighbors(X)
        idx    = idx[:, 1:]
        cons   = np.array([
            (y[idx[i]] == y[i]).mean()
            for i in range(len(X))])
        return (cons >= self.cons_thresh) & (y==0), \
               (cons >= self.cons_thresh) & (y==1)

    def _density_score(self, X, anchor_be, anchor_ma, y):
        if anchor_be.sum() < 3 or anchor_ma.sum() < 3:
            return np.zeros(len(X), dtype=np.float32)
        ratios = np.zeros(len(X), dtype=np.float32)
        cnt    = 0
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
            ratios - (1 - y.astype(float))
        ).astype(np.float32)

    def fit(self, X, y_noisy, y_true=None):
        N = len(X)
        print("  [ACD] sub-clustering N={:,}...".format(N))

        # be/ma 각각 sub-clustering
        be_main, be_minor = self._subcluster(
            X, y_noisy, 0, N)
        ma_main, ma_minor = self._subcluster(
            X, y_noisy, 1, N)

        # main mask (density map 구성용)
        main_mask = np.zeros(N, dtype=bool)
        main_mask[be_main] = True
        main_mask[ma_main] = True

        # U mask (zero-day 후보, 그대로 통과)
        U_mask = np.zeros(N, dtype=bool)
        if len(be_minor) > 0:
            U_mask[be_minor] = True
        if len(ma_minor) > 0:
            U_mask[ma_minor] = True

        n_main = main_mask.sum()
        n_U    = U_mask.sum()
        print("  main={:,}  U={:,}  "
              "(be_main={:,}  ma_main={:,})".format(
              n_main, n_U,
              len(be_main), len(ma_main)))

        # density noisy score (main만)
        noisy_score = np.zeros(N, dtype=np.float32)
        if n_main > 20 and \
           (y_noisy[main_mask]==0).sum() > 5 and \
           (y_noisy[main_mask]==1).sum() > 5:
            X_m  = X[main_mask]
            y_m  = y_noisy[main_mask]
            ab, am = self._knn_anchor(X_m, y_m)
            sc_m   = self._density_score(X_m, ab, am, y_m)
            noisy_score[main_mask] = sc_m

        self.noisy_score = noisy_score
        self.U_mask_     = U_mask
        self.main_mask_  = main_mask

        # DetAUC
        det_auc = 0.0
        if y_true is not None:
            nm = (y_noisy != y_true).astype(int)
            if nm.sum() > 0 and nm.sum() < N:
                try:
                    det_auc = roc_auc_score(nm, noisy_score)
                except:
                    pass
        self.det_auc_ = det_auc
        print("  [ACD] DetAUC={:.4f}  "
              "U후보={:,}개".format(det_auc, n_U))
        return self

    def clean(self, X, y_noisy):
        scores     = self.noisy_score
        new_labels = y_noisy.copy().astype(int)
        keep       = np.ones(len(scores), dtype=bool)

        # main cluster에서만 제거/flip
        for cls in [0, 1]:
            idx = np.where(
                (y_noisy==cls) & self.main_mask_)[0]
            if len(idx) == 0:
                continue
            s = scores[idx]
            if self.top_flip_pct > 0 and len(idx) > 10:
                ft = np.percentile(
                    s, 100 - self.top_flip_pct)
                new_labels[idx[s >= ft]] ^= 1
            if self.top_remove_pct > 0:
                rt = np.percentile(
                    s, 100 - self.top_remove_pct)
                keep[idx[s >= rt]] = False

        # U cluster 항상 통과
        keep[self.U_mask_] = True

        if (new_labels[keep]==0).sum() == 0 or \
           (new_labels[keep]==1).sum() == 0:
            print("  [ACD] 한쪽 클래스 없음 → 전체 반환")
            self._keep_mask = np.ones(len(scores), dtype=bool)
            return X, y_noisy.copy().astype(int)

        self._keep_mask = keep
        print("  [ACD] remove={:,}  retained={:,}  "
              "U통과={:,}".format(
              (~keep).sum(), keep.sum(),
              self.U_mask_.sum()))
        return X[keep], new_labels[keep]

    def get_unknown(self, X, y_noisy):
        """zero-day 후보 반환"""
        return X[self.U_mask_], y_noisy[self.U_mask_]
