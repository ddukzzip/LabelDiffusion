"""
최종 실험 파이프라인 - 3가지 비교

실험 A: Baseline vs Ours (CT→Density+Aug)
실험 B: Text(1D) vs Image(2D) 전처리
실험 C: CoTeaching 교체 → GCE filter
"""

import sys, os, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/oper/LabelDiffusion")
from models.density_map      import DensityMapBuilder
from models.density_detector import DensityNoisyDetector
from models.baselines        import CoTeaching, LossFiltering
from models.augmentation     import DataAugmentor
from models.gce_filter       import GCEFilter
from models.adaptive_cluster_density import AdaptiveClusterDensity

DOH_TRAIN    = "/home/oper/DoHBrw2020/npz/doh_train.npz"
DOH_TEST     = "/home/oper/DoHBrw2020/npz/doh_test.npz"
GOTHAM_NPZ   = "/home/oper/LabelDiffusion/data/gotham_seqs_pcap.npz"
RESULTS_DIR  = "/home/oper/LabelDiffusion/results/final"
IMG_DIR      = "/home/oper/LabelDiffusion/results/final/images"
N_SIZES      = [250, 500, 1000]
SEEDS        = [42, 0, 1]
NOISE_LEVELS = [0.00, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55]


# ── 데이터 로드 ──────────────────────────────────────────

def load_doh(seed, n_sample):
    tr  = np.load(DOH_TRAIN)
    te  = np.load(DOH_TEST)
    rng = np.random.default_rng(seed)
    def sample(d, n, rng):
        Xl = d['seqs_len'].astype(np.float32)
        Xi = d['seqs_iat'].astype(np.float32)
        Xs = d['seqs_signed'].astype(np.float32)
        y  = d['labels'].astype(int)
        be = rng.permutation(np.where(y==0)[0])
        ma = rng.permutation(np.where(y==1)[0])
        n2 = min(n//2, len(be), len(ma))
        idx = np.concatenate([be[:n2], ma[:n2]])
        rng.shuffle(idx)
        return Xl[idx], Xi[idx], Xs[idx], y[idx]
    Xl_tr,Xi_tr,Xs_tr,y_tr = sample(tr, n_sample, rng)
    Xl_te,Xi_te,Xs_te,y_te = sample(te, 4000, rng)
    return Xl_tr, Xi_tr, Xs_tr, y_tr, Xl_te, y_te


def load_gotham(seed, n_sample):
    d   = np.load(GOTHAM_NPZ, allow_pickle=True)
    Xl  = d['seqs_len'].astype(np.float32)
    Xi  = d['seqs_iat'].astype(np.float32)
    Xf  = d['seqs_flags'].astype(np.float32)
    y   = d['labels'].astype(int)
    rng = np.random.default_rng(seed)
    be  = rng.permutation(np.where(y==0)[0])
    ma  = rng.permutation(np.where(y==1)[0])
    n2  = min(n_sample//2, len(be), len(ma))
    tr  = np.concatenate([be[:n2], ma[:n2]])
    rng.shuffle(tr)
    n_te = min(2000, len(be)-n2, len(ma)-n2)
    te   = np.concatenate([be[n2:n2+n_te], ma[n2:n2+n_te]])
    return Xl[tr],Xi[tr],Xf[tr],y[tr],Xl[te],y[te]


# ── 유틸 ────────────────────────────────────────────────

def build_joint_density(Xl, Xi, y, n_bins=50):
    """IAT x LEN joint density (50x50)"""
    joint_be = np.zeros((n_bins, n_bins), dtype=np.float32)
    joint_ma = np.zeros((n_bins, n_bins), dtype=np.float32)
    len_min = float(Xl.min()); len_max = float(Xl.max()) + 1e-8
    iat_min = float(Xi.min()); iat_max = float(Xi.max()) + 1e-8
    be_idx = np.where(y==0)[0]
    ma_idx = np.where(y==1)[0]
    for f in range(min(Xl.shape[1], Xi.shape[1])):
        l_idx = np.floor(
            (Xl[:, f] - len_min) / (len_max - len_min) * n_bins
        ).astype(int).clip(0, n_bins-1)
        i_idx = np.floor(
            (Xi[:, f] - iat_min) / (iat_max - iat_min) * n_bins
        ).astype(int).clip(0, n_bins-1)
        for n in be_idx:
            joint_be[i_idx[n], l_idx[n]] += 1
        for n in ma_idx:
            joint_ma[i_idx[n], l_idx[n]] += 1
    if joint_be.sum() > 0:
        joint_be /= joint_be.sum()
    if joint_ma.sum() > 0:
        joint_ma /= joint_ma.sum()
    return joint_be, joint_ma


def add_noise(labels, p, seed):
    rng   = np.random.default_rng(seed)
    noisy = labels.copy()
    noisy[rng.random(len(labels)) < p] ^= 1
    return noisy


def make_rf(seed=42):
    return RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=seed)


def eval_f1(m, X_te, y_te):
    return float(f1_score(
        y_te, m.predict(X_te),
        average="macro", zero_division=0))


def ct_filter(X, y_noisy, p, seed):
    """CoTeaching clean sample 선택"""
    ct = CoTeaching(noise_rate=p, n_rounds=5,
                    keep_pct_start=70.0,
                    keep_pct_end=85.0,
                    n_estimators=100)
    ct.fit_clean(X, y_noisy)
    if not hasattr(ct, '_keep_mask'):
        return None, None, None
    km = ct._keep_mask
    if km.sum() < 20:
        return None, None, None
    if (y_noisy[km]==0).sum() < 5 or \
       (y_noisy[km]==1).sum() < 5:
        return None, None, None
    return X[km], y_noisy[km], km


def density_filter(X_clean, y_clean, seed):
    """Density image 기반 2차 필터"""
    db = DensityMapBuilder(
        n_bins=50, cons_thresh=0.7,
        k=min(20, len(X_clean)//4))
    db.fit(X_clean, y_clean)
    dimg = db.build()
    dd = DensityNoisyDetector(
        top_remove_pct=20.0, top_flip_pct=10.0)
    dd.fit(dimg, db.bin_edges)
    dd.detect(X_clean, y_clean)
    return dd.clean(X_clean, y_clean), dimg, db


def text_filter_1d(X_clean, y_clean):
    """Text(1D KDE) 기반 전처리 - 실험 B 비교용"""
    be_idx = np.where(y_clean==0)[0]
    ma_idx = np.where(y_clean==1)[0]
    if len(be_idx) < 5 or len(ma_idx) < 5:
        return X_clean, y_clean

    noisy_score = np.zeros(len(X_clean), dtype=np.float32)
    cnt = 0
    for f in range(X_clean.shape[1]):
        try:
            b = X_clean[be_idx, f]
            m = X_clean[ma_idx, f]
            if b.std() < 1e-6 or m.std() < 1e-6:
                continue
            kbd = gaussian_kde(b)
            kmd = gaussian_kde(m)
            bd  = kbd(X_clean[:, f])
            md  = kmd(X_clean[:, f])
            ratio = bd / (bd + md + 1e-8)
            noisy_score += np.abs(
                ratio - (1 - y_clean.astype(float)))
            cnt += 1
        except:
            continue
    if cnt > 0:
        noisy_score /= cnt

    keep = np.ones(len(X_clean), dtype=bool)
    for cls in [0, 1]:
        idx = np.where(y_clean==cls)[0]
        if len(idx) == 0:
            continue
        s = noisy_score[idx]
        thresh = np.percentile(s, 80.0)
        keep[idx[s >= thresh]] = False

    if (y_clean[keep]==0).sum() < 5 or \
       (y_clean[keep]==1).sum() < 5:
        return X_clean, y_clean
    return X_clean[keep], y_clean[keep]


def save_density_images(dimg, dataset, p, seed, N,
                        Xl=None, Xi=None, y=None):
    """채널별 density image 저장 + IAT×LEN joint"""
    os.makedirs(IMG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Benign Density', 'Malicious Density',
              'Density Ratio (Benign/Total)']
    cmaps  = ['Blues', 'Reds', 'RdBu']

    be_map    = dimg[:, :, 0]
    ma_map    = dimg[:, :, 1]
    ratio_map = be_map / (be_map + ma_map + 1e-8)
    maps = [be_map, ma_map, ratio_map]

    for i, (ax, data, title, cmap) in enumerate(
            zip(axes, maps, titles, cmaps)):
        vmin = 0; vmax = 1 if i == 2 else None
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                       origin='lower',
                       vmin=vmin, vmax=vmax)
        ax.set_title(
            "{}\n{} p={} N={}".format(
                title, dataset, p, N))
        ax.set_xlabel('Bin')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fname = "{}/{}_p{:03d}_N{}_s{}.png".format(
        IMG_DIR, dataset, int(p*100), N, seed)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()

    # joint density 이미지 저장
    if Xl is not None and Xi is not None and y is not None:
        jbe, jma = build_joint_density(Xl, Xi, y, n_bins=50)
        jratio = jbe / (jbe + jma + 1e-8)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, data, title, cmap in zip(
            axes,
            [jbe, jma, jratio],
            ['Joint Benign (IAT x LEN)',
             'Joint Malicious (IAT x LEN)',
             'Joint Ratio'],
            ['Blues', 'Reds', 'RdBu']):
            im = ax.imshow(data, aspect='auto', cmap=cmap,
                           origin='lower')
            ax.set_title("{} {} p={} N={}".format(
                title, dataset, p, N))
            ax.set_xlabel('LEN bin')
            ax.set_ylabel('IAT bin')
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fname2 = "{}/{}_joint_p{:03d}_N{}_s{}.png".format(
            IMG_DIR, dataset, int(p*100), N, seed)
        plt.savefig(fname2, dpi=100, bbox_inches='tight')
        plt.close()


# ── 핵심 실험 함수 ───────────────────────────────────────

def run_one(Xl, Xi, Xa, y_true, Xl_te, y_te,
            p, seed, dataset, N, save_img=False):
    """
    3가지 비교 실험 동시 수행

    실험 A: Baseline vs Ours
    실험 B: Text(1D) vs Image(2D)
    실험 C: GCE filter → Density+Aug
    """
    y_noisy = add_noise(y_true, p, seed)
    aug     = DataAugmentor(seed=seed)
    res     = {}

    sc    = StandardScaler()
    X_s   = sc.fit_transform(Xl).astype(np.float32)
    X_ts  = sc.transform(Xl_te).astype(np.float32)

    # ── 실험 A ───────────────────────────────────────────
    # A1: Baseline (noisy 그대로)
    clf = make_rf(seed)
    clf.fit(X_s, y_noisy)
    res["A1_Baseline"] = eval_f1(clf, X_ts, y_te)

    # A1b: Baseline + Aug
    Xba, yba = aug.augment(X_s, y_noisy)
    clf = make_rf(seed)
    clf.fit(Xba, yba)
    res["A1b_Baseline_Aug"] = eval_f1(clf, X_ts, y_te)

    # A2: Ours (CT→Density+Aug)
    X_ct, y_ct, km = ct_filter(X_s, y_noisy, p, seed)
    if X_ct is not None:
        (X_den, y_den), dimg, db = density_filter(
            X_ct, y_ct, seed)
        if save_img and seed == 42:
            # Xi_ct: CT 필터 후 seqs_iat
            Xi_ct_img = Xi[km].astype(np.float32)                 if km is not None else Xi.astype(np.float32)
            save_density_images(dimg, dataset, p, seed, N,
                                Xl=X_ct, Xi=Xi_ct_img,
                                y=y_ct)
        if len(X_den) > 10 and \
           (y_den==0).sum() > 5 and (y_den==1).sum() > 5:
            Xa2, ya2 = aug.augment(X_den, y_den)
            clf = make_rf(seed)
            clf.fit(Xa2, ya2)
            res["A2_Ours_CT_Density"] = eval_f1(
                clf, X_ts, y_te)

    # ── A3: CT→Joint(IAT×LEN)+Aug ───────────────────────
    try:
        if X_ct is not None and            len(X_ct) > 20 and            (y_ct==0).sum() > 5 and (y_ct==1).sum() > 5:
            # seqs_iat 로드 (Xi)
            # Xi는 run_one의 두 번째 인자
            # km은 boolean mask → Xi 인덱싱
            if km is not None:
                Xi_ct = Xi[km].astype(np.float32)
            else:
                Xi_ct = Xi.astype(np.float32)
            jbe, jma = build_joint_density(
                X_ct, Xi_ct, y_ct, n_bins=50)
            # joint density ratio → noisy score
            jratio = jbe / (jbe + jma + 1e-8)
            # per-sample noisy score from joint
            j_noisy = np.zeros(len(X_ct), dtype=np.float32)
            cnt_j = 0
            len_min = X_ct.min(); len_max = X_ct.max()+1e-8
            iat_min = Xi_ct.min(); iat_max = Xi_ct.max()+1e-8
            for f in range(min(X_ct.shape[1],
                               Xi_ct.shape[1])):
                l_idx = np.floor(
                    (X_ct[:,f]-len_min)/
                    (len_max-len_min)*50
                ).astype(int).clip(0,49)
                i_idx = np.floor(
                    (Xi_ct[:,f]-iat_min)/
                    (iat_max-iat_min)*50
                ).astype(int).clip(0,49)
                r = jratio[i_idx, l_idx]
                j_noisy += np.abs(
                    r - (1 - y_ct.astype(float)))
                cnt_j += 1
            if cnt_j > 0:
                j_noisy /= cnt_j
            # top 20% remove
            keep_j = np.ones(len(X_ct), dtype=bool)
            for cls in [0, 1]:
                idx_c = np.where(y_ct==cls)[0]
                if len(idx_c) == 0: continue
                s = j_noisy[idx_c]
                t = np.percentile(s, 80)
                keep_j[idx_c[s>=t]] = False
            Xj = X_ct[keep_j]; yj = y_ct[keep_j]
            if len(Xj)>10 and                (yj==0).sum()>5 and (yj==1).sum()>5:
                Xja, yja = aug.augment(Xj, yj)
                clf = make_rf(seed)
                clf.fit(Xja, yja)
                res["A3_CT_Joint"] = eval_f1(
                    clf, X_ts, y_te)
    except Exception as e:
        print("  Joint 오류: {}".format(e))

    # ── 실험 B: Text vs Image ────────────────────────────
    # B1: CT→Text(1D)+Aug
    if X_ct is not None:
        X_txt, y_txt = text_filter_1d(X_ct, y_ct)
        if len(X_txt) > 10 and \
           (y_txt==0).sum() > 5 and (y_txt==1).sum() > 5:
            Xta, yta = aug.augment(X_txt, y_txt)
            clf = make_rf(seed)
            clf.fit(Xta, yta)
            res["B1_CT_Text1D"] = eval_f1(clf, X_ts, y_te)

    # B2: CT→Image(2D)+Aug = A2와 동일
    if "A2_Ours_CT_Density" in res:
        res["B2_CT_Image2D"] = res["A2_Ours_CT_Density"]

    # ── 실험 C: GCE filter → Density+Aug ────────────────
    try:
        gce = GCEFilter(
            n_estimators=100,
            q=0.7,
            keep_pct=80.0,
            k=min(20, len(X_s)//10),
            seed=seed)
        X_gce, y_gce, _ = gce.fit_clean(
            X_s, y_noisy, y_true=y_true)
        if len(X_gce) > 20 and \
           (y_gce==0).sum() > 5 and (y_gce==1).sum() > 5:
            (X_gd, y_gd), _, _ = density_filter(
                X_gce, y_gce, seed)
            if len(X_gd) > 10 and \
               (y_gd==0).sum() > 5 and (y_gd==1).sum() > 5:
                Xga, yga = aug.augment(X_gd, y_gd)
                clf = make_rf(seed)
                clf.fit(Xga, yga)
                res["C1_GCE_Density"] = eval_f1(
                    clf, X_ts, y_te)
    except Exception as e:
        print("  GCE 오류: {}".format(e))

    # ── 실험 C2: GCE+class_weight → Image ──────────────
    try:
        gce_w = GCEFilter(
            n_estimators=100,
            q=0.7,
            keep_pct=80.0,
            k=min(20, len(X_s)//10),
            use_class_weight=True,
            seed=seed)
        X_gcew, y_gcew, _ = gce_w.fit_clean(
            X_s, y_noisy, y_true=y_true)
        if len(X_gcew) > 20 and            (y_gcew==0).sum() > 5 and (y_gcew==1).sum() > 5:
            # C2: GCE_W → Image
            (X_gwd, y_gwd), _, _ = density_filter(
                X_gcew, y_gcew, seed)
            if len(X_gwd) > 10 and                (y_gwd==0).sum() > 5 and (y_gwd==1).sum() > 5:
                Xgwa, ygwa = aug.augment(X_gwd, y_gwd)
                clf = make_rf(seed)
                clf.fit(Xgwa, ygwa)
                res["C2_GCEW_Image"] = eval_f1(clf, X_ts, y_te)
            # C3: GCE_W → Text1D
            X_gwt, y_gwt = text_filter_1d(X_gcew, y_gcew)
            if len(X_gwt) > 10 and                (y_gwt==0).sum() > 5 and (y_gwt==1).sum() > 5:
                Xgta, ygta = aug.augment(X_gwt, y_gwt)
                clf = make_rf(seed)
                clf.fit(Xgta, ygta)
                res["C3_GCEW_Text"] = eval_f1(clf, X_ts, y_te)
    except Exception as e:
        print("  GCE_W 오류: {}".format(e))

    # ── 실험 C4: GCE(no weight) → Text1D ────────────────
    try:
        if "C1_GCE_Density" in res:
            X_gce2, y_gce2, _ = GCEFilter(
                n_estimators=100, q=0.7, keep_pct=80.0,
                k=min(20, len(X_s)//10),
                use_class_weight=False,
                seed=seed).fit_clean(X_s, y_noisy,
                                     y_true=y_true)
            if len(X_gce2) > 20 and                (y_gce2==0).sum() > 5 and (y_gce2==1).sum() > 5:
                X_gt2, y_gt2 = text_filter_1d(X_gce2, y_gce2)
                if len(X_gt2) > 10 and                    (y_gt2==0).sum() > 5 and                    (y_gt2==1).sum() > 5:
                    Xgt2a, ygt2a = aug.augment(X_gt2, y_gt2)
                    clf = make_rf(seed)
                    clf.fit(Xgt2a, ygt2a)
                    res["C4_GCE_Text"] = eval_f1(
                        clf, X_ts, y_te)
    except Exception as e:
        print("  GCE_Text 오류: {}".format(e))

    # ── 실험 D: ACD (Adaptive Cluster Density) ──────────
    # D1: CT → ACD+Aug
    try:
        X_ct2, y_ct2, km2 = ct_filter(X_s, y_noisy, p, seed)
        if X_ct2 is not None:
            acd = AdaptiveClusterDensity(
                max_k=3,
                minor_thresh=0.15,
                pure_thresh=0.80,
                cons_thresh=0.7,
                k_knn=min(15, len(X_ct2)//10),
                top_remove_pct=20.0,
                top_flip_pct=10.0,
                seed=seed)
            acd.fit(X_ct2, y_ct2, y_true=y_true[km2])
            X_acd, y_acd = acd.clean(X_ct2, y_ct2)
            if len(X_acd) > 10 and                (y_acd==0).sum() > 5 and (y_acd==1).sum() > 5:
                Xaa, yaa = aug.augment(X_acd, y_acd)
                clf = make_rf(seed)
                clf.fit(Xaa, yaa)
                res["D1_CT_ACD"] = eval_f1(clf, X_ts, y_te)

                # U cluster 크기 기록
                n_u = acd.U_mask_.sum()
                if n_u > 0:
                    print("  [ACD] zero-day 후보 "
                          "U cluster: {:,}개".format(n_u))
    except Exception as e:
        print("  ACD 오류: {}".format(e))

    # D3: CT → ACD + Joint density
    try:
        X_ct3, y_ct3, km3 = ct_filter(X_s, y_noisy, p, seed)
        if X_ct3 is not None:
            acd3 = AdaptiveClusterDensity(
                max_k=3,
                minor_thresh=0.15,
                pure_thresh=0.80,
                cons_thresh=0.7,
                k_knn=min(15, len(X_ct3)//10),
                top_remove_pct=20.0,
                top_flip_pct=10.0,
                seed=seed)
            acd3.fit(X_ct3, y_ct3, y_true=y_true[km3])

            # B/M main에서 Joint density noisy score
            main_mask = acd3.main_mask_
            U_mask    = acd3.U_mask_
            X_main    = X_ct3[main_mask]
            y_main    = y_ct3[main_mask]
            Xi_ct3    = Xi[km3].astype(np.float32)
            Xi_main   = Xi_ct3[main_mask]

            if len(X_main) > 20 and                (y_main==0).sum() > 5 and                (y_main==1).sum() > 5:
                # joint density
                jbe3, jma3 = build_joint_density(
                    X_main, Xi_main, y_main, n_bins=50)
                jratio3 = jbe3 / (jbe3 + jma3 + 1e-8)

                # per-sample noisy score
                j3_score = np.zeros(
                    len(X_main), dtype=np.float32)
                len_min = X_main.min()
                len_max = X_main.max() + 1e-8
                iat_min = Xi_main.min()
                iat_max = Xi_main.max() + 1e-8
                cnt3 = 0
                for f in range(min(X_main.shape[1],
                                   Xi_main.shape[1])):
                    l_idx = np.floor(
                        (X_main[:,f]-len_min)/
                        (len_max-len_min)*50
                    ).astype(int).clip(0,49)
                    i_idx = np.floor(
                        (Xi_main[:,f]-iat_min)/
                        (iat_max-iat_min)*50
                    ).astype(int).clip(0,49)
                    r = jratio3[i_idx, l_idx]
                    j3_score += np.abs(
                        r - (1-y_main.astype(float)))
                    cnt3 += 1
                if cnt3 > 0:
                    j3_score /= cnt3

                # top 20% remove from main
                keep3 = np.ones(len(X_main), dtype=bool)
                for cls in [0, 1]:
                    idx_c = np.where(y_main==cls)[0]
                    if len(idx_c) == 0:
                        continue
                    s = j3_score[idx_c]
                    t = np.percentile(s, 80)
                    keep3[idx_c[s >= t]] = False

                # U cluster 합류
                X_main_clean = X_main[keep3]
                y_main_clean = y_main[keep3]
                X_u = X_ct3[U_mask]
                y_u = y_ct3[U_mask]

                X_d3 = np.vstack([X_main_clean, X_u])                     if len(X_u) > 0 else X_main_clean
                y_d3 = np.concatenate([
                    y_main_clean, y_u])                     if len(X_u) > 0 else y_main_clean

                if len(X_d3) > 10 and                    (y_d3==0).sum() > 5 and                    (y_d3==1).sum() > 5:
                    Xd3a, yd3a = aug.augment(X_d3, y_d3)
                    clf = make_rf(seed)
                    clf.fit(Xd3a, yd3a)
                    res["D3_CT_ACD_Joint"] = eval_f1(
                        clf, X_ts, y_te)
    except Exception as e:
        print("  D3 오류: {}".format(e))

    # D2: GCE → ACD+Aug
    try:
        gce_acd = GCEFilter(
            n_estimators=100, q=0.7, keep_pct=80.0,
            k=min(20, len(X_s)//10),
            use_class_weight=True, seed=seed)
        X_ga, y_ga, _ = gce_acd.fit_clean(
            X_s, y_noisy, y_true=y_true)
        if len(X_ga) > 20 and            (y_ga==0).sum() > 5 and (y_ga==1).sum() > 5:
            acd2 = AdaptiveClusterDensity(
                max_k=3,
                minor_thresh=0.15,
                pure_thresh=0.80,
                cons_thresh=0.7,
                k_knn=min(15, len(X_ga)//10),
                top_remove_pct=20.0,
                top_flip_pct=10.0,
                seed=seed)
            acd2.fit(X_ga, y_ga, y_true=y_true)
            X_acd2, y_acd2 = acd2.clean(X_ga, y_ga)
            if len(X_acd2) > 10 and                (y_acd2==0).sum() > 5 and                (y_acd2==1).sum() > 5:
                Xaa2, yaa2 = aug.augment(X_acd2, y_acd2)
                clf = make_rf(seed)
                clf.fit(Xaa2, yaa2)
                res["D2_GCEW_ACD"] = eval_f1(
                    clf, X_ts, y_te)
    except Exception as e:
        print("  ACD2 오류: {}".format(e))

    for k, v in res.items():
        print("    {:<28}: {:.4f}".format(k, v))
    return res


# ── 실험 실행 ─────────────────────────────────────────────

def run_experiment(name, load_fn):
    methods = ["A1_Baseline",
               "A1b_Baseline_Aug",
               "A2_Ours_CT_Density",
               "A3_CT_Joint",
               "B1_CT_Text1D",
               "B2_CT_Image2D",
               "C1_GCE_Density",
               "C2_GCEW_Image",
               "C3_GCEW_Text",
               "C4_GCE_Text",
               "D1_CT_ACD",
               "D2_GCEW_ACD",
               "D3_CT_ACD_Joint"]
    all_res = {}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for N in N_SIZES:
        n_key = "N{}".format(N)
        all_res[n_key] = {}
        print()
        print("#"*55)
        print("  {} N={}".format(name, N))
        print("#"*55)

        for p in NOISE_LEVELS:
            p_key    = "p{:03d}".format(int(p*100))
            seed_res = {m: [] for m in methods}
            save_img = (N == 1000)

            for seed in SEEDS:
                print("  -- p={}  seed={} --".format(p, seed))
                data = load_fn(seed, N)
                res  = run_one(*data, p, seed, name, N,
                               save_img=save_img)
                for m, v in res.items():
                    if m in seed_res:
                        seed_res[m].append(v)

            all_res[n_key][p_key] = {
                m: {"mean": float(np.mean(v)),
                    "std":  float(np.std(v))}
                for m, v in seed_res.items() if v}

    # 결과 출력
    print()
    print("="*70)
    print("  {} 결과 (macro F1, N=1k)".format(name))
    print("="*70)

    # 실험 A: N별 출력
    for N in N_SIZES:
        n_key = "N{}".format(N)
        print()
        print("  [실험 A] Baseline vs Ours  (N={})".format(N))
        print("  {:<28} {:>7}  {:>7}  {:>7}  {:>7}".format(
            "방법", "p=0.20", "p=0.30", "p=0.40", "p=0.45"))
        print("  " + "-"*60)
        for m in ["A1_Baseline",
                  "A2_Ours_CT_Density",
                  "A3_CT_Joint",
                  "D1_CT_ACD"]:
            row = "  {:<28}".format(m)
            for p in [0.20, 0.30, 0.40, 0.45]:
                p_key = "p{:03d}".format(int(p*100))
                val = all_res[n_key].get(p_key, {}).get(
                    m, {}).get("mean", 0)
                row += "  {:.4f}".format(val)
            print(row)

    # Before vs After (N=1k만, p=0.20~0.45)
    print()
    print("  [Before vs After] N=1k  (p=0.20~0.45)")
    print("  {:<6}  {:<10}  {:<12}  {:<12}  {:<10}".format(
        "p", "Baseline", "Ours(CT)", "Ours(ACD)", "d(CT)"))
    print("  " + "-"*58)
    for p in [0.20, 0.30, 0.40, 0.45]:
        p_key = "p{:03d}".format(int(p*100))
        b0  = all_res["N1000"].get(p_key, {}).get(
            "A1_Baseline", {}).get("mean", 0)
        ct  = all_res["N1000"].get(p_key, {}).get(
            "A2_Ours_CT_Density", {}).get("mean", 0)
        acd = all_res["N1000"].get(p_key, {}).get(
            "D1_CT_ACD", {}).get("mean", 0)
        d_ct  = ct  - b0
        mk = " **" if d_ct>0.10 else " *" if d_ct>0.05              else " ~" if d_ct>0 else " v"
        print("  {:.2f}   {:.4f}    {:.4f}      {:.4f}      "
              "{:+.4f}{}".format(p, b0, ct, acd, d_ct, mk))

    # CH3 비교: Density Ratio vs Joint density
    print()
    print("  [CH3 비교] Density Ratio vs Joint(IAT x LEN)")
    print("  {:<28} {:>7}  {:>7}  {:>7}  {:>7}".format(
        "방법", "p=0.20", "p=0.30", "p=0.40", "p=0.45"))
    print("  " + "-"*60)
    for m in ["A1_Baseline",
              "A2_Ours_CT_Density",
              "A3_CT_Joint",
              "D1_CT_ACD"]:
        label = {
            "A1_Baseline":        "Baseline",
            "A2_Ours_CT_Density": "CT+Density(ratio CH3)",
            "A3_CT_Joint":        "CT+Joint(IAT x LEN CH3)",
            "D1_CT_ACD":          "CT+ACD+Density"
        }.get(m, m)
        for N in N_SIZES:
            n_key = "N{}".format(N)
            row = "  {:<28}  N={:<5}".format(label, N)
            for p in [0.20, 0.30, 0.40, 0.45]:
                p_key = "p{:03d}".format(int(p*100))
                val = all_res[n_key].get(p_key, {}).get(
                    m, {}).get("mean", 0)
                row += "  {:.4f}".format(val)
            print(row)
        print()

    # 실험 B
    print()
    print("  [실험 B] Text(1D) vs Image(2D)")
    print("  {:<28} {}  {}  {}  {}  {}  {}".format(
        "방법", "p=0.20", "p=0.30", "p=0.40",
        "p=0.45", "p=0.50", "p=0.55"))
    print("  " + "-"*70)
    for m in ["B1_CT_Text1D", "B2_CT_Image2D"]:
        row = "  {:<28}".format(m)
        for p in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55]:
            p_key = "p{:03d}".format(int(p*100))
            val = all_res["N1000"].get(p_key, {}).get(
                m, {}).get("mean", 0)
            row += "  {:.4f}".format(val)
        print(row)

    # 실험 D
    print()
    print("  [실험 D] ACD (Adaptive Cluster Density)")
    print("  {:<28} {}  {}  {}  {}  {}  {}".format(
        "방법", "p=0.20", "p=0.30", "p=0.40",
        "p=0.45", "p=0.50", "p=0.55"))
    print("  " + "-"*70)
    for m in ["A1_Baseline",
              "A2_Ours_CT_Density",
              "A3_CT_Joint",
              "D1_CT_ACD",
              "D3_CT_ACD_Joint"]:
        row = "  {:<28}".format(m)
        for p in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55]:
            p_key = "p{:03d}".format(int(p*100))
            val = all_res["N1000"].get(p_key, {}).get(
                m, {}).get("mean", 0)
            row += "  {:.4f}".format(val)
        print(row)

    # 실험 C
    print()
    print("  [실험 C] CoTeaching vs GCE filter")
    print("  {:<28} {}  {}  {}  {}  {}  {}".format(
        "방법", "p=0.20", "p=0.30", "p=0.40",
        "p=0.45", "p=0.50", "p=0.55"))
    print("  " + "-"*70)
    for m in ["A2_Ours_CT_Density",
                  "B1_CT_Text1D",
                  "C1_GCE_Density",
                  "C4_GCE_Text",
                  "C2_GCEW_Image",
                  "C3_GCEW_Text"]:
        row = "  {:<28}".format(m)
        for p in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55]:
            p_key = "p{:03d}".format(int(p*100))
            val = all_res["N1000"].get(p_key, {}).get(
                m, {}).get("mean", 0)
            row += "  {:.4f}".format(val)
        print(row)

    return all_res


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print()
    print("="*55)
    print("  [DoH2020]")
    print("="*55)
    doh_res = run_experiment("DoH2020", load_doh)
    with open(os.path.join(
            RESULTS_DIR, "doh_final.json"), "w") as f:
        json.dump(doh_res, f, indent=2)

    print()
    print("="*55)
    print("  [Gotham2025]")
    print("="*55)
    got_res = run_experiment("Gotham2025", load_gotham)
    with open(os.path.join(
            RESULTS_DIR, "gotham_final.json"), "w") as f:
        json.dump(got_res, f, indent=2)

    print()
    print("이미지:", IMG_DIR)


if __name__ == "__main__":
    main()
