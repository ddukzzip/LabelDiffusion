"""
pipeline_mc.py
==============
CICIDS2018 Multiclass Noisy Label Detection Pipeline.

Experiments:
  A : Baseline – train on noisy labels directly
  D : 3-channel group image detection + Z-score clean

Noise levels: p=0.30, p=0.45
Metric: macro F1, per-class F1

Output: /home/oper/LabelDiffusion/multiclass/results/
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

# Multiclass modules (same directory)
sys.path.insert(0, "/home/oper/LabelDiffusion/multiclass")
from extract_features import extract_batch, OUT_PATH as FEAT_PATH
from group_image_mc   import MulticlassGroupImageBuilder, LABEL_NAMES
from noisy_detector_mc import MulticlassNoisyDetector

CIC_DATA    = "/home/oper/CICIDS2018/merged_final/cicids2018_train_T64.npz"
RESULTS_DIR = "/home/oper/LabelDiffusion/multiclass/results"
SEED        = 42
N_PER_CLASS = 3_333   # equal per class → 15 × 3333 = 49995 ≈ 50k
N_TEST_PER  = 666     # test per class  → 15 × 666  = 9990
NOISE_LEVELS = [0.30, 0.45]

# GroupImageBuilder params (same as binary best)
N_BINS      = 20
CONS_THRESH = 0.8
MAX_ANCHORS = 100

# XGBoost params
XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=SEED, verbosity=0,
    use_label_encoder=False,
)

K_CLASSES = 15    # 0..14


# ══════════════════════════════════════════════════════════════════════
# Data loading / feature extraction
# ══════════════════════════════════════════════════════════════════════

def load_features():
    """
    Load (or extract) feature matrix.
    Returns: features (N, 39) = stat(33) + metas[2:](6), labels_fine (N,)
    metas[2:] = [duration, pkt_count, total_bytes, protocol, port_src, port_dst]
    """
    if os.path.exists(FEAT_PATH):
        c = np.load(FEAT_PATH)
        if c["features"].shape == (1_679_986, 33):
            print(f"캐시 로드: stat features {c['features'].shape}")
            stat_feats  = c["features"].astype(np.float32)
            labels_fine = c["labels_fine"].astype(int)
            # Append metas[2:] from raw data
            print(f"  metas[2:] 로드: {CIC_DATA}")
            d = np.load(CIC_DATA)
            metas = d["metas"][:, 2:].astype(np.float32)  # (N, 6)
            feats = np.concatenate([stat_feats, metas], axis=1)  # (N, 39)
            print(f"  최종 features: {feats.shape}")
            return feats, labels_fine

    print("통계 feature 추출 중 (전체 1.68M)…")
    d = np.load(CIC_DATA)
    seqs_len    = d["seqs_len"]
    seqs_iat    = d["seqs_iat"]
    seqs_signed = d["seqs_signed"]
    labels_fine = d["labels_fine"].astype(int)
    metas       = d["metas"][:, 2:].astype(np.float32)
    N = len(labels_fine)

    BATCH = 100_000
    stat_feats = np.zeros((N, 33), dtype=np.float32)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        if start % 300_000 == 0:
            print(f"  {start:,}/{N:,}…")
        stat_feats[start:end] = extract_batch(
            seqs_len[start:end], seqs_iat[start:end], seqs_signed[start:end])

    np.savez(FEAT_PATH, features=stat_feats, labels_fine=labels_fine)
    print(f"  저장: {FEAT_PATH}")
    feats = np.concatenate([stat_feats, metas], axis=1)
    return feats, labels_fine


def stratified_sample(labels_fine, n_per_class, n_test_per, seed):
    """
    Equal-per-class stratified sampling (클래스별 균등).
    n_per_class train + n_test_per test per class, non-overlapping.
    """
    rng     = np.random.default_rng(seed)
    classes = sorted(np.unique(labels_fine).tolist())
    assert len(classes) == K_CLASSES

    train_idx, test_idx = [], []
    for cls in classes:
        cls_idx = np.where(labels_fine == cls)[0]
        perm    = rng.permutation(cls_idx)
        need    = n_per_class + n_test_per
        assert len(perm) >= need, \
            f"class {cls}: need {need} but only {len(perm)} available"
        train_idx.append(perm[:n_per_class])
        test_idx.append(perm[n_per_class:n_per_class + n_test_per])

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


# ══════════════════════════════════════════════════════════════════════
# Noise simulation
# ══════════════════════════════════════════════════════════════════════

def add_noise_multiclass(labels, p, n_classes=K_CLASSES, seed=42):
    """
    Symmetric multiclass flip:
    each label flipped to a uniformly random OTHER class with probability p.
    """
    rng       = np.random.default_rng(seed)
    noisy     = labels.copy().astype(int)
    flip_mask = rng.random(len(labels)) < p
    n_flip    = flip_mask.sum()
    if n_flip == 0:
        return noisy

    # For each flipped sample, draw from all classes != given
    orig_flip  = labels[flip_mask].astype(int)
    rand_vals  = rng.integers(0, n_classes - 1, size=n_flip)
    new_labels = np.where(rand_vals >= orig_flip,
                          rand_vals + 1, rand_vals)  # skip original class
    noisy[flip_mask] = new_labels
    return noisy


# ══════════════════════════════════════════════════════════════════════
# Classifier helpers
# ══════════════════════════════════════════════════════════════════════

def _make_xgb(n_classes):
    p = dict(**XGB_PARAMS)
    p["objective"]    = "multi:softmax"
    p["num_class"]    = n_classes
    try:
        return xgb.XGBClassifier(device="cuda", tree_method="hist", **p)
    except Exception:
        return xgb.XGBClassifier(tree_method="hist", **p)


def fit_eval(X_tr, y_tr, X_te, y_te, label="", n_classes=K_CLASSES):
    """Fit XGBoost, evaluate, return result dict + per-class F1."""
    clf = _make_xgb(n_classes)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    acc      = float(accuracy_score(y_te, y_pred))

    per_cls  = f1_score(y_te, y_pred, average=None,
                        labels=list(range(n_classes)), zero_division=0)

    return {
        "label":    label,
        "n_train":  int(len(X_tr)),
        "macro_f1": macro_f1,
        "acc":      acc,
        "per_class_f1": per_cls.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════
# Single noise-level run
# ══════════════════════════════════════════════════════════════════════

def run_one(X_tr, y_tr_fine, X_te, y_te_fine, noise_p):
    print(f"\n{'#'*60}")
    print(f"  p={noise_p:.2f}")
    print(f"{'#'*60}")

    y_noisy = add_noise_multiclass(y_tr_fine, noise_p)
    n_flip  = (y_noisy != y_tr_fine).sum()
    print(f"  노이즈: flipped={n_flip:,}/{len(y_tr_fine):,} "
          f"({n_flip/len(y_tr_fine)*100:.1f}%)")

    results = {}

    # ── A: Baseline ──────────────────────────────────────────────────
    print("\n  [A] Baseline")
    r = fit_eval(X_tr, y_noisy, X_te, y_te_fine, "A")
    results["A"] = r
    print(f"  macro F1={r['macro_f1']:.4f}  acc={r['acc']:.4f}")

    # ── D: Group image detection + Z-score clean ─────────────────────
    print("\n  [D] Group image 구축 (ch1ch2ch3, Z-score clean)")

    gib = MulticlassGroupImageBuilder(
        n_bins=N_BINS, cons_thresh=CONS_THRESH,
        k=20, min_anchors=10, max_anchors=MAX_ANCHORS)
    gib.fit(X_tr, y_noisy)
    imgs = gib.build()

    det = MulticlassNoisyDetector()
    det.fit(imgs, gib.bin_edges)
    det.detect(X_tr, y_noisy)

    # Detection evaluation vs ground truth
    det_eval = det.evaluate(y_tr_fine)

    # Z-score clean
    X_dz, y_dz, info_dz = det.clean_zscore(
        X_tr, y_noisy, z_flip=2.0, z_remove=1.5)

    print(f"\n  [D] Train on cleaned data: {len(X_dz):,} samples")
    r = fit_eval(X_dz, y_dz, X_te, y_te_fine, "D+Z")
    r["det_auc"]    = det_eval.get("auc", None)
    r["clean_info"] = info_dz
    results["DZ"] = r
    print(f"  macro F1={r['macro_f1']:.4f}  acc={r['acc']:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════

def print_per_class(res_p030, res_p045, key):
    if key not in res_p030 or key not in res_p045:
        return
    label = res_p030[key]["label"]
    print(f"\n  Per-class F1  [{label}]")
    print(f"  {'Class':<22}  {'p=0.30':>8}  {'p=0.45':>8}")
    print("  " + "─" * 44)
    for cls in range(K_CLASSES):
        name = LABEL_NAMES.get(cls, f"cls{cls}")
        f030 = res_p030[key]["per_class_f1"][cls]
        f045 = res_p045[key]["per_class_f1"][cls]
        print(f"  {cls:2d} {name:<20}  {f030:>8.4f}  {f045:>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Feature load / extract ────────────────────────────────────────
    print("=== CICIDS2018 Multiclass Pipeline ===\n")
    feats, labels_fine = load_features()
    print(f"Features: {feats.shape}  labels_fine: {labels_fine.shape}")

    # ── Stratified sampling ───────────────────────────────────────────
    print("\n샘플링…")
    tr_idx, te_idx = stratified_sample(labels_fine, N_PER_CLASS, N_TEST_PER, SEED)

    X_tr_raw = feats[tr_idx]
    y_tr_fine = labels_fine[tr_idx]
    X_te_raw  = feats[te_idx]
    y_te_fine = labels_fine[te_idx]

    print(f"  train={len(X_tr_raw):,}  test={len(X_te_raw):,}")
    print(f"  train: {N_PER_CLASS:,}/class × {K_CLASSES} classes  "
          f"features={X_tr_raw.shape[1]} (stat33 + metas6)")

    # ── Standardise ──────────────────────────────────────────────────
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_raw).astype(np.float32)
    X_te = sc.transform(X_te_raw).astype(np.float32)

    # ── Run experiments ───────────────────────────────────────────────
    all_res = {}
    for p in NOISE_LEVELS:
        pkey = f"p{int(p*100):03d}"
        all_res[pkey] = run_one(X_tr, y_tr_fine, X_te, y_te_fine, p)

    # ── Save results ──────────────────────────────────────────────────
    serial = {}
    for pkey, res in all_res.items():
        serial[pkey] = {}
        for exp, r in res.items():
            s = dict(r)
            serial[pkey][exp] = s

    with open(os.path.join(RESULTS_DIR, "results_mc.json"), "w") as fh:
        json.dump(serial, fh, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # Final comparison table
    # ══════════════════════════════════════════════════════════════════
    SEP = "═" * 72
    print(f"\n\n{SEP}")
    print(f"  CICIDS2018 Multiclass  최종 결과 "
          f"(cons={CONS_THRESH}, anchors={MAX_ANCHORS}, bins={N_BINS})")
    print(f"{SEP}")

    EXPS  = [("A", "A-Baseline"), ("DZ", "D+Z")]
    PKEYS = [f"p{int(p*100):03d}" for p in NOISE_LEVELS]

    # Header
    hdr = f"  {'방법':<16}"
    for pk in PKEYS:
        hdr += f"  {'F1_'+pk:>9}  {'N_train':>7}"
    print(hdr)
    print("  " + "─" * 56)

    for exp_key, exp_label in EXPS:
        row = f"  {exp_label:<16}"
        for pk in PKEYS:
            r  = all_res[pk].get(exp_key, {})
            f1 = r.get("macro_f1", float("nan"))
            nt = r.get("n_train", 0)
            row += f"  {f1:>9.4f}  {nt:>7,}"
        print(row)
    print("  " + "─" * 56)

    # ΔF1 row
    print(f"\n  ΔF1 vs Baseline A:")
    row = f"  {'D+Z':<16}"
    for pk in PKEYS:
        a_f1 = all_res[pk].get("A",  {}).get("macro_f1", float("nan"))
        d_f1 = all_res[pk].get("DZ", {}).get("macro_f1", float("nan"))
        delta = d_f1 - a_f1 if (a_f1 == a_f1 and d_f1 == d_f1) else float("nan")
        row += f"  {delta:>+9.4f}"
    print(row)

    # Detection AUC
    print(f"\n  Detection AUC:")
    for pk in PKEYS:
        auc = all_res[pk].get("DZ", {}).get("det_auc", None)
        print(f"    {pk}: AUC={auc:.4f}" if auc else f"    {pk}: AUC=N/A")

    print(f"\n{SEP}")

    # Per-class F1 tables
    for pk in PKEYS:
        r_a  = all_res[pk].get("A", {})
        r_dz = all_res[pk].get("DZ", {})
        if not r_a or not r_dz:
            continue
        print(f"\n  Per-class F1  [{pk}]")
        print(f"  {'Class':<22}  {'A':>8}  {'D+Z':>8}  {'Δ':>8}")
        print("  " + "─" * 52)
        for cls in range(K_CLASSES):
            name = LABEL_NAMES.get(cls, f"cls{cls}")
            fa   = r_a["per_class_f1"][cls]
            fd   = r_dz["per_class_f1"][cls]
            print(f"  {cls:2d} {name:<20}  {fa:>8.4f}  {fd:>8.4f}  {fd-fa:>+8.4f}")
        print(f"  {'macro avg':<22}  "
              f"{r_a['macro_f1']:>8.4f}  "
              f"{r_dz['macro_f1']:>8.4f}  "
              f"{r_dz['macro_f1']-r_a['macro_f1']:>+8.4f}")

    print(f"\n  결과 저장: {RESULTS_DIR}/results_mc.json")


if __name__ == "__main__":
    main()
