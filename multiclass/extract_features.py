"""
extract_features.py  (multiclass version)
==========================================
Vectorized 33-feature extraction from CICIDS2018 sequences.
Output: /home/oper/CICIDS2018/statistical_features_cic.npz
        features   : (1679986, 33) float32
        labels_fine: (1679986,)    int8
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np

CIC_DATA   = "/home/oper/CICIDS2018/merged_final/cicids2018_train_T64.npz"
OUT_PATH   = "/home/oper/CICIDS2018/statistical_features_cic.npz"
BATCH_SIZE = 100_000   # process in 100k chunks


# ══════════════════════════════════════════════════════════════════════
# Vectorized stats helpers
# ══════════════════════════════════════════════════════════════════════

def _safe_stats9(arr_nan: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    arr_nan : (N, 64) float64 with np.nan for invalid entries
    mask    : (N, 64) bool valid positions
    Returns : (N, 9) float32
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean   = np.nanmean(arr_nan,        axis=1)
        std    = np.nanstd(arr_nan,         axis=1)
        mx     = np.nanmax(arr_nan,         axis=1)
        mn     = np.nanmin(arr_nan,         axis=1)
        median = np.nanmedian(arr_nan,      axis=1)
        p25    = np.nanpercentile(arr_nan,  25, axis=1)
        p75    = np.nanpercentile(arr_nan,  75, axis=1)

    ratio = mask.sum(axis=1).astype(np.float32) / 64.0
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(std > 1e-10, 3.0 * (mean - median) / std, 0.0)

    out = np.stack([mean, std, mx, mn, median, ratio, p25, p75, skew], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _entropy_vec(arr_nan: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Batch entropy using fixed-grid histogram.  (N, 64) → (N,) float32"""
    N = arr_nan.shape[0]
    mn = np.nanmin(arr_nan, axis=1, keepdims=True)
    mx = np.nanmax(arr_nan, axis=1, keepdims=True)
    rng = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(rng > 1e-10, (arr_nan - mn) / rng, 0.5)
    norm = np.nan_to_num(norm, nan=0.5)
    binned = np.clip(np.floor(norm * n_bins).astype(np.int32), 0, n_bins - 1)

    counts = np.zeros((N, n_bins), dtype=np.float32)
    for b in range(n_bins):
        counts[:, b] = (binned == b).sum(axis=1)
    counts += 1e-10
    counts /= counts.sum(axis=1, keepdims=True)
    return (-(counts * np.log(counts)).sum(axis=1)).astype(np.float32)


def _burst_vec(seqs_signed: np.ndarray) -> np.ndarray:
    """
    Max consecutive same-direction burst per row.
    seqs_signed: (N, 64). Returns (N,) float32 (normalised by 64).
    """
    N, T = seqs_signed.shape
    nz   = seqs_signed != 0
    dirs = np.sign(seqs_signed)

    # same[i, j] = True if position j continues the same direction as j-1
    same = np.zeros((N, T), dtype=bool)
    same[:, 1:] = (dirs[:, 1:] == dirs[:, :-1]) & nz[:, 1:] & nz[:, :-1]

    # DP: streak length ending at j, vectorised over N (loop over T=64)
    streak  = nz[:, 0].astype(np.int32)    # initial streak = 1 if nz else 0
    max_str = streak.copy()
    for j in range(1, T):
        streak = np.where(same[:, j], streak + 1,
                          nz[:, j].astype(np.int32))
        max_str = np.maximum(max_str, streak)

    return (max_str / 64.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Main extraction (batched)
# ══════════════════════════════════════════════════════════════════════

def extract_batch(sl_b, si_b, ss_b):
    """Extract 33 features for one batch. Returns (B, 33) float32."""
    mask_l = sl_b > 0
    mask_i = si_b > 0
    mask_s = ss_b != 0

    sl = np.where(mask_l, sl_b.astype(np.float64), np.nan)
    si = np.where(mask_i, si_b.astype(np.float64), np.nan)
    ss = np.where(mask_s, np.abs(ss_b).astype(np.float64), np.nan)

    f_len  = _safe_stats9(sl, mask_l)   # (B, 9)
    f_iat  = _safe_stats9(si, mask_i)   # (B, 9)
    f_sign = _safe_stats9(ss, mask_s)   # (B, 9)

    # Flow-level features (6)
    flow_len_norm = mask_l.sum(axis=1).astype(np.float32) / 64.0
    sign_pos  = (ss_b > 0).sum(axis=1)
    sign_nz   = mask_s.sum(axis=1)
    sign_ratio = np.where(sign_nz > 0,
                          sign_pos / (sign_nz.astype(np.float32) + 1e-8),
                          0.5).astype(np.float32)

    # Correlation (len × iat, vectorised)
    mn_l = np.nanmean(sl, axis=1, keepdims=True)
    mn_i = np.nanmean(si, axis=1, keepdims=True)
    valid = mask_l & mask_i
    dl = np.where(valid, sl - mn_l, 0.0)
    di = np.where(valid, si - mn_i, 0.0)
    num = (dl * di).sum(axis=1)
    den = np.sqrt((dl**2).sum(axis=1) * (di**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(den > 1e-10, num / den, 0.0).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    len_ent = _entropy_vec(sl)
    iat_ent = _entropy_vec(si)
    burst   = _burst_vec(ss_b)

    flow_feats = np.stack(
        [flow_len_norm, sign_ratio, corr, len_ent, iat_ent, burst], axis=1)

    return np.concatenate([f_len, f_iat, f_sign, flow_feats], axis=1)


def main():
    # ── Check cache ──────────────────────────────────────────────────
    if os.path.exists(OUT_PATH):
        c = np.load(OUT_PATH)
        if c["features"].shape == (1_679_986, 33):
            print(f"캐시 유효: {OUT_PATH}  shape={c['features'].shape}")
            return
        print(f"캐시 불완전 ({c['features'].shape[0]:,} rows) → 재추출")

    print(f"데이터 로드: {CIC_DATA}")
    d = np.load(CIC_DATA)
    seqs_len    = d["seqs_len"]
    seqs_iat    = d["seqs_iat"]
    seqs_signed = d["seqs_signed"]
    labels_fine = d["labels_fine"]
    N = len(labels_fine)
    print(f"  N={N:,}")

    all_feats = np.zeros((N, 33), dtype=np.float32)
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        if start % 200_000 == 0:
            print(f"  배치 {start:,} / {N:,}…")
        sl_b = seqs_len[start:end]
        si_b = seqs_iat[start:end]
        ss_b = seqs_signed[start:end]
        all_feats[start:end] = extract_batch(sl_b, si_b, ss_b)

    nan_cnt = np.isnan(all_feats).sum()
    print(f"  NaN count: {nan_cnt}  (should be 0)")

    print(f"저장: {OUT_PATH}")
    np.savez(OUT_PATH,
             features=all_feats,
             labels_fine=labels_fine)
    print(f"  완료. shape={all_feats.shape}")


if __name__ == "__main__":
    main()
