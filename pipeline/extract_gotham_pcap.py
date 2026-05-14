"""
Gotham 2025 pcap → seqs_len, seqs_iat, seqs_flags
DoH2020과 동일한 파이프라인
폴더 구조로 레이블링:
  raw/benign/ → label=0
  raw/malicious/*/ → label=1 (binary)
  또는 멀티클래스
"""

import os, glob
import dpkt
import socket
import numpy as np
from collections import defaultdict

BASE_DIR = "/home/oper/LabelDiffusion/data/gotham2025/raw/"
OUT_PATH = "/home/oper/LabelDiffusion/data/gotham_seqs_pcap.npz"
MAX_LEN  = 50
MAX_PKT  = 1999
MAX_IAT  = 9999

# 멀티클래스 레이블
ATTACK_MAP = {
    "benign":           0,
    "mirai-dos":        1,
    "mirai-infection":  2,
    "network-scanning": 3,
    "merlin":           4,
    "coap-amplificator":5,
}

def encode_flags(tcp):
    f = 0
    try:
        if tcp.flags & dpkt.tcp.TH_SYN:  f |= 1
        if tcp.flags & dpkt.tcp.TH_FIN:  f |= 2
        if tcp.flags & dpkt.tcp.TH_RST:  f |= 4
        if tcp.flags & dpkt.tcp.TH_PUSH: f |= 8
        if tcp.flags & dpkt.tcp.TH_ACK:  f |= 16
    except: pass
    return f

def extract_flows_from_pcap(pcap_path):
    """pcap → flow별 패킷 리스트"""
    flows = defaultdict(list)
    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP): continue
                    ip = eth.data
                    pkt_len = ip.len

                    # TCP
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        tcp = ip.data
                        src = socket.inet_ntop(socket.AF_INET, ip.src)
                        dst = socket.inet_ntop(socket.AF_INET, ip.dst)
                        fid = tuple(sorted([
                            (src, tcp.sport), (dst, tcp.dport)]))
                        flags = encode_flags(tcp)
                        flows[fid].append((ts, pkt_len, flags))

                    # UDP
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        udp = ip.data
                        src = socket.inet_ntop(socket.AF_INET, ip.src)
                        dst = socket.inet_ntop(socket.AF_INET, ip.dst)
                        fid = tuple(sorted([
                            (src, udp.sport), (dst, udp.dport)]))
                        flows[fid].append((ts, pkt_len, 0))

                    # ICMP
                    elif isinstance(ip.data, dpkt.icmp.ICMP):
                        src = socket.inet_ntop(socket.AF_INET, ip.src)
                        dst = socket.inet_ntop(socket.AF_INET, ip.dst)
                        fid = (src, dst, 'icmp')
                        flows[fid].append((ts, pkt_len, 0))

                except: continue
    except Exception as e:
        print(f"    pcap 에러: {e}")
    return flows

def flows_to_seqs(flows):
    seqs_len   = []
    seqs_iat   = []
    seqs_flags = []

    for fid, pkts in flows.items():
        if len(pkts) < 3: continue
        pkts = sorted(pkts, key=lambda x: x[0])

        ts_arr  = np.array([p[0] for p in pkts])
        len_arr = np.array([p[1] for p in pkts])
        flg_arr = np.array([p[2] for p in pkts])
        iats    = np.diff(ts_arr) * 1000  # ms

        seq_len   = np.zeros(MAX_LEN, dtype=np.float32)
        seq_iat   = np.zeros(MAX_LEN, dtype=np.float32)
        seq_flags = np.zeros(MAX_LEN, dtype=np.float32)

        n = min(len(pkts), MAX_LEN)
        seq_len[:n]   = np.clip(len_arr[:n], 0, MAX_PKT)
        seq_flags[:n] = flg_arr[:n]
        if len(iats) > 0:
            ni = min(len(iats), MAX_LEN-1)
            seq_iat[1:ni+1] = np.clip(iats[:ni], 0, MAX_IAT)

        seqs_len.append(seq_len)
        seqs_iat.append(seq_iat)
        seqs_flags.append(seq_flags)

    return (np.array(seqs_len,   dtype=np.float32),
            np.array(seqs_iat,   dtype=np.float32),
            np.array(seqs_flags, dtype=np.float32))

def main():
    all_len, all_iat, all_flags, all_labels = [], [], [], []

    # benign
    benign_dir = os.path.join(BASE_DIR, "benign")
    benign_pcaps = glob.glob(os.path.join(benign_dir, "*.pcap"))
    print(f"\n[Benign] {len(benign_pcaps)}개 pcap...")
    for i, pf in enumerate(benign_pcaps):
        flows = extract_flows_from_pcap(pf)
        sl, si, sf = flows_to_seqs(flows)
        if len(sl) == 0: continue
        all_len.append(sl)
        all_iat.append(si)
        all_flags.append(sf)
        all_labels.append(np.zeros(len(sl), dtype=np.int32))
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(benign_pcaps)}  "
                  f"누적 benign={sum(len(x) for x in all_len if len(x)>0):,}")

    be_total = sum(len(x) for x in all_len)
    print(f"Benign 완료: {be_total:,} flows")

    # malicious
    mal_dir = os.path.join(BASE_DIR, "malicious")
    for attack_name, cls_id in ATTACK_MAP.items():
        if attack_name == "benign": continue
        attack_dir = os.path.join(mal_dir, attack_name)
        if not os.path.exists(attack_dir): continue
        pcap_files = glob.glob(os.path.join(attack_dir, "*.pcap"))
        print(f"\n[{attack_name}] cls={cls_id}  {len(pcap_files)}개 pcap...")

        atk_len = []
        for i, pf in enumerate(pcap_files):
            flows = extract_flows_from_pcap(pf)
            sl, si, sf = flows_to_seqs(flows)
            if len(sl) == 0: continue
            all_len.append(sl)
            all_iat.append(si)
            all_flags.append(sf)
            all_labels.append(
                np.full(len(sl), cls_id, dtype=np.int32))
            atk_len.append(len(sl))
            if (i+1) % 5 == 0:
                print(f"  {i+1}/{len(pcap_files)}  "
                      f"이번 공격 누적={sum(atk_len):,}")

        print(f"  {attack_name} 완료: {sum(atk_len):,} flows")

    # 합치기
    X_len   = np.vstack(all_len)
    X_iat   = np.vstack(all_iat)
    X_flags = np.vstack(all_flags)
    y       = np.concatenate(all_labels)

    print(f"\n{'='*50}")
    print(f"총 flows: {len(y):,}")
    print(f"shape: {X_len.shape}")
    for cls_id, cls_name in {v:k for k,v in ATTACK_MAP.items()}.items():
        cnt = (y==cls_id).sum()
        if cnt > 0:
            print(f"  cls{cls_id} {cls_name}: {cnt:,}")

    # binary label도 저장
    y_binary = (y > 0).astype(np.int32)
    print(f"\nbinary: benign={( y_binary==0).sum():,}  "
          f"malicious={(y_binary==1).sum():,}")

    np.savez(OUT_PATH,
             seqs_len=X_len,
             seqs_iat=X_iat,
             seqs_flags=X_flags,
             labels=y_binary,        # binary
             labels_multi=y,         # multiclass
             class_names=np.array(list(ATTACK_MAP.keys())))
    print(f"\n저장: {OUT_PATH}")

if __name__ == "__main__":
    main()
