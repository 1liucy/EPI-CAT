import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Tuple


REGION_FILE = 'Modified_pairs_Regions_with_Strand.csv'   # 含 enhancer/promoter 区域与 label
TF_NARROWPEAK_DIR = 'narrowPeak_files(K562)'                     # 存放 *.narrowPeak / *.bed
TF_CODE_JSON = 'tf_name_to_code.json'                                             # 复用/持久化 TF -> 单字符编码
FINAL_OUTPUT = 'processed_train_data_with_binding.csv'  # 最终输出


REVERSE_BINDINGS_WITH_STRAND = True
# =====================================

# 全局字典用于映射转录因子名到单字符编码
tf_name_to_code = {}
current_code = ord('A')  # 起始编码

def load_or_initialize_code(json_path: str = TF_CODE_JSON):
    global tf_name_to_code, current_code
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            tf_name_to_code = json.load(f)
        if tf_name_to_code:
            current_code = max(ord(code) for code in tf_name_to_code.values()) + 1
        else:
            current_code = ord('A')
    except FileNotFoundError:
        tf_name_to_code = {}
        current_code = ord('A')

def save_code(json_path: str = TF_CODE_JSON):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tf_name_to_code, f, ensure_ascii=False)

def parse_narrowpeak_summit(narrowpeak_file: str, tf_name: str) -> pd.DataFrame:
   
    global current_code
    if tf_name not in tf_name_to_code:
        tf_name_to_code[tf_name] = chr(current_code)
        current_code += 1

    rows = []
    with open(narrowpeak_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 10:
                continue
            chrom = fields[0]
            start = int(fields[1])
            signal_value = float(fields[6])
            summit_offset = int(fields[9])
            summit_pos = start + summit_offset
            rows.append([chrom, start, summit_pos, tf_name, tf_name_to_code[tf_name], signal_value])

    return pd.DataFrame(rows, columns=["chrom", "start", "summit", "tf_name", "tf_code", "signal_value"])

def load_regions(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required = [
        'enhancer_chrom', 'enhancer_start', 'enhancer_end',
        'promoter_chrom', 'promoter_start', 'promoter_end', 'strand'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"区域文件缺少必要列: {missing}")
    if 'label' not in df.columns:

        df['label'] = np.nan
    return df

def map_summit_to_regions(
    summit_df: pd.DataFrame,
    region_df: pd.DataFrame
) -> Tuple[List[str], List[str], List[float], List[float], List[List[float]], List[List[float]]]:

    enhancer_codes, promoter_codes = [], []
    enhancer_scores, promoter_scores = [], []
    enhancer_tf_bindings, promoter_tf_bindings = [], []

    
    summit_df = summit_df.sort_values(by=['chrom', 'summit', 'start', 'tf_name']).reset_index(drop=True)

    for _, r in region_df.iterrows():
        # Enhancer
        e_chrom, e_start, e_end = r['enhancer_chrom'], int(r['enhancer_start']), int(r['enhancer_end'])
        e_hits = summit_df[(summit_df['chrom'] == e_chrom) &
                           (summit_df['summit'] >= e_start) &
                           (summit_df['summit'] <= e_end)]

        if e_hits.empty:
            e_code = 'n'
            e_bind = []
        else:
            e_hits_sorted = e_hits.sort_values(by=['summit', 'start', 'tf_name'])
            e_code = ''.join(e_hits_sorted['tf_code'].tolist())
            e_bind = e_hits_sorted['signal_value'].astype(float).tolist()

        enhancer_codes.append(e_code)
        enhancer_tf_bindings.append(e_bind)
        enhancer_scores.append(sum(e_bind) if e_bind else 0.0)

        # Promoter
        p_chrom, p_start, p_end, strand = r['promoter_chrom'], int(r['promoter_start']), int(r['promoter_end']), str(r['strand'])
        p_hits = summit_df[(summit_df['chrom'] == p_chrom) &
                           (summit_df['summit'] >= p_start) &
                           (summit_df['summit'] <= p_end)]

        if p_hits.empty:
            p_code = 'n'
            p_bind = []
        else:
            p_hits_sorted = p_hits.sort_values(by=['summit', 'start', 'tf_name'])
            p_code = ''.join(p_hits_sorted['tf_code'].tolist())
            p_bind = p_hits_sorted['signal_value'].astype(float).tolist()


        if strand == '-':
            p_code = p_code[::-1]
            if REVERSE_BINDINGS_WITH_STRAND and p_bind:
                p_bind = p_bind[::-1]

        promoter_codes.append(p_code)
        promoter_tf_bindings.append(p_bind)
        promoter_scores.append(sum(p_bind) if p_bind else 0.0)

    return (enhancer_codes, promoter_codes,
            enhancer_scores, promoter_scores,
            enhancer_tf_bindings, promoter_tf_bindings)

def flatten_all_bindings(list_of_lists: List[List[float]]) -> List[float]:
    out = []
    for lst in list_of_lists:
        if lst:
            out.extend(lst)
    return out

def global_min_max(enh_bind: List[List[float]], pro_bind: List[List[float]]) -> Tuple[float, float]:
    all_vals = flatten_all_bindings(enh_bind) + flatten_all_bindings(pro_bind)
    if not all_vals:
       
        return 0.0, 1.0
    return float(min(all_vals)), float(max(all_vals))

def normalize_list(vals: List[float], vmin: float, vmax: float) -> List[float]:
    if not vals:
        return []
    if vmax == vmin:
       
        return [0.0 for _ in vals]
    return [(x - vmin) / (vmax - vmin) for x in vals]

def build_tf_to_index_from_codes(codes: List[str]) -> dict:

    charset = set()
    for s in codes:
        if isinstance(s, str):
            for ch in s:
                if ch and ch != 'n':
                    charset.add(ch)
    # 固定顺序以保证可复现
    sorted_chars = sorted(charset)
    return {ch: i for i, ch in enumerate(sorted_chars)}

def one_hot_encode_with_binding(sequence: str, tf_bindings: List[float], tf_to_index: dict) -> np.ndarray:

    L = len(sequence) if isinstance(sequence, str) else 0
    D = len(tf_to_index) + 1
    mat = np.zeros((L, D), dtype=float)
    if L == 0:
        return mat
    for i, ch in enumerate(sequence):
        if ch in tf_to_index:
            mat[i, tf_to_index[ch]] = 1.0
        # 绑定强度列
        if tf_bindings and i < len(tf_bindings):
            mat[i, -1] = float(tf_bindings[i])
        else:
            mat[i, -1] = 0.0
    return mat

def pad_sequences_custom(mats: List[np.ndarray], max_length: int, width: int) -> List[np.ndarray]:
    padded = []
    for M in mats:
        L = M.shape[0]
        if L < max_length:
            pad = np.zeros((max_length - L, width), dtype=float)
            padded.append(np.vstack([M, pad]))
        else:
            padded.append(M[:max_length, :])
    return padded

def calculate_max_len_from_codes(codes: List[str]) -> int:
    return max((len(s) for s in codes if isinstance(s, str)), default=0)

def main():
    # 1) 读入/初始化 TF 编码，加载区域
    load_or_initialize_code(TF_CODE_JSON)
    regions = load_regions(REGION_FILE)

    # 2) 遍历 TF 目录，汇总峰信息
    all_summits_list = []
    for fname in os.listdir(TF_NARROWPEAK_DIR):
        if not (fname.endswith('.narrowPeak') or fname.endswith('.bed')):
            continue

        m = re.search(r'\(([^)]+)\)', fname)
        tf_name = m.group(1) if m else None
        if not tf_name:

            continue
        fpath = os.path.join(TF_NARROWPEAK_DIR, fname)
        if not (os.path.exists(fpath) and os.access(fpath, os.R_OK)):
            continue
        try:
            df_tf = parse_narrowpeak_summit(fpath, tf_name)
            if not df_tf.empty:
                all_summits_list.append(df_tf)
        except Exception as e:
            print(f"解析 {fpath} 出错：{e}")

    if not all_summits_list:
        raise RuntimeError("没有成功读取到任何 TF 峰数据，请检查目录与文件名格式。")

    all_summits = pd.concat(all_summits_list, ignore_index=True)
    all_summits = all_summits.sort_values(by=['chrom', 'summit', 'start', 'tf_name']).reset_index(drop=True)

    # 3) 映射到区域，得到 code 与原始绑定强度列表
    (enh_codes, pro_codes,
     enh_scores, pro_scores,
     enh_bind_lists, pro_bind_lists) = map_summit_to_regions(all_summits, regions)

    regions['enhancer_code'] = enh_codes
    regions['promoter_code'] = pro_codes
    regions['enhancer_score'] = enh_scores
    regions['promoter_score'] = pro_scores
    regions['enhancer_tf_bindings'] = enh_bind_lists
    regions['promoter_tf_bindings'] = pro_bind_lists

    
    gmin, gmax = global_min_max(enh_bind_lists, pro_bind_lists)
    regions['enhancer_tf_bindings'] = regions['enhancer_tf_bindings'].apply(lambda lst: normalize_list(lst, gmin, gmax))
    regions['promoter_tf_bindings'] = regions['promoter_tf_bindings'].apply(lambda lst: normalize_list(lst, gmin, gmax))

    # 5) 动态构建 one-hot 维度（基于所有出现过的 code 字符）
    tf_to_index = build_tf_to_index_from_codes(regions['enhancer_code'].tolist() + regions['promoter_code'].tolist())
    vec_width = len(tf_to_index) + 1  # +1 为 binding 列

    # 6) 生成 one-hot 矩阵（含绑定强度列）
    enh_mats = []
    pro_mats = []
    for _, row in regions.iterrows():
        e_mat = one_hot_encode_with_binding(row['enhancer_code'], row['enhancer_tf_bindings'], tf_to_index)
        p_mat = one_hot_encode_with_binding(row['promoter_code'], row['promoter_tf_bindings'], tf_to_index)
        enh_mats.append(e_mat)
        pro_mats.append(p_mat)

    # 7) 计算最大长度并 padding
    max_len_en = calculate_max_len_from_codes(regions['enhancer_code'].tolist())
    max_len_pr = calculate_max_len_from_codes(regions['promoter_code'].tolist())
    enh_padded = pad_sequences_custom(enh_mats, max_len_en, vec_width)
    pro_padded = pad_sequences_custom(pro_mats, max_len_pr, vec_width)

    # 8) 保存到 CSV（将矩阵以 JSON 字符串存列）
    regions['enhancer_code_padded'] = [json.dumps(m.tolist()) for m in enh_padded]
    regions['promoter_code_padded'] = [json.dumps(m.tolist()) for m in pro_padded]

    # 输出最终文件
    regions.to_csv(FINAL_OUTPUT, index=False)
    print(f" 已输出：{FINAL_OUTPUT}")
    print(f"  - enhancer 最大长度: {max_len_en}, promoter 最大长度: {max_len_pr}")

    # 持久化 TF 编码映射
    save_code(TF_CODE_JSON)

if __name__ == '__main__':
    main()