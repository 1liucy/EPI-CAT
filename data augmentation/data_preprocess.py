import os, re, json
import numpy as np
import pandas as pd
from typing import List, Tuple

REGION_FILE = 'Modi_K_train.csv'
TF_NARROWPEAK_DIR = 'narrowPeak_files(K562)'
FINAL_OUT = 'K-train.csv'
TF_CODE_JSON = 'tf_name_to_code.json'   
SAVE_INTERMEDIATE = False                  
REVERSE_BINDINGS_WITH_STRAND = True      

# -------- 读取/维护 TF 编码 --------
tf_name_to_code = {}
current_code = ord('A')

def load_or_initialize_code(json_path: str = TF_CODE_JSON):
    global tf_name_to_code, current_code
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            tf_name_to_code = json.load(f)
        current_code = max((ord(v) for v in tf_name_to_code.values()), default=ord('A')-1) + 1
        if current_code < ord('A'):
            current_code = ord('A')
    except FileNotFoundError:
        tf_name_to_code, current_code = {}, ord('A')

def save_code(json_path: str = TF_CODE_JSON):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tf_name_to_code, f, ensure_ascii=False)

# -------- 解析 narrowPeak，提取 summit 与 signalValue --------
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

# -------- 加载区域 & 映射峰到 enhancer/promoter --------
def load_regions_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def map_summit_to_regions(summit_df: pd.DataFrame, region_df: pd.DataFrame):
    enhancer_codes, promoter_codes = [], []
    enhancer_scores, promoter_scores = [], []
    enhancer_tf_bindings, promoter_tf_bindings = [], []

    summit_df = summit_df.sort_values(by=['chrom','summit','start','tf_name']).reset_index(drop=True)

    for _, region in region_df.iterrows():
        # enhancer
        e_chrom, e_start, e_end = region['enhancer_chrom'], int(region['enhancer_start']), int(region['enhancer_end'])
        e_hits = summit_df[(summit_df['chrom']==e_chrom) & (summit_df['summit']>=e_start) & (summit_df['summit']<=e_end)]
        if e_hits.empty:
            e_code = 'n'; e_bind = []
        else:
            e_hits = e_hits.sort_values(by=['summit','start','tf_name'])
            e_code = ''.join(e_hits['tf_code'].tolist())
            e_bind = e_hits['signal_value'].astype(float).tolist()
        enhancer_codes.append(e_code)
        enhancer_tf_bindings.append(e_bind)
        enhancer_scores.append(sum(e_bind) if e_bind else 0.0)

        # promoter
        p_chrom, p_start, p_end, strand = region['promoter_chrom'], int(region['promoter_start']), int(region['promoter_end']), str(region['strand'])
        p_hits = summit_df[(summit_df['chrom']==p_chrom) & (summit_df['summit']>=p_start) & (summit_df['summit']<=p_end)]
        if p_hits.empty:
            p_code = 'n'; p_bind = []
        else:
            p_hits = p_hits.sort_values(by=['summit','start','tf_name'])
            p_code = ''.join(p_hits['tf_code'].tolist())
            p_bind = p_hits['signal_value'].astype(float).tolist()

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


def fill_tf_bindings_with_zero_inplace(df: pd.DataFrame):
    def fix_bindings(code: str, bindings: List[float]):
        if isinstance(code, str) and ('n' in code):
            return [0.0] * len(code)
        return bindings
    df['enhancer_tf_bindings'] = df.apply(lambda r: fix_bindings(r['enhancer_code'], r['enhancer_tf_bindings']), axis=1)
    df['promoter_tf_bindings'] = df.apply(lambda r: fix_bindings(r['promoter_code'], r['promoter_tf_bindings']), axis=1)


def flatten_all_bindings(series_of_lists: pd.Series) -> List[float]:
    all_vals = []
    for lst in series_of_lists:
        if isinstance(lst, list):
            all_vals.extend(lst)
    return all_vals

def normalize_inplace_global(df: pd.DataFrame, cols=('enhancer_tf_bindings','promoter_tf_bindings')):
    all_vals = flatten_all_bindings(df[cols[0]]) + flatten_all_bindings(df[cols[1]])
    if len(all_vals) == 0:
        return
    gmin, gmax = float(min(all_vals)), float(max(all_vals))
    if gmax == gmin:
        df[cols[0]] = df[cols[0]].apply(lambda lst: [0.0]*len(lst))
        df[cols[1]] = df[cols[1]].apply(lambda lst: [0.0]*len(lst))
        return
    def norm_list(lst: List[float]):
        return [ (x - gmin) / (gmax - gmin) for x in lst ]
    df[cols[0]] = df[cols[0]].apply(norm_list)
    df[cols[1]] = df[cols[1]].apply(norm_list)

# --------  One-hot + 强度列 + Padding--------
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')           
TF_TO_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}  
VEC_WIDTH = len(TF_TO_INDEX) + 1                       

def one_hot_encode_with_binding(sequence: str, tf_bindings: List[float]) -> np.ndarray:
    L = len(sequence) if isinstance(sequence, str) else 0
    mat = np.zeros((L, VEC_WIDTH), dtype=float)
    if L == 0:
        return mat
    for i, ch in enumerate(sequence):
        if ch == 'n':
            continue
        if ch in TF_TO_INDEX:
            mat[i, TF_TO_INDEX[ch]] = 1.0
        if tf_bindings and i < len(tf_bindings):
            mat[i, -1] = float(tf_bindings[i])
        else:
            mat[i, -1] = 0.0
    return mat

def pad_sequences_custom(mats: List[np.ndarray], max_length: int) -> List[np.ndarray]:
    padded = []
    for M in mats:
        L = M.shape[0]
        if L < max_length:
            pad = np.zeros((max_length - L, VEC_WIDTH), dtype=float)
            padded.append(np.vstack([M, pad]))
        else:
            padded.append(M[:max_length, :])
    return padded

def calculate_max_length(series: pd.Series) -> int:
    return max((len(s) for s in series.astype(str)), default=0)


def main():
    #  载入/初始化 TF 映射 & 区域表
    load_or_initialize_code(TF_CODE_JSON)
    regions = load_regions_file(REGION_FILE)

    #  扫描 TF 目录，汇总峰
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
        raise RuntimeError("未读取到任何 TF 峰数据，请检查目录与文件名格式。")

    all_summits = pd.concat(all_summits_list, ignore_index=True)
    all_summits = all_summits.sort_values(by=['chrom','summit','start','tf_name']).reset_index(drop=True)

    #  映射到区域
    (enh_codes, pro_codes,
     enh_scores, pro_scores,
     enh_bind_lists, pro_bind_lists) = map_summit_to_regions(all_summits, regions)

    regions['enhancer_code'] = enh_codes
    regions['promoter_code'] = pro_codes
    regions['enhancer_score'] = enh_scores
    regions['promoter_score'] = pro_scores
    regions['enhancer_tf_bindings'] = enh_bind_lists
    regions['promoter_tf_bindings'] = pro_bind_lists

    if SAVE_INTERMEDIATE:
        regions.to_csv(STEP1_OUT, index=False)
        print(f"Step1 已保存: {STEP1_OUT}")

    fill_tf_bindings_with_zero_inplace(regions)

    if SAVE_INTERMEDIATE:
        regions.to_csv(STEP2_OUT, index=False)
        print(f"Step2 已保存: {STEP2_OUT}")

    normalize_inplace_global(regions, cols=('enhancer_tf_bindings','promoter_tf_bindings'))

    if SAVE_INTERMEDIATE:
        regions.to_csv(STEP3_OUT, index=False)
        print(f"Step3 已保存: {STEP3_OUT}")

    max_len_en = calculate_max_length(regions['enhancer_code'])
    max_len_pr = calculate_max_length(regions['promoter_code'])

    enh_mats = [one_hot_encode_with_binding(c, b) for c, b in zip(regions['enhancer_code'], regions['enhancer_tf_bindings'])]
    pro_mats = [one_hot_encode_with_binding(c, b) for c, b in zip(regions['promoter_code'], regions['promoter_tf_bindings'])]

    enh_padded = pad_sequences_custom(enh_mats, max_len_en)
    pro_padded = pad_sequences_custom(pro_mats, max_len_pr)

    regions['enhancer_code_padded'] = [json.dumps(m.tolist()) for m in enh_padded]
    regions['promoter_code_padded']  = [json.dumps(m.tolist()) for m in pro_padded]

    #  输出最终文件
    regions.to_csv(FINAL_OUT, index=False)
    print(f" 全流程完成，已输出: {FINAL_OUT}")
    print(f"enhancer 最大长度: {max_len_en}, promoter 最大长度: {max_len_pr}")

    #  持久化 TF 编码映射
    save_code(TF_CODE_JSON)

if __name__ == '__main__':
    main()
