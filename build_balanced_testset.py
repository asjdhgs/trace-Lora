import os
import json
import re
import math
import random

# ================= 配置区域 =================
BASE_DIR = "/home/fengxiaoyu/lx/LLAMA_NEW/data_transform/data/raw/total"
INPUT_DIRS = {
    "正常": [os.path.join(BASE_DIR, "normal")],
    "异常": [os.path.join(BASE_DIR, "abnormal")],
}
SAMPLES_PER_CLASS = 500
RANDOM_SEED = 42
OUTPUT_FILE = "/home/fengxiaoyu/lx/LLAMA_NEW/data_transform/data/cooked/prediction/predict_balanced_pure.jsonl"

# ⚠️ 核心：把逻辑写在 Prompt 里，而不是 Python 代码里
# 我们告诉模型我们在 EDA 中发现的规律，让模型自己去比对
SYSTEM_PROMPT = (
    "你是一个专家级调用链分析师。请根据给定的 Trace 统计数据，逻辑推理并判断其是“正常”还是“异常”。\n\n"
    "【判断标准】\n"
    "1. **完整性检查**：正常的 Trace 必须包含完整的调用链路（num_edges >= 9）。如果 num_edges < 9，属于链路中断（异常）。\n"
    "2. **时延检查**：正常的业务处理耗时通常在 **500ms 到 2000ms** 之间。\n"
    "   - 如果 total_latency_ms < 500ms：通常意味着请求未完成即报错返回（异常）。\n"
    "   - 如果 total_latency_ms > 2000ms：通常意味着系统严重超时（异常）。\n\n"
    "【输出要求】\n"
    "请一步步思考，将数据的数值与上述标准进行比对，最后输出结论。\n"
    "结论行必须严格为：“====结论====\n正常” 或 “====结论====\n异常”。"
)
# ===========================================

START_FINISH_RE = re.compile(r"starts\s+at\s*(\d+)\s*ms.*?finishes\s+at\s*(\d+)\s*ms", re.IGNORECASE)

def parse_trace_and_stats(txt: str):
    """只负责计算基础特征，不包含任何判断逻辑"""
    durations = []
    earliest, latest = None, None
    ann_lines = []
    lines = [l for l in txt.splitlines() if l.strip()]
    
    for raw in lines:
        line = raw
        if line.startswith('[') and 'starts at' in line and 'finishes at' in line:
            m = START_FINISH_RE.search(line)
            if m:
                s = int(m.group(1)); f = int(m.group(2))
                dur = max(0, f-s)
                durations.append(dur)
                if line.rstrip().endswith('].'):
                    line = line[:-2] + f", duration={dur} ms]."
                elif line.rstrip().endswith(']'):
                    line = line[:-1] + f", duration={dur} ms]"
                else:
                    line = line + f" (duration={dur} ms)"
                earliest = s if earliest is None else min(earliest, s)
                latest   = f if latest   is None else max(latest, f)
        ann_lines.append(line)

    num_edges = len(durations)
    total = max(0, latest - earliest) if (earliest is not None and latest is not None and latest >= earliest) else sum(durations)
    mx   = max(durations) if durations else 0
    avg  = int(sum(durations)/len(durations)) if durations else 0
    vs   = sorted(durations)
    idx  = max(0, min(len(vs)-1, math.ceil(0.95 * len(vs)) - 1)) if vs else 0
    p95  = int(vs[idx]) if vs else 0
    max_ratio = (mx/total) if total>0 else 0.0
    b_idx     = durations.index(mx) if durations else -1

    header = (
        f"# 统计特征\n"
        f"num_edges={num_edges}\n"
        f"total_latency_ms={total}\n"
        f"max_edge_latency_ms={mx}\n"
        f"mean_edge_latency_ms={avg}\n"
        f"p95_edge_latency_ms={p95}\n"
        f"max_edge_ratio={max_ratio:.4f}\n"
        f"bottleneck_index={b_idx}\n"
    )
    
    return header + "\n" + "\n".join(ann_lines)

def main():
    random.seed(RANDOM_SEED)
    all_samples = []
    print(f"🚀 开始构建纯模型推理数据集...")

    for label_name, dir_list in INPUT_DIRS.items():
        file_paths = []
        for d in dir_list:
            if not os.path.exists(d): continue
            for fn in os.listdir(d):
                if fn.endswith(".txt"):
                    file_paths.append(os.path.join(d, fn))
        
        if SAMPLES_PER_CLASS and len(file_paths) > SAMPLES_PER_CLASS:
            selected_files = random.sample(file_paths, SAMPLES_PER_CLASS)
        else:
            selected_files = file_paths

        for fp in selected_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                
                cooked_content = parse_trace_and_stats(raw)
                
                # 构造 User Input
                # 依然加上 === Data End === 防止续写，但内容里没有任何“提示”
                input_text = (
                    "=== Trace Data Start ===\n"
                    f"{cooked_content}\n"
                    "=== Trace Data End ===\n\n"
                    "请根据 System Prompt 中的标准，分析上述数据的 num_edges 和 total_latency_ms，并给出结论。"
                )

                sample = {
                    "instruction": SYSTEM_PROMPT, # 规则在这里
                    "input": input_text,          # 数据在这里
                    "output": "",
                    "label": label_name,
                    "id": os.path.basename(fp)
                }
                all_samples.append(sample)
            except Exception as e:
                print(f"Error {fp}: {e}")

    random.shuffle(all_samples)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ 生成完毕: {OUTPUT_FILE}")
    print(f"👀 Prompt 预览 (User):\n{all_samples[0]['input'][-200:]}")

if __name__ == "__main__":
    main()