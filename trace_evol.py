# tools/synth/trace_evol.py
import os, re, json, math, random, hashlib
from typing import List, Dict, Any, Tuple

random.seed(2025)

# ====== 你的目录：可以混合真实与合成 ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIRS: Dict[str, List[str]] = {
    "正常": [
        os.path.join(BASE_DIR, "data/raw/total/normal"),   # 真实
        #os.path.join(BASE_DIR, "data/synth/raw/normal"),   # 合成
    ],
    "异常": [
        os.path.join(BASE_DIR, "data/raw/total/abnormal"), # 真实
        #os.path.join(BASE_DIR, "data/synth/raw/abnormal"), # 合成
    ],
}

OUT_DIR = os.path.join(BASE_DIR, "data/cooked/evol")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON  = os.path.join(OUT_DIR, "trace_evol_sharegpt4.json")
OUT_JSONL = os.path.join(OUT_DIR, "trace_evol_sharegpt4.jsonl")

MAX_PER_CLASS = 500
RANDOM_SEED   = 2025

# ====== 正则与工具 ======
START_FINISH_RE = re.compile(
    r"starts\s+at\s*(\d+)\s*ms.*?finishes\s+at\s*(\d+)\s*ms",
    re.IGNORECASE
)

INSTRUCTION_CORE = "你是一个调用链异常检测器。请根据给定的特征和调用链，仅输出“正常”或“异常”，不要输出其它任何文字。"

def parse_trace_and_stats(txt: str) -> Dict[str, Any]:
    durations = []
    earliest, latest = None, None
    ann_lines = []
    for raw in txt.splitlines():
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
    total = max(0, (latest - earliest)) if (earliest and latest and latest>=earliest) else sum(durations)
    mx   = max(durations) if durations else 0
    avg  = int(sum(durations)/len(durations)) if durations else 0
    vs   = sorted(durations)
    idx  = max(0, min(len(vs)-1, math.ceil(0.95 * len(vs)) - 1)) if vs else 0
    p95  = int(vs[idx]) if vs else 0

    # 辅助特征
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
        f"bottleneck_index={b_idx}\n\n"
    )
    cooked_input = header + "\n".join(ann_lines)
    return {
        "durations": durations,
        "max_edge": mx,
        "bottleneck_index": b_idx,
        "max_edge_ratio": max_ratio, 
        "cooked_user": cooked_input,
        "num_edges": num_edges,       
        "total_latency_ms": total
    }

# ====== Evol 问题模板：训练期“多风格指令” ======
# 最终标签仍只输出“正常/异常”，只是训练时让模型习惯不同问法
QUESTION_POOL = [
    "请判断该调用链是否异常，只输出“正常”或“异常”。",
    "根据统计特征与调用序列，判断该请求是否异常（仅输出两个字）。",
    "请依据瓶颈与延迟特征，判断是否异常（正常/异常）。",
    "只输出分类结果：正常 或 异常。",
    "判断：正常 还是 异常？（不要输出其它内容）",
]

def build_sharegpt(sample_txt: str, label: str) -> Dict[str, Any]:
    st = parse_trace_and_stats(sample_txt)
    q = random.choice(QUESTION_POOL)
    
    system_prompt = "你是一个专家级调用链分析师。请先分析Trace的特征，最后给出“正常”或“异常”的结论。"
    user_content = st["cooked_user"] 

    # === 动态生成分析文案（核心修改）===
    
    # 1. 分析链路长度
    if st['num_edges'] <= 2:
        analysis_len = f"链路节点数 num_edges={st['num_edges']}，链路过短。"
    else:
        analysis_len = f"链路节点数 num_edges={st['num_edges']}，结构完整。"

    # 2. 分析耗时瓶颈 (根据真实数据说话，不撒谎)
    ratio = st['max_edge_ratio']
    if ratio > 0.9:
        analysis_ratio = f"发现第 {st['bottleneck_index']} 条调用耗时占比高达 {ratio*100:.1f}%，存在显著的单点耗时。"
    elif ratio > 0.5:
        analysis_ratio = f"存在主要耗时节点，占比 {ratio*100:.1f}%。"
    else:
        analysis_ratio = "各节点耗时分布相对均匀。"

    # 3. 生成最终推理逻辑 (这里才是区分正常/异常的关键)
    if label == "异常":
        # 异常的理由：要么短，要么慢且非核心业务，要么报错(虽然这里没体现)
        # 既然数据里都是 ratio=1.0，说明区别可能在于 total_latency 或者链路结构
        if st['num_edges'] <= 2:
            reason = "结合极短链路特征，判定为异常中断。"
        else:
            reason = "尽管链路完整，但结合高延迟与瓶颈特征，判定为性能异常。"
            
        explanation = (
            f"分析：\n"
            f"1. {analysis_len}\n"
            f"2. {analysis_ratio}\n"
            f"3. {reason}\n"
            f"====结论====\n异常"
        )
    else:
        # 正常的理由：即使慢，也是正常的慢
        reason = "该长耗时节点属于核心业务逻辑，符合预期，整体链路正常。"
        
        explanation = (
            f"分析：\n"
            f"1. {analysis_len}\n"
            f"2. {analysis_ratio}\n" # 即使是正常样本，如果ratio高，也要承认它高！
            f"3. {reason}\n"
            f"====结论====\n正常"
        )
    # =================================

    return {
        "conversations": [
            {"from": "system",    "value": system_prompt},
            {"from": "user",      "value": user_content + "\n" + q},
            {"from": "assistant", "value": explanation}
        ]
    }

def iter_txt_files(dirs: List[str]):
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".txt"):
                yield os.path.join(d, fn)

def main():
    random.seed(RANDOM_SEED)
    recs: List[Dict[str, Any]] = []
    for label, dirs in INPUT_DIRS.items():
        c = 0
        for p in iter_txt_files(dirs):
            if c >= MAX_PER_CLASS:
                break
            try:
                txt = open(p, "r", encoding="utf-8").read().strip()
                recs.append(build_sharegpt(txt, label))
                c += 1
            except Exception as e:
                print(f"[WARN] skip {p}: {e}")

    random.shuffle(recs)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid_endings = 0
    for r in recs:
        output_text = r["conversations"][-1]["value"]
        # 检查是否包含结论分隔符，且以正常或异常结尾
        if "====结论====\n正常" in output_text or "====结论====\n异常" in output_text:
            valid_endings += 1
            
    print(f"[OK] total={len(recs)} -> {OUT_JSON}")
    print(f"[OK] Valid format count: {valid_endings}/{len(recs)}")
    print(f"[OK] JSONL -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
