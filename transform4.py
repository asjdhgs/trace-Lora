import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import List
import networkx as nx
import os

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str  # 如果没有，则为None
    children_span_list: List['Span']

    start_time: datetime
    duration: float  # 单位为毫秒
    service_name: str
    anomaly: int = 0  # normal:0/anomaly:1
    status_code: str = None

    operation_name: str = None
    root_cause: bool = None  # True
    latency: int = None  # normal:None/anomaly:1
    structure: int = None  # normal:None/anomaly:1
    extra: dict = None


@dataclass
class Trace:
    trace_id: str
    root_span: Span
    span_count: int = 0
    anomaly_type: int = None  # normal:0/only_latency_anomaly:1/only_structure_anomaly:2/both_anomaly:3


def get_communication_type(operation_name: str) -> str:
    """根据操作名称推断通信类型"""
    if operation_name is None:
        return "UNKNOWN"
    if "HTTP" in operation_name.upper():
        return "HTTP"
    if "grpc" in operation_name.lower():
        return "GRPC"
    if "DB" in operation_name.upper() or "SQL" in operation_name.upper():
        return "DATABASE"
    return "RPC"


def format_edge(span: Span, parent_span: Span = None) -> str:
    """将单个 Span 转换为论文要求的 Edge 编码格式"""
    edge_id = span.span_id
    source = parent_span.service_name if parent_span else "Client"
    destination = span.service_name
    communication_type = get_communication_type(span.operation_name)
    start_time = int(span.start_time.timestamp() * 1000)  # 转毫秒
    finish_time = start_time + int(span.duration)

    return (f"[Edge ID is {edge_id}, Source is {source}, Destination is {destination}, "
            f"Type is {communication_type}, Communication starts at {start_time} ms, "
            f"Communication finishes at {finish_time} ms].")


def extract_spans(span: Span, spans=None):
    """递归提取 Span 树中的所有 Span"""
    if spans is None:
        spans = []
    spans.append(span)
    for child_span in span.children_span_list:
        extract_spans(child_span, spans)
    return spans


def build_dependency_graph(spans: List[Span]) -> nx.DiGraph:
    """构建跨 Trace 的依赖关系图 (Dependency Graph)"""
    graph = nx.DiGraph()

    # 添加节点
    for span in spans:
        graph.add_node(span.span_id, span=span)

    # 添加父子依赖关系
    for span in spans:
        if span.parent_span_id and span.parent_span_id in graph.nodes:
            graph.add_edge(span.parent_span_id, span.span_id, type='parent')

    # 添加时间邻接依赖关系
    sorted_spans = sorted(spans, key=lambda s: s.start_time)
    for i in range(len(sorted_spans) - 1):
        current_span = sorted_spans[i]
        next_span = sorted_spans[i + 1]
        time_gap = (next_span.start_time - current_span.start_time).total_seconds()
        if 0 < time_gap <= 2:  # 两个 Span 间隔小于2秒视为有关联
            graph.add_edge(current_span.span_id, next_span.span_id, type='time')

    return graph


def generate_sequence(graph: nx.DiGraph) -> List[str]:
    """根据依赖关系图生成完整的 Trace 序列"""
    try:
        ordered_nodes = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        print("检测到循环依赖！")
        ordered_nodes = list(graph.nodes)

    sequence = []
    for node_id in ordered_nodes:
        span = graph.nodes[node_id]['span']
        parent_node = list(graph.predecessors(node_id))
        parent_span = graph.nodes[parent_node[0]]['span'] if parent_node else None
        sequence.append(format_edge(span, parent_span))
    return sequence


def main(pkl_file, output_dir):
    """读取 PKL 文件，生成多条 Trace 的序列并输出为论文格式"""
    with open(pkl_file, 'rb') as f:
        traces = pickle.load(f)

    # 创建输出目录（如果没有的话）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 针对每个 Trace，生成对应的文本文件
    for trace in traces:
        output_file = os.path.join(output_dir, f"{trace.trace_id}.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Trace ID is {trace.trace_id}\n")
            f.write("<Trace Sequence>\n")

            # 提取所有 Span
            spans = extract_spans(trace.root_span)

            # 构建跨 Trace 的依赖关系图
            dependency_graph = build_dependency_graph(spans)

            # 生成序列
            sequence_lines = generate_sequence(dependency_graph)

            for line in sequence_lines:
                f.write(line + "\n")

            f.write("</Trace Sequence>\n")
            f.write("=" * 80 + "\n")

        print(f"调用链序列已生成并保存到 {output_file}")



pkl_file = 'D:/文档/trace项目/total/train_normal.pkl'       # 输入 PKL 文件
output_dir = '../transformed_data/total_transform4/train_normal'  # 输出文件夹
main(pkl_file, output_dir)
