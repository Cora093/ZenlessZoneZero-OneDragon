"""校验 BagelEnter 的节点图：打印全部边、可达性和无入边／无出边节点。

背景：合并 operation_node／node_from 时曾误删 `选择雅努斯 -> 选择高危` 的边，
导致首次进入成功后无路可走，卡在选关页反复重试。教训是「节点名不重复」不够，
还得查「无入边」和「从起点是否可达」。

用法（在仓库根目录）：
    python .workbuddy/tools/check_bagel_enter_graph.py [模块路径]
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TARGET = 'src/zzz_od/application/bagel/bagel_enter.py'


def parse_graph(path: Path) -> tuple[dict[str, set[str]], set[str], str | None]:
    """从 AST 解析 operation_node／node_from，返回 (边集合, 节点集合, 起始节点)。

    Args:
        path: 待解析的操作类文件。
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    edges: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    start: str | None = None

    # 先收集节点名：`@node_from` 写在 `@operation_node` 之前，单遍扫描时节点名还未知。
    node_of_func: dict[str, str] = {}
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for deco in func.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            # 装饰器可能写成 `@operation_node` 或 `@operation.operation_node`。
            deco_name = getattr(deco.func, 'id', None) or getattr(deco.func, 'attr', None)
            if deco_name != 'operation_node':
                continue
            kwargs = {kw.arg: kw.value for kw in deco.keywords}
            if 'name' not in kwargs or not isinstance(kwargs['name'], ast.Constant):
                continue
            name = str(kwargs['name'].value)
            nodes.add(name)
            node_of_func[func.name] = name
            is_start = kwargs.get('is_start_node')
            if isinstance(is_start, ast.Constant) and is_start.value is True:
                start = name

    # 再收集边。
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        node_name = node_of_func.get(func.name)
        if node_name is None:
            continue
        for deco in func.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            deco_name = getattr(deco.func, 'id', None) or getattr(deco.func, 'attr', None)
            if deco_name != 'node_from':
                continue
            kwargs = {kw.arg: kw.value for kw in deco.keywords}
            from_name = kwargs.get('from_name')
            if not isinstance(from_name, ast.Constant):
                continue
            src = str(from_name.value)
            # `status`／`success` 是边上的匹配条件，标出来便于排查「节点存在但走不到」。
            labels = [src]
            if isinstance(kwargs.get('status'), ast.Constant):
                labels.append(f'{src}[status={kwargs["status"].value}]')
            if isinstance(kwargs.get('success'), ast.Constant):
                labels.append(f'{src}[success={kwargs["success"].value}]')
            for label in labels:
                edges[label].add(node_name)
    return edges, nodes, start


def main() -> int:
    """打印边、可达性和孤立节点；有问题时返回非零退出码。"""
    target = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    if not target.is_file():
        print(f'找不到文件：{target}')
        return 2

    edges, nodes, start = parse_graph(target)
    node_names = {n.split('[')[0] for n in nodes}

    print(f'=== 节点（{len(node_names)} 个）===')
    for name in sorted(node_names):
        print(f'  {name}')

    print(f'\n=== 边（{sum(len(v) for v in edges.values())} 条）===')
    for src in sorted(edges):
        for dst in sorted(edges[src]):
            print(f'  {src} -> {dst}')

    targets = {n.split('[')[0] for dsts in edges.values() for n in dsts}
    no_in = sorted(n for n in node_names if n not in targets)
    no_out = sorted(n for n in node_names if not any(
        k.split('[')[0] == n for k in edges
    ))

    reachable: set[str] = set()
    if start is not None:
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            for src, dsts in edges.items():
                if src.split('[')[0] == cur:
                    stack.extend(d.split('[')[0] for d in dsts)
    unreachable = sorted(node_names - reachable) if start is not None else []

    print('\n=== 检查 ===')
    print(f'  起始节点：{start}')
    print(f'  无入边节点：{no_in or "无"}')
    print(f'  无出边节点：{no_out or "无"}')
    print(f'  起点不可达节点：{unreachable or "无"}')

    problems = []
    # 起始节点之外无出边是正常的（终点）；无入边则是可疑的孤立节点。
    suspects = [n for n in no_in if n != start]
    if suspects:
        problems.append(f'无入边节点（可能误删了入边）：{suspects}')
    if start is not None and unreachable:
        problems.append(f'起点不可达：{unreachable}')
    if start is None:
        problems.append('未找到 is_start_node=True 的起始节点')

    if problems:
        print('\n发现问题：')
        for p in problems:
            print(f'  ! {p}')
        return 1
    print('\n节点图检查通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
