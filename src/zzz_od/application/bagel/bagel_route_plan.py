# -*- coding: utf-8 -*-
"""按底图可走区域给路线打补丁：每条腿规划一条离墙够远的最短路。

底图是 1:1 的地图画布（定位就是拿小地图在它上面做模板匹配），alpha 通道即可走区域。
实测 5 条路线 64 个点全部落在不透明像素上、59 条腿全部连通，且每条腿都能规划出
离墙 ≥6px 的路径（端点 5px 内放宽），而原始直线在 17 条腿上只有 3~5px 净空。
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np

from one_dragon.utils.log_utils import log

DEVIATION_LIMIT: float = 5.0
"""允许偏离原路线多少像素（px）：超过就禁止。原路线常常是故意贴墙走的
（走廊窄、绕怪、切内角），所以不许为了「居中」把路线推到走廊中间去。"""

CLEAR_FLOOR: float = 1.5
"""离墙的最小硬下限（px）：只用来避免像素级贴边判死，不做居中优化。"""

END_EXEMPT: float = 5.0
"""端点附近这么远内放宽约束（px）：路线点本身常常就离墙 3~5px。"""

MARGIN: int = 26
"""A* 裁剪边距（px）。"""

PLAN_VERSION: int = 3


def _layer_dir() -> Path:
    from zzz_od.game_data.map_locator.map_locator import default_layer_dir
    return default_layer_dir()


def load_layer_masks() -> dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    """读全部可走区域掩码：(可走掩码, 距离变换, 原点)。"""
    directory = _layer_dir()
    data = json.loads((directory / 'layers.json').read_text(encoding='utf-8'))
    out = {}
    for item in data['layers']:
        key = f"{item['map_id']}/{item['name']}"
        image = cv2.imread(str(directory / item['image']), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            continue
        walk = (image[:, :, 3] > 0).astype(np.uint8)
        out[key] = (walk, cv2.distanceTransform(walk, cv2.DIST_L2, 5),
                    (int(item['origin'][0]), int(item['origin'][1])))
    return out


def _shortcut(points: list[tuple[float, float]], walk, dist, origin,
              leg_start: tuple[float, float], leg_end: tuple[float, float]
              ) -> list[tuple[float, float]]:
    """把 A* 的 45 度锯齿拉直：能一眼望到头就直接连，只留必要拐点。

    判据只看「可走」和「离原路线不超过 DEVIATION_LIMIT」——不追求居中。
    """
    def clear(p, q) -> bool:
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        steps = max(int(length), 1)
        for i in range(steps + 1):
            t = i / steps
            sx_, sy_ = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
            px = int(round(sx_ - origin[0]))
            py = int(round(sy_ - origin[1]))
            if not (0 <= py < walk.shape[0] and 0 <= px < walk.shape[1]):
                return False
            if walk[py, px] == 0:
                return False
            # 端点和顶点本身允许贴墙；中段只要不贴到像素级就行。
            vx, vy = px + origin[0], py + origin[1]
            near_end = (math.hypot(vx - leg_start[0], vy - leg_start[1]) <= END_EXEMPT
                        or math.hypot(vx - leg_end[0], vy - leg_end[1]) <= END_EXEMPT)
            if not near_end and _deviation(vx, vy, leg_start, leg_end) > DEVIATION_LIMIT:
                return False
            if float(dist[py, px]) < CLEAR_FLOOR:
                return False
        return True

    if len(points) <= 2:
        return points
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not clear(points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


def _deviation(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    """点到「原路线这条线段」的距离（px）。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:
        return math.hypot(x - a[0], y - a[1])
    t = ((x - a[0]) * dx + (y - a[1]) * dy) / seg2
    t = max(0.0, min(1.0, t))
    return math.hypot(x - (a[0] + dx * t), y - (a[1] + dy * t))


def _deviation_grid(xs, ys, a: tuple[float, float], b: tuple[float, float]):
    """整张裁剪网格到原路线线段的距离（向量化版本）。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:
        return np.hypot(xs - a[0], ys - a[1])
    t = ((xs - a[0]) * dx + (ys - a[1]) * dy) / seg2
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(xs - (a[0] + dx * t), ys - (a[1] + dy * t))


def plan_leg(walk, dist, origin, a, b) -> list[tuple[float, float]] | None:
    """A* 规划一条从 a 到 b 的路径：以原路线为基准，只做最小偏移把它变成可走。

    代价 = 步长 ×（1 + 偏离原路线的惩罚），并在允许掩码里硬性禁止「不可走」和
    「偏离超过 DEVIATION_LIMIT」的像素。所以走廊够宽时结果就是原直线，
    只在原直线真的被墙切到时才会绕出一小段。
    """
    ax, ay = int(round(a[0] - origin[0])), int(round(a[1] - origin[1]))
    bx, by = int(round(b[0] - origin[0])), int(round(b[1] - origin[1]))
    x0 = max(0, min(ax, bx) - MARGIN)
    y0 = max(0, min(ay, by) - MARGIN)
    x1 = min(walk.shape[1], max(ax, bx) + MARGIN + 1)
    y1 = min(walk.shape[0], max(ay, by) + MARGIN + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    sub_walk = walk[y0:y1, x0:x1]
    sub_dist = dist[y0:y1, x0:x1]
    h, w = sub_walk.shape
    sx, sy, gx, gy = ax - x0, ay - y0, bx - x0, by - y0
    if not (0 <= sy < h and 0 <= sx < w and 0 <= gy < h and 0 <= gx < w):
        return None
    ys, xs = np.mgrid[0:h, 0:w]
    near_end = (((xs - sx) ** 2 + (ys - sy) ** 2 <= END_EXEMPT ** 2)
                | ((xs - gx) ** 2 + (ys - gy) ** 2 <= END_EXEMPT ** 2))
    # 每个像素到「原路线这条线段」的距离：超过 DEVIATION_LIMIT 一律不许走。
    dev = _deviation_grid(xs + x0 + origin[0], ys + y0 + origin[1], a, b)
    allowed = (sub_walk > 0) & (near_end | ((sub_dist >= CLEAR_FLOOR) & (dev <= DEVIATION_LIMIT)))
    if not allowed[sy, sx] or not allowed[gy, gx]:
        return None
    best = np.full((h, w), np.inf, dtype=np.float64)
    best[sy, sx] = 0.0
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(0.0, 0.0, sx, sy)]
    neigh = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.41421), (1, -1, 1.41421), (-1, 1, 1.41421), (-1, -1, 1.41421))
    while heap:
        _f, g, x, y = heapq.heappop(heap)
        if x == gx and y == gy:
            break
        if g > best[y, x] + 1e-9:
            continue
        for dx, dy, step in neigh:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h or not allowed[ny, nx]:
                continue
            # 偏离原路线越多代价越高（平方惩罚）：能在原线附近解决就绝不往外绕。
            penalty = (float(dev[ny, nx]) / DEVIATION_LIMIT) ** 2
            ng = g + step * (1.0 + 3.0 * penalty)
            if ng + 1e-9 < best[ny, nx]:
                best[ny, nx] = ng
                parent[(nx, ny)] = (x, y)
                heapq.heappush(heap, (ng + math.hypot(gx - nx, gy - ny), ng, nx, ny))
    if not np.isfinite(best[gy, gx]):
        return None
    path = [(gx, gy)]
    cur = (gx, gy)
    while cur != (sx, sy):
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    global_path = [(x + x0 + origin[0], y + y0 + origin[1]) for x, y in path]
    return _shortcut(global_path, walk, dist, origin, a, b)


SMOOTH_SPACING: float = 4.0
"""圆弧采样的间距（px）：与定位精度、前视距离同一量级。"""

SMOOTH_MIN_SCALE: float = 0.25
"""顶点切线下限：折角越大切线越短，避免曲线甩进墙里。"""


def _turn_scale(prev_pt, vertex, next_pt) -> float:
    """顶点折角对应的切线缩放：直行 1.0，折角越大越小。"""
    ax, ay = vertex[0] - prev_pt[0], vertex[1] - prev_pt[1]
    bx, by = next_pt[0] - vertex[0], next_pt[1] - vertex[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na <= 0 or nb <= 0:
        return SMOOTH_MIN_SCALE
    cos_v = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return max(SMOOTH_MIN_SCALE, math.cos(math.acos(cos_v) / 2.0))


def smooth_leg(prev_pt, a, b, next_pt, spacing: float = SMOOTH_SPACING
               ) -> list[tuple[float, float]]:
    """把 a→b 换成经过 a、b 的圆弧（Hermite/Catmull-Rom 采样）。

    曲线严格经过两个路线点，所以到点判据不用改；切线长度按相邻腿长和折角缩放，
    折角越大越接近直线——这样 60~90 度的转角变成一段弧，控制器按住前进跟着切线转就能过去，
    不用停下来"停车转正"。
    """
    scale_a = _turn_scale(prev_pt, a, b)
    scale_b = _turn_scale(a, b, next_pt)
    seg = math.hypot(b[0] - a[0], b[1] - a[1])
    if seg <= 0:
        return [a, b]
    total_a = math.hypot(b[0] - prev_pt[0], b[1] - prev_pt[1])
    total_b = math.hypot(next_pt[0] - a[0], next_pt[1] - a[1])
    # 切线长度不超过相邻腿长的一半，避免长短腿交替时过冲
    tan_a = min(scale_a * 0.5 * total_a, 0.5 * seg)
    tan_b = min(scale_b * 0.5 * total_b, 0.5 * seg)
    m1 = ((b[0] - prev_pt[0]) / total_a * tan_a,
          (b[1] - prev_pt[1]) / total_a * tan_a) if total_a > 0 else (0.0, 0.0)
    m2 = ((next_pt[0] - a[0]) / total_b * tan_b,
          (next_pt[1] - a[1]) / total_b * tan_b) if total_b > 0 else (0.0, 0.0)
    steps = max(2, int(seg / max(spacing, 1.0)))
    out: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        out.append((
            h00 * a[0] + h10 * m1[0] + h01 * b[0] + h11 * m2[0],
            h00 * a[1] + h10 * m1[1] + h01 * b[1] + h11 * m2[1],
        ))
    out[0] = (a[0], a[1])
    out[-1] = (b[0], b[1])
    return out


def _leg_walkable(path, walk, origin) -> bool:
    for x, y in path:
        px, py = int(round(x - origin[0])), int(round(y - origin[1]))
        if not (0 <= py < walk.shape[0] and 0 <= px < walk.shape[1]):
            return False
        if walk[py, px] == 0:
            return False
    return True


def plan_route(points: list[dict], masks) -> dict[str, list[list[float]]]:
    """整条路线做圆弧过渡：返回 {腿序号: [[x, y], ...]}（含两端路线点）。

    弧线落进墙里就退回直线；直线也不可走时才交给 A* 规划（那是真的被墙挡住）。
    """
    pts = [(float(p['x']), float(p['y'])) for p in points]
    if len(pts) < 2:
        return {}
    planned: dict[str, list[list[float]]] = {}
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        layer = points[i]['layer']
        if layer != points[i + 1]['layer']:
            continue
        entry = masks.get(layer)
        if entry is None:
            continue
        walk, dist, origin = entry
        prev_pt = pts[i - 1] if i > 0 else a
        next_pt = pts[i + 2] if i + 2 < len(pts) else b
        path = smooth_leg(prev_pt, a, b, next_pt)
        if not _leg_walkable(path, walk, origin):
            path = [a, b]
        if not _leg_walkable(path, walk, origin):
            planned_path = plan_leg(walk, dist, origin, a, b)
            if planned_path is None:
                continue
            path = planned_path
        planned[str(i)] = [[round(x, 1), round(y, 1)] for x, y in path]
    return planned


def _points_signature(points: list[dict]) -> str:
    """这条路线点位内容的指纹。

    缓存只认点数是不够的：路线点数不变、坐标改了（重新录制同一张图的路线很容易出现），
    旧计划会被整条复用——每腿只把末点纠正回新坐标，中间的规划点全是旧的。
    """
    raw = json.dumps([[p['x'], p['y']] for p in points], sort_keys=True)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def load_plan(route_id: str, points: list[dict], cache_dir: Path) -> dict[int, list[tuple[float, float]]]:
    """取这条路线的规划结果：有缓存用缓存，没有就现算并写缓存。"""
    cache_file = cache_dir / f'{route_id}.json'
    signature = _points_signature(points)
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            if (data.get('version') == PLAN_VERSION
                    and data.get('signature') == signature):
                return {int(k): [(float(p[0]), float(p[1])) for p in v]
                        for k, v in data['legs'].items()}
        except Exception:
            log.warning('读路线规划缓存失败，重新规划', exc_info=True)
    try:
        masks = load_layer_masks()
    except Exception:
        log.warning('读地图可走区域失败，本局按原直线走', exc_info=True)
        return {}
    planned = plan_route(points, masks)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            'version': PLAN_VERSION,
            'route_id': route_id,
            'points': len(points),
            'signature': signature,
            'deviation_limit': DEVIATION_LIMIT,
            'clear_floor': CLEAR_FLOOR,
            'legs': planned,
        }, ensure_ascii=False, indent=1), encoding='utf-8')
    except Exception:
        log.warning('写路线规划缓存失败', exc_info=True)
    return {int(k): [(float(p[0]), float(p[1])) for p in v] for k, v in planned.items()}
