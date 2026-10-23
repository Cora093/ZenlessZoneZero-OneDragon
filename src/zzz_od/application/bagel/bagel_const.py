APP_ID: str = 'bagel'
APP_NAME: str = '贝果计划'
DEFAULT_GROUP: bool = True
# 排在驱动盘拆解之后、通知之前。数值越小越靠前。
PRIORITY: int = 1800
NEED_NOTIFY: bool = False
MAP_TITLE: str = '高危雅努斯幻境'
RECOMMENDED_VALUE: str = '258000'

# 距节点超过此像素按住前进；以内改为短步。电子保险箱最后一段始终碎步。
NAV_CRUISE_DISTANCE: float = 10
# 电子保险箱巷口拐点在这个距离内就切下一段。切晚了会斜穿停车位；切得太早会在巷口原地大转弯，小地图对不上。
NAV_SAFE_CORNER_DISTANCE: float = 12
# 楼角和贴墙中间路点的切段距离。
NAV_SAFE_APPROACH_DISTANCE: float = 2
# 最后接近点前提前松键，给截图延迟与停步前移留出余量；横向仍须贴墙。
NAV_SAFE_BRAKE_DISTANCE: float = 6
NAV_SAFE_BRAKE_WAIT: float = 0.5
# 朝向偏差（度）：以内不转；不超过停车阈值时边走边转；再大则松键停车转。
NAV_TURN_DEADBAND: float = 7
NAV_CRUISE_TURN_LIMIT: float = 45
NAV_CRUISE_TURN_CAP: float = 30
# 边走边转的最小间隔（秒）。间隔内只按住前进，避免转向还没落地又往反方向补。
NAV_CRUISE_TURN_GAP: float = 0.4
NAV_STOP_TURN_CAP: float = 90
# 短步、校准、电子保险箱最后碎步的按下时长（秒）。
NAV_FORWARD_PRESS: float = 0.2
NAV_ALIGN_PRESS: float = 0.08
NAV_SAFE_APPROACH_PRESS: float = 0.08
# 首次导航位置尚未取得时，只对几何证据不足松键等待。
NAV_INITIAL_LOCATE_MISS_LIMIT: int = 5
NAV_INITIAL_LOCATE_WAIT: float = 0.3
# 已经定位成功后，允许连续几帧对不上再停。单帧遮挡不再结束本段。
NAV_LOCATE_MISS_LIMIT: int = 3

# 电子保险箱完成四轮点按后等待搜索结果，点按时机按光圈状态判断。
SAFE_UNLOCK_HITS: int = 4

# 容器类型 -> 搜查面板标题区域。三种标题在同一位置、不同文案，必须逐个判：
# 面板已开时目标名可能对不上（标题渲染慢一帧），只看目标那一种会漏判成「无面板」。
CONTAINER_TITLE_AREAS: dict[str, str] = {
    'box': '搜查容器标题',
    'safe': '电子保险箱标题',
    'mech': '机械保险箱标题',
}
# 容器类型 -> 局内交互提示区域。
CONTAINER_PROMPT_AREAS: dict[str, str] = {
    'box': '武备箱交互',
    'safe': '电子保险箱交互',
    'mech': '机械保险箱交互',
}
# 需要按住交互键才开箱的容器（按不够箱子不响应）；其余容器点按即可。
CONTAINER_HOLD_TYPES: frozenset[str] = frozenset({'mech'})
# 机械保险箱要按住 3.5 秒才开 —— 按不够箱子完全不响应，会被当成开箱失败。
# 恢复预算 30 秒、交互上限 3 次，三次按压共 10.5 秒，预算够用。
MECH_HOLD_PRESS_TIME: float = 3.5
# 容器类型 -> 到达该容器时返回的任务状态。导航与流程执行器都要用同一份，
# 两边各写一份必然对不上（会表现成「明明到了却被判成没到」）。
CONTAINER_ARRIVE_STATUS: dict[str, str] = {
    'box': '已到武备箱',
    'safe': '已到电子保险箱',
    'mech': '已到机械保险箱',
}
# 容器类型 -> 中文名，用于操作名与提示文案。
CONTAINER_LABELS: dict[str, str] = {
    'box': '武备箱',
    'safe': '电子保险箱',
    'mech': '机械保险箱',
}

# 仓库清理：default 用下面的固定组合；custom 按 clean_steps 顺序，每步用该步的类型和品质。
CLEAN_MODE_DEFAULT: str = 'default'
CLEAN_MODE_CUSTOM: str = 'custom'
CLEAN_TYPES: tuple[str, ...] = ('贵重物品', '战术棱镜', '其他', '装备', '战术道具', '门禁卡')
CLEAN_QUALITIES: tuple[str, ...] = ('C', 'B', 'A', 'S', 'Z')
DEFAULT_CLEAN_TYPES: tuple[str, ...] = ('贵重物品', '战术棱镜', '其他')
DEFAULT_CLEAN_QUALITIES: tuple[str, ...] = ('C', 'B', 'A', 'S')
