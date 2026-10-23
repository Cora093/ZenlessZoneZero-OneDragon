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

# 到达后转向时，连续多少帧读不到角色箭头就放弃转向、改为按计时走完。
# 局内画面被面板或特效盖住时箭头会短暂消失，容忍几帧；持续读不到则不该把整条
# 路线卡在这里。
NAV_HOOK_HEADING_MISS_LIMIT: int = 3
# 到达后转向单独用更紧的判定：转角常是 90 度量级，巡航那套 7 度死区会让
# 「差 6 度」就算转到位，走十几格后横向偏出两格，蹭到墙或箱子。
# 3 度足够小，转不到位就继续补；上限放宽到 120 度，让 90 度转角一次转完，
# 不必拆成两次（拆开后余量落在死区里会被误判成已完成）。
NAV_ARRIVE_TURN_DEADBAND: float = 3.0
NAV_ARRIVE_TURN_CAP: float = 120.0
# 到达后转向最多补几轮。小地图上的箭头是角色朝向，转镜头不会立刻改变它，
# 要靠角色继续走一步、朝向跟上来才读得到新角度；万一读到的角度一直不变
# （角色被卡住、面板挡住箭头等），到轮数上限就放弃转向按原逻辑走完，
# 不能把整条路线卡在这里。
NAV_ARRIVE_TURN_MAX_ROUNDS: int = 12
# 到达后动作每轮间隔（秒）。既是转向的检查节奏，也是位移的计时步长。
NAV_HOOK_STEP_WAIT: float = 0.2
# 到达后动作可以按住哪些移动键。这些键只按住不放（press 不带时长），
# 角色在移动中朝向才会跟上来 —— 停在原地转镜头的话，小地图上的角色朝向
# 箭头根本不会变，闭环永远收敛不了。
NAV_ARRIVE_MOVE_KEYS: frozenset[str] = frozenset({'move_w', 'move_a', 'move_s', 'move_d'})

# 出生识别兜底：入场动画未结束时定位小地图偏暗，可用特征不足，任何底图都定位不了。
# 实测此时有效像素 2282，正常帧 4800 以上，此时退出重开会白白浪费一局。
# 画面偏暗且全图未命中时先等几帧；画面正常仍未命中说明确实未建档，按原逻辑退出。
# 「偏暗」只是候选条件：稳定态的未建档布局同样落在门槛下方，靠下面的帧间差收敛
# 判据提前退出，不能只靠重试次数耗尽。
SPAWN_DIM_VALID_PIXELS: int = 3000
SPAWN_DIM_RETRY_LIMIT: int = 6
# 连续多少帧定位小地图的平均灰度差小于下面这个值，即认为画面已停止重绘。入场
# 过渡态每帧都在刷新（有效像素从 2282 一路涨到 4800 以上），稳定后差值趋近 0，
# 两者相差 20 倍以上。要求连续两帧都收敛：加载卡顿时过渡期可能恰好停住一两帧，
# 只看单帧会把仍在入场的一局误杀。
SPAWN_STABLE_MAX_DIFF: float = 1.5
SPAWN_STABLE_FRAMES: int = 2
SPAWN_STABLE_WAIT: float = 0.3

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

# 长按前进时是否顺带点一下冲刺键。冲刺键与闪避共用（键表里 dodge 默认 shift）：
# 探索跑图时是冲刺，战斗中是闪避，贝果跑图流程本身不按闪避所以可以开。
# 一次点按后游戏会保持冲刺状态，不需要按住不放；每个路点只点一次，理由见
# 导航侧 `_start_forward` 的说明。
NAV_SPRINT_ENABLED: bool = True

# 「原地开始」：先读小地图，认得出支持出生点就直接进该路线流程，不重开。
START_IN_PLACE_HUD_MISS_LIMIT: int = 4
START_IN_PLACE_HUD_WAIT: float = 0.3
# 认出的位置离出生点多远仍算「在出生点附近」。超出说明已经走远了，
# 直接按该路线跑会从错误位置起步，不如照常重开。
START_IN_PLACE_SPAWN_RADIUS: float = 12.0

# 仓库清理：default 用下面的固定组合；custom 按 clean_steps 顺序，每步用该步的类型和品质。
CLEAN_MODE_DEFAULT: str = 'default'
CLEAN_MODE_CUSTOM: str = 'custom'
CLEAN_TYPES: tuple[str, ...] = ('贵重物品', '战术棱镜', '其他', '装备', '战术道具', '门禁卡')
CLEAN_QUALITIES: tuple[str, ...] = ('C', 'B', 'A', 'S', 'Z')
DEFAULT_CLEAN_TYPES: tuple[str, ...] = ('贵重物品', '战术棱镜', '其他')
DEFAULT_CLEAN_QUALITIES: tuple[str, ...] = ('C', 'B', 'A', 'S')
