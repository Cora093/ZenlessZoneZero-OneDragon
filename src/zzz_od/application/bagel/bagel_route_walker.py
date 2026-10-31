"""贝果路线执行状态机：按打点顺序走到最后。

入场后先识别出生点：连续 2 次新截图命中同一个已登记出生点才确认，最多
观察 10 秒。正式运行执行该出生点当前开启的路线；试跑执行指定路线，但角色
也必须站在该路线绑定的出生点。确认后从路线第 1 点开始，游标顺序推进：

- 下一格是途径点：正常导航，进入 3px 到达半径即算路过。
- 下一格是箱类：按距离慢步或碎步靠近。当前目标是电子保险箱时，进入
  15px 后保持 0.08 秒短步；可信提示出现后松键，按类型交互。
- 路线走完：原地判断现场有无交互物，没有就结束；由应用层走既有退出结算链。

出生点不在已登记范围内、或出生点没有开启路线时，返回 ``STATUS_ANCHOR_MISS``，
由应用层退出重开。

导航与交互复用旧实现里与旧流程解耦的部分：分层定位、朝向识别、转向
补偿、光圈解锁、逐件入箱、关闭面板。旧流程（flow.yml 步骤、写死的
动作模板）不再参与。
"""

from __future__ import annotations

import os
import time
from collections import deque
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import cal_utils, cv2_utils, str_utils
from one_dragon.utils.i18_utils import gt
from one_dragon.utils.log_utils import log
from zzz_od.application.bagel.bagel_close_search import BagelCloseSearch
from zzz_od.application.bagel.bagel_input_worker import (
    TURN_DEADBAND,
    TURN_DEADBAND_FAST,
    BagelInputWorker,
)
from zzz_od.application.bagel.bagel_operation import BagelOperation
from zzz_od.application.bagel.bagel_route_model import (
    BEACON,
    ELEVATOR,
    ELEVATOR_EXIT_BACKWARD,
    BagelRoute,
    BagelRoutePoint,
    BagelRouteRef,
    BagelSpawnPoint,
    check_route_start,
)
from zzz_od.application.bagel.bagel_route_recorder import is_foreground
from zzz_od.application.bagel.bagel_route_store import (
    BagelRouteConfigError,
    BagelRouteStore,
)
from zzz_od.application.bagel.bagel_spawn import (
    SPAWN_AMBIGUOUS,
    SPAWN_HIT,
    SPAWN_MISS,
    BagelSpawnMatch,
    match_spawn,
)
from zzz_od.application.bagel.bagel_stats import (
    BagelSafeLedger,
    format_value,
    parse_value,
)
from zzz_od.application.bagel.bagel_store import BagelStoreSafe
from zzz_od.application.bagel.bagel_unlock_safe import BagelUnlockSafe
from zzz_od.game_data.map_locator.map_locator import (
    MultiMapLocator,
    load_map_locators,
)
from zzz_od.game_data.map_locator.minimap_angle import analyze

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.matcher.ocr.ocr_match_result import OcrMatchResult
    from one_dragon.base.operation.operation_base import OperationResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

ELEVATOR_OPTION_TAP_SECONDS: float = 1.0
"""电梯选项还没出现时，点一次屏幕中间推进对话的间隔（用户给定每秒一次）。"""

ELEVATOR_ENTRY_TIMEOUT: float = 10.0
"""从电梯门口往前走多久还没进电梯就判失败（秒）。

只限制「门口走向电梯」这一段，此时还在场景里，失败后能正常退出。
进电梯之后的三段一律不设超时：卡在里面连游戏都退不出来。
"""

ELEVATOR_PHASE_ENTER: str = 'enter'
ELEVATOR_PHASE_WAIT_OPTION: str = 'wait_option'
ELEVATOR_PHASE_CLICK_OPTION: str = 'click_option'
ELEVATOR_PHASE_EXIT: str = 'exit'
"""电梯四个阶段的标识：进门、等选项、点选项、出电梯。"""

SPAWN_CONFIRM_FRAMES: int = 2
"""出生点识别需要连续几次新截图得出相同结论。"""

SPAWN_OBSERVE_SECONDS: float = 10.0
"""出生点识别的最长观察时间（秒）；期间不移动角色。"""

BEACON_ARRIVE_RADIUS: float = 3.0
"""途径点（纯路点）的到点半径；仍按 3px 精准收口。"""

BEACON_CROSS_TOLERANCE: float = 1.0
"""线段穿越判定的额外容差（px）：定位是 1px 量化，留一个像素余量。"""


ARRIVE_RADIUS: float = 3.0
"""到达半径：3px 内算到点（用户给定）。"""

INTERACTION_TRUST_DISTANCE: float = 40.0
"""交互采信距离：提示出现时距目标点超过此值不采信。"""

SPRINT_DISTANCE: float = 30.0
"""多少像素外开冲刺（用户给定 30px）。

30px 外就冲刺，边跑边点闪避；30px 内不冲刺，松键改用走路逼近。
"""

NEAR_STEP_TURN_RADIUS: float = 8.0
"""距离目标点小于它就不再按角度停车转。

这个距离上目标方位完全由 1~2px 的定位抖动决定：实测 5px 处相邻两轮算出的偏差
是 78° 和 59°，按它停车转只会原地打摆。改成像束角内那样边走边转，转向幅度受
CRUISE_TURN_CAP 限制。
"""

BOX_SMALL_STEP_RADIUS: float = 3.0
"""箱子碎步半径：进到它以内用碎步（用户给定 3px）。"""

ELECTRONIC_SMALL_STEP_RADIUS: float = 15.0
"""电子保险箱短步半径：进入后保持短步，直到当前目标结束。"""

BOX_WAIT_RADIUS: float = 1.9
"""箱子原地等待半径：距离小于 1.9px 就停步原地等提示（用户给定）。

用户要求「0px 才停」，但坐标不一定正好落在目标点上：
编辑器拖点会保留 1 位小数，定位又是整数像素，实际常停在 0.5px。
按 <= 0 判定永远进不去原地等待，只能一路碎步撞到 75 秒兜底超时。
所以用「小于 1.9」作判据，0.5px 这种就能正确进入等待。
"""

SMALL_STEP_TIMEOUT: float = 5.0
"""碎步超时（秒）：3px 内碎步前进超过它还没贴到目标就判超时（用户给定）。

碎步是不看角度硬迈的，方向偏了会一直贴着原地蹭。给一个短时限，
免得它在这个距离上耗到 75 秒的单点兜底。
"""

SLOW_STEP_PRESS: float = 0.5
"""慢步单次按压时长（秒）：比碎步长，一次走得多一点（用户给定 0.5）。"""

SLOW_STEP_WAIT: float = 0.1
"""慢步投递后重新观察的间隔（秒）；短按未结束时不叠发下一步。"""

TURN_SCALE_LESSON_SAMPLES: int = 8
"""单局至少攒到这么多个有效样本，才把本局样本并入跨局累积池。

每局都会新建 walker 和输入线程，样本天然按局清零；统计能力靠跨局累积池提供，
所以这里只挡明显不够一局（例如才走了两三个点）的情况。"""

TURN_SCALE_MIN_CHANGE: float = 0.05
"""写回门槛的下限：偏离 1.0 不到 5% 一律不写，再低就只是噪声。"""

TURN_SCALE_SIGNIFICANCE: float = 3.0
"""写回门槛按标准误算：偏离要超过「这么多倍标准误」才认。

样本是逐局累积的（存在 config/bagel/turn_scale_samples.yml），标准误随样本数开方下降：
刚装机、灵敏度差 20% 的机器第一局就会写；累计够了以后门槛降到 2~3%，才追得动真实漂移。
"""

TURN_SCALE_EVIDENCE_SAMPLES: float = 40.0
"""写回步长的「半程点」：实际走 peak_count/(peak_count+它) 步。

刚装机只有一局证据时只走 1/3，证据攒够才敢整步靠近，避免把单局噪声抄进 turn_dx。
"""

TURN_SCALE_BOUNDS: tuple[float, float] = (0.55, 1.95)
"""可信的补偿比例范围；超出说明样本被别的动作污染，存了也是错的。"""

TURN_SCALE_SEED_SAMPLES: int = 8
"""从累积池取初值至少要这么多样本：再少，中位数还不如上一局的比例准。"""

TURN_SCALE_SEED_SPREAD: float = 0.15
"""累积池离散度上限：超过它说明混进了被打断的脏样本，退回上一局的比例。"""

MAX_POSITION_JUMP: float = 60.0
"""单帧定位跳变上限（px）：实测正常帧位移 5~7px，几百像素的跳变只能是定位串层。"""

POSITION_JUMP_CONFIRM_FRAMES: int = 2
"""同一处新位置连续这么多帧才认可跳变（开门、电梯换区域确实会跳一次）。"""

POSITION_JUMP_HOLD_SECONDS: float = 1.5
"""跳变一直对不上时，最多沿用旧位置这么久，之后接受新位置，避免卡死。"""

ANGLE_FAIL_SECONDS: float = 10.0
"""连续读不到镜头朝向超过这个时长才判失败（用户给定 10 秒）。

单帧读不到是常事（转视角、贴墙、扇形被挡），不能立刻中断整条路线。
"""

WALK_TURN_ANGLE: float = 90.0
"""束角：偏差超过它才松前进键停车转，其余靠边走边转掰过去。

仓内既有实现（world_patrol、bagel_navigate）同样取 90 度。"""

CRUISE_ALIGN_LIMIT: float = 30.0
"""前进闸门：偏差超过它就松键先原地对准，不按住前进。

取 30 度＝边走边转的单笔上限，一笔即可回到闸门内；偏差更大时按住前进是斜着
往墙里冲。"""

SPRINT_ALIGN_LIMIT: float = 15.0
"""冲刺闸门：偏差在它以内才冲刺，闸门外走普通速度。

冲刺会把边走边转的横向漂移成倍放大。"""

BEHIND_BAND: float = 112.5
"""目标转到身后（超过它）就停止前进；只有刚走过头时才用后退纠偏。"""

OVERSHOOT_RADIUS: float = 12.0
"""目标在身后、距离又在这以内，视为刚走过头：短促后退一步。

只在八向走位已经关掉的贴身区生效（更远处由八向的后退档处理），
避免两种后退机制在点附近互相打断、造成可见的顿挫。"""

OVERSHOOT_BACK_STEP: float = 0.18
"""走过头后退一步的按压时长（秒）：给短一点，退多了会退到墙里。"""

OVERSHOOT_STEP_LIMIT: int = 4
"""单点最多后退几步；退不动说明后面是墙，交回停车转。"""

MOVE_FORWARD_BAND: float = 22.5
"""斜走兜底：偏差在这以内只按前进，超过它才考虑用侧移吃掉一部分。"""

TURN_MIX_MID: float = 45.0
"""八向混合的中间档（度）：0/45/90 三档之间的分界。"""

MIX_CYCLE_SECONDS: float = 0.4
"""八向混合一个循环的时长（秒）：档位按比例交替，一个循环内覆盖两档方向。

取 0.4 是因为主循环一轮约 0.12 秒：循环太短会让按键每轮都翻，游戏里看着毛糙。"""

VECTOR_STALL_SECONDS: float = 0.6
"""八向走位时距离这么久没缩小，认为被挡住，改成停车转镜头。"""

VECTOR_STALL_DISTANCE: float = 2.0
"""距离缩小不到这么多像素算没进展。"""

VECTOR_FALLBACK_LIMIT: int = 6
"""单点最多因受阻回退几次；再挡就彻底停车转。"""

DIAGONAL_TIME_LIMIT: float = 6.0
"""单点最多八向走位这么多秒；之后交回「前进＋转镜头」，免得一直横着走。"""

DIAGONAL_ROUND_SECONDS: float = 0.2
"""主循环一轮的估计耗时，用来给斜走时间预算记账。"""

DIAGONAL_MIN_DISTANCE: float = 12.0
"""八向走位一直用到离点这么近为止：途径点附近也不停下来转镜头。

实测 60px 就开始停着对准，角色在点附近会顿一下；缩到 12px（碎步半径上方）
让八向一直走到贴身，视觉上不停顿。"""

BLIND_WALK_SECONDS: float = 0.8
"""定位刚丢时先按原方向盲走这么久再停：高危图里站着 2 秒就被打死。"""

STRAFE_MIN_SEGMENT: float = 8.0
"""路径线短于它就不拿它当基准（px）：两点几乎重合时方向会被量化放大。"""

LOOKAHEAD_DISTANCE: float = 22.0
"""前视距离（px）：沿「上一个点→当前目标点」往前取瞄准点，不直接瞄点位。

人偏离路线时，瞄点位会让视线又被拉向点位（近点方位抖动大）；瞄线上前方
一个点，行进方向贴着路线，横向偏差收敛更早。剩的距离不足它就退回瞄点位。"""




ARROW_MIN_DISPLACEMENT: float = 5.0
"""位移至少这么多像素才认为行进方向可信（px）。"""

ARROW_MAX_AGE: float = 0.6
"""只拿这么久以内的定位估行进方向（秒）。"""

SMALL_STEP_PRESS: float = 0.08
"""碎步单次按压时长（秒）。"""

LOCATE_FAIL_SECONDS: float = 10.0
"""连续定位失败超过这个时长才判失败（用户给定 10 秒）。

给 MultiMapLocator 的最后一档降级留足时间：它要丢失满 1 秒
才开始搜索全部层，全层搜索本身也慢。
"""

STUCK_SECONDS: float = 3.0
STUCK_DISTANCE: float = 2.0
"""按住前进时判卡位：窗口内位移不足此值即视为走不动。"""

SIDE_STEP_SECONDS: float = 0.2
"""卡位时侧移一小步的按键时长（秒）。短按一下，不挪过头。"""

STUCK_STEP_LIMIT: int = 5
"""同一个点连续侧移的上限；挪这么多步还脱不了身，说明不是偏线而是真走不通。"""

POINT_TIMEOUT: dict[str, float] = {BEACON: 45.0, 'default': 75.0}
"""单点超时（秒）：途径点 45，箱类 75。"""

ACTION_LIMIT: int = 100
"""单个点允许的输入次数上限。"""

BOX_PROMPT_SECONDS: float = 1.0
"""箱子点贴到 0px 后原地等提示的上限（用户给定 1 秒）。

箱子一定会刷出交互提示，所以到 0px 还读不到，只可能是这个点本身
没东西，直接跳到下一个点，不判失败、也不原地死等。
"""

SURROUND_CHECK_SECONDS: float = 2.0
"""走完路线后原地观察交互物的时长。"""

MECHANICAL_HOLD_SECONDS: float = 5.0
"""机械保险箱长按 F 的最长时间（秒）。"""

INTERACT_TAP_SECONDS: float = 0.2
"""点按交互键 F 的按住时长（秒）。"""

FAIL_SURVEY_SECONDS: float = 3.0
"""装箱或关闭面板失败后，等 HUD 调查收益读数停稳的最长时间（秒）。

战斗中普通攻击按键可能被动画遮住，涨幅浮字也可能干扰读数，所以要多等几帧。
"""

FAIL_SURVEY_WAIT: float = 0.3
"""失败后补读调查收益时，两次读数的间隔（秒）。"""

SURVEY_LINE_TOLERANCE: int = 6
"""读调查收益时，同一行文字中心的纵向容差（像素）。

物品入箱后，HUD 数字上方可能浮出该件涨幅（如「+10K」），会落进读数区域。
"""

ICON_THRESHOLD: float = 0.8
"""交互图标与刷卡图标的模板匹配阈值；实测同类 1.0、异类 0.33。"""

ENTRY_TEXT_MARGIN: int = 6
"""读取条目文字时，纵向向上下各留的像素。"""

PICKUP_WORDS: tuple[str, ...] = ('药粉', '拾取')
"""可拾取物而非容器的提示词；按 F 会捡起地上的东西，不能当箱子处理。"""


def _closest_on_segment(
    start: BagelRoutePoint, end: BagelRoutePoint, position: tuple[float, float],
) -> tuple[float, float]:
    """线段上离角色位置最近的点，夹在线段两端之间。

    Args:
        start: 线段起点，路径上的上一个点。
        end: 线段终点，当前目标点。
        position: 角色当前坐标。

    Returns:
        线段上的最近点坐标。
    """
    dx, dy = end.x - start.x, end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return (float(start.x), float(start.y))
    t = ((position[0] - start.x) * dx + (position[1] - start.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return (start.x + t * dx, start.y + t * dy)


def _robust_spread(samples: list[float]) -> float:
    """样本的稳健标准差估计（四分位距换算），用来算写回门槛。

    用 IQR 而不是标准差：单局里偶尔会有 2.16 这种被别的动作污染的样本，
    标准差会被它放大好几倍，IQR 不受影响。

    Args:
        samples: 补偿比例样本。

    Returns:
        稳健标准差，夹在 [0.05, 0.60] 之间，样本不足时按上限保守处理。
    """
    values = sorted(samples)
    if len(values) < 8:
        return 0.60
    low = values[len(values) // 4]
    high = values[len(values) * 3 // 4]
    return min(0.60, max(0.05, (high - low) / 1.349))


class BagelRouteWalker(BagelOperation):
    """按打点顺序执行路线，箱类按类型交互。"""

    STATUS_ANCHOR_MISS: str = '出生点没有可执行路线，退出重开'
    STATUS_SPAWN_FAILED: str = '出生点识别失败'
    STATUS_CONFIG_INVALID: str = '路线配置无效'
    STATUS_NOT_FOREGROUND: str = '游戏不在前台，已停止'
    STATUS_DONE: str = '路线走完'
    STATUS_FAIL_SURVEY: str = '失败前补读调查收益'

    def __init__(
        self, ctx: ZContext, blueprint_priority: bool = True,
        route_ref: BagelRouteRef | None = None,
    ) -> None:
        """加载出生点清单并准备定位。

        Args:
            ctx: 运行上下文。
            blueprint_priority: 入箱时蓝图是否优先。
            route_ref: 试跑时指定的路线，关闭的路线也可以试跑；None 表示正式运行，
                识别出生点后执行该出生点当前开启的路线。

        Raises:
            ValueError: 出生点清单无法读取，或指定路线不存在、损坏、绑定的出生点未登记。
        """
        super().__init__(ctx, op_name='贝果-路线执行', timeout_seconds=-1)
        self.store: BagelRouteStore = BagelRouteStore()
        try:
            self.spawns: list[BagelSpawnPoint] = self.store.load_spawns()
            self.trial_route: BagelRoute | None = (
                None if route_ref is None else self.store.load_route(route_ref)
            )
        except BagelRouteConfigError as error:
            raise ValueError(str(error)) from error
        if self.trial_route is not None and not any(
            spawn.spawn_id == self.trial_route.spawn_id for spawn in self.spawns
        ):
            raise ValueError(f'路线绑定的出生点未登记：{self.trial_route.spawn_id}')
        self.routes: list[BagelRoute] = []
        """本局执行的路线快照；识别出生点后才确定，只含一条。"""
        self.spawn: BagelSpawnPoint | None = None
        """本局识别到的出生点。"""
        self.spawn_started_at: float | None = None
        """出生点识别的起始时刻。"""
        self.spawn_last_match: BagelSpawnMatch | None = None
        """上一帧的出生点匹配结论；定位失败时清空。"""
        self.spawn_streak: int = 0
        """连续得出相同出生点结论的新截图次数。"""
        self.blueprint_priority: bool = blueprint_priority
        self.locator: MultiMapLocator = MultiMapLocator(load_map_locators())
        self.route_index: int = 0
        self.cursor: int = 0
        self.current_kind: str | None = None
        self.f_pressed: bool = False
        self.mechanical_hold_until: float | None = None
        self.mechanical_retried: bool = False
        self.position: tuple[float, float] | None = None
        self.last_map_id: str | None = None
        self.last_good_position: tuple[float, float] | None = None
        """上一个可信定位，用于判单帧跳变。"""
        self._jump_candidate: tuple[float, float] | None = None
        """当前怀疑是跳变的新位置。"""
        self._jump_candidate_frames: int = 0
        """该新位置连续出现了几帧。"""
        self._jump_reject_since: float | None = None
        """开始拒绝跳变的时刻，用来兜底放行。"""
        self.locate_misses: int = 0
        self.locate_miss_since: float | None = None
        """连续定位失败的起始时刻，恢复定位后清空。"""
        self.angle_miss_since: float | None = None
        """连续读不到镜头朝向的起始时刻，读到朝向后清空。"""
        self.last_position: tuple[float, float] | None = None
        self.prev_position: tuple[float, float] | None = None
        """再上一帧定位，用于「线段是否穿过点位」的到达判定。"""
        self.steps: int = 0
        self.stuck_since: tuple[tuple[float, float], float] | None = None
        self.stuck_steps: int = 0
        """当前点已经侧移了几次，进入新点时清零。"""
        self.overshoot_steps: int = 0
        """当前点已经后退纠偏几次，进入新点时清零；防止对着墙一直后退。"""
        self.diagonal_seconds: float = 0.0
        """当前点已用掉的八向走位时间预算，进入新点时清零。"""
        self.mix_phase: float = 0.0
        """八向混合的相位（0~1）：小于占空比时用深一档方向。"""
        self.mix_last_at: float = 0.0
        """上一轮推进混合相位的时刻；0 表示还没开始。"""
        self.vector_last_distance: float | None = None
        """八向走位上一轮的距离，用来判有没有进展。"""
        self.vector_stall_since: float | None = None
        """八向走位开始走不动的时刻。"""
        self.vector_fallbacks: int = 0
        """当前点因受阻回退转镜头的次数，进入新点时清零。"""
        self.pos_history: deque[tuple[float, float, float]] = deque(maxlen=12)
        """最近几帧定位 (时刻, x, y)，用位移反推实际行进方向。"""
        """因为行进方向还没跟上镜头而停下的连续轮数，用来兜底防止卡住。"""
        self.target_wait_since: float | None = None
        self.electronic_small_steps: bool = False
        """当前电子保险箱已进入短步阶段，定位回到半径外时也不恢复长步。"""
        self.small_step_since: float | None = None
        """进入 3px 碎步段的起始时刻，用于碎步超时判定。"""
        self.point_started_at: float | None = None
        self.last_input_frame: float | None = None
        self.decision: str = '准备开始'
        """最近一轮的决策文字，供执行态悬浮窗显示。"""
        self.sprinting: bool = False
        """当前是否在冲刺。"""
        log.info(
            '贝果转向本局按 turn_dx=%.4f 起步，校准由转向线程自己学，局末写回同一字段',
            self.ctx.game_config.turn_dx,
        )
        self.elevator_phase: str | None = None
        """电梯当前阶段：进门 / 选楼层 / 出电梯；不在电梯里时为 None。"""
        self.elevator_since: float | None = None
        """当前电梯阶段的起始时刻，用于计时。"""
        self.elevator_last_tap: float = 0.0
        """上次为推进对话而点按的时刻，控制每秒一次。"""
        self.input_worker: BagelInputWorker = BagelInputWorker(
            ctx, seed_ratio=self._seed_turn_ratio(),
        )
        """按键单独线程：主线投递即返回；同种操作不打断自己，不同操作打断旧的。"""
        self.ledger: BagelSafeLedger = BagelSafeLedger()
        """本局安全箱物品，只用于统计日志。"""
        self.survey_value: int = 0
        """本局调查收益的最后一次读数，只用于统计日志。

        零携带入场，开局为 0；每次关闭搜查面板后读 HUD 更新，装箱或关闭面板失败时再补读一次。
        """
        self.fail_reason: str | None = None
        """装箱或关闭面板失败的原因；补读调查收益后按它结束本局。"""
        self.fail_screenshot: tuple[float, MatLike] | None = None
        """失败时的截图时间和截图；补读结束后恢复，现场截图仍是失败画面。"""
        self.fail_survey_started: float | None = None
        """开始补读调查收益的时刻。"""
        self.fail_survey_last: int | None = None
        """上一帧补读到的 HUD 读数；连续两帧相同才采用。"""

    def state_snapshot(self) -> dict[str, object]:
        """给悬浮窗用的执行状态快照。

        每次调用都按当前游标现算，不维护第二份真值，避免两边对不上。

        Returns:
            含决策、路线进度、点位与当前位置的字典。
        """
        route = self.routes[self.route_index] if self.routes else None
        point = self._current_point()
        distance: float | None = None
        if point is not None and self.position is not None:
            distance = hypot(point.x - self.position[0], point.y - self.position[1])
        return {
            'decision': self.decision,
            'route_id': route.route_id if route is not None else '',
            'route_name': route.name if route is not None else '',
            'spawn_name': self.spawn.name if self.spawn is not None else '',
            'route_index': self.route_index,
            'route_count': len(self.routes),
            'cursor': self.cursor,
            'total': len(route.points) if route is not None else 0,
            'points': [(point.kind, (point.x, point.y)) for point in route.points]
                      if route is not None else [],
            'layer': route.points[0].layer if route is not None and route.points else '',
            'position': self.position,
            'map_id': self.last_map_id,
            'distance': distance,
        }

    def _execute_one_round(self) -> OperationRoundResult:
        """每轮前先确认游戏在前台，结束后更新决策文字。

        用户切出游戏时，按键会打进别的窗口，必须先松键并停下，
        否则会在后台一直空转。
        """
        if not self._game_in_foreground():
            self._release_forward()
            self.decision = '游戏不在前台 · 已停止'
            return self.round_fail(self.STATUS_NOT_FOREGROUND)
        result = super()._execute_one_round()
        node_name = '未知' if self._current_node is None else self._current_node.cn
        status = result.status or result.status_display or ''
        self.decision = f'{node_name} · {status}' if status else node_name
        self._log_decision(node_name, status)
        return result

    def _log_decision(self, node_name: str, status: str) -> None:
        """每轮打一条决策日志，带点位、距离与角度，便于核对停在哪一步。

        同样的文字连续重复时也照打：卡住的表现正是同一条反复出现，
        静默掉就看不到「一直短步」「一直转圈」这类症状。

        到点那一轮不能把「当前游标」和「位置」直接配对：游标已经推进到
        下一个点，位置还是走到上一个点时的位置，两边一算就得到一个
        几十像素的假距离，看起来像「离老远就说到点」。所以到点这一轮
        只打位置，判定距离由 navigate 自己在到点时打一条独立的 log。
        """
        point = self._current_point()
        parts = [f'[{node_name}]']
        if status:
            parts.append(status)
        if point is not None:
            parts.append(f'第{self.cursor + 1}点 {point.kind}')
        arrived = node_name == '导航移动' and status.startswith('已到点')
        if point is not None and self.position is not None and not arrived:
            distance = hypot(point.x - self.position[0], point.y - self.position[1])
            parts.append(f'距 {distance:.1f}px')
        if self.position is not None:
            parts.append(f'位置 ({self.position[0]:.0f},{self.position[1]:.0f})')
        if self.current_kind is not None:
            parts.append(f'提示 {self.current_kind}')
        log.info('贝果执行 %s', ' · '.join(parts))

    def _game_in_foreground(self) -> bool:
        """游戏窗口是否在前台；拿不到窗口句柄时按前台处理，避免误停。"""
        game_win = self.ctx.controller.game_win
        hwnd = game_win.get_hwnd()
        if hwnd is None:
            return True
        return is_foreground(hwnd)

    # ---------- 定位 ----------

    def minimap(self) -> MatLike:
        """按原生区域裁出小地图；框架截图为 RGB，定位与朝向直接使用。"""
        area = self.ctx.screen_loader.get_area('贝果-局内', '定位小地图')
        if area is None:
            raise ValueError('缺少贝果定位小地图区域')
        crop = cv2_utils.crop_image_only(self.last_screenshot, area.pc_rect)
        if crop is None or crop.size == 0:
            raise ValueError('小地图裁剪为空')
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)
        return crop

    def locate(self) -> tuple[float, float] | None:
        """定位一次并更新所在地图。

        单帧跳变超过 MAX_POSITION_JUMP 时先当作不可信、沿用上一个可信位置：
        开门/电梯换区域时定位会串层，实测跳过 800 像素，照它算距离就会朝一个
        不存在的点走。同一处新位置连续出现够多帧、或已经僵持够久，才认可换区域。
        """
        result = self.locator.locate(self.minimap())
        if result is None:
            return None
        locator, score, center = result
        new_position = (float(center[0]), float(center[1]))
        last = self.last_good_position
        if last is not None:
            jump = hypot(new_position[0] - last[0], new_position[1] - last[1])
            if jump > MAX_POSITION_JUMP:
                if (
                    self._jump_candidate is not None
                    and hypot(new_position[0] - self._jump_candidate[0],
                              new_position[1] - self._jump_candidate[1]) <= 12.0
                ):
                    self._jump_candidate_frames += 1
                else:
                    self._jump_candidate = new_position
                    self._jump_candidate_frames = 1
                now = time.monotonic()
                if self._jump_reject_since is None:
                    self._jump_reject_since = now
                confirmed = self._jump_candidate_frames >= POSITION_JUMP_CONFIRM_FRAMES
                expired = now - self._jump_reject_since >= POSITION_JUMP_HOLD_SECONDS
                if not confirmed and not expired:
                    log.warning(
                        '贝果定位单帧跳变 %.0fpx（层 %s 得分 %.2f），先沿用上一处位置 %s',
                        jump, locator.map_id, score, last,
                    )
                    self._save_locator_jump(new_position, jump)
                    return last
                log.info(
                    '贝果定位跳变 %.0fpx 已确认（连续 %s 帧，层 %s），按换区域接受',
                    jump, self._jump_candidate_frames, locator.map_id,
                )
        self._jump_candidate = None
        self._jump_candidate_frames = 0
        self._jump_reject_since = None
        self.last_map_id = locator.map_id
        self.last_good_position = new_position
        self.position = new_position
        return self.position

    def _save_locator_jump(self, position: tuple[float, float], jump: float) -> None:
        """把跳变现场的小地图裁剪存下来，供离线查是哪一层匹配错了。"""
        dump_dir = os.environ.get('BAGEL_LOCATOR_DUMP')
        if not dump_dir:
            return
        try:
            import cv2
            directory = Path(dump_dir)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime('%H%M%S')
            name = f'{stamp}_{jump:.0f}px_{position[0]:.0f}_{position[1]:.0f}'
            cv2.imwrite(
                str(directory / f'{name}.png'),
                cv2.cvtColor(self.minimap(), cv2.COLOR_RGB2BGR),
            )
            log.info('贝果定位跳变现场已存：%s', directory / f'{name}.png')
        except Exception:
            log.warning('存定位跳变现场失败', exc_info=True)

    def player_angle(self) -> float | None:
        """识别镜头朝向角，正右为 0、逆时针增加。

        必须取 ``analyze`` 的第二个返回值（扇形方向 = 镜头朝向），
        不能用第一个（菱形尖端 = 角色朝向）。角色站着不动时，
        鼠标转动只改变镜头，角色朝向几乎不变，取错就会出现
        「不管怎么转，角度完全没更新」。
        """
        _marker_vertices, _marker_edges, sector, _radius, _center = analyze(self.minimap())
        return sector

    # ---------- 节点 ----------

    @operation_node(name='出生点识别', is_start_node=True)
    def anchor(self) -> OperationRoundResult:
        """识别出生点并确定本局路线，期间不移动角色。

        每轮使用新截图定位一次。连续 ``SPAWN_CONFIRM_FRAMES`` 次得出相同结论才采用；
        定位缺失或命中多个出生点时等下一帧，超过 ``SPAWN_OBSERVE_SECONDS`` 仍无结论则失败，
        不按路线顺序任取一条。
        """
        now = time.monotonic()
        if self.spawn_started_at is None:
            self.spawn_started_at = now
        if now - self.spawn_started_at > SPAWN_OBSERVE_SECONDS:
            reason = '定位失败' if self.spawn_last_match is None else self.spawn_last_match.describe()
            log.warning('贝果出生点识别：%.0f 秒内没有稳定结论（%s）', SPAWN_OBSERVE_SECONDS, reason)
            return self.round_fail(self.STATUS_SPAWN_FAILED, data=reason)
        position = self.locate()
        if position is None:
            self.spawn_last_match = None
            self.spawn_streak = 0
            return self.round_wait('定位失败，等下一帧', wait=0.2)
        match = match_spawn(self.spawns, self.last_map_id or '', position)
        self.spawn_streak = self.spawn_streak + 1 if match.same_as(self.spawn_last_match) else 1
        self.spawn_last_match = match
        if match.kind == SPAWN_AMBIGUOUS or self.spawn_streak < SPAWN_CONFIRM_FRAMES:
            return self.round_wait(match.describe(), wait=0.2)
        if match.kind == SPAWN_MISS:
            log.info(
                '贝果出生点识别：位置 (%.0f,%.0f) 不在任何已登记出生点范围内',
                position[0], position[1],
            )
            return self.round_fail(self.STATUS_ANCHOR_MISS)
        if match.kind != SPAWN_HIT or match.spawn is None:
            return self.round_fail(self.STATUS_SPAWN_FAILED, data=match.describe())
        return self._start_route_at(match.spawn)

    def _start_route_at(self, spawn: BagelSpawnPoint) -> OperationRoundResult:
        """按识别到的出生点确定路线快照，并从第 1 点开始。

        Args:
            spawn: 已确认的出生点。
        """
        self.spawn = spawn
        if self.trial_route is not None:
            route = self.trial_route
            if route.spawn_id != spawn.spawn_id:
                bound = next(
                    (item.name for item in self.spawns if item.spawn_id == route.spawn_id),
                    route.spawn_id,
                )
                return self.round_fail(
                    self.STATUS_SPAWN_FAILED,
                    data=f'角色在{spawn.name}，路线「{route.name}」绑定的是{bound}',
                )
        else:
            try:
                route = self.store.active_route(spawn.spawn_id)
            except BagelRouteConfigError as error:
                log.error('贝果出生点 %s 的路线配置无效：%s', spawn.name, error)
                return self.round_fail(self.STATUS_CONFIG_INVALID, data=str(error))
            if route is None:
                log.info('贝果出生点识别：%s 没有开启的路线', spawn.name)
                return self.round_fail(self.STATUS_ANCHOR_MISS)
        problem = check_route_start(route, spawn)
        if problem is not None:
            return self.round_fail(self.STATUS_CONFIG_INVALID, data=f'路线「{route.name}」{problem}')
        self.routes = [route]
        self._enter_point(0, 0)
        log.info(
            '贝果出生点识别：%s，执行路线「%s」（%s），共 %s 点',
            spawn.name, route.name, route.ref, len(route.points),
        )
        return self.round_success('已锚定')

    @node_from(from_name='游标推进', status='已推进')
    @node_from(from_name='导航移动', status='已到点')
    @node_from(from_name='导航移动', status='跳过此点')
    @node_from(from_name='碎步移动', status='跳过此点')
    @node_from(from_name='进出电梯', status='已离开电梯')
    @operation_node(name='点调度', mute=True)
    def dispatch(self) -> OperationRoundResult:
        """按游标当前点分派到导航、碎步或现场判断。"""
        point = self._current_point()
        if point is None:
            return self.round_success('路线走完')
        if point.kind == BEACON:
            return self.round_success('下一格是途径点')
        if point.kind == ELEVATOR:
            return self.round_success('下一格是电梯')
        return self.round_success('下一格是箱类')

    @node_from(from_name='点调度', status='路线走完')
    @node_from(from_name='导航移动', status='路线走完')
    @node_from(from_name='碎步移动', status='路线走完')
    @operation_node(name='现场交互判断')
    def check_surroundings(self) -> OperationRoundResult:
        """原地扫 2 秒：识别到已知提示就按提示交互，否则换路线。"""
        if self._nearby_kind() is not None:
            self.target_wait_since = None
            return self.round_success('现场有交互物')
        if self.target_wait_since is None:
            self.target_wait_since = time.monotonic()
        elif time.monotonic() - self.target_wait_since >= SURROUND_CHECK_SECONDS:
            self.target_wait_since = None
            return self.round_success('现场无交互物')
        return self.round_wait('原地观察交互提示', wait=0.25)

    def _nearby_kind(self) -> str | None:
        """右侧最上面一条交互提示的类型，识别不到时返回 None。

        先模板匹配交互图标或刷卡图标定位条目，再对该条右侧文字分类：
        含「保险箱」按电子/机械分，含「门」按门，其余都是可搜查容器。
        """
        entry = self._top_entry()
        if entry is None:
            return None
        return self._classify(entry[0], entry[1])

    def _top_entry(self) -> tuple[str, str] | None:
        """取最上面一条提示，返回（图标类型, 条目文字）。

        多条提示同时出现时，最上面那条才是按 F 生效的目标。
        只在提示区矩形内匹配，避免全屏误命中。
        """
        area = self.ctx.screen_loader.get_area('贝果-局内', '交互图标')
        if area is None:
            return None
        crop = cv2_utils.crop_image_only(self.last_screenshot, area.pc_rect)
        if crop is None or crop.size == 0:
            return None
        best: tuple[int, str, str] | None = None
        for icon, kind in (('interact_icon', 'interact'), ('card_key_icon', 'card')):
            result = self.ctx.tm.match_template(
                crop, 'bagel', icon, threshold=ICON_THRESHOLD, only_best=False,
            )
            for match in result:
                if best is None or match.center.y < best[0]:
                    best = (match.center.y, kind, self._entry_text(match, area.pc_rect.x1))
        if best is None:
            return None
        return best[1], best[2]

    def _entry_text(self, match, offset_x: int) -> str:
        """读取一条提示右侧的文字，去掉空白。

        只裁这一条的高度：整块提示区一起识别会漏读，按条裁剪才能稳定读出。

        Args:
            match: 提示区裁剪内的图标匹配结果。
            offset_x: 提示区裁剪相对整屏的左边距，用于换算回整屏坐标。
        """
        area = self.ctx.screen_loader.get_area('贝果-局内', '交互条目文字')
        if area is None:
            return ''
        left = max(area.pc_rect.x1, offset_x + match.x + match.w)
        right = area.pc_rect.x2
        top = area.pc_rect.y1 + match.y - ENTRY_TEXT_MARGIN
        bottom = area.pc_rect.y1 + match.y + match.h + ENTRY_TEXT_MARGIN
        if right <= left or bottom <= top:
            return ''
        texts = self.ctx.ocr_service.get_ocr_result_list(
            image=self.last_screenshot, rect=Rect(left, top, right, bottom),
        )
        return ''.join(str_utils.remove_whitespace(item.data) for item in texts)

    def _classify(self, icon_kind: str, text: str) -> str | None:
        """按图标与条目文字判断交互类型。

        特殊分支只有两类保险箱：电子保险箱要解锁，机械保险箱要长按。
        其余「箱 / 包 / 柜」都是按一下 F 开始搜宝；门和刷卡处按 F 直接开门。
        """
        if icon_kind == 'card':
            return 'card_reader'
        if gt('电子保险箱', 'game') in text:
            return 'electronic_safe'
        if gt('机械保险箱', 'game') in text:
            return 'mechanical_safe'
        if gt('门', 'game') in text:
            return 'door'
        if any(name in text for name in PICKUP_WORDS):
            return None
        if text:
            return 'box'
        return None

    @node_from(from_name='现场交互判断', status='现场有交互物')
    @operation_node(name='现场交互')
    def interact_nearby(self) -> OperationRoundResult:
        """首次按提示确定类型，之后检查本次交互结果，不要求场景提示仍可见。"""
        if self.current_kind is None:
            self.current_kind = self._nearby_kind()
        if self.current_kind is None:
            return self.round_retry('提示消失，再看一帧', wait=0.2)
        return self._interact_by_kind(self.current_kind)

    @node_from(from_name='现场交互判断', status='现场无交互物')
    @operation_node(name='路线结束')
    def finish_route(self) -> OperationRoundResult:
        """本局只执行一条路线，走完且现场没有交互物即结束。"""
        log.info('贝果路线执行：路线「%s」走完', self.routes[self.route_index].name)
        return self.round_success(self.STATUS_DONE)

    # ---------- 导航 ----------

    @node_from(from_name='出生点识别', status='已锚定')
    @node_from(from_name='点调度', status='下一格是途径点')
    @operation_node(name='导航移动', mute=True)
    def navigate(self) -> OperationRoundResult:
        """朝当前点移动；途径点 3px 到达，箱类见到提示就停。

        到点后必须回点调度重新分派，不能直接再进本节点：游标已经推进到
        下一个点，那个点可能是电梯或箱子，得由点调度按种类交给对应节点。
        直接自环会让电梯点落到箱类分支，被当成箱子慢步逼近。
        """
        return self._move_toward()

    @node_from(from_name='点调度', status='下一格是箱类')
    @operation_node(name='碎步移动', mute=True)
    def small_step(self) -> OperationRoundResult:
        """按距离逼近箱类点，统一检查到位、提示与单点超时。"""
        return self._move_toward()

    # ---------- 交互 ----------

    @node_from(from_name='点调度', status='下一格是电梯')
    @operation_node(name='进出电梯')
    def elevator(self) -> OperationRoundResult:
        """进出电梯：走向电梯，点楼层选项，再走出来。

        电梯点在进门前能正常定位，进电梯后小地图整块消失，所以不能按
        3px 判到达（用户定义：读不到朝向就是进了电梯）。进去之后分三段，
        每段只等它自己的信号，统一不设超时——卡在电梯里连游戏都退不出来。

        1. 进门：朝电梯点走，读不到朝向即进入电梯。
        2. 等选项：一直点屏幕中间推进对话，直到楼层选项出现。
        3. 点选项：一直点那个选项，直到它从画面上消失。
        4. 出梯：一直按住配置的方向，直到小地图回来。

        点上没有楼层文字（`elevator_option` 为空）的电梯没有第 2、3 段，
        进门后直接进出梯段。
        """
        point = self._current_point()
        if point is None:
            return self.round_success('路线走完')
        if self.elevator_phase is None:
            # 首次进入本节点必须从「进门」阶段开始。之前漏了这一步，
            # phase 一直是 None，直接落到出梯阶段，「朝电梯走」整个没执行过。
            self.elevator_phase = ELEVATOR_PHASE_ENTER
            self.elevator_since = None
        phase = self.elevator_phase
        if phase == ELEVATOR_PHASE_ENTER:
            return self._elevator_enter(point)
        if phase == ELEVATOR_PHASE_WAIT_OPTION:
            return self._elevator_wait_option(point)
        if phase == ELEVATOR_PHASE_CLICK_OPTION:
            return self._elevator_click_option(point)
        return self._elevator_exit(point)

    def _elevator_enter(self, point: BagelRoutePoint) -> OperationRoundResult:
        """朝电梯点走，途中读不到朝向就是进了电梯。

        电梯点没有「到达」这回事：小地图会整块消失，坐标和朝向都读不到，
        3px 这种判据在电梯口根本不成立。所以这里只朝点走，一旦
        `player_angle()` 读不到就当已进电梯（用户定义）。
        """
        if self.player_angle() is None:
            self._release_forward()
            self.elevator_since = None
            if point.elevator_option:
                self.elevator_phase = ELEVATOR_PHASE_WAIT_OPTION
                self.elevator_last_tap = 0.0
                return self.round_wait('小地图已消失，视为进入电梯', wait=0.2)
            self.elevator_phase = ELEVATOR_PHASE_EXIT
            return self.round_wait('小地图已消失，这电梯没有楼层选项', wait=0.2)
        if self.elevator_since is None:
            self.elevator_since = time.monotonic()
        if time.monotonic() - self.elevator_since >= ELEVATOR_ENTRY_TIMEOUT:
            self._release_forward()
            self.elevator_phase = None
            self.elevator_since = None
            return self.round_recoverable_fail('电梯门口走了 10 秒还没进入电梯')
        position = self.locate()
        if position is not None:
            angle = self.player_angle()
            if angle is not None:
                target_angle = self._target_angle(position, point)
                angle_diff = cal_utils.angle_delta(angle % 360, target_angle)
                self._submit_turn(target_angle, deadband=TURN_DEADBAND_FAST)
                if abs(angle_diff) > WALK_TURN_ANGLE:
                    # 方向偏太多先停下等转，不然会走到旁边去。
                    self._release_forward()
                    return self.round_wait(f'朝电梯转向 {angle_diff:.0f}度', wait=0.05)
                self.input_worker.hold_forward()
                return self.round_wait('走向电梯', wait=0.15)
        # 定位暂时丢失但朝向还在（贴墙等）：先不松前进，等下一帧。
        self.input_worker.hold_forward()
        return self.round_wait('走向电梯，定位暂缺', wait=0.15)

    def _elevator_wait_option(self, point: BagelRoutePoint) -> OperationRoundResult:
        """一直点屏幕中间推进对话，直到楼层选项出现。不设超时。"""
        found = self._find_elevator_option(point.elevator_option)
        if found is not None:
            self.elevator_phase = ELEVATOR_PHASE_CLICK_OPTION
            return self.round_wait(f'楼层选项已出现：{found.data}', wait=0.1)
        if time.monotonic() - self.elevator_last_tap >= ELEVATOR_OPTION_TAP_SECONDS:
            self.elevator_last_tap = time.monotonic()
            self.input_worker.tap_center()
        return self.round_wait('等待电梯楼层选项', wait=0.2)

    def _elevator_click_option(self, point: BagelRoutePoint) -> OperationRoundResult:
        """一直点那个选项，直到它从画面上消失。不设超时。

        选项点中后会收起，收起即表示楼层已选、电梯开始移动。
        中途某一帧识别不到也算收起，所以这里用「识别到才点、识别不到就
        进下一段」，不需要连续几帧确认。
        """
        found = self._find_elevator_option(point.elevator_option)
        if found is None:
            self.elevator_phase = ELEVATOR_PHASE_EXIT
            return self.round_wait('楼层选项已消失，准备出梯', wait=0.1)
        self.ctx.controller.click(found.rect.center)
        return self.round_wait(f'点选楼层：{found.data}', wait=0.2)

    def _find_elevator_option(self, option: str) -> OcrMatchResult | None:
        """在楼层列表里找配置的文字。

        Args:
            option: 要选的楼层文字，如「负二层」。

        Returns:
            匹配到的 OCR 结果；没有则返回 None。
        """
        area = self.ctx.screen_loader.get_area('贝果-局内', '电梯选项')
        if area is None:
            return None
        results = self.ctx.ocr_service.get_ocr_result_list(
            image=self.last_screenshot, rect=area.pc_rect,
        )
        target = gt(option, 'game')
        for item in results:
            if target in str_utils.remove_whitespace(item.data):
                return item
        return None

    def _elevator_exit(self, point: BagelRoutePoint) -> OperationRoundResult:
        """一直按住配置的方向往前走，直到小地图回来。不设超时。"""
        if self.player_angle() is not None:
            self._release_forward()
            self._release_backward()
            self.elevator_phase = None
            self.elevator_since = None
            self._enter_point(self.route_index, self.cursor + 1)
            return self.round_success('已离开电梯')
        if self.elevator_since is None:
            self.elevator_since = time.monotonic()
        if point.elevator_exit == ELEVATOR_EXIT_BACKWARD:
            self.input_worker.hold_backward()
        else:
            self.input_worker.hold_forward()
        elapsed = time.monotonic() - self.elevator_since
        return self.round_wait(f'出电梯 {elapsed:.1f}s', wait=0.15)

    def _release_backward(self) -> None:
        """松开后退键。"""
        self.input_worker.stop_backward()

    @node_from(from_name='碎步移动', status='提示已出现')
    @node_from(from_name='导航移动', status='提示已出现')
    @operation_node(name='按类型交互')
    def interact(self) -> OperationRoundResult:
        """按当前点类型执行交互。"""
        if self.current_kind is None:
            return self.round_retry('未知交互类型', wait=0.2)
        return self._interact_by_kind(self.current_kind)

    def _interact_by_kind(self, kind: str) -> OperationRoundResult:
        """分派交互动作。"""
        if kind == 'electronic_safe':
            return self._run_unlock_safe()
        if kind == 'mechanical_safe':
            return self._hold_mechanical()
        return self._press_f(kind)

    def _run_unlock_safe(self) -> OperationRoundResult:
        """电子保险箱：光圈解锁子操作。"""
        self._release_forward()
        op = BagelUnlockSafe(self.ctx, phase='full')
        result = op.execute()
        self._adopt_screenshot(op)
        if result.success:
            return self.round_success('电子保险箱已解锁')
        return self.round_recoverable_fail(f'电子保险箱解锁失败：{result.status}')

    def _hold_mechanical(self) -> OperationRoundResult:
        """机械保险箱：提示出现时长按 F，直到搜查面板出现。

        长按发生在提示出现时，不是面板打开后；面板一出就松手，
        后面与普通箱走同一套收集入箱。面板用底部共用的安全箱标记确认，
        不依赖专有标题。
        """
        if self.mechanical_hold_until is None:
            self._release_forward()
            self.mechanical_hold_until = time.monotonic() + MECHANICAL_HOLD_SECONDS
            self.input_worker.hold_interact()
            return self.round_wait('长按F中')
        if self._has('搜查安全箱'):
            self.input_worker.release_interact()
            self._reset_point_state()
            return self.round_success('机械保险箱已打开')
        if time.monotonic() >= self.mechanical_hold_until:
            self.input_worker.release_interact()
            if self.mechanical_retried:
                self._reset_point_state()
                return self.round_recoverable_fail('机械保险箱长按F超时')
            self.mechanical_retried = True
            self.mechanical_hold_until = None
            return self.round_wait('长按F超时，释放后重试', wait=0.3)
        return self.round_wait('长按F等待面板', wait=0.3)

    def _press_f(self, kind: str) -> OperationRoundResult:
        """点按 F：普通容器、门、刷卡处、电梯。

        容器类以搜查面板出现为完成；门、刷卡处、电梯以提示消失为完成。
        按 F 前先松前进键，避免按住前行漂移。
        """
        self._release_forward()
        if kind == 'box' and self._has('搜查容器标题'):
            # 容器面板已开：立刻交给收集入箱，不在这里判完成。
            # 用「提示消失」推断完成会漏掉收集和关闭面板，面板一直开着
            # 盖住小地图，后面整段导航都定位失败。
            self._reset_point_state()
            return self.round_success('面板已打开')
        if self.f_pressed and self._top_entry() is None:
            self._reset_point_state()
            return self.round_success('已完成')
        self.input_worker.tap_interact(INTERACT_TAP_SECONDS)
        self.f_pressed = True
        return self.round_wait('已点按F，等待结果', wait=0.4)

    @node_from(from_name='按类型交互', status='电子保险箱已解锁')
    @node_from(from_name='按类型交互', status='机械保险箱已打开')
    @node_from(from_name='按类型交互', status='面板已打开')
    @node_from(from_name='现场交互', status='电子保险箱已解锁')
    @node_from(from_name='现场交互', status='机械保险箱已打开')
    @node_from(from_name='现场交互', status='面板已打开')
    @operation_node(name='收集入箱')
    def collect(self) -> OperationRoundResult:
        """逐件装入安全箱。"""
        op = BagelStoreSafe(self.ctx, blueprint_priority=self.blueprint_priority, ledger=self.ledger)
        result = op.execute()
        self._adopt_screenshot(op)
        if result.success:
            return self.round_success('已入箱')
        return self._fail_after_survey(f'入箱失败：{result.status}')

    @node_from(from_name='收集入箱', status='已入箱')
    @operation_node(name='关闭面板')
    def close_panel(self) -> OperationRoundResult:
        """关闭搜查面板并确认回到局内。"""
        op = BagelCloseSearch(self.ctx)
        result = op.execute()
        self._adopt_screenshot(op)
        if result.success:
            self._read_survey_value()
            return self.round_success('已关闭面板')
        return self._fail_after_survey(f'关闭面板失败：{result.status}')

    def _fail_after_survey(self, reason: str) -> OperationRoundResult:
        """记下失败原因和失败画面，先补读调查收益，再按原因结束本局。

        Args:
            reason: 失败原因；补读结束后原样作为可结算重开的失败返回。
        """
        log.info('贝果%s；先补读调查收益，再结束本局', reason)
        self.fail_reason = reason
        if self.last_screenshot is not None:
            self.fail_screenshot = (self.last_screenshot_time, self.last_screenshot)
        return self.round_fail(self.STATUS_FAIL_SURVEY)

    @node_from(from_name='收集入箱', success=False, status=STATUS_FAIL_SURVEY)
    @node_from(from_name='关闭面板', success=False, status=STATUS_FAIL_SURVEY)
    @operation_node(name='失败前补读收益')
    def read_survey_before_fail(self) -> OperationRoundResult:
        """已回到局内时补读一次 HUD 调查收益，再按原失败原因结束本局。

        连续两帧读数相同才采用，避开涨幅浮字刚出现时的错读。
        以下情况不读，保留最后一次读数：
        - 已是 DEFEAT 画面；
        - 搜查面板仍开着：底栏数字左侧的金币图标会被 OCR 读成「9」（如「187K」读成「9187K」），
          错值每帧相同，连续两帧相同也挡不住；
        - 超过 ``FAIL_SURVEY_SECONDS`` 仍读不到稳定读数。
        """
        now = time.monotonic()
        if self.fail_survey_started is None:
            self.fail_survey_started = now
        if self.is_bagel_result():
            return self._finish_fail('已是失败结算画面')
        if self._has('搜查安全箱'):
            return self._finish_fail('搜查面板仍开着，不读底栏')
        value: int | None = None
        if self.round_by_find_area(self.last_screenshot, '战斗画面', '按键-普通攻击').is_success:
            value = self._read_hud_survey()
        if value is not None and value == self.fail_survey_last:
            log.info(
                '贝果失败前补读调查收益：HUD 读数 %s，上一次读数 %s',
                format_value(value), format_value(self.survey_value),
            )
            self.survey_value = value
            return self._finish_fail(None)
        self.fail_survey_last = value
        if now - self.fail_survey_started >= FAIL_SURVEY_SECONDS:
            return self._finish_fail(f'{FAIL_SURVEY_SECONDS:.0f} 秒内读数未停稳')
        return self.round_wait('等调查收益读数停稳', wait=FAIL_SURVEY_WAIT)

    def _finish_fail(self, skipped: str | None) -> OperationRoundResult:
        """恢复失败画面，按原失败原因结束本局。

        Args:
            skipped: 没有采用补读结果的原因；已采用时为 None。
        """
        if skipped is not None:
            log.info('贝果失败前补读调查收益：%s，沿用最后一次读数 %s', skipped, format_value(self.survey_value))
        if self.fail_screenshot is not None:
            self.last_screenshot_time, self.last_screenshot = self.fail_screenshot
        return self.round_recoverable_fail(self.fail_reason or self.STATUS_FAIL_SURVEY)

    @node_from(from_name='按类型交互', status='已完成')
    @node_from(from_name='现场交互', status='已完成')
    @node_from(from_name='关闭面板', status='已关闭面板')
    @operation_node(name='游标推进')
    def advance(self) -> OperationRoundResult:
        """游标加一并回点调度。"""
        self._enter_point(self.route_index, self.cursor + 1)
        return self.round_success('已推进')

    # ---------- 移动核心 ----------

    def _move_toward(self) -> OperationRoundResult:
        """朝当前游标点移动一轮。"""
        point = self._current_point()
        if point is None:
            return self.round_success('路线走完')
        timeout = self._check_timeout(point)
        if timeout is not None:
            return timeout
        hud = self._check_hud()
        if hud is not None:
            return hud
        position = self.locate()
        if position is None:
            # 定位丢失立刻停步：继续按着前进会走到不知道哪里去，
            # 恢复定位后再接着走。判失败要等满 10 秒，给降级搜索留时间。
            self._release_forward()
            if self.is_bagel_result():
                return self.round_fail(self.STATUS_DEFEATED)
            if self.last_position is None:
                return self.round_retry('起步定位失败，重试', wait=0.3)
            if self.locate_miss_since is None:
                self.locate_miss_since = time.monotonic()
                # 记录丢掉定位前是不是在走：在走就盲走一小段，站着更容易被打死。
                self.blind_walk_ok = self.sprinting or self.input_worker.forward_active
            missing = time.monotonic() - self.locate_miss_since
            if missing < BLIND_WALK_SECONDS and self.blind_walk_ok:
                self.input_worker.hold_forward()
                self._record_input()
                return self.round_wait(f'定位丢失，盲走 {missing:.1f}s', wait=0.05)
            if missing < LOCATE_FAIL_SECONDS:
                self.locate_misses += 1
                return self.round_wait('定位丢失，停下等待', wait=0.1)
            return self.round_recoverable_fail(
                f'连续 {LOCATE_FAIL_SECONDS:.0f}s 定位失败，停止移动',
            )
        self.locate_misses = 0
        self.locate_miss_since = None
        self.prev_position = self.last_position
        self.last_position = position
        self.pos_history.append((time.monotonic(), position[0], position[1]))
        distance = hypot(point.x - position[0], point.y - position[1])
        if (
            point.kind == 'electronic_safe'
            and distance <= ELECTRONIC_SMALL_STEP_RADIUS
            and not self.electronic_small_steps
        ):
            self.electronic_small_steps = True
            self._release_forward()
            log.info(
                '贝果电子保险箱进入短步阶段：距离 %.1fpx，半径 %.0fpx，按压 %.2fs',
                distance, ELECTRONIC_SMALL_STEP_RADIUS, SMALL_STEP_PRESS,
            )
        if self.last_input_frame is not None and self.last_screenshot_time <= self.last_input_frame:
            return self.round_wait('等待动作后的新截图', wait=0.05)
        if distance <= SPRINT_DISTANCE and self.sprinting:
            self._exit_sprint()
        if point.kind == BEACON:
            # 途径点是纯路点：3px 内算到点，全程用八向巡航走过去（不切慢步）。
            # 每轮位移 4~7px 比 3px 判据还大，所以再加一条「线段穿过」判定：
            # 只要这一帧的位移线段从点位旁边经过就算到，不必真的落进圈里。
            crossed = self._path_crosses_point(self.prev_position, position, point)
            if distance <= BEACON_ARRIVE_RADIUS or crossed:
                self._release_forward()
                log.info(
                    '贝果到点：第 %s 点判定距离 %.2fpx（判据 %.0fpx%s，位置 %s）',
                    self.cursor + 1, distance, BEACON_ARRIVE_RADIUS,
                    '，线段穿过' if crossed else '', position,
                )
                self._enter_point(self.route_index, self.cursor + 1)
                # 状态文字必须正好是「已到点」：节点边按状态精确匹配，
                # 掺了距离就匹配不上，导航链在这里断掉、整条路线直接判走完。
                return self.round_success('已到点')
            return self._cruise(point, position)
        # 电子保险箱进入 15px 后保持短步；其他箱类仍在 3px 内使用碎步。
        # 可信提示出现就转入交互，不要求先走到原地等待半径。
        # 箱类分支不用侧移：八向走位残留的 A/D 会让慢步逼近时一直横滑。
        self.input_worker.stop_strafe()
        prompt = self._prompt_stop(distance)
        if prompt is not None:
            return prompt
        if distance < BOX_WAIT_RADIUS:
            self.small_step_since = None
            return self._wait_here(point)
        self.target_wait_since = None
        if distance <= BOX_SMALL_STEP_RADIUS:
            timeout = self._check_small_step_timeout()
            if timeout is not None:
                return timeout
            return self._short_step()
        self.small_step_since = None
        return self._slow_step(point, position, short_press=self.electronic_small_steps)

    def _wait_here(self, point: BagelRoutePoint) -> OperationRoundResult:
        """已站在箱子上：原地不动，等提示出现或超时跳到下一个点。

        箱子一定会刷出交互提示，所以到时间还没提示，只可能是这个点
        本身没有东西（或站位被挡住），直接跳过，不判失败、也不死等。

        Args:
            point: 当前游标点。
        """
        self._release_forward()
        if self.target_wait_since is None:
            self.target_wait_since = time.monotonic()
            return self.round_wait('原地等待交互提示', wait=0.2)
        if time.monotonic() - self.target_wait_since >= BOX_PROMPT_SECONDS:
            self.target_wait_since = None
            log.info(
                '贝果第 %s 点（%s）原地等 %ss 无提示，跳到下一个点',
                self.cursor + 1, point.kind, f'{BOX_PROMPT_SECONDS:.0f}',
            )
            self._enter_point(self.route_index, self.cursor + 1)
            return self.round_success('跳过此点')
        return self.round_wait('原地等待交互提示', wait=0.2)

    def _prompt_stop(self, distance: float) -> OperationRoundResult | None:
        """可信提示出现后松键并转入交互，不另做停稳复核。

        Args:
            distance: 当前可靠定位到目标点的距离。

        Returns:
            允许交互的结果；未识别到可信提示时返回 None。
        """
        kind = self._nearby_kind()
        if kind is None or distance > INTERACTION_TRUST_DISTANCE:
            return None
        self._release_forward()
        self.current_kind = kind
        self.target_wait_since = None
        return self.round_success('提示已出现')

    def _angle_miss_result(self, angle: float | None) -> OperationRoundResult | None:
        """读不到镜头朝向时停下等；连续读不到满 10 秒才判失败。

        单帧读不到是常事（转视角、贴墙、扇形被墙切掉），一帧失败就中断
        整条路线太脆。读不到时转向线程也转不了（没有当前角算偏差），
        这里只松键等画面回来，不下发任何转向。

        Args:
            angle: 本轮读到的镜头朝向；读到就不做处理。

        Returns:
            需要等待或失败时返回本轮结果；读到朝向后返回 None。
        """
        if angle is not None:
            self.angle_miss_since = None
            return None
        self._release_forward()
        # 读不到朝向时不知道偏哪里，侧移/后退也要一起松，否则会一直朝一个方向横移。
        self.input_worker.stop_strafe()
        self.input_worker.stop_backward()
        if self.angle_miss_since is None:
            self.angle_miss_since = time.monotonic()
        missed = time.monotonic() - self.angle_miss_since
        if missed >= ANGLE_FAIL_SECONDS:
            return self.round_recoverable_fail(
                f'连续 {ANGLE_FAIL_SECONDS:.0f}s 识别不到镜头朝向，停止移动',
            )
        self.input_worker.submit_turn(None)
        return self.round_wait(f'未识别到朝向，停下等待（{missed:.1f}s）', wait=0.1)

    def _cruise(
        self, point: BagelRoutePoint, position: tuple[float, float],
    ) -> OperationRoundResult:
        """远距离移动：只提交目标方向，前进键自己管。

        转向全由转向线程负责（稳 3 帧、5 度以上才转）。这里只按本轮
        偏差管前进：超过闸门松键先对准，闸门内按住走，对准了才冲刺。

        近距离不再按角度停车转：这个距离上的目标方位由定位抖动决定，
        停下车来按它转只会原地打摆，改成边走边转（线程内限幅 15 度）。
        """
        angle = self.player_angle()
        miss = self._angle_miss_result(angle)
        if miss is not None:
            return miss
        aim_x, aim_y = self._aim_point(position, point)
        target_angle = self._angle_to(position, aim_x, aim_y)
        controller_angle = angle % 360
        angle_diff = cal_utils.angle_delta(controller_angle, target_angle)
        distance = hypot(point.x - position[0], point.y - position[1])
        near = distance <= NEAR_STEP_TURN_RADIUS
        if (
            abs(angle_diff) > BEHIND_BAND and distance <= OVERSHOOT_RADIUS
            and self.overshoot_steps < OVERSHOOT_STEP_LIMIT
        ):
            # 刚走过头：短促后退一步步退回路径上，比转 180 度再走一遍省事。
            # 只后退、不侧移不斜走：侧移在走廊里只会顶墙（实测侧移纠偏会把卡死率拉高）。
            self.overshoot_steps += 1
            self._release_forward()
            self.input_worker.stop_strafe()
            self._stop_sprint()
            self.input_worker.back_step(OVERSHOOT_BACK_STEP)
            self._record_input()
            return self.round_wait(f'走过头，后退纠偏 {angle_diff:.0f}度', wait=OVERSHOOT_BACK_STEP + 0.1)
        if (
            abs(angle_diff) > MOVE_FORWARD_BAND
            and distance > DIAGONAL_MIN_DISTANCE
            and self.diagonal_seconds < DIAGONAL_TIME_LIMIT
            and self.vector_fallbacks < VECTOR_FALLBACK_LIMIT
        ):
            return self._vector_move(
                point, position, controller_angle, angle_diff, target_angle, distance,
            )
        self.input_worker.stop_strafe()
        self.input_worker.stop_backward()
        self.vector_last_distance = None
        self.vector_stall_since = None
        stopping = abs(angle_diff) > WALK_TURN_ANGLE and not near
        aligning = abs(angle_diff) > CRUISE_ALIGN_LIMIT and not near
        # 停车转要对准到 2 度；边走边修不追 2 度——移动中下发实测散布 5.6%（静止 1.1%），
        # 死区比自身噪声还小就会一直补笔，每个点下发 2~7 笔。
        self._submit_turn(
            target_angle, cruise_only=near,
            deadband=TURN_DEADBAND_FAST if stopping else TURN_DEADBAND,
        )
        if aligning:
            # 松键把镜头掰进闸门再走：边走边转期间角色沿镜头方向前进，
            # 大偏差下按住前进就是斜着冲出去撞墙。
            self._release_forward()
            if stopping:
                return self.round_wait(f'停车等转向 {angle_diff:.0f}度')
            return self.round_wait(f'停车对准 {angle_diff:.0f}度')
        stuck = self._check_stuck(position, point, controller_angle)
        if stuck is not None:
            return stuck
        # 这里原本有一道「等行进方向跟上」的闸门：镜头对准、人还在旧方向时停一轮。
        # 八向混合上线后方向由按键混合直接保证，闸门在每次转弯都误判并停一轮
        # （实测 3 局停了 55 次，每局白站 3~5 秒），已删除。
        if self.input_worker.hold_forward():
            self._record_input()
        self._set_sprint(distance, aligned=abs(angle_diff) <= SPRINT_ALIGN_LIMIT)
        return self.round_wait('冲刺前进' if self.sprinting else '按住前进')

    @staticmethod
    def _path_crosses_point(
        start: tuple[float, float] | None, end: tuple[float, float], point: BagelRoutePoint,
    ) -> bool:
        """这一帧的位移线段是否从目标点旁边穿过。

        Args:
            start: 上一帧定位；没有就判不穿过。
            end: 当前定位。
            point: 当前目标点。

        Returns:
            线段到目标点的垂距在「到点半径 + 容差」以内、且垂足落在两帧之间时返回 True。
        """
        if start is None:
            return False
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 <= 0:
            return False
        t = ((point.x - ax) * dx + (point.y - ay) * dy) / length2
        if not 0.0 <= t <= 1.0:
            return False
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        return hypot(point.x - closest_x, point.y - closest_y) <= (
            BEACON_ARRIVE_RADIUS + BEACON_CROSS_TOLERANCE
        )

    def _vector_move(
        self, point: BagelRoutePoint, position: tuple[float, float], controller_angle: float,
        angle_diff: float, target_angle: float, distance: float,
    ) -> OperationRoundResult:
        """八向走位：按偏差选 W/A/S/D 组合走过去，同时让镜头继续转。

        分档（偏差绝对值）：22.5 度内只前进（由 _cruise 处理）、到 67.5 度前进＋侧移、
        到 112.5 度只侧移、再往后按后退——路线折返时就是倒着走到下一个点，不用停下来
        转镜头。**镜头不停**：继续提交目标角，位移方向跟着镜头一起收敛，避免长时间
        背身横移那种「眩晕」，也不站在转角挨打。

        只有距离一直不缩小（被墙或箱子挡住）才回退成停车转镜头，单点有次数上限。

        Args:
            point: 当前目标点。
            position: 当前可靠定位。
            controller_angle: 当前镜头朝向（度）。
            angle_diff: 目标方位与镜头朝向的有符号差，正数表示目标在左侧。
            target_angle: 目标方位（度）。
            distance: 到目标点的距离（px）。

        Returns:
            本轮结果。
        """
        magnitude = abs(angle_diff)
        side = -1 if angle_diff > 0 else 1
        """侧移方向：角度为正表示目标在镜头左侧，按 A。"""
        if self.vector_last_distance is None or distance < self.vector_last_distance - VECTOR_STALL_DISTANCE:
            self.vector_last_distance = distance
            self.vector_stall_since = None
        else:
            now = time.monotonic()
            if self.vector_stall_since is None:
                self.vector_stall_since = now
            elif now - self.vector_stall_since >= VECTOR_STALL_SECONDS:
                self.vector_fallbacks += 1
                self.vector_last_distance = None
                self.vector_stall_since = None
                self.input_worker.stop_strafe()
                self.input_worker.stop_backward()
                self._release_forward()
                self._stop_sprint()
                self._submit_turn(target_angle, deadband=TURN_DEADBAND_FAST)
                return self.round_wait(f'八向受阻，改转镜头 {angle_diff:.0f}度')
        self.diagonal_seconds += DIAGONAL_ROUND_SECONDS
        # 镜头继续转：位移方向随镜头一起收敛，偏差比背身横移时小得多。
        self._submit_turn(target_angle, deadband=TURN_DEADBAND)
        if magnitude > BEHIND_BAND:
            # 目标在身后：不按后退（实测倒着走进死角会顶墙、卡住），
            # 停车把镜头转过来再走——这段是急弯/折返，停一下换方向比倒着蹭安全。
            self.input_worker.stop_strafe()
            self.input_worker.stop_backward()
            self._release_forward()
            self._stop_sprint()
            self._submit_turn(target_angle, deadband=TURN_DEADBAND_FAST)
            return self.round_wait(f'身后目标，停车转 {angle_diff:.0f}度')
        self.input_worker.stop_backward()
        # 八向混合：镜头偏 Δ 时按比例交替 0/45/90 度三档按键，把实际行进方向压在
        # 目标方位上。镜头的转动只管最终对齐，跑位不再等它——弧和事后横滑都不用发生。
        use_forward, use_strafe = self._eight_way_step(magnitude)
        if use_forward:
            self.input_worker.hold_forward()
        else:
            self.input_worker.stop_forward()
        if use_strafe:
            self.input_worker.hold_strafe(side)
        else:
            self.input_worker.stop_strafe()
        self._stop_sprint()
        self._record_input()
        suffix = '·侧移' if use_strafe else ''
        lag = self._arrow_lag(point)
        if lag is not None:
            # 行进偏：实际行进方向与路线段的夹角，混合有没有把方向掰过去就看这个。
            suffix += f'·行进偏{lag:.0f}度'
        return self.round_wait(f'八向混合 {angle_diff:.0f}度{suffix}')

    def _eight_way_step(self, magnitude: float) -> tuple[bool, bool]:
        """按镜头偏差做八向混合，返回本轮 (是否按前进, 是否按侧移)。

        镜头偏差 Δ 要走的实际方向就在镜头和侧移之间：Δ≤45 度时在「前进」与
        「前进+侧移」之间按 Δ/45 交替，45~112.5 度时在「前进+侧移」与「只侧移」
        之间按 (Δ-45)/45 交替。按比例交替出来的平均方向就是 Δ，所以不用等镜头转完。
        """
        now = time.monotonic()
        delta = now - self.mix_last_at if self.mix_last_at > 0 else DIAGONAL_ROUND_SECONDS
        self.mix_last_at = now
        self.mix_phase = (self.mix_phase + delta / MIX_CYCLE_SECONDS) % 1.0
        if magnitude <= TURN_MIX_MID:
            depth = magnitude / TURN_MIX_MID
            shallow, deep = (True, False), (True, True)
        else:
            depth = (magnitude - TURN_MIX_MID) / TURN_MIX_MID
            shallow, deep = (True, True), (False, True)
        depth = max(0.0, min(depth, 1.0))
        return deep if self.mix_phase < depth else shallow

    def _submit_turn(self, target: float, cruise_only: bool = False, deadband: float = TURN_DEADBAND) -> None:
        """提交目标方向给转向线程；只存不定夺，不记输入。

        转不转、转多少、校准，全由转向线程自己采样、稳 3 帧后定。
        这里不调 _record_input：提交只是写内存，不是输入，
        记了会逼主线每轮多等一张新截图，把主循环限速。

        Args:
            target: 目标方向（度）。
            cruise_only: 近距离只做边走边转，线程内限幅 15 度。
            deadband: 本次目标的对准死区；快速移动传 2 度，其余用默认 5 度。
        """
        self.input_worker.submit_turn(target, cruise_only=cruise_only, deadband=deadband)

    def _seed_turn_ratio(self) -> float:
        """取本局补偿比例初值：累积池够用就用池中位数，否则退回上一局的比例。

        上一局存的是单局估计（局间方差约 ±5%），池中位数收敛得多，用它起步能省掉
        开局几笔大角的偏差。

        Returns:
            本局转向线程的补偿比例初值。
        """
        turn_dx = self.ctx.game_config.turn_dx
        samples = self._load_turn_scale_samples(turn_dx)
        if len(samples) >= TURN_SCALE_SEED_SAMPLES:
            seed = float(median(samples))
            spread = _robust_spread(samples)
            min_scale, max_scale = TURN_SCALE_BOUNDS
            if spread <= TURN_SCALE_SEED_SPREAD and min_scale <= seed <= max_scale:
                log.info(
                    '贝果转向本局用累积池中位数 %.3f 起步（%s 样本，离散 %.3f）',
                    seed, len(samples), spread,
                )
                return seed
        try:
            ratio = self.store.load_turn_ratio(turn_dx)
        except Exception:
            log.warning('读转向补偿比例失败，本局从 1.0 开始学', exc_info=True)
            return 1.0
        log.info('贝果转向本局沿用上局补偿比例 %.3f 起步', ratio)
        return ratio

    def persist_turn_scale(self) -> None:
        """把累积学到的补偿比例写回灵敏度字段 turn_dx，供下一局直接用。

        样本逐局累积到 config/bagel/turn_scale_samples.yml：单局只有二十来个样本时
        补偿估计的 1σ 有 ±10~20%，拿单局结果直接写就会让 turn_dx 在两局之间来回摆
        （实测两分钟内 -3.527 → -3.333 → -3.645）。所以：
        - 每局把静止下发的样本并进累积池（turn_dx 变了就作废重来）；
        - 用峰值箱估补偿，脏样本够不着峰值自然丢掉；
        - 偏离超过「3 倍标准误」（且不低于 5%）才写，标准误随样本累计开方下降，
          所以刚装机灵敏度差 20% 的第一局就会写，累计够了才追得动 2~3% 的真实漂移；
        - 步长按 峰值样本数/(峰值样本数+40) 走，证据少时只挪一小步；
        - 写成功后清空累积池，让新灵敏度重新累积。
        """
        samples = self.input_worker.turn_scale_samples
        if len(samples) < TURN_SCALE_LESSON_SAMPLES:
            log.info('贝果转向补偿本局只有 %s 条静止下发样本，不够 %s，跳过写回',
                     len(samples), TURN_SCALE_LESSON_SAMPLES)
            return
        old_dx = float(self.ctx.game_config.turn_dx)
        history = self._load_turn_scale_samples(old_dx)
        merged = history + samples
        scale = median(merged)
        min_scale, max_scale = TURN_SCALE_BOUNDS
        if not min_scale <= scale <= max_scale:
            log.info('贝果转向校准值 %.3f 超出可信范围，不写回 turn_dx', scale)
            return

        spread = _robust_spread(merged)
        need = max(TURN_SCALE_MIN_CHANGE, TURN_SCALE_SIGNIFICANCE * spread / sqrt(len(merged)))
        if abs(scale - 1.0) < need:
            self._save_turn_scale_samples(old_dx, merged)
            self._save_turn_ratio(old_dx)
            log.info(
                '贝果转向补偿 %.3f 偏离不足门槛 %.3f（累计 %s 样本），先累积不写回',
                scale, need, len(merged),
            )
            return

        weight = len(merged) / (len(merged) + TURN_SCALE_EVIDENCE_SAMPLES)
        step = float(scale) ** weight
        new_dx = round(old_dx * step, 6)
        try:
            self.ctx.game_config.turn_dx = new_dx
        except Exception:
            log.error('转向补偿写回 turn_dx 失败', exc_info=True)
            return
        self._save_turn_scale_samples(new_dx, [])
        log.info(
            '贝果转向补偿 %.3f 已写回 turn_dx（%.4f → %.4f，走 %.3f 步，门槛 %.3f，'
            '累计 %s 样本），下一局直接用',
            scale, old_dx, new_dx, step, need, len(merged),
        )
        self._save_turn_ratio(new_dx)

    def _save_turn_ratio(self, turn_dx: float) -> None:
        """把本局收敛的实时补偿比例存下来，供下一局开局直接用。"""
        try:
            self.store.save_turn_ratio(turn_dx, self.input_worker.turn_ratio)
        except Exception:
            log.warning('写转向补偿比例失败', exc_info=True)

    def _load_turn_scale_samples(self, turn_dx: float) -> list[float]:
        """读跨局累积样本；损坏或灵敏度被改过都当空池，不影响本局运行。"""
        try:
            return self.store.load_turn_scale_samples(turn_dx)
        except Exception:
            log.warning('读转向补偿累积样本失败，按空池处理', exc_info=True)
            return []

    def _save_turn_scale_samples(self, turn_dx: float, samples: list[float]) -> None:
        """写跨局累积样本；写失败只影响下一局的收敛速度。"""
        try:
            self.store.save_turn_scale_samples(turn_dx, samples)
        except Exception:
            log.warning('写转向补偿累积样本失败', exc_info=True)

    def _set_sprint(self, distance: float, aligned: bool = True) -> None:
        """按距离和朝向决定是否冲刺。

        绝区零里冲刺是「按住方向键 + 连点闪避」，所以冲刺时持续点闪避，
        不需要按住闪避键。

        Args:
            distance: 距当前目标点的距离。
            aligned: 朝向是否已经进入冲刺闸门；没对准就先走普通速度。
        """
        want = distance > SPRINT_DISTANCE and aligned
        if want:
            self.sprinting = True
            self._dodge_tap()
        elif self.sprinting:
            self._exit_sprint()

    def _dodge_tap(self) -> None:
        """点一下闪避：冲刺期间每轮都点，维持冲刺状态。"""
        self.input_worker.dodge()

    def _exit_sprint(self) -> None:
        """退出冲刺：松开方向键。"""
        self.sprinting = False
        self.input_worker.stop_forward()

    def _stop_sprint(self) -> None:
        """立刻松键。"""
        self.sprinting = False
        self.input_worker.stop_forward()

    def _prepare_forward_step(self) -> OperationRoundResult | None:
        """上一短按结束并取得松键后的截图，才允许发起下一步。"""
        if self.input_worker.forward_step_active():
            return self.round_wait('等待本次前进短按结束', wait=0.02)
        self._release_forward()
        released_at = self.input_worker.forward_released_at
        if released_at is None or self.last_screenshot_time <= released_at:
            return self.round_wait('等待松键后的新截图', wait=0.02)
        return None

    def _short_step(self) -> OperationRoundResult:
        """近距离移动：不停下来对准，直接短按前进。

        3px 内不再转向：这么近的距离，目标方向完全由 1px 级的坐标抖动
        决定，算出来的角度没有意义。一旦按角度判停，就会出现「每轮都在
        转向、一步也不走」的原地打摆。

        方向真的偏了也不怕：走一步后距离变大，自然回到慢步分支去对准。
        超时由调用方 _check_small_step_timeout 管，这里只负责迈步。
        """
        ready = self._prepare_forward_step()
        if ready is not None:
            return ready
        limited = self._fail_if_action_limit()
        if limited is not None:
            return limited
        if not self._step_forward(SMALL_STEP_PRESS):
            return self.round_wait('等待前进短按可用', wait=0.02)
        self._record_input()
        return self.round_wait('短步前进')

    def _check_small_step_timeout(self) -> OperationRoundResult | None:
        """碎步段超时检查：3px 内迈步超过 5 秒还没贴到就跳过这个点。

        碎步不看角度硬迈，方向偏了会一直原地蹭，靠单点 75 秒兜底太慢，
        所以这一小段单独计时。超时说明这个点贴不上（站位被挡、坐标偏），
        直接推进到下一个点，不判失败、也不死等。

        Returns:
            跳过结果；未超时返回 None。
        """
        if self.small_step_since is None:
            self.small_step_since = time.monotonic()
            return None
        if time.monotonic() - self.small_step_since < SMALL_STEP_TIMEOUT:
            return None
        self.small_step_since = None
        self._release_forward()
        log.info(
            '贝果第 %s 点（%s）碎步 %ss 没贴上，跳到下一个点',
            self.cursor + 1, self._current_kind_name(), f'{SMALL_STEP_TIMEOUT:.0f}',
        )
        self._enter_point(self.route_index, self.cursor + 1)
        return self.round_success('跳过此点')

    def _current_kind_name(self) -> str:
        """当前游标点的类型名，只用于日志。"""
        point = self._current_point()
        return point.kind if point is not None else '无'

    def _slow_step(
        self, point: BagelRoutePoint, position: tuple[float, float],
        short_press: bool = False,
    ) -> OperationRoundResult:
        """按目标方向步进，电子箱短步阶段使用更短的按压时间。

        只提交目标方向给转向线程；偏差超过 10 度时停车等转，
        其余情况短按前进。每步结束后使用新截图重新判断，
        避免多次短按相互覆盖。

        近距离不再按角度停车转：这个距离上的目标方位由定位抖动决定，
        改成边走边转（线程内限幅 15 度）。

        Args:
            point: 当前目标点。
            position: 当前可靠定位。
            short_press: 是否使用 0.08 秒短步；否则使用 0.5 秒慢步。
        """
        ready = self._prepare_forward_step()
        if ready is not None:
            return ready
        action = '短步' if short_press else '慢步'
        angle = self.player_angle()
        miss = self._angle_miss_result(angle)
        if miss is not None:
            return miss
        target_angle = self._target_angle(position, point)
        angle_diff = cal_utils.angle_delta(angle % 360, target_angle)
        distance = hypot(point.x - position[0], point.y - position[1])
        near = distance <= NEAR_STEP_TURN_RADIUS
        self._submit_turn(target_angle, cruise_only=near)
        if abs(angle_diff) > WALK_TURN_ANGLE and not near:
            # 大偏差：松键停车等转向线程转到位，不短按前进。
            self._release_forward()
            return self.round_wait(f'{action}停车等转向 {angle_diff:.0f}度')
        limited = self._fail_if_action_limit()
        if limited is not None:
            return limited
        duration = SMALL_STEP_PRESS if short_press else SLOW_STEP_PRESS
        if not self._step_forward(duration):
            return self.round_wait('等待前进短按可用', wait=0.02)
        self._record_input()
        wait = None if short_press else SLOW_STEP_WAIT
        return self.round_wait(f'{action}前进', wait=wait)

    @staticmethod
    def _target_angle(
        position: tuple[float, float], point: BagelRoutePoint,
    ) -> float:
        """当前位置指向目标点的角度。

        与朝向识别（analyze）同一套坐标系：屏幕上方为正、逆时针增加，
        所以纵向取负号；符号写反会让角色一直原地打转。
        """
        return BagelRouteWalker._angle_to(position, point.x, point.y)

    @staticmethod
    def _angle_to(position: tuple[float, float], x: float, y: float) -> float:
        """当前位置指向坐标 (x, y) 的角度，与 _target_angle 同一套坐标系。"""
        return degrees(atan2(-(y - position[1]), x - position[0])) % 360

    def _movement_heading(self) -> float | None:
        """用最近位移反推实际行进方向（度，与 _target_angle 同一坐标系）。

        镜头对准不等于人在朝那个方向走：角色箭头比镜头滞后约 0.35 秒。
        位移不足 ARROW_MIN_DISPLACEMENT 或定位太旧时返回 None。
        """
        if len(self.pos_history) < 2:
            return None
        now = time.monotonic()
        x1, y1 = self.pos_history[-1][1], self.pos_history[-1][2]
        for t0, x0, y0 in reversed(self.pos_history):
            if now - t0 > ARROW_MAX_AGE:
                break
            if hypot(x1 - x0, y1 - y0) >= ARROW_MIN_DISPLACEMENT:
                return self._angle_to((x0, y0), x1, y1)
        return None

    def _arrow_lag(self, point: BagelRoutePoint) -> float | None:
        """实际行进方向与路线段方向的夹角（度，取绝对值）；量不到返回 None。

        比对的是「上一个点→当前目标点」这条段的方向，不是瞬时瞄点方位：
        离点 5px 内瞄点方位自身就会摆 60~80 度，拿它当基准全是假滞后。
        """
        origin = self._previous_point()
        if origin is None:
            return None
        if hypot(point.x - origin.x, point.y - origin.y) < STRAFE_MIN_SEGMENT:
            return None
        heading = self._movement_heading()
        if heading is None:
            return None
        path_heading = self._angle_to((origin.x, origin.y), point.x, point.y)
        return abs(cal_utils.angle_delta(heading, path_heading))

    def _aim_point(
        self, position: tuple[float, float], point: BagelRoutePoint,
    ) -> tuple[float, float]:
        """巡航瞄准点：路径线上前方 LOOKAHEAD_DISTANCE 处的点。

        没有上一个点、路径线太短、或剩余距离不足前视距离时，直接返回目标点本身。
        """
        origin = self._previous_point()
        if origin is None:
            return (point.x, point.y)
        length = hypot(point.x - origin.x, point.y - origin.y)
        if length < STRAFE_MIN_SEGMENT:
            return (point.x, point.y)
        foot_x, foot_y = _closest_on_segment(origin, point, position)
        remain = hypot(point.x - foot_x, point.y - foot_y)
        if remain <= LOOKAHEAD_DISTANCE:
            return (point.x, point.y)
        ux = (point.x - origin.x) / length
        uy = (point.y - origin.y) / length
        return (foot_x + ux * LOOKAHEAD_DISTANCE, foot_y + uy * LOOKAHEAD_DISTANCE)

    def _check_stuck(
        self, position: tuple[float, float], point: BagelRoutePoint, angle: float,
    ) -> OperationRoundResult | None:
        """按住前进却走不动时，往路径线那一侧侧移一小步。

        走不动一般不是被挡住，而是已经偏到路径线外，正顶着墙或箱子。
        这时继续判失败没意义。拿「上一个点 → 当前目标点」这条连线当路径线，
        朝线上的最近点挪一小步，让前进方向重新对上线。
        """
        if self.stuck_since is None:
            self.stuck_since = (position, time.monotonic())
            return None
        origin, started = self.stuck_since
        if hypot(position[0] - origin[0], position[1] - origin[1]) >= STUCK_DISTANCE:
            self.stuck_since = (position, time.monotonic())
            self.stuck_steps = 0
            return None
        if time.monotonic() - started < STUCK_SECONDS:
            return None
        if self.stuck_steps >= STUCK_STEP_LIMIT:
            self._release_forward()
            return self.round_recoverable_fail('持续前进但位置未变化，侧移多次仍无法脱离')
        direction = self._path_side(position, point, angle)
        self.stuck_steps += 1
        self.stuck_since = None
        if direction is None:
            # 没有上一个点，或已经站在线上，判断不出该往哪挪，不瞎动。
            return self.round_wait('前进受阻，等待重新判定路径线', wait=0.2)
        self.input_worker.side_step(direction, SIDE_STEP_SECONDS)
        self._record_input()
        return self.round_wait(
            f'前进受阻，往路径线{"右" if direction > 0 else "左"}侧移动一步',
            wait=SIDE_STEP_SECONDS + 0.1,
        )

    def _path_side(
        self, position: tuple[float, float], point: BagelRoutePoint, angle: float,
    ) -> int | None:
        """该往哪一侧挪：1 按 D（角色右手边），-1 按 A（角色左手边）。

        A/D 是相对角色朝向的左右，不是地图上的左右，所以要先把「指向路径线」
        的方向换算到角色局部坐标：角色朝向角 angle 下，右手方向是
        `(sin angle, cos angle)`（地图 y 轴向下），与它点乘为正就是往右挪。

        路径线取上一个点到当前目标点的连线；角色在线的哪一侧都不必先判，
        直接朝线上最近点走，偏多偏少都成立。
        """
        origin = self._previous_point()
        if origin is None:
            return None
        foot = _closest_on_segment(origin, point, position)
        to_line = (foot[0] - position[0], foot[1] - position[1])
        if hypot(to_line[0], to_line[1]) < 0.5:
            return None
        right_x, right_y = sin(radians(angle)), cos(radians(angle))
        return 1 if to_line[0] * right_x + to_line[1] * right_y >= 0 else -1

    def _previous_point(self) -> BagelRoutePoint | None:
        """上一个点；路线起点没有上一个点时返回 None。"""
        if not self.routes:
            return None
        route = self.routes[self.route_index]
        if self.cursor <= 0:
            return None
        return route.points[self.cursor - 1]

    def _check_timeout(self, point: BagelRoutePoint) -> OperationRoundResult | None:
        """单点超时检查。"""
        if self.point_started_at is None:
            self.point_started_at = time.monotonic()
            return None
        limit = POINT_TIMEOUT.get(point.kind, POINT_TIMEOUT['default'])
        if time.monotonic() - self.point_started_at >= limit:
            self._release_forward()
            return self.round_recoverable_fail(
                f'第 {self.cursor + 1} 点（{point.kind}）超时 {limit:.0f}s',
            )
        return None

    def _check_hud(self) -> OperationRoundResult | None:
        """局内 HUD 检查：暂停菜单直接失败。

        这里不能因为没识别到普通攻击按钮就松前进键。长距离行进时
        只要某一帧模板没命中，前进键就被松开，表现就是走两步停一下。
        只有确认打开了暂停菜单才停。
        """
        if self.round_by_find_area(
            self.last_screenshot, '战斗-菜单', '按钮-退出战斗',
        ).is_success:
            self._release_forward()
            return self.round_recoverable_fail('移动中打开了暂停菜单')
        return None

    def _fail_if_action_limit(self) -> OperationRoundResult | None:
        """单点输入次数上限。"""
        if self.steps < ACTION_LIMIT:
            return None
        self._release_forward()
        return self.round_recoverable_fail('单点输入达到上限')

    def _current_point(self) -> BagelRoutePoint | None:
        """当前路线游标处的点；还没确定路线时返回 None。"""
        if not self.routes:
            return None
        route = self.routes[self.route_index]
        if self.cursor >= len(route.points):
            return None
        return route.points[self.cursor]

    def _enter_point(self, route_index: int, point_index: int) -> None:
        """进入某个点：切换游标并清空单点状态。"""
        self.route_index = route_index
        self.cursor = point_index
        self.point_started_at = time.monotonic()
        self._reset_point_state()

    def _reset_point_state(self) -> None:
        """清空与单个点绑定的交互状态。

        转向比例不重置：它是整局共享的校准结果，换点后继续沿用。
        只清掉待学习样本，避免拿跨点的旧样本反推。
        """
        self.current_kind = None
        self.f_pressed = False
        self.mechanical_hold_until = None
        self.mechanical_retried = False
        self.target_wait_since = None
        self.electronic_small_steps = False
        self.small_step_since = None
        self.steps = 0
        self.stuck_since = None
        self.stuck_steps = 0
        self.overshoot_steps = 0
        self.diagonal_seconds = 0.0
        self.mix_phase = 0.0
        self.mix_last_at = 0.0
        self.blind_walk_ok = False
        self.vector_last_distance = None
        self.vector_stall_since = None
        self.vector_fallbacks = 0
        self.pos_history.clear()
        self.elevator_phase = None
        self.elevator_since = None

    def _step_forward(self, duration: float) -> bool:
        """请求一次前进短按，返回输入线程是否接受。

        Args:
            duration: 这一次按住的时长（秒）。
        """
        return self.input_worker.step_forward(duration)

    def _record_input(self) -> None:
        """记录一次输入并拒绝复用旧截图。"""
        self.steps += 1
        self.last_input_frame = max(time.time(), self.last_screenshot_time)

    def _release_forward(self) -> None:
        """松开前进键与冲刺键，并清卡位计时与步进计时。"""
        self.stuck_since = None
        self._stop_sprint()
        self.input_worker.stop_forward()
        self.input_worker.stop_backward()

    def _read_survey_value(self) -> None:
        """关闭面板回到局内后读 HUD 调查收益；读不到时保留上一次读数。"""
        value = self._read_hud_survey()
        if value is not None:
            self.survey_value = value

    def _read_hud_survey(self) -> int | None:
        """读 HUD 调查收益区域最下面一行的数字，只用于统计日志。

        物品入箱后，HUD 数字上方可能浮出该件涨幅（如「+10K」），会落进读数区域。
        OCR 读不出浮字前的「+」，整块拼起来会先读到涨幅，所以只取最下面一行。

        Returns:
            读数；读不出数字或读取出错时为 None。
        """
        try:
            area = self.ctx.screen_loader.get_area('贝果-局内', '调查收益')
            if area is None:
                raise ValueError('缺少贝果画面区域：贝果-局内/调查收益')
            matches = self.ctx.ocr_service.get_ocr_result_list(
                self.last_screenshot, rect=area.pc_rect, crop_first=True, color_range=area.color_range,
            )
        except Exception:
            log.error('读取调查收益失败，本次不计入统计', exc_info=True)
            return None
        if not matches:
            return None
        bottom = max(match.center.y for match in matches)
        line = sorted(
            (match for match in matches if bottom - match.center.y <= SURVEY_LINE_TOLERANCE),
            key=lambda match: match.x,
        )
        return parse_value(''.join(match.data for match in line))

    def _adopt_screenshot(self, op: BagelOperation) -> None:
        """子操作结束后同步其最后截图。"""
        if op.last_screenshot is not None:
            self.last_screenshot = op.last_screenshot
            self.last_screenshot_time = op.last_screenshot_time

    def handle_pause(self) -> None:
        """暂停时把所有键松开，并让转向线程立刻停手。

        暂停期间主线不再提交，靠提交新鲜度线程 1 秒后也会自己停；这里主动清一次，
        避免暂停瞬间按旧目标转镜头。按键是注入到系统全局的，暂停（尤其是人工暂停）
        后可能长时间不恢复，残留的 W/A/S/D 会被一直按住，所以一律全松。
        """
        self.input_worker.release_all()
        self.input_worker.submit_turn(None)
        super().handle_pause()

    def after_operation_done(self, result: OperationResult) -> None:
        """退出路径统一松键，含冲刺键，并停掉输入线程。"""
        self._stop_sprint()
        self.input_worker.shutdown()
        super().after_operation_done(result)
