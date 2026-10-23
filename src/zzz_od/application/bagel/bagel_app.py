from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING

import cv2
import numpy as np

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import cv2_utils
from one_dragon.utils.log_utils import log
from zzz_od.application.bagel import bagel_const, bagel_usage
from zzz_od.application.bagel.bagel_deposit import BagelDeposit
from zzz_od.application.bagel.bagel_enter import BagelEnter
from zzz_od.application.bagel.bagel_exit import BagelExit
from zzz_od.application.bagel.bagel_flow import BagelFlow, load_published_flow
from zzz_od.application.bagel.bagel_map_locator import minimap_mask
from zzz_od.application.bagel.bagel_operation import (
    BagelOperation,
    BagelRecoverableFailure,
    bagel_run_events,
    execute_bagel_round,
    is_bagel_result,
)
from zzz_od.application.bagel.bagel_return import BagelReturn
from zzz_od.application.bagel.bagel_route import SUPPORTED_MAP_IDS
from zzz_od.application.bagel.bagel_route_vision import BagelSpawnMatcher
from zzz_od.application.bagel.bagel_run_flow import BagelRunFlow, release_flow_inputs
from zzz_od.application.bagel.bagel_settle import BagelSettleWarehouse
from zzz_od.application.bagel.bagel_slots import safe_occupied_indices
from zzz_od.application.zzz_application import ZApplication

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from one_dragon.base.operation.operation_base import OperationResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.application.bagel.bagel_config import BagelConfig
    from zzz_od.application.bagel.bagel_run_record import BagelRunRecord
    from zzz_od.context.zzz_context import ZContext


class BagelApp(ZApplication):
    """支持出生点收集并按成功入仓局数循环。"""

    STATUS_SKIP: str = '非支持出生点，退出重开'
    STATUS_A: str = '录像店复活点，开始收集'
    STATUS_B: str = '白鸽工地地铁站复活点，开始收集'
    STATUS_C: str = '单廊双容器复活点，开始收集'
    # 底图标识 -> (状态名, 出生点名称)。原地开始与正常入场共用这一张表。
    SPAWN_STATUS: dict[str, tuple[str, str]] = {
        'janus_high_a': (STATUS_A, '录像店复活点'),
        'janus_high_b': (STATUS_B, '白鸽工地地铁站复活点'),
        'janus_high_c': (STATUS_C, '单廊双容器复活点'),
    }

    def __init__(
        self, ctx: ZContext, config: BagelConfig, record: BagelRunRecord,
    ) -> None:
        """注入工厂对应账号、分组的配置。"""
        super().__init__(
            ctx,
            app_id=bagel_const.APP_ID,
            op_name=bagel_const.APP_NAME,
            need_check_game_win=True,
            run_record=record,
        )
        self.config: BagelConfig = config
        self.spawn_matcher: BagelSpawnMatcher | None = None
        self.matched_map_id: str | None = None
        # 原地开始是否已接管当前这一局；接管后入场不再把它当遗留局退出。
        self.started_in_place: bool = False
        self.attempts: int = 0
        self.success_rounds: int = 0
        self.rounds_since_sell: int = 0
        self.empty_rounds: int = 0
        self.defeat_rounds: int = 0
        self.failure_retries_used: int = 0
        self.failure_retry_pending: bool = False
        self.failure_reason: str | None = None
        self.failure_history: list[dict[str, object]] = []
        self._spawn_hud_misses: int = 0
        # 出生识别暗帧兜底：连续偏暗未命中的轮数、上一帧小地图签名、已连续收敛帧数。
        self._spawn_dim_misses: int = 0
        self._spawn_prev_signature: np.ndarray | None = None
        self._spawn_stable_frames: int = 0
        self.flow_snapshot: dict[str, BagelFlow] = {}
        self.initial_clear_pending: bool = True
        self.last_return_status: str | None = None

    def execute(self) -> OperationResult:
        """正式应用返回前完成本次运行的事件收尾。"""
        with bagel_run_events(self):
            try:
                result = super().execute()
                if self.run_record is not None:
                    self.run_record.update('retry_summary', {
                        'used': self.failure_retries_used,
                        'limit': self.config.max_failure_retries,
                        'failures': self.failure_history,
                        'success_rounds': self.success_rounds,
                        'stop_reason': result.status,
                        'stop_detail': str(result.data) if result.data is not None else None,
                    })
                log.info('贝果任务结束：%s；整体重试已用 %s/%s',
                         result.status, self.failure_retries_used, self.config.max_failure_retries)
                return result
            finally:
                release_flow_inputs(self.ctx)

    def after_operation_done(self, result: OperationResult) -> None:
        """正式任务停止也保存现场，程序错误或收尾失败不能只留下旧的局内截图。"""
        try:
            if not result.success and self.last_screenshot is not None:
                self.screenshot()
                path = self.save_screenshot()
                log.error('贝果任务停止：%s；原因：%s', result.status, result.data)
                bagel_usage.log_screenshot(path)
            elif not result.success:
                log.warning('没有可保存的现场截图。请查看相关日志。')
        except Exception:
            log.error('保存贝果任务停止截图时发生异常', exc_info=True)
            log.warning('本次未确认保存现场截图。请查看相关日志。')
        finally:
            if not result.success:
                log.error('停止原因：%s；详情：%s', result.status, result.data)
                log.info(bagel_usage.stop_guidance(result.status))
                bagel_usage.log_feedback()
            super().after_operation_done(result)

    def _execute_one_round(self) -> OperationRoundResult:
        """初始化以外的程序异常也直接停止，不获得业务重开额度。"""
        return execute_bagel_round(self)

    def handle_pause(self) -> None:
        """暂停释放按键，不打断当前拖拽，也不重置累计状态。"""
        release_flow_inputs(self.ctx, release_mouse=False)
        super().handle_pause()

    def handle_init(self) -> None:
        """新任务重置计数；子操作切换和暂停恢复不调用此初始化。"""
        super().handle_init()
        self.attempts = 0
        self.success_rounds = 0
        self.rounds_since_sell = 0
        self.empty_rounds = 0
        self.defeat_rounds = 0
        self.failure_retries_used = 0
        self.failure_retry_pending = False
        self.failure_reason = None
        self.failure_history = []
        self.matched_map_id = None
        self._spawn_hud_misses = 0
        self._reset_spawn_frame_state()
        self.spawn_matcher = None
        self.flow_snapshot = {}
        self.initial_clear_pending = True
        self.last_return_status = None

    def _spawn_matcher(self) -> BagelSpawnMatcher:
        """延迟加载出生匹配，避免无资源环境构造失败。"""
        if self.spawn_matcher is None:
            self.spawn_matcher = BagelSpawnMatcher()
        return self.spawn_matcher

    @operation_node(
        name='检查贝果运行条件', is_start_node=True, screenshot_before_round=False,
    )
    def check_ready(self) -> OperationRoundResult:
        """校验配置和路线；是否零携带由入场时的当前画面确认。"""
        try:
            self.config.validate()
            self.flow_snapshot = {map_id: load_published_flow(map_id) for map_id in SUPPORTED_MAP_IDS}
            self._spawn_matcher()  # 入场前验证固定地图资源，避免进入局内才发现文件损坏。
        except (OSError, ValueError) as error:
            return self.round_fail(f'贝果配置无效：{error}')
        bagel_usage.log_start(self.config)
        if self.config.start_in_place:
            started = self._try_start_in_place()
            if started is not None:
                return started
        return self.round_success()

    def _try_start_in_place(self) -> OperationRoundResult | None:
        """已在某个支持出生点附近时直接进该路线，不重开；不适用时返回 None。

        开启 `start_in_place` 后，站在已支持路线的出生点附近即可直接开始那条路线，
        避免反复重开被随机到别的地图。判定放在「检查贝果运行条件」末尾 ——
        那是本次任务的唯一出口，不会与「零携带入场」的既有出边产生歧义。
        """
        self.screenshot()
        if is_bagel_result(self):
            return None
        if not self.round_by_find_area(
            self.last_screenshot, '战斗画面', '按键-普通攻击',
        ).is_success:
            self._spawn_hud_misses += 1
            if self._spawn_hud_misses < bagel_const.START_IN_PLACE_HUD_MISS_LIMIT:
                return self.round_wait(
                    f'局内 HUD 暂未识别（{self._spawn_hud_misses}/'
                    f'{bagel_const.START_IN_PLACE_HUD_MISS_LIMIT}），再看一帧',
                    wait=bagel_const.START_IN_PLACE_HUD_WAIT,
                )
            log.info('原地开始：未识别到局内画面，照常零携带入场')
            return None
        self._spawn_hud_misses = 0
        area = self.ctx.screen_loader.get_area('贝果-局内', '定位小地图')
        if area is None:
            return self.round_fail('缺少贝果定位小地图区域')
        crop = cv2_utils.crop_image_only(self.last_screenshot, area.pc_rect)
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        map_id = self._spawn_matcher().match(crop)
        spawn_status = self.SPAWN_STATUS.get(map_id) if map_id is not None else None
        if spawn_status is None:
            log.info('原地开始：当前小地图认不出支持出生点，照常零携带入场')
            return None
        # 认出了地图，还要确认玩家确实站在出生点附近：走远了直接跑该路线会从
        # 错误位置起步，比重开更糟。
        vision = self._spawn_matcher().vision(map_id)
        location = vision.last_location
        if location is None or location.position is None:
            log.info('原地开始：认出%s但取不到坐标，照常零携带入场', spawn_status[1])
            return None
        distance = hypot(location.position[0] - vision.spawn[0], location.position[1] - vision.spawn[1])
        if distance > bagel_const.START_IN_PLACE_SPAWN_RADIUS:
            log.info(
                '原地开始：认出%s但离出生点 %.1f 格（上限 %.1f），照常零携带入场',
                spawn_status[1], distance, bagel_const.START_IN_PLACE_SPAWN_RADIUS,
            )
            return None
        self.matched_map_id = map_id
        self.attempts += 1
        self.started_in_place = True
        log.info('原地开始：已在%s附近（距出生点 %.1f 格），直接开始收集', spawn_status[1], distance)
        return self.round_success(spawn_status[0])

    @node_from(from_name='检查贝果运行条件')
    @node_from(from_name='返回入口')
    @node_from(from_name='判断成功局数', status='继续入场')
    @operation_node(name='零携带入场', screenshot_before_round=False)
    def enter(self) -> OperationRoundResult:
        """非支持出生点可继续重开，不受尝试次数限制。"""
        self.screenshot()
        allow_clear_loadout = self.initial_clear_pending
        # 资格在调用入场前消耗，不能等出生识别增加 attempts 才关闭。
        self.initial_clear_pending = False
        if not allow_clear_loadout and all(
            self.round_by_find_area(self.last_screenshot, '贝果-仓库', name).is_success
            for name in ('放入仓库', '返回研究站')
        ):
            occupied = safe_occupied_indices(self.last_screenshot)
            if occupied is None:
                return self.round_fail('结算仓库安全箱格子状态不明，禁止开始新局')
            if occupied:
                return self.round_fail('结算仓库安全箱仍有物资，先核对去向，禁止开始新局')
        self._spawn_hud_misses = 0
        retrying_failure = self.failure_retry_pending
        if retrying_failure:
            self.failure_retry_pending = False
            self.failure_retries_used += 1
            log.warning('贝果失败重开：%s；整体重试已用 %s/%s',
                        self.failure_reason, self.failure_retries_used, self.config.max_failure_retries)
        log.info('贝果收集：准备第 %s 次入场', self.attempts + 1)
        result = BagelEnter(
            self.ctx, allow_clear_loadout=allow_clear_loadout, allow_world_recovery=True,
            prepare_shortcut_entry=allow_clear_loadout,
            recovery_auto_clean=self.config.auto_clean_warehouse,
            recovery_filter_groups=self.config.clean_filter_groups(),
            keep_started_round=self.started_in_place,
        ).execute()
        if retrying_failure and not result.success:
            return self._finish_failed_round(result, '额外入场')
        return self.round_by_op_result(result, wait=1)

    def _reset_spawn_frame_state(self) -> None:
        """清掉跨帧状态，避免下一局沿用上一局的暗帧计数与签名。"""
        self._spawn_dim_misses = 0
        self._spawn_prev_signature = None
        self._spawn_stable_frames = 0

    def _spawn_frame_is_settled(self, minimap: MatLike) -> bool:
        """画面是否已停止重绘（定位小地图几乎不变，且不再偏暗）。

        判据是帧间灰度差而不是「够不够亮」：稳定态的未建档布局同样落在偏暗门槛
        下方，但它是静止的，等下去永远不会变亮；入场过渡态则每帧都在刷新。两者
        相差 20 倍以上，因此画面一旦收敛就立刻退出，不再耗满重试次数。

        要求连续两帧都收敛：加载卡顿时过渡期可能恰好停住一两帧，只看单帧会把仍在
        入场的一局误杀。

        收敛还必须同时「不偏暗」：入场动画后段每帧变化都很小（实测有效像素 2264 →
        2265，帧间差已低于阈值），但画面其实远未稳定 —— 实测此时有效像素只有
        2265，正常帧4800 以上。只看帧间差会在这时候误判成已稳定，丢掉这一局。
        """
        if minimap is None or not minimap.size:
            return True
        signature = cv2.resize(
            cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        previous = self._spawn_prev_signature
        self._spawn_prev_signature = signature
        if previous is None or previous.shape != signature.shape:
            self._spawn_stable_frames = 0
            return False
        if float(np.abs(signature - previous).mean()) > bagel_const.SPAWN_STABLE_MAX_DIFF:
            self._spawn_stable_frames = 0
            return False
        if self._spawn_valid_pixels(minimap) < bagel_const.SPAWN_DIM_VALID_PIXELS:
            # 帧间差已经很小，但画面仍然偏暗：还在入场，不算稳定。
            self._spawn_stable_frames = 0
            return False
        self._spawn_stable_frames += 1
        return self._spawn_stable_frames >= bagel_const.SPAWN_STABLE_FRAMES

    def _spawn_valid_pixels(self, minimap: MatLike) -> int:
        """定位小地图里可用于匹配的有效像素数（各图口径一致，取第一张即可）。"""
        try:
            vision = next(iter(self._spawn_matcher().routes.values()))
            return int((minimap_mask(minimap, vision.map.mask_settings) > 0).sum())
        except (StopIteration, AttributeError):
            return -1

    @node_from(from_name='零携带入场')
    @operation_node(name='识别出生点', timeout_seconds=10)
    def check_spawn(self) -> OperationRoundResult:
        """支持点进入短线收集；其余正常退出重开。"""
        if is_bagel_result(self):
            return self.round_fail(BagelOperation.STATUS_DEFEATED)
        if not self.round_by_find_area(
            self.last_screenshot, '战斗画面', '按键-普通攻击',
        ).is_success:
            self._spawn_hud_misses += 1
            if self._spawn_hud_misses < 4:
                return self.round_wait('局内 HUD 暂未识别，再看一帧', wait=0.3)
            return self.round_fail(BagelOperation.STATUS_ROUND_FAILED, data='未识别贝果局内画面')
        self._spawn_hud_misses = 0
        area = self.ctx.screen_loader.get_area('贝果-局内', '定位小地图')
        if area is None:
            return self.round_fail('缺少贝果定位小地图区域')
        crop = cv2_utils.crop_image_only(self.last_screenshot, area.pc_rect)
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        self.attempts += 1
        map_id = self._spawn_matcher().match(crop)
        if map_id is None and not self._spawn_frame_is_settled(crop):
            # 画面还在重绘：入场动画未结束，定位小地图偏暗且特征不足。此时退出重开
            # 会白白浪费一局，等它稳定下来再判。
            self._spawn_dim_misses += 1
            if self._spawn_dim_misses < bagel_const.SPAWN_DIM_RETRY_LIMIT:
                log.info(
                    '出生点画面偏暗（%s 有效像素），等画面稳定后再判',
                    self._spawn_valid_pixels(crop),
                )
                return self.round_wait(
                    f'出生点画面仍在入场过渡（{self._spawn_dim_misses}/'
                    f'{bagel_const.SPAWN_DIM_RETRY_LIMIT}），再看一帧',
                    wait=bagel_const.SPAWN_STABLE_WAIT,
                )
        self._reset_spawn_frame_state()
        self.matched_map_id = map_id
        if map_id == 'janus_high_a':
            log.info('第 %s 次抽到录像店复活点，开始收集', self.attempts)
            return self.round_success(self.STATUS_A)
        if map_id == 'janus_high_b':
            log.info('第 %s 次抽到白鸽工地地铁站复活点，开始收集', self.attempts)
            return self.round_success(self.STATUS_B)
        if map_id == 'janus_high_c':
            log.info('第 %s 次抽到单廊双容器复活点，开始收集', self.attempts)
            return self.round_success(self.STATUS_C)
        log.info('第 %s 次非支持出生点，退出重开', self.attempts)
        log.info('当前出生点不支持收集，正在退出重开。本次不计成功，不清理仓库，也不消耗整体重试次数。出生点重选没有次数上限。')
        log.info('抽到录像店或白鸽工地地铁站出生点后，程序开始收集。如需结束任务，请使用程序的停止操作。')
        return self.round_success(self.STATUS_SKIP)

    @node_from(from_name='识别出生点', status=STATUS_SKIP)
    @operation_node(name='退出非支持局', screenshot_before_round=False)
    def exit_unsupported(self) -> OperationRoundResult:
        """空跑退出；失败时留图。"""
        return self.round_by_op_result(BagelExit(self.ctx).execute())

    @node_from(from_name='退出非支持局')
    @node_from(from_name='判断整体重试次数', status='继续入场')
    @operation_node(name='返回入口', screenshot_before_round=False)
    def return_entry(self) -> OperationRoundResult:
        """返回贝果主界面后开下一局；大世界落点由入场操作重建入口。"""
        result = BagelReturn(self.ctx).execute()
        return (
            self._finish_failed_round(result, '返回入口') if self.failure_retry_pending
            else self.round_by_op_result(result)
        )

    @node_from(from_name='识别出生点', status=STATUS_A)
    @node_from(from_name='识别出生点', status=STATUS_B)
    @node_from(from_name='识别出生点', status=STATUS_C)
    # 原地开始直接在「检查贝果运行条件」返回同样的状态，不经过识别出生点。
    # 缺这两条时状态匹配会退化到 status=None 那条边，被送进零携带入场。
    @node_from(from_name='检查贝果运行条件', status=STATUS_A)
    @node_from(from_name='检查贝果运行条件', status=STATUS_B)
    @node_from(from_name='检查贝果运行条件', status=STATUS_C)
    @operation_node(name='执行局内流程', screenshot_before_round=False)
    def run_flow(self) -> OperationRoundResult:
        """按出生地执行发布流程；与开发工具共用同一执行器。"""
        if self.matched_map_id not in self.flow_snapshot:
            return self.round_fail('缺少已识别出生地的发布流程')
        result = BagelRunFlow(
            self.ctx, self.flow_snapshot[self.matched_map_id],
            map_snapshot=self._spawn_matcher().vision(self.matched_map_id).map,
            continuous_safe_unlock=True,
            on_location_wait=lambda: log.info(bagel_usage.LOCATION_WAIT_HINT),
        )
        outcome = result.execute()
        if result.last_screenshot is not None:
            self.last_screenshot = result.last_screenshot
            self.last_screenshot_time = result.last_screenshot_time
        if not outcome.success and (
            isinstance(outcome.data, BagelRecoverableFailure)
            or outcome.status == self.STATUS_TIMEOUT
        ):
            reason = outcome.data.reason if isinstance(outcome.data, BagelRecoverableFailure) else outcome.status
            return self.round_fail(BagelOperation.STATUS_ROUND_FAILED, data=reason)
        return self.round_by_op_result(outcome)

    @node_from(from_name='执行局内流程')
    @operation_node(name='结算仓库', screenshot_before_round=False)
    def settle(self) -> OperationRoundResult:
        """只有安全箱有物且确实入仓才计成功；空箱局仍要返回入口。"""
        finishing = (self.config.max_success_rounds > 0
                     and self.success_rounds + 1 >= self.config.max_success_rounds)
        result = self._settle_warehouse(finishing=finishing)
        if result.success:
            if result.status == BagelDeposit.STATUS_DONE:
                self.success_rounds += 1
                self.empty_rounds = 0
            elif result.status == BagelDeposit.STATUS_EMPTY:
                self.empty_rounds += 1
            log.info('贝果单局结算：%s；已入仓 %s 局，连续空箱 %s 局',
                     result.status, self.success_rounds, self.empty_rounds)
        return self.round_by_op_result(result)

    def _settle_warehouse(self, finishing: bool = False) -> OperationResult:
        """应用持有本任务出售计数，结算成功后再更新；失败局也复用。"""
        due = self.rounds_since_sell + 1 >= self.config.sell_interval
        log.info('贝果出售计划：累计入仓 %s/%s；正常结束补卖=%s',
                 self.rounds_since_sell, self.config.sell_interval, finishing)
        result = BagelSettleWarehouse(
            self.ctx, self.config.auto_clean_warehouse,
            filter_groups=self.config.clean_filter_groups(),
            sell_due=due or finishing,
        ).execute()
        if result.success:
            if isinstance(result.data, dict) and result.data.get('sale_completed'):
                self.rounds_since_sell = 0
            elif result.status == BagelDeposit.STATUS_DONE:
                self.rounds_since_sell += 1
        return result

    @node_from(from_name='结算仓库')
    @operation_node(name='成功局返回入口', screenshot_before_round=False)
    def return_after_success(self) -> OperationRoundResult:
        """最后一局也返回，并记录实际到达的页面。"""
        result = BagelReturn(self.ctx).execute()
        self.last_return_status = result.status if result.success else None
        return self.round_by_op_result(result)

    @node_from(from_name='成功局返回入口')
    @operation_node(name='判断成功局数', screenshot_before_round=False)
    def decide_success_rounds(self) -> OperationRoundResult:
        """只有正常入仓的局数计入上限。"""
        if self.config.max_success_rounds > 0 and self.success_rounds >= self.config.max_success_rounds:
            location = {'已返回贝果入口': '贝果入口', '已返回大世界': '大世界'}.get(self.last_return_status)
            if location is not None:
                log.info('已达到成功次数上限。已成功入仓 %s 局。程序已完成结算检查，并返回%s。本次任务已结束。', self.success_rounds, location)
            else:
                log.info('已达到成功次数上限。已成功入仓 %s 局。返回结果：%s。本次任务已结束。', self.success_rounds, self.last_return_status or '位置未确认')
            return self.round_success(f'已成功入仓 {self.success_rounds} 局')
        if self.empty_rounds >= 3:
            return self.round_fail('连续 3 局安全箱为空，未计成功，停止自动重开')
        return self.round_success('继续入场')

    @node_from(from_name='识别出生点', success=False, status=BagelOperation.STATUS_DEFEATED)
    @node_from(from_name='识别出生点', success=False, status=BagelOperation.STATUS_ROUND_FAILED)
    @node_from(from_name='识别出生点', success=False, status=ZApplication.STATUS_TIMEOUT)
    @node_from(from_name='执行局内流程', success=False, status=BagelOperation.STATUS_ROUND_FAILED)
    @node_from(from_name='执行局内流程', success=False, status=ZApplication.STATUS_TIMEOUT)
    @node_from(from_name='执行局内流程', success=False, status=BagelOperation.STATUS_CONTAINER_FAILED)
    @node_from(from_name='执行局内流程', success=False, status=BagelOperation.STATUS_DEFEATED)
    @node_from(from_name='执行局内流程', success=False, status=BagelOperation.STATUS_INTERRUPTED)
    @operation_node(name='失败局退出', screenshot_before_round=False)
    def exit_after_defeat(self) -> OperationRoundResult:
        """记住原始失败，释放输入并从新画面选择退出和结算路径。"""
        previous = self._previous_round_result
        self.failure_reason = (
            f'{previous.status}：{previous.data}' if previous is not None and previous.data
            else previous.status if previous is not None else '局内失败'
        )
        release_flow_inputs(self.ctx)
        path = self.save_screenshot()
        self.failure_history.append({'reason': self.failure_reason, 'screenshot': path,
                                     'used': self.failure_retries_used})
        log.error('本局失败：%s。程序正在退出并处理仓库结算。本局不计成功。整体重试次数已用 %s/%s。',
                  self.failure_reason, self.failure_retries_used, self.config.max_failure_retries)
        bagel_usage.log_screenshot(path)
        return self._finish_failed_round(BagelExit(self.ctx).execute(), '退出')

    def _finish_failed_round(self, result: OperationResult, phase: str) -> OperationRoundResult:
        """收尾失败同时报告原始原因，不能继续返回或开局。"""
        if not result.success:
            return self.round_fail(
                f'本局失败：{self.failure_reason}；{phase}失败：{result.status}',
                data=result.data,
            )
        return self.round_by_op_result(result)

    @node_from(from_name='失败局退出')
    @operation_node(name='失败局结算仓库', screenshot_before_round=False)
    def settle_after_defeat(self) -> OperationRoundResult:
        """失败局同样入仓清理；清理失败则保留其错误，不改口成撤离失败。"""
        if self.config.auto_clean_warehouse:
            log.info('失败局有物入仓计入出售间隔，不补卖；满仓或残留允许提前出售并重试一次入仓。')
        else:
            log.info('清理仓库已关闭。失败局只入仓，不出售物品。')
        result = self._settle_warehouse()
        if result.success:
            self.defeat_rounds += 1
            log.info('贝果失败局结算：%s；累计失败 %s 局，未计成功',
                     result.status, self.defeat_rounds)
        return self._finish_failed_round(result, '入仓或清理')

    @node_from(from_name='失败局结算仓库')
    @operation_node(name='判断整体重试次数', screenshot_before_round=False)
    def decide_defeat_rounds(self) -> OperationRoundResult:
        """结算后决定是否再开；额度在实际调用额外入场时才消耗。"""
        if self.failure_retries_used >= self.config.max_failure_retries:
            return self.round_fail(
                f'整体重试已用 {self.failure_retries_used}/{self.config.max_failure_retries}，'
                f'已完成仓库结算，停止自动重开；本局失败：{self.failure_reason}',
            )
        self.failure_retry_pending = True
        return self.round_success('继续入场')
