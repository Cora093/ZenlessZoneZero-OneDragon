from __future__ import annotations

from typing import TYPE_CHECKING

from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from zzz_od.application.bagel.bagel_const import CONTAINER_LABELS
from zzz_od.application.bagel.bagel_container import BagelContainerOperation

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.application.bagel.bagel_container import ContainerRecovery
    from zzz_od.context.zzz_context import ZContext


class BagelOpenContainer(BagelContainerOperation):
    """普通开箱的公共逻辑：点按或按住交互键，确认搜查面板出现。

    武备箱点按即可；机械保险箱的交互键要按住 3.5 秒才响应，按不够箱子不动。
    两种容器都不需要光圈解锁 —— 那是电子保险箱独有的流程。
    """

    def __init__(
        self, ctx: ZContext, target: str, recovery: ContainerRecovery | None = None,
    ) -> None:
        """与靠近导航共享一次容器恢复预算。"""
        super().__init__(
            ctx, target=target,
            op_name=f'贝果-打开{CONTAINER_LABELS.get(target, "武备箱")}',
            recovery=recovery,
        )

    @property
    def label(self) -> str:
        """容器中文名，用于提示文案。"""
        return CONTAINER_LABELS.get(self.target, '容器')

    def open_container(self) -> OperationRoundResult:
        """已有面板直接继续，否则按当前目标提示交互。"""
        return self.interact_container()

    def wait_container_search(self) -> OperationRoundResult:
        """限次补按或向执行器申请重新靠近。"""
        result = self.interact_container()
        if result.is_success:
            if result.status == '容器已打开':
                return self.round_success(f'已进入{self.label}搜查')
            return self.round_wait(f'{self.label}未打开，补按一次交互', wait=0.25)
        return result


class BagelOpenBox(BagelOpenContainer):
    """打开武备箱并确认搜查面板，不触发全部拾取。"""

    def __init__(self, ctx: ZContext, recovery: ContainerRecovery | None = None) -> None:
        """武备箱点按即可。"""
        super().__init__(ctx, target='box', recovery=recovery)

    @operation_node(name='打开武备箱', is_start_node=True)
    def open_box(self) -> OperationRoundResult:
        """已有面板直接继续，否则按当前目标提示交互。"""
        return self.open_container()

    @node_from(from_name='打开武备箱')
    @operation_node(name='等待武备箱搜查面板')
    def wait_search(self) -> OperationRoundResult:
        """限次补按或向执行器申请重新靠近。"""
        return self.wait_container_search()


class BagelOpenMech(BagelOpenContainer):
    """打开机械保险箱：交互键要按住 3.5 秒，点按不响应。"""

    def __init__(self, ctx: ZContext, recovery: ContainerRecovery | None = None) -> None:
        """机械保险箱要按住交互键，按不够箱子不响应。"""
        super().__init__(ctx, target='mech', recovery=recovery)

    @operation_node(name='打开机械保险箱', is_start_node=True)
    def open_box(self) -> OperationRoundResult:
        """已有面板直接继续，否则按住交互键。"""
        return self.open_container()

    @node_from(from_name='打开机械保险箱')
    @operation_node(name='等待机械保险箱搜查面板')
    def wait_search(self) -> OperationRoundResult:
        """限次补按或向执行器申请重新靠近。"""
        return self.wait_container_search()
