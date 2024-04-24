import numpy as np
from pybts import *
from bts_base import BaseHomeNode
from constanst import *
from messages import *
from bts_builder import FIRE_BT_BUILDER


@FIRE_BT_BUILDER.register_node
class HomeUpdateMemoryGrid(BaseHomeNode):
    """基地更新记忆矩阵，并将最新的记忆矩阵下发给无人机"""

    def update(self) -> Status:
        old_memory_grid = np.copy(self.memory_grid)
        for msg in self.platform.read_messages(DroneViewUpdateMessage):
            lt_pos, rt_pos = msg.rect
            self.memory_grid[lt_pos[0]:rt_pos[0], lt_pos[1]:rt_pos[1]] = self.env.grid[lt_pos[0]:rt_pos[0],
                                                                         lt_pos[1]:rt_pos[1]]
            self.platform.memory_grid_set_time[lt_pos[0]:rt_pos[0], lt_pos[1]:rt_pos[1]] = msg.time
            self.platform.send_message_to_all_drone(message=msg)

        self.memory_grid[self.platform.unreachable_grid == 1] = Objects.Empty  # 不可到达区域设置成空
        # 下发记忆矩阵
        # self.platform.send_message_to_all_drone(message=MemoryGridUpdateMessage(memory_grid=self.memory_grid))
        self.env.render_grid = self.memory_grid

        # 计算新火点数量
        new_find_fire_count = np.sum((old_memory_grid != Objects.Fire) & (self.memory_grid == Objects.Fire))
        new_extinguish_fire_count = np.sum((old_memory_grid == Objects.Fire) & (self.memory_grid != Objects.Fire))
        new_explore_unseen_count = np.sum((old_memory_grid == Objects.Unseen) & (self.memory_grid != Objects.Unseen))
        self.context['new_find_fire_count'] = new_find_fire_count
        self.context['new_extinguish_fire_count'] = new_extinguish_fire_count
        self.context['new_explore_unseen_count'] = new_explore_unseen_count
        return Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeUpdateUnreachableGrid(BaseHomeNode):
    """基地更新无法到达点，并下发给其他无人机"""

    def update(self) -> Status:
        for msg in self.platform.read_messages(DroneUnreachableUpdateMessage):
            self.platform.unreachable_grid[msg.point] = 1
            self.platform.send_message_to_all_drone(message=msg)

        # 下发记忆矩阵
        return Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeHasDroneNeedRecharge(BaseHomeNode):
    """基地当前是否有无人机需要补充"""

    def update(self) -> Status:
        for drone in self.env.drones:
            if drone.pos == self.platform.pos:
                return Status.SUCCESS
        return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class HomeRechargeDrone(BaseHomeNode):
    """基地给无人机补充"""

    def update(self) -> Status:
        self.put_action(HomeActions.RECHARGE)
        return Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeGreedyAction(BaseHomeNode):

    def update(self) -> Status:
        from policies import home_greedy_action
        action = home_greedy_action(self.platform)
        self.put_action(action)
        return Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeHasUnseenArea(BaseHomeNode):
    """是否还有不可视区域"""

    def update(self) -> Status:
        unseen_mask = self.platform.memory_grid == Objects.Unseen
        if unseen_mask.any():
            return Status.SUCCESS
        return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class HomeHasFireArea(BaseHomeNode):
    """是否还有火区"""

    def update(self) -> Status:
        fire_mask = self.platform.memory_grid == Objects.Fire
        if fire_mask.any():
            return Status.SUCCESS
        return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class HomeAssignUnseenExplorationAreas(BaseHomeNode):
    """基地分配探索区域给各无人机。"""

    @property
    def repeat_count(self):
        return self.converter.int(self.attrs.get('repeat_count', 10))

    def updater(self) -> typing.Iterator[Status]:
        for i in range(self.repeat_count):
            memory_grid = self.memory_grid
            unseen_mask = memory_grid == Objects.Unseen

            # 获取所有未探测区域的坐标
            unseen_indices = np.argwhere(unseen_mask)
            if unseen_indices.size == 0:
                yield Status.FAILURE  # 如果没有未探测区域，则返回失败状态
                return

            # 尝试均等分配未探测区域
            num_drones = len(self.env.drones)
            area_per_drone = len(unseen_indices) // num_drones
            extra = len(unseen_indices) % num_drones

            start_idx = 0
            for i, drone in enumerate(self.env.drones):
                # 为每个无人机计算分配区域的大小
                end_idx = start_idx + area_per_drone + (1 if i < extra else 0)
                if start_idx < len(unseen_indices):
                    drone_area_indices = unseen_indices[start_idx:end_idx]
                    # 计算区域的边界
                    x_min, x_max = np.min(drone_area_indices[:, 0]), np.max(drone_area_indices[:, 0])
                    y_min, y_max = np.min(drone_area_indices[:, 1]), np.max(drone_area_indices[:, 1])
                    # 创建并发送消息
                    area_message = MoveToAreaMessage(rect=((x_min, y_min), (x_max + 1, y_max + 1)))
                    self.platform.send_message(message=area_message, to_platform=drone)
                start_idx = end_idx

            yield Status.RUNNING


@FIRE_BT_BUILDER.register_node
class HomeAssignFireExplorationAreas(BaseHomeNode):
    """基地分配火场探索区域给各无人机。"""

    @property
    def repeat_count(self):
        return self.converter.int(self.attrs.get('repeat_count', 10))

    def updater(self) -> typing.Iterator[Status]:
        for i in range(self.repeat_count):
            memory_grid = self.memory_grid
            target_mask = memory_grid == Objects.Fire

            # 获取所有未探测区域的坐标
            target_indices = np.argwhere(target_mask)
            if target_indices.size == 0:
                yield Status.FAILURE  # 如果没有未探测区域，则返回失败状态
                return
            # 尝试均等分配未探测区域
            num_drones = len(self.env.drones)
            area_per_drone = len(target_indices) // num_drones
            extra = len(target_indices) % num_drones

            start_idx = 0
            for i, drone in enumerate(self.env.drones):
                # 为每个无人机计算分配区域的大小
                end_idx = start_idx + area_per_drone + (1 if i < extra else 0)
                if start_idx < len(target_indices):
                    drone_area_indices = target_indices[start_idx:end_idx]
                    # 计算区域的边界
                    x_min, x_max = np.min(drone_area_indices[:, 0]), np.max(drone_area_indices[:, 0])
                    y_min, y_max = np.min(drone_area_indices[:, 1]), np.max(drone_area_indices[:, 1])
                    # 创建并发送消息
                    area_message = MoveToAreaMessage(rect=((x_min, y_min), (x_max + 1, y_max + 1)))
                    self.platform.send_message(message=area_message, to_platform=drone)
                start_idx = end_idx

            yield Status.RUNNING


@FIRE_BT_BUILDER.register_node
class IsFindNewFire(BaseHomeNode):
    """
    是否发现了新的火点
    """

    def update(self) -> Status:
        if self.context['new_find_fire_count'] > 0:
            # print(f"Detected {self.context['new_find_fire_count']} new fire points.")
            return Status.SUCCESS  # 发现新火点
        else:
            return Status.FAILURE  # 没有新火点发现


@FIRE_BT_BUILDER.register_node
class IsExtinguishNewFire(BaseHomeNode):
    """
    是否扑灭了新的火点
    """

    def update(self) -> Status:
        if self.context['new_extinguish_fire_count'] > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class IsExploreNewUnseen(BaseHomeNode):
    """
    是否扑灭了新的火点
    """

    def update(self) -> Status:
        if self.context['new_explore_unseen_count'] > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE
