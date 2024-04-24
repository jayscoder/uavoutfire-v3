import numpy as np
from pybts import *
from bts_base import BaseHomeNode
from constanst import *
from messages import *
from bts_builder import FIRE_BT_BUILDER


@FIRE_BT_BUILDER.register_node
class HomeUpdateMemoryGrid(BaseHomeNode):
    """基地更新记忆矩阵，并将最新的记忆矩阵下发给无人机"""

    @property
    def outdated_time(self) -> int:
        return self.converter.int(self.attrs.get('outdated_time', 300))

    def update(self) -> Status:
        old_memory_grid = np.copy(self.memory_grid)
        for msg in self.platform.read_messages(DroneViewUpdateMessage):
            lt_pos, rt_pos = msg.rect
            self.memory_grid[lt_pos[0]:rt_pos[0], lt_pos[1]:rt_pos[1]] = self.env.grid[lt_pos[0]:rt_pos[0],
                                                                         lt_pos[1]:rt_pos[1]]
            self.platform.memory_grid_set_time[lt_pos[0]:rt_pos[0], lt_pos[1]:rt_pos[1]] = msg.time
            self.platform.outdated_memory_grid[lt_pos[0]:rt_pos[0], lt_pos[1]:rt_pos[1]] = self.env.grid[
                                                                                           lt_pos[0]:rt_pos[0],
                                                                                           lt_pos[1]:rt_pos[1]]
            self.platform.send_message_to_all_drone(message=msg)

        # 超过30步的区域重新视为过期
        self.memory_grid[self.platform.unreachable_grid == 1] = Objects.Obstacle  # 不可到达区域设置成障碍物
        self.platform.outdated_memory_grid[self.platform.unreachable_grid == 1] = Objects.Obstacle  # 不可到达区域设置成障碍物
        self.platform.outdated_memory_grid[
            self.env.time - self.platform.memory_grid_set_time > self.outdated_time] = Objects.Unseen

        # 下发记忆矩阵
        # self.platform.send_message_to_all_drone(message=MemoryGridUpdateMessage(memory_grid=self.memory_grid))
        self.env.render_grid = self.platform.outdated_memory_grid

        # 计算新火点数量
        new_find_fire_count = np.sum((old_memory_grid != Objects.Fire) & (self.memory_grid == Objects.Fire))
        extinguished_fire_mask = (old_memory_grid == Objects.Fire) & (self.memory_grid != Objects.Fire)

        new_extinguish_fire_count = np.sum(extinguished_fire_mask)
        new_explore_unseen_count = np.sum((old_memory_grid == Objects.Unseen) & (self.memory_grid != Objects.Unseen))

        # 检查新灭掉的火周围是否有可燃物
        new_extinguish_fire_nearby_flammable_count = 0
        for x in range(self.memory_grid.shape[0]):
            for y in range(self.memory_grid.shape[0]):
                if extinguished_fire_mask[x, y]:  # 如果这是新灭掉的火
                    # 检查周围的格子
                    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]  # 四个方向的邻居
                    for nx, ny in neighbors:
                        if 0 <= nx < self.memory_grid.shape[0] and 0 <= ny < self.memory_grid.shape[1]:  # 确保不越界
                            if self.memory_grid[nx, ny] == Objects.Flammable:
                                new_extinguish_fire_nearby_flammable_count += 1
                                break  # 仅统计每个灭火点附近是否有可燃物，有即可停止检查

        self.context['new_find_fire_count'] = new_find_fire_count
        self.context['new_extinguish_fire_count'] = new_extinguish_fire_count
        self.context['new_extinguish_fire_nearby_flammable_count'] = new_extinguish_fire_nearby_flammable_count
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
class HomeAssignUnseenExplorationTasks(BaseHomeNode):
    """基地分配探索区域给各探索无人机。"""

    @property
    def repeat_count(self):
        return self.converter.int(self.attrs.get('repeat_count', 10))

    def updater(self) -> typing.Iterator[Status]:
        for i in range(self.repeat_count):
            target_mask = self.platform.outdated_memory_grid == Objects.Unseen

            # 获取所有未探测区域的坐标
            target_indices = np.argwhere(target_mask)
            if target_indices.size == 0:
                yield Status.FAILURE  # 如果没有未探测区域，则返回失败状态
                return

            # 尝试均等分配未探测区域
            num_drones = len(self.env.alive_explore_drones)
            # 每个无人机分配至少一个探测区域，如果区域数小于无人机数，一些无人机将共享同一探测区域
            num_assigned_areas = min(num_drones, len(target_indices))
            area_per_drone = max(1, len(target_indices) // num_drones)
            extra = len(target_indices) % num_drones

            start_idx = 0
            for i, drone in enumerate(self.env.alive_explore_drones):
                assign_i = i % num_assigned_areas
                # 为每个无人机计算分配区域的大小
                end_idx = start_idx + area_per_drone + (1 if assign_i < extra else 0)
                drone_area_indices = target_indices[start_idx:end_idx]
                if len(drone_area_indices) == 0:
                    # 清除区域任务
                    self.platform.send_message(message=MoveToAreaMessage(rect=None), to_platform=drone)
                    continue
                # 计算区域的边界
                x_min, x_max = np.min(drone_area_indices[:, 0]), np.max(drone_area_indices[:, 0])
                y_min, y_max = np.min(drone_area_indices[:, 1]), np.max(drone_area_indices[:, 1])
                # 创建并发送消息
                area_message = MoveToAreaMessage(rect=((x_min, y_min), (x_max + 1, y_max + 1)))
                self.platform.send_message(message=area_message, to_platform=drone)
                start_idx = end_idx

            yield Status.RUNNING
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeAssignFireExplorationTasks(BaseHomeNode):
    """基地分配火场探索区域给各无人机。"""

    @property
    def repeat_count(self):
        return self.converter.int(self.attrs.get('repeat_count', 10))

    def updater(self) -> typing.Iterator[Status]:
        for i in range(self.repeat_count):
            target_mask = self.platform.memory_grid == Objects.Fire

            # 获取所有未探测区域的坐标
            target_indices = np.argwhere(target_mask)
            if target_indices.size == 0:
                yield Status.FAILURE  # 如果没有未探测区域，则返回失败状态
                return
            # 尝试均等分配未探测区域
            num_drones = len(self.env.alive_extinguish_drones)
            # 每个无人机分配至少一个探测区域，如果区域数小于无人机数，一些无人机将共享同一探测区域
            num_assigned_areas = min(num_drones, len(target_indices))
            area_per_drone = max(1, len(target_indices) // num_drones)
            extra = len(target_indices) % num_drones

            start_idx = 0
            for i, drone in enumerate(self.env.alive_extinguish_drones):
                assign_i = i % num_assigned_areas
                # 为每个无人机计算分配区域的大小
                end_idx = start_idx + area_per_drone + (1 if assign_i < extra else 0)
                drone_area_indices = target_indices[start_idx:end_idx]
                if len(drone_area_indices) == 0:
                    # 清除区域任务
                    self.platform.send_message(message=MoveToAreaMessage(rect=None), to_platform=drone)
                    continue
                # 计算区域的边界
                x_min, x_max = np.min(drone_area_indices[:, 0]), np.max(drone_area_indices[:, 0])
                y_min, y_max = np.min(drone_area_indices[:, 1]), np.max(drone_area_indices[:, 1])

                x_min = max(0, x_min - 2)
                x_max = min(self.env.size, x_max + 2)
                y_min = max(0, y_min - 2)
                y_max = min(self.env.size, y_max + 2)

                # 创建并发送消息
                area_message = MoveToAreaMessage(rect=((x_min, y_min), (x_max, y_max)))
                self.platform.send_message(message=area_message, to_platform=drone)
                start_idx = end_idx

            yield Status.RUNNING
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeClearFireExplorationTasks(BaseHomeNode):

    def updater(self) -> typing.Iterator[Status]:
        self.platform.send_message_to_all_extinguish_drone(message=MoveToAreaMessage(rect=None))
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeClearUnseenExplorationTasks(BaseHomeNode):

    def updater(self) -> typing.Iterator[Status]:
        self.platform.send_message_to_all_explore_drone(message=MoveToAreaMessage(rect=None))
        yield Status.SUCCESS


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
