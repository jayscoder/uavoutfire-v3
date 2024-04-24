import numpy as np

from bts_base import *
from policies import *
from messages import *
from bts_builder import FIRE_BT_BUILDER
from dstar import DStarLiteManager


@FIRE_BT_BUILDER.register_node
class DoAction(BaseDroneNode):
    @property
    def action(self) -> int:
        return self.converter.int(self.attrs.get('action', 0))

    @classmethod
    def do_action(cls, platform: Drone, action: int) -> typing.Iterator[Status]:
        if action in [DroneActions.MOVE_UP, DroneActions.MOVE_RIGHT, DroneActions.MOVE_DOWN, DroneActions.MOVE_LEFT,
                      DroneActions.MOVE_FORWARD]:
            # 移动
            old_pos = platform.pos
            old_dir = platform.direction
            platform.put_action(action)
            yield Status.RUNNING

            if old_pos == platform.pos and old_dir == platform.direction:
                # 移动失败
                yield Status.FAILURE
            else:
                yield Status.SUCCESS
        elif action in [DroneActions.TURN_LEFT, DroneActions.TURN_RIGHT]:
            old_dir = platform.direction
            platform.put_action(action)
            yield Status.RUNNING
            if old_dir == platform.direction:
                # 旋转失败
                yield Status.FAILURE
            else:
                yield Status.SUCCESS
        elif action == DroneActions.EXTINGUISH:
            old_obs = platform.pos_obs
            platform.put_action(action)
            yield Status.RUNNING
            new_obs = platform.pos_obs
            if old_obs == Objects.Fire and new_obs == Objects.Empty:
                # 灭火成功
                yield Status.SUCCESS
            else:
                # 灭火成功
                yield Status.FAILURE

    def updater(self) -> typing.Iterator[Status]:
        yield from DoAction.do_action(platform=self.platform, action=self.action)


@FIRE_BT_BUILDER.register_node
class MoveUp(DoAction):
    @property
    def action(self) -> int:
        return DroneActions.MOVE_UP


@FIRE_BT_BUILDER.register_node
class MoveLeft(DoAction):
    @property
    def action(self) -> int:
        return DroneActions.MOVE_LEFT


@FIRE_BT_BUILDER.register_node
class MoveRight(DoAction):
    @property
    def action(self) -> int:
        return DroneActions.MOVE_RIGHT


@FIRE_BT_BUILDER.register_node
class MoveDown(DoAction):
    @property
    def action(self) -> int:
        return DroneActions.MOVE_DOWN


@FIRE_BT_BUILDER.register_node
class MoveForward(BaseDroneNode):
    @property
    def action(self) -> int:
        return DroneActions.MOVE_FORWARD


@FIRE_BT_BUILDER.register_node
class TurnLeft(BaseDroneNode):
    @property
    def action(self) -> int:
        return DroneActions.TURN_LEFT


@FIRE_BT_BUILDER.register_node
class TurnRight(BaseDroneNode):
    @property
    def action(self) -> int:
        return DroneActions.TURN_RIGHT


@FIRE_BT_BUILDER.register_node
class Extinguish(DoAction):
    @property
    def action(self) -> int:
        return DroneActions.EXTINGUISH


@FIRE_BT_BUILDER.register_node
class DroneSendViewUpdateToHome(BaseDroneNode):
    """
    向基地传递观测
    """

    def updater(self) -> typing.Iterator[Status]:
        self.platform.send_message_to_home(
                DroneViewUpdateMessage(rect=self.platform.visible_area(), obs=self.platform.view_obs))
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class DroneReceiveUnreachableUpdate(BaseDroneNode):
    """
    无人机接收无法到达点更新
    """

    def update(self) -> Status:
        for msg in self.platform.read_messages(DroneUnreachableUpdateMessage):
            self.platform.unreachable_grid[msg.point] = 1
            return Status.SUCCESS
        return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class DroneReceiveMoveToAreaMessage(BaseDroneNode):

    def update(self) -> Status:
        for msg in self.platform.read_messages(MoveToAreaMessage):
            self.move_to_area_task = msg.rect
            return Status.SUCCESS
        return Status.FAILURE

    def to_data(self):
        return {
            'move_to_area_task': self.move_to_area_task
        }


@FIRE_BT_BUILDER.register_node
class DroneUpdateMemoryGrid(BaseDroneNode):
    """
    无人机更新记忆矩阵，同时会接收其他无人机发来的记忆矩阵更新
    """

    def update(self) -> Status:
        for msg in self.platform.read_messages(DroneViewUpdateMessage):
            lt_pos, rb_pos = msg.rect
            self.memory_grid[lt_pos[0]:rb_pos[0], lt_pos[1]:rb_pos[1]] = self.env.grid[lt_pos[0]:rb_pos[0],
                                                                         lt_pos[1]:rb_pos[1]]

        lt_pos, rb_pos = self.platform.visible_area()
        self.memory_grid[lt_pos[0]:rb_pos[0], lt_pos[1]:rb_pos[1]] = self.env.grid[lt_pos[0]:rb_pos[0],
                                                                     lt_pos[1]:rb_pos[1]]
        return Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class DroneGreedyMove(BaseDroneNode):
    """
    贪婪移动策略
    """

    @property
    def memory(self):
        return self.converter.bool(self.attrs.get('memory', False))

    def updater(self) -> typing.Iterator[Status]:
        """贪婪策略"""
        if self.memory:
            action = drone_greedy_action_with_memory(self.platform, grid=self.memory_grid)
        else:
            action = drone_greedy_action(self.platform)
        return DoAction.do_action(self.platform, action)


@FIRE_BT_BUILDER.register_node
class HasFireAtLocation(BaseDroneNode):
    """当前无人机所在位置是否有火源）"""

    def updater(self) -> typing.Iterator[Status]:
        if self.env.grid[self.platform.pos] == Objects.Fire:
            yield Status.SUCCESS
        else:
            yield Status.FAILURE


@FIRE_BT_BUILDER.register_node
class GreedyExploreUnseen(BaseDroneNode):
    """探索未知区域。寻找并向最近的未探测区域移动（会导致所有无人机一起去一个地方）。"""

    def updater(self) -> typing.Iterator[Status]:
        # 当前无人机的位置
        current_pos = self.platform.pos

        # 检查当前位置是否是未探测区域
        if self.memory_grid[current_pos] == Objects.Unseen:
            # 如果当前位置未探测，则认为已经到达目标，尝试探测
            yield Status.SUCCESS
        else:
            # 寻找最近的未探测区域，使用 NumPy 加速查找
            unseen_indices = np.argwhere(self.memory_grid == Objects.Unseen)

            if unseen_indices.size > 0:
                # 计算所有未探测区域到当前位置的曼哈顿距离
                distances = np.abs(unseen_indices - current_pos).sum(axis=1)
                nearest_index = np.argmin(distances)
                nearest_unseen: tuple[int, int] = tuple(unseen_indices[nearest_index])

                # 使用 A* 或其他路径规划算法生成移动方向
                next_move_vec = astar_find_next_direction(self.env.grid, start=current_pos, target=nearest_unseen)
                # 移动无人机
                yield from DoAction.do_action(self.platform, DroneActions.move_action_from_direction_vec(next_move_vec))
            else:
                # 如果没有未探测区域，返回失败状态
                yield Status.FAILURE


@FIRE_BT_BUILDER.register_node
class IsObstacleInFront(BaseDroneNode):
    """前方是否有障碍物"""

    def update(self) -> Status:
        obstacles = self.platform.sense_obstacles()
        if obstacles['front']:
            return Status.SUCCESS
        else:
            return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class WallAround(BaseDroneNode):
    """
    执行沿墙（障碍物）导航的行为。
    """

    @property
    def move_distance(self):
        return self.converter.int(self.attrs.get('move_distance', 1))

    @classmethod
    def move(cls, platform: Drone, distance: int) -> typing.Iterator[Status]:
        for _ in range(distance):
            # 假设：platform.sense() 返回前、左、右方向是否有障碍物
            obstacles = platform.sense_obstacles()
            front_blocked, left_blocked, right_blocked = obstacles['front'], obstacles['left'], obstacles['right']

            # 检测前方是否有障碍
            if front_blocked:
                # 前方有障碍，需要决定转向方向
                if not right_blocked:
                    # 右转
                    yield from DoAction.do_action(platform,
                                                  DroneActions.move_action_from_direction(platform.direction + 1))
                elif not left_blocked:
                    yield from DoAction.do_action(platform,
                                                  DroneActions.move_action_from_direction(platform.direction - 1))
                else:
                    # 前、左、右均被堵，尝试后退
                    yield from DoAction.do_action(platform,
                                                  DroneActions.move_action_from_direction(platform.direction + 2))
            else:
                # 前方无障碍，继续向前移动
                yield from DoAction.do_action(platform, DroneActions.MOVE_FORWARD)

    def updater(self) -> typing.Iterator[Status]:
        yield from self.move(platform=self.platform, distance=self.move_distance)


@FIRE_BT_BUILDER.register_node
class AStarMoveToAreaTask(BaseDroneNode):

    @classmethod
    def move(cls, grid: np.array, platform: Drone, target: tuple[tuple[int, int], tuple[int, int]]):
        while not is_in_rect(platform.pos, target):
            astar = FireGridAStar(obs=grid, goal=target, start=platform.pos)
            path = astar.astar(platform.pos, rect_center(target))
            if path is None:
                return None
            for i, p in enumerate(path):
                if i == 0:
                    continue
                if grid[p] == Objects.Obstacle:
                    break
                next_move_vec = p[0] - platform.pos[0], p[1] - platform.pos[1]
                yield from DoAction.do_action(platform=platform,
                                              action=DroneActions.move_action_from_direction_vec(next_move_vec))

    def updater(self) -> typing.Iterator[Status]:
        if self.move_to_area_task is None:
            yield Status.FAILURE
            return
        if is_in_rect(pos=self.platform.pos, rect=self.move_to_area_task):
            yield Status.SUCCESS
            return
        yield from self.move(grid=self.memory_grid, platform=self.platform, target=self.move_to_area_task)


@FIRE_BT_BUILDER.register_node
class HasMoveToAreaTask(BaseDroneNode):

    def update(self) -> Status:
        if self.move_to_area_task is None:
            return Status.FAILURE
        return Status.SUCCESS

    def to_data(self):
        return {
            'move_to_area_task': self.move_to_area_task,
        }


@FIRE_BT_BUILDER.register_node
class IsCompleteMoveToAreaTask(BaseDroneNode):

    def update(self) -> Status:
        if self.move_to_area_task is None:
            return Status.FAILURE
        if is_in_rect(self.platform.pos, self.move_to_area_task):
            return Status.SUCCESS
        return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class AStarMoveToAreaTaskTarget(BaseDroneNode):

    def pick_target_point(self, picked: set):
        p1 = rect_nearest(rect=self.move_to_area_task, pos=self.platform.pos)
        if p1 not in picked and self.memory_grid[p1] != Objects.Obstacle:
            picked.add(p1)
            return p1

        lt_pos, rb_pos = self.move_to_area_task

        for x in range(lt_pos[0], rb_pos[0]):
            for y in range(lt_pos[1], rb_pos[1]):
                p = (x, y)
                if p not in picked and self.memory_grid[p] != Objects.Obstacle:
                    picked.add(p)
                    return p
        return None

    def updater(self) -> typing.Iterator[Status]:
        if self.move_to_area_task is None:
            yield Status.FAILURE
            return
        if is_in_rect(pos=self.platform.pos, rect=self.move_to_area_task):
            yield Status.SUCCESS
            return
        picked = set()
        retry_count = 0
        while not is_in_rect(self.platform.pos, self.move_to_area_task) and retry_count < 4:
            if self.move_to_area_task is None:
                break
            goal = self.pick_target_point(picked)
            # print(f'移动到目标点', goal)
            if goal is None:
                yield Status.FAILURE
                return
            for status in AStarMoveToPoint.move_to(platform=self.platform, point=goal):
                if status == Status.FAILURE:
                    # print('换一个点')
                    retry_count += 1
                    yield Status.RUNNING
                    break
                yield Status.RUNNING
                if is_in_rect(pos=self.platform.pos, rect=self.move_to_area_task):
                    yield Status.SUCCESS
                    return
                if not is_in_rect(pos=goal, rect=self.move_to_area_task):
                    retry_count += 1
                    break
                if self.move_to_area_task is None:
                    break
        if not is_in_rect(self.platform.pos, self.move_to_area_task):
            yield Status.FAILURE
            return
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class DStarLiteMoveToAreaTaskTarget(BaseDroneNode):

    def pick_target_point(self, picked: set):
        p1 = rect_nearest(rect=self.move_to_area_task, pos=self.platform.pos)
        if p1 not in picked and self.memory_grid[p1] != Objects.Obstacle:
            picked.add(p1)
            return p1

        lt_pos, rb_pos = self.move_to_area_task

        for x in range(lt_pos[0], rb_pos[0]):
            for y in range(lt_pos[1], rb_pos[1]):
                p = (x, y)
                if p not in picked and self.memory_grid[p] != Objects.Obstacle:
                    picked.add(p)
                    return p
        return None

    def updater(self) -> typing.Iterator[Status]:
        if self.move_to_area_task is None:
            yield Status.FAILURE
            return
        if is_in_rect(pos=self.platform.pos, rect=self.move_to_area_task):
            yield Status.SUCCESS
            return
        picked = set()
        retry_count = 0
        while not is_in_rect(self.platform.pos, self.move_to_area_task) and retry_count < 4:
            if self.move_to_area_task is None:
                break
            goal = self.pick_target_point(picked)
            # print(f'移动到目标点', goal)
            if goal is None:
                yield Status.FAILURE
                return
            for status in DStarLiteMoveToPoint.move_to(platform=self.platform, point=goal):
                if status == Status.FAILURE:
                    # print('换一个点')
                    retry_count += 1
                    yield Status.RUNNING
                    break
                yield Status.RUNNING
                if is_in_rect(pos=self.platform.pos, rect=self.move_to_area_task):
                    yield Status.SUCCESS
                    return
                if not is_in_rect(pos=goal, rect=self.move_to_area_task):
                    retry_count += 1
                    break
                if self.move_to_area_task is None:
                    break
        if not is_in_rect(self.platform.pos, self.move_to_area_task):
            yield Status.FAILURE
            return
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class DStarLiteMoveToPoint(BaseDroneNode):

    @classmethod
    def move_to(cls, platform: Drone, point: tuple[int, int]):
        dstar = DStarLiteManager(platform=platform, goal=point)
        retry_count = 0
        while not (platform.pos == point) and retry_count < 4:
            # print(f'移动到目标点', point, retry_count)
            dstar.update_all()
            path = dstar.move_and_replan()
            if path is None:
                # print(f'找不到路径', point, retry_count)
                platform.unreachable_grid[dstar.goal] = 1
                platform.send_message_to_home(DroneUnreachableUpdateMessage(point=dstar.goal))
                yield Status.FAILURE
                return
            for i, p in enumerate(path):
                # print('移动', p)
                if i == 0 and p != platform.pos:
                    raise Exception('搓酥')
                if i == 0:
                    continue
                if platform.memory_grid[p] == Objects.Obstacle:
                    # print('碰到墙了', retry_count)
                    retry_count += 1
                    break
                next_move_vec = p[0] - platform.pos[0], p[1] - platform.pos[1]
                yield from DoAction.do_action(
                        platform=platform,
                        action=DroneActions.move_action_from_direction_vec(next_move_vec))
                dstar.update_view()
        if not not (platform.pos == point):
            yield Status.FAILURE
            return
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class AStarMoveToPoint(BaseDroneNode):

    @classmethod
    def move_to(cls, platform: Drone, point: tuple[int, int]):

        retry_count = 0
        while not (platform.pos == point) and retry_count < 4:
            # print(f'移动到目标点', point, retry_count)
            astar = FireGridAStar(obs=platform.memory_grid, goal=point, start=platform.pos)
            path = astar.astar(platform.pos, point)
            if path is None:
                # print(f'找不到路径', point, retry_count)
                platform.unreachable_grid[point] = 1
                platform.send_message_to_home(DroneUnreachableUpdateMessage(point=point))
                yield Status.FAILURE
                return
            for i, p in enumerate(path):
                # print('移动', p)
                if i == 0 and p != platform.pos:
                    raise Exception('错误')
                if i == 0:
                    continue
                if platform.memory_grid[p] == Objects.Obstacle:
                    # print('碰到墙了，重新规划路线', retry_count)
                    retry_count += 1
                    break
                next_move_vec = p[0] - platform.pos[0], p[1] - platform.pos[1]
                yield from DoAction.do_action(
                        platform=platform,
                        action=DroneActions.move_action_from_direction_vec(next_move_vec))
        if not (platform.pos == point):
            yield Status.FAILURE
            return
        yield Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class WallAroundMoveToTaskTarget(BaseDroneNode):
    @classmethod
    def move(cls, grid: np.array, platform: Drone, target: tuple[tuple[int, int], tuple[int, int]]):
        target_pos = rect_center(target)

        while not is_in_rect(platform.pos, target):
            # 如果移动方向上没有障碍物，则直接前进，否则WallAround
            move_vec = target_pos[0] - platform.pos[0], target_pos[1] - platform.pos[1]
            direction = Directions.calculate(move_vec)
            direction_pos = platform.direction_pos(direction)

            if grid[direction_pos] == Objects.Obstacle:
                yield from WallAround.move(platform=platform, distance=1)
            else:
                yield from DoAction.do_action(platform=platform,
                                              action=DroneActions.move_action_from_direction(direction))


@FIRE_BT_BUILDER.register_node
class GoToNearestFireInAreaTask(BaseDroneNode):
    def updater(self) -> typing.Iterator[Status]:
        # 查找最近的火点
        target = self.platform.find_nearest_reachable_obj_pos(obj=Objects.Fire, in_task_area=True)
        if target is None:
            yield Status.FAILURE
            return
        yield from AStarMoveToPoint.move_to(platform=self.platform, point=target)


@FIRE_BT_BUILDER.register_node
class GoToNearestFire(BaseDroneNode):
    def updater(self) -> typing.Iterator[Status]:
        # 查找最近的火点
        target = self.platform.find_nearest_reachable_obj_pos(obj=Objects.Fire, in_task_area=False)
        if target is None:
            yield Status.FAILURE
            return
        yield from AStarMoveToPoint.move_to(platform=self.platform, point=target)


@FIRE_BT_BUILDER.register_node
class GoToNearestUnseenInAreaTask(BaseDroneNode):

    def updater(self) -> typing.Iterator[Status]:
        target = self.platform.find_nearest_reachable_obj_pos(obj=Objects.Unseen, in_task_area=True)
        if target is None:
            yield Status.FAILURE
            return
        yield from AStarMoveToPoint.move_to(platform=self.platform, point=target)


@FIRE_BT_BUILDER.register_node
class IsExtinguisherOver(BaseDroneNode):

    def update(self) -> Status:
        if self.platform.fire_extinguisher <= 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class IsBatteryBingo(BaseDroneNode):

    def update(self) -> Status:
        if self.platform.battery < manhattan_distance(self.env.home.pos, self.platform.pos) * 1.5:
            return Status.SUCCESS
        return Status.FAILURE


@FIRE_BT_BUILDER.register_node
class GoHome(BaseDroneNode):
    def updater(self) -> typing.Iterator[Status]:
        yield from AStarMoveToPoint.move_to(platform=self.platform, point=self.env.home.pos)

# @FIRE_BT_BUILDER.register_node
# class DroneMoveToTaskTarget(BaseDroneNode):
#     """
#     无人机飞到分配给它的区域。
#     """
#
#     def updater(self) -> typing.Iterator[Status]:
#         for msg in self.platform.read_messages(MoveToAreaMessage):
#             self.move_to_area_task = msg.rect
#
#         if self.move_to_area_task is None:
#             yield Status.FAILURE
#             return
#
#         if is_in_rect(pos=self.platform.pos, rect=self.move_to_area_task):
#             yield Status.SUCCESS
#             return
#         print('move_to_area_task', self.move_to_area_task)
#         yield from AStarMoveToTaskTarget.move(
#                 grid=self.memory_grid,
#                 target=self.move_to_area_task,
#                 platform=self.platform)
