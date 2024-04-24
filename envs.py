from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import matplotlib
import matplotlib.colors as mcolors
import random

import pybts
from gymnasium.core import ActType, ObsType

from agent import QLearningAgent
import gymnasium as gym
import typing
from typing import Any, SupportsFloat
from constanst import *
import math
from matplotlib.animation import FuncAnimation
import pygame
from utils import *
from tqdm import tqdm
from collections import deque, defaultdict
from messages import *
from typing import Type, Optional
import time


def now_time():
    return time.time()


# TODO: 无人机不能在一个格子里，会碰撞
# TODO: 记忆矩阵添加过期时间

class Platform:
    def __init__(self, x, y, id: int, env: FireEnvironment):
        self.id = id
        self.x = x
        self.y = y
        self.env = env
        self.messages = defaultdict(deque)
        self.parent: Platform | None = None
        self.children: typing.List[Platform] = []
        self.waiting_actions = deque()  # 等待执行的动作
        self.consume_actions = []  # 已经执行的动作
        self.is_alive = True
        self.context = { }  # 平台环境空间，platform会重建，但是行为树不会，所以行为树的context和这里的不是同一个

        self.unreachable_grid = np.zeros_like(self.env.grid)  # 1为不可到达点
        self.move_to_area_task: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
        self.role: int = 0  # 平台角色

    @property
    def pos(self):
        return self.x, self.y

    @property
    def pos_obs(self):
        # 当前位置上的物体
        return self.env.grid[self.x, self.y]

    @property
    def is_leader(self):
        return len(self.children) > 0

    def put_action(self, action):
        self.waiting_actions.append(action)
        # assert len(self.waiting_actions) <= 1, f"Action {action} 一个tick只能提交一次"

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def send_message_to_home(self, message: BaseMessage):
        # 向基地传递消息
        self.send_message(message=message, to_platform=self.env.home)

    def send_message_to_parent(self, message: BaseMessage):
        self.send_message(message=message, to_platform=self.parent)

    def send_message_to_children(self, message: BaseMessage):
        for child in self.children:
            if child.is_alive:
                self.send_message(message=message, to_platform=child)

    def send_message_to_all_drone(self, message: BaseMessage):
        for drone in self.env.drones:
            if drone.is_alive and message.sender_id != drone.id:
                self.send_message(message=message, to_platform=drone)

    def send_message_to_all_explore_drone(self, message: BaseMessage):
        for drone in self.env.drones:
            if drone.is_alive and message.sender_id != drone.id and drone.role == DroneRole.Explore:
                self.send_message(message=message, to_platform=drone)

    def send_message_to_all_extinguish_drone(self, message: BaseMessage):
        for drone in self.env.drones:
            if drone.is_alive and message.sender_id != drone.id and drone.role == DroneRole.Extinguish:
                self.send_message(message=message, to_platform=drone)

    def send_message(self, message: BaseMessage, to_platform: Platform | int) -> None:
        if isinstance(to_platform, int):
            to_platform = self.env.platforms[to_platform]
        message.to_id = to_platform.id
        message.sender_id = self.id
        message.time = self.env.time
        to_platform.messages[message.__class__.__name__].append(message)

    def read_messages(self, msg_type: Type[MSG_TYPE]) -> Iterable[MSG_TYPE]:
        # messages = deque()
        while self.messages[msg_type.__name__]:
            yield self.messages[msg_type.__name__].popleft()

    def iterate(self):
        for child in self.children:
            yield from child.iterate()
        yield self

    def update(self):
        assert len(self.waiting_actions) <= 1, "动作在一个周期里最多只有1个"
        while self.waiting_actions:
            action = self.waiting_actions.popleft()
            self.step(action)
            self.consume_actions.append(action)

    def step(self, action):
        pass

    @property
    def any_child_is_alive(self) -> bool:
        return any(m.is_alive for m in self.children)

    @property
    def alive_children(self) -> list[Platform]:
        return [m for m in self.children if m.is_alive]

    @property
    def alive_children_count(self) -> int:
        return len(self.alive_children)


class Home(Platform):
    def __init__(self, x, y, id: int, env: FireEnvironment):
        super().__init__(x, y, id, env)
        self.memory_grid = np.full(shape=self.env.grid.shape, fill_value=Objects.Unseen)  # 记忆矩阵
        self.memory_grid_set_time = np.full(shape=self.env.grid.shape, fill_value=0)  # 每个记忆矩阵更新的时间
        self.outdated_memory_grid = np.full(shape=self.env.grid.shape, fill_value=Objects.Unseen)  # 过期的记忆矩阵

    def recharge(self, drone):
        # print(f"Recharging drone {drone.id} at home located at ({self.x}, {self.y})")
        drone.battery = drone.max_battery  # 重置电量到最大值
        drone.fire_extinguisher = drone.max_fire_extinguisher  # 重置灭火剂到最大值

    def step(self, action):
        if action == HomeActions.RECHARGE:
            # 给当前基地上的无人机充电
            for drone in self.env.drones:
                if drone.pos == self.pos:
                    self.recharge(drone)

    def greedy_act(self):
        from policies import home_greedy_action
        self.step(home_greedy_action(self))


class Drone(Platform):
    def __init__(self, x, y, id: int, role: int, env: FireEnvironment):
        super().__init__(x, y, id=id, env=env)
        self.role = role

        if role == DroneRole.Extinguish:
            # 灭火无人机的主要任务是有效地扑灭火源，因此需要更多的灭火剂和足够的电量来支持持续的操作，
            # 但其视野范围可以相对较小，因为它通常会在探索无人机发现的火源区域操作。
            # 视野范围: 1 - 2（灭火时需要的精准操作不需要太广的视野）
            # 最大电量: 1200（确保有足够的电量支持持续操作）
            # 灭火剂容量: 20（增加灭火剂容量以扑灭更多的火源）
            # 消耗电量: 设为每次移动或操作消耗5单位电量（因操作较多，消耗较快）
            # 消耗灭火剂: 每次灭火操作消耗2单位灭火剂（确保可以多次执行灭火任务）
            self.view_range = 1  # 视野范围
            self.max_battery = 3000  # 最大电量
            self.battery = self.max_battery  # 当前电量
            self.cost_battery = 3  # 每次移动消耗电量3
            self.max_fire_extinguisher = 20  # 最大灭火剂容量
            self.fire_extinguisher = self.max_fire_extinguisher  # 当前灭火剂量
            self.cost_extinguisher = 1  # 每次灭火操作消耗1单位灭火剂
        else:
            # 探索无人机的主要任务是广泛地侦查和映射区域，识别火源和其他障碍物。
            # 视野范围: 3-4（增加视野范围以便覆盖更大的区域进行探索）
            # 最大电量: 1000（足够支持长时间操作）
            # 消耗电量: 每次移动或操作消耗1单位电量（考虑到主要任务是移动探索，消耗较少）
            # 无需携带灭火剂（探索无人机不参与灭火）
            self.view_range = 3  # 视野范围
            self.max_battery = 1500  # 最大电量
            self.battery = self.max_battery  # 当前电量
            self.cost_battery = 1  # 每次移动消耗电量1
            self.max_fire_extinguisher = 0  # 最大灭火剂容量
            self.fire_extinguisher = self.max_fire_extinguisher  # 当前灭火剂量
            self.cost_extinguisher = 1  # 每次灭火操作消耗1单位灭火剂

        self.is_alive = True  # 无人机是否存活
        self.extinguished_fire_count = 0  # 扑灭火的数量
        self.direction = Directions.Right  # 当前正在移动的方向
        self.move_count_grid = np.zeros_like(env.grid)  # 每个位置移动次数统计

    @property
    def memory_grid(self):
        return self.env.home.memory_grid

    @property
    def outdated_memory_grid(self):
        return self.env.home.outdated_memory_grid

    def move(self, dx, dy):
        if not self.is_alive:
            return
        # 一次最多移动1格，而且只能沿着格子移动
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        if abs(dx) + abs(dy) > 1:
            # print('移动的格子不对')
            if random.random() < 0.5:
                dx = 0
            else:
                dy = 0
        if abs(dx) + abs(dy) == 0:
            return

        self.direction = DIRECTION_VECS.index((dx, dy))
        new_x = max(0, min(self.env.size - 1, self.x + dx))
        new_y = max(0, min(self.env.size - 1, self.y + dy))
        if self.env.grid[new_x, new_y] == Objects.Obstacle:
            self.is_alive = False  # 碰撞到障碍物，无人机死亡
            print('无人机碰撞到障碍物死亡')
            # print(f"Drone {self.id} crashed into an obstacle and is now disabled.")
        else:
            self.battery -= (abs(dx) + abs(dy)) * self.cost_battery
            self.x = new_x
            self.y = new_y
            if self.battery <= 0:
                self.battery = 0
                print('无人机电量耗尽死亡')
                self.is_alive = False
                # print(f"Drone {self.id} has run out of battery.")
        self.move_count_grid[new_x, new_y] += 1

    def extinguish_fire(self):
        if self.fire_extinguisher <= 0:
            return
        # 尝试灭火
        if self.env.grid[self.x, self.y] == Objects.Fire:
            self.env.grid[self.x, self.y] = 0  # 灭火后将网格值设置为0
            self.fire_extinguisher -= self.cost_extinguisher  # 灭火剂数量减少1点
            self.extinguished_fire_count += 1

    def visible_area(self) -> tuple[tuple[int, int], tuple[int, int]]:
        # 计算并返回无人机的可视区域，左上角的点和右下角的点
        x_min = max(0, self.x - self.view_range)
        x_max = min(self.env.size, self.x + self.view_range + 1)
        y_min = max(0, self.y - self.view_range)
        y_max = min(self.env.size, self.y + self.view_range + 1)
        return (x_min, y_min), (x_max, y_max)

    def is_battery_bingo(self, battery: int | None = None, pos: (int, int) = None) -> bool:
        if battery is None:
            battery = self.battery
        if pos is None:
            pos = self.pos
        return battery < manhattan_distance(self.env.home.pos, pos) * self.cost_battery * 1.2

    def predict_can_move_to_target(self, target: tuple[int, int]) -> bool:
        """预测一下能否移动到目标点"""
        dis = manhattan_distance(self.pos, target)
        battery = self.battery - dis * self.cost_battery  # 移动到目标点后预计消耗电量
        return not self.is_battery_bingo(battery=battery, pos=target)

    def find_nearest_reachable_obj_pos(
            self,
            obj: int, in_task_area: bool = True,
            near_obj: int | None = None,
            threshold: int = 1
    ) -> tuple[
             int, int] | None:
        # 查找最近的某个点
        nearest_pos = None
        min_distance = float('inf')
        left_top = (0, 0)
        right_bottom = (self.env.size, self.env.size)
        if in_task_area and self.move_to_area_task is not None:
            left_top, right_bottom = self.move_to_area_task
        # 确定搜索范围内的所有满足条件的点
        mask = (self.memory_grid == obj) | (self.outdated_memory_grid == obj)
        mask = mask[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        indices = np.argwhere(mask) + np.array([left_top[0], left_top[1]])

        if near_obj is not None:
            # 检查每个点附近是否有 `near_obj`
            valid_indices = []
            for idx in indices:
                x, y = idx
                sub_grid = grid_area_extract(grid=self.memory_grid, center=(x, y), offset=1)
                if np.any(sub_grid == near_obj):
                    valid_indices.append(idx)
            indices = np.array(valid_indices)

        if indices.size == 0:
            return None

        # 计算所有有效点到当前位置的曼哈顿距离
        distances = np.abs(indices[:, 0] - self.pos[0]) + np.abs(indices[:, 1] - self.pos[1])

        # 筛选出可到达的点
        reachable = np.array([self.predict_can_move_to_target(target=(idx[0], idx[1])) for idx in indices])
        reachable_distances = distances[reachable]

        if reachable_distances.size == 0:
            return None

        # 找到最近的点
        min_index = np.argmin(reachable_distances)
        nearest_pos = tuple(indices[reachable][min_index])

        # if reachable_distances[min_index] <= threshold:
        #     return nearest_pos

        return nearest_pos

        for x in range(left_top[0], right_bottom[0]):
            for y in range(left_top[1], right_bottom[1]):
                if self.memory_grid[x][y] == obj or self.outdated_memory_grid[x][y] == obj:
                    if near_obj is not None:
                        if not np.any(
                                grid_area_extract(grid=self.memory_grid, center=(x, y), offset=1) == near_obj):
                            continue

                    distance = manhattan_distance((x, y), self.pos)

                    can_move_to = self.predict_can_move_to_target(target=(x, y))
                    if distance < min_distance and can_move_to:
                        min_distance = distance
                        nearest_pos = (x, y)
                        if min_distance <= threshold:
                            return nearest_pos
        return nearest_pos

    @property
    def need_go_home(self) -> bool:
        """
        判断当前无人机是否需要回基地
        :return:
        """
        if self.is_battery_bingo():
            # 需要留够回基地的电量
            return True
        if self.role == DroneRole.Extinguish and self.fire_extinguisher == 0:
            return True
        return False

    def step(self, action):
        if action == DroneActions.MOVE_UP:
            self.move(*DIRECTION_VECS[Directions.Up])
        elif action == DroneActions.MOVE_DOWN:
            self.move(*DIRECTION_VECS[Directions.Down])
        elif action == DroneActions.MOVE_LEFT:
            self.move(*DIRECTION_VECS[Directions.Left])
        elif action == DroneActions.MOVE_RIGHT:
            self.move(*DIRECTION_VECS[Directions.Right])
        elif action == DroneActions.EXTINGUISH:
            self.extinguish_fire()
        elif action == DroneActions.MOVE_FORWARD:
            self.move(*DIRECTION_VECS[self.direction])
        elif action == DroneActions.TURN_LEFT:
            self.direction = (self.direction - 1) % len(Directions)
        elif action == DroneActions.TURN_RIGHT:
            self.direction = (self.direction + 1) % len(Directions)

        # self.update_memory()

    @property
    def front_pos(self):
        return calculate_direction_position(self.pos, self.direction)

    @property
    def left_pos(self):
        return calculate_direction_position(self.pos, self.direction - 1)

    @property
    def right_pos(self):
        return calculate_direction_position(self.pos, self.direction + 1)

    @property
    def back_pos(self):
        return calculate_direction_position(self.pos, self.direction + 2)

    def direction_pos(self, direction):
        return calculate_direction_position(self.pos, direction)

    def sense_obstacles(self):
        """判断前、左、右方向是否有障碍物"""
        # 检查这些位置是否有障碍物
        obstacles = {
            'front': self.env.over_boundary(self.front_pos) or self.env.grid[self.front_pos] == Objects.Obstacle,
            'left' : self.env.over_boundary(self.front_pos) or self.env.grid[self.left_pos] == Objects.Obstacle,
            'right': self.env.over_boundary(self.front_pos) or self.env.grid[self.right_pos] == Objects.Obstacle,
            'back' : self.env.over_boundary(self.front_pos) or self.env.grid[self.back_pos] == Objects.Obstacle
        }
        return obstacles

    @property
    def view_obs(self):
        # 可视范围内的矩阵
        lt_pos, rb_pos = self.visible_area()
        return self.env.grid[lt_pos[0]:rb_pos[0], lt_pos[1]:rb_pos[1]]

    def greedy_act(self):
        """贪婪策略"""
        from policies import drone_greedy_action
        action = drone_greedy_action(self)
        self.step(action)


class FireEnvironment(gym.Env):
    def __init__(self,
                 size=50,
                 init_empty_fires=5,  # 空地上的火扩散的速度会比较慢
                 init_flammable_fires=2,  # 草地上的火扩散速度会比较快
                 num_explore_drones=2,
                 num_extinguish_drones=2,
                 num_obstacles=30,
                 num_flammables=4,
                 max_step: int = 3000,
                 ):
        self.size = size
        self.init_empty_fires = init_empty_fires
        self.init_flammable_fires = init_flammable_fires

        self.empty_fire_spread_chance = 0
        self.flammable_fire_spread_chance = 0.005
        self.num_obstacles = num_obstacles
        self.num_flammables = num_flammables
        self.num_explore_drones = num_explore_drones
        self.num_extinguish_drones = num_extinguish_drones

        self.max_duration = max_step
        self.terminated = False
        self.truncated = False
        self.time = 0
        self.grid = np.zeros((size, size))
        self.fire_weight = np.zeros((size, size))  # 火量
        self.render_grid: Union[np.array, None] = None
        self.platforms = []
        self.episode = 0

        self.area_size_per_flammables = 50
        self.area_size_per_obstacles = 7

        self.accum_reward = 0
        self.last_render_time = 0
        self.last_update_time = 0
        self.paused = False  # 是否暂停

        self.down_sample_factor = 4
        self.observation_space = gym.spaces.Box(low=0, high=5,
                                                shape=(2, self.size // self.down_sample_factor,
                                                       self.size // self.down_sample_factor))

        self.action_space = gym.spaces.Box(low=0, high=1,
                                           shape=(self.num_explore_drones + self.num_extinguish_drones, 2))

        self.trees: list[pybts.Tree] = []
        self.reset()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the simulation to start a new episode."""
        for tree in self.trees:
            tree.reset()

        if self.time > 0:
            self.episode += 1
        self.grid.fill(0)
        self.terminated = False
        self.truncated = False
        self.init_objects()
        self.time = 0
        self.accum_reward = 0

        return self.gen_obs(), self.gen_info()

    def gen_info(self):
        return {
            'terminated': self.terminated, 'truncated': self.truncated, 'time': self.time,
            # 'is_success': self.alive_fires == 0,
            # 'episode': self.episode,
        }

    @property
    def home(self) -> Home:
        return self.platforms[0]

    def drone(self, id: int) -> Drone:
        return self.platforms[id]

    @property
    def drones(self) -> list[Drone]:
        return self.platforms[1:]

    @property
    def explore_drones(self) -> list[Drone]:
        return list(filter(lambda drone: drone.role == DroneRole.Explore, self.drones))

    @property
    def alive_explore_drones(self) -> list[Drone]:
        return list(filter(lambda drone: drone.role == DroneRole.Explore and drone.is_alive, self.drones))

    @property
    def alive_extinguish_drones(self) -> list[Drone]:
        return list(filter(lambda drone: drone.role == DroneRole.Extinguish and drone.is_alive, self.drones))

    @property
    def extinguish_drones(self) -> list[Drone]:
        return list(filter(lambda drone: drone.role == DroneRole.Extinguish, self.drones))

    def init_objects(self):
        self.platforms.clear()
        self.platforms.append(Home(0, 0, id=0, env=self))
        spread_init(self.grid,
                    obj=Objects.Obstacle,
                    count=self.num_obstacles,
                    area_size=self.area_size_per_obstacles)
        spread_init(self.grid,
                    obj=Objects.Flammable,
                    count=self.num_flammables,
                    area_size=self.area_size_per_flammables)

        spread_init(self.grid, obj=Objects.Fire, count=self.init_flammable_fires, area_size=1, at_obj=Objects.Flammable)
        # 添加一下距离草地的距离
        spread_init(self.grid, obj=Objects.Fire, count=self.init_empty_fires, area_size=1, at_obj=Objects.Empty,
                    at_area_offset=2)

        for _ in range(self.num_explore_drones):
            # 在基地位置初始化无人机
            x, y = self.home.x, self.home.y
            drone = Drone(x, y, id=len(self.platforms), env=self, role=DroneRole.Explore)
            self.platforms.append(drone)
            self.home.add_child(drone)

        for _ in range(self.num_extinguish_drones):
            # 在基地位置初始化无人机
            x, y = self.home.x, self.home.y
            drone = Drone(x, y, id=len(self.platforms), env=self, role=DroneRole.Extinguish)
            self.platforms.append(drone)
            self.home.add_child(drone)

    def is_far_from_fire(self, x, y, min_distance):
        # Check the area around (x, y) within the min_distance for any fire
        start_x = max(0, x - min_distance)
        end_x = min(self.size, x + min_distance + 1)
        start_y = max(0, y - min_distance)
        end_y = min(self.size, y + min_distance + 1)
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if self.grid[i, j] == Objects.Fire:  # Check for fires
                    return False
        return True

    def spread_fire(self):
        """火势蔓延策略"""
        new_grid = self.grid.copy()
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == Objects.Fire:  # Existing fire spreads normally
                    for di, dj in DIRECTION_VECS:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni, nj] in [Objects.Empty,
                                                                                                 Objects.Flammable]:
                            if self.grid[ni, nj] == Objects.Flammable:
                                if np.random.rand() < self.flammable_fire_spread_chance:
                                    # 碰到易燃物品后扩散速度加快20倍
                                    new_grid[ni, nj] = Objects.Fire
                            elif np.random.rand() < self.empty_fire_spread_chance:
                                new_grid[ni, nj] = Objects.Fire
        self.grid = new_grid

    @property
    def done(self):
        return self.terminated or self.truncated

    @property
    def alive_drones_count(self) -> int:
        return sum(q.is_alive for q in self.drones)

    @property
    def alive_explore_drones_count(self) -> int:
        return sum(q.is_alive and q.role == DroneRole.Explore for q in self.drones)

    @property
    def alive_extinguish_drones_count(self) -> int:
        return sum(q.is_alive and q.role == DroneRole.Extinguish for q in self.drones)

    @property
    def extinguish_fire_count(self) -> int:
        # 无人机扑灭火的数量
        return sum([d.extinguished_fire_count for d in self.drones])

    @property
    def alive_fires(self) -> int:
        # 剩余火量
        return np.sum(self.grid == Objects.Fire)

    @property
    def alive_flammables(self) -> int:
        # 剩余草地数量
        return np.sum(self.grid == Objects.Flammable)

    @property
    def alive_fires_ratio(self) -> float:
        # 剩余火量比例
        return self.alive_fires / (self.size * self.size)

    @property
    def alive_near_flammable_fires(self) -> int:
        # 剩余靠近草地的火的数量
        fire_positions = np.argwhere(self.grid == Objects.Fire)
        near_flammable_count = 0

        for fire_x, fire_y in fire_positions:
            # 检查火源的上下左右邻居
            neighbors = [
                (fire_x - 1, fire_y), (fire_x + 1, fire_y),
                (fire_x, fire_y - 1), (fire_x, fire_y + 1)
            ]
            # 确保邻居坐标在网格范围内并检查是否为草地
            for nx, ny in neighbors:
                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                    if self.grid[nx, ny] == Objects.Flammable:
                        near_flammable_count += 1
                        break  # 如果找到至少一个草地邻居，就停止检查其他邻居

        return near_flammable_count

    @property
    def alive_near_flammable_fires_ratio(self) -> float:
        # 剩余草地比例
        return self.alive_near_flammable_fires / (self.size * self.size)

    @property
    def alive_flammables_ratio(self) -> float:
        # 剩余草地比例
        return self.alive_flammables / (self.size * self.size)

    @property
    def alive_drones_ratio(self) -> float:
        # 剩余无人机比例
        return self.alive_drones_count / len(self.drones)

    def in_boundary(self, pos: tuple) -> bool:
        """
        Returns true if the position is inside the boundary of the
        :param pos:
        :return:
        """
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def over_boundary(self, pos: tuple) -> bool:
        return not self.in_boundary(pos)

    def update(self):
        """Advance the simulation by one step."""

        old_grid = np.copy(self.grid)
        old_memory_grid = np.copy(self.home.memory_grid)

        self.last_update_time = now_time()

        self.time += 1
        self.spread_fire()
        reward = 0

        for t in self.trees:
            t.tick()

        for platform in self.platforms:
            platform.update()

        if self.time > self.max_duration:
            self.truncated = True

        # Check if all drones are disabled
        # if random.random() < 1 / 500:
        #     spread_init(self.grid, obj=Objects.Fire, count=self.init_flammable_fires, area_size=1,
        #                 at_obj=Objects.Flammable)

        if self.alive_drones_count == 0 or self.alive_extinguish_drones_count == 0 or self.alive_fires == 0:
            # 灭火无人机数量为0也代表结束了
            self.terminated = True
            # print("All drones are disabled. Simulation ends.")

        # Check if all fires have been extinguished
        if self.alive_fires == 0:
            self.terminated = True
            # print(f"All fires have been extinguished. Simulation ends. rewards={rewards} done={self.done}")

        if self.alive_fires >= 500:
            self.terminated = True
            # 剩余火太多了，直接结束

        if self.terminated or self.truncated:
            time_ratio = self.max_duration / self.time
            # 剩余无人机越多、剩余火越少、剩余草地越多，则奖励越多，同时考虑时间系数，用时越短，则奖励越高
            a, b, c = 0.1, -20, -1
            reward += (self.alive_flammables_ratio * a
                       + self.alive_near_flammable_fires_ratio * b
                       + self.alive_fires_ratio * c) * time_ratio * 1000

        else:
            # 计算新火点数量
            new_find_fire_count = np.sum((old_memory_grid != Objects.Fire) & (self.home.memory_grid == Objects.Fire))
            extinguished_fire_mask = (old_grid == Objects.Fire) & (self.grid != Objects.Fire)
            new_extinguish_fire_count = np.sum(extinguished_fire_mask)
            # new_explore_unseen_count = np.sum(
            #         (old_memory_grid == Objects.Unseen) & (self.home.memory_grid != Objects.Unseen))

            # 检查新灭掉的火周围是否有可燃物
            new_extinguish_fire_nearby_flammable_count = 0
            if new_extinguish_fire_count > 0:
                for x in range(self.grid.shape[0]):
                    for y in range(self.grid.shape[0]):
                        if extinguished_fire_mask[x, y]:  # 如果这是新灭掉的火
                            # 检查周围的格子
                            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]  # 四个方向的邻居
                            for nx, ny in neighbors:
                                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[
                                    1]:  # 确保不越界
                                    if self.grid[nx, ny] == Objects.Flammable:
                                        new_extinguish_fire_nearby_flammable_count += 1
                                        break  # 仅统计每个灭火点附近是否有可燃物，有即可停止检查

            reward += new_extinguish_fire_count * 0.1 + new_extinguish_fire_nearby_flammable_count * 1 + new_find_fire_count * 0.05

        # Simulation continues
        self.accum_reward += reward
        # print('reward', reward)
        return self.gen_obs(), reward, self.terminated, self.truncated, self.gen_info()

    # def matplot_render(self):
    #     # plt.clf()  # Clear the entire figure to remove previous drawings
    #     # 创建一个临时的显示矩阵，包含火源和障碍物的信息
    #     # 配置颜色映射：0 = 无火无障碍（黑色），1 = 火源（红色），2 = 障碍物（灰色）
    #     cmap = mcolors.ListedColormap(['white', 'red', 'gray'])
    #     bounds = [0, 1, 2, 3]
    #     norm = mcolors.BoundaryNorm(bounds, cmap.N)
    #     # 使用自定义的颜色映射渲染场景
    #     plt.imshow(self.grid, cmap=cmap, norm=norm)
    #     plt.colorbar()  # 显示颜色条
    #
    #     plt.scatter(self.home.x, self.home.y, color='orange', s=200)  # 基地位置
    #
    #     # 标记无人机的位置
    #     for drone in self.drones:
    #         plt.scatter(drone.y, drone.x, color='blue', s=100)  # 无人机位置
    #         # 画出无人机的视野范围
    #         x_min, x_max, y_min, y_max = drone.visible_area()
    #         plt.plot([y_min, y_max - 1, y_max - 1, y_min, y_min], [x_min, x_min, x_max - 1, x_max - 1, x_min],
    #                  'b--')
    #
    #     plt.show()

    def pygame_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.size * 10, self.size * 10))
        pygame.display.set_caption("Fire Simulation")

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        for i in range(len(action)):
            x, y = action[i]
            w, h = 10, 10
            x = int(x * (self.size - w)) + w / 2
            y = int(y * (self.size - h)) + h / 2
            # x = int(x * (self.env.size - 10)) + 5
            # y = int(y * (self.env.size - 10)) + 5

            # print('HomeRLAssignFireExplorationAreas', x, y, size)
            rects = []
            rect = build_rect_from_center((x, y), (w, h), max_size=self.size)
            rects.append(rect)
            area_message = MoveToAreaMessage(rect=rect)
            self.home.send_message(message=area_message, to_platform=self.drones[i])

        return self.update()

    def gen_obs(self):
        image = downsample_grid(self.grid, factor=self.down_sample_factor)
        # print('gen_obs', image.shape, self.observation_space)
        # if self.obs_uav:
        uav_id = np.zeros_like(image)
        # uav_need_go_home = np.zeros_like(image)
        for i, drone in enumerate(self.drones):
            new_x = drone.x // self.down_sample_factor
            new_y = drone.y // self.down_sample_factor
            if new_x >= uav_id.shape[0]:
                new_x -= 1
            if new_y >= uav_id.shape[1]:
                new_y -= 1
            uav_id[new_x, new_y] = drone.role
            # uav_need_go_home[drone.x, drone.y] = int(drone.need_go_home)
        # outdated = (self.time - self.home.memory_grid_set_time) / 600
        image = np.stack([image, uav_id], axis=0)
        # print(image.shape)
        return image

    def pygame_render(self):
        self.last_render_time = time.time()
        render_grid = self.render_grid
        if render_grid is None:
            render_grid = self.grid
        render_grid = self.grid
        screen = self.screen
        screen.fill((0, 0, 0))  # Clear the screen with black
        for x in range(self.size):
            for y in range(self.size):
                color = 'white'  # white
                if self.home.unreachable_grid[x, y] == 1:
                    color = 'black'
                elif render_grid[x, y] == Objects.Fire:
                    color = '#ff4500'  # 火：使用橙红色，与纯红色相比，色彩更鲜明，对色盲用户友好
                elif render_grid[x, y] == Objects.Obstacle:
                    color = '#808080'  # 障碍物：使用中灰色，这里保留原始的灰色，因为它通常表示不可通过区域
                elif render_grid[x, y] == Objects.Flammable:
                    color = '#32cd32'  # 易燃物品：使用鲜明的草绿色，更易于与其他元素区分
                elif render_grid[x, y] == Objects.Unseen:
                    color = '#1e1e1e'  # 未探测区域：使用接近黑色的深灰色，表示这一区域尚未探索
                pygame.draw.rect(screen, color, (x * 10, y * 10, 10, 10))
        # Draw home
        base_pos = (self.home.x * 10, self.home.y * 10)
        screen.blit(PYGAME_IMAGES['BASE'], base_pos)

        for drone in self.drones:
            member_pos = (drone.x * 10, drone.y * 10)
            if drone.role == DroneRole.Explore:
                img = PYGAME_IMAGES['UAV_EXPLORE']
                rotate = 0
                if drone.direction == Directions.Right:
                    rotate = -90
                elif drone.direction == Directions.Down:
                    rotate = 180
                elif drone.direction == Directions.Left:
                    rotate = 90
                img = pygame.transform.rotate(img, rotate)
            else:
                img = PYGAME_IMAGES['UAV_EXTINGUISH']
            img_rect = img.get_rect()
            img_rect.topleft = member_pos
            screen.blit(img, img_rect)

            # 创建字体对象
            font = pygame.font.Font(None, 16)  # 使用默认字体，大小为36

            # 渲染文本到 Surface 对象
            text_surface = font.render(f'{drone.id}', True, (0, 0, 0))

            # 获取文本区域的矩形
            text_rect = text_surface.get_rect()
            text_rect.center = [img_rect.center[0], img_rect.bottom + 10]
            screen.blit(text_surface, text_rect)

            # screen.blit(PYGAME_IMAGES['MEMBER'], member_pos)
            if not drone.is_alive:
                screen.blit(PYGAME_IMAGES['EXPLOSION'], member_pos)
            if drone.move_to_area_task is not None:
                left_top, right_bottom = drone.move_to_area_task
                size_w = right_bottom[0] - left_top[0]
                size_h = right_bottom[1] - left_top[1]
                pygame.draw.rect(screen, 'blue', (left_top[0] * 10, left_top[1] * 10, size_w * 10, size_h * 10),
                                 width=1)
            # # Draw direction arrow
            # arrow_color = (255, 0, 0)  # Red arrow
            # dx, dy = 0, 0
            # if drone.direction == Directions.Up:
            #     dx, dy = 0, -arrow_length
            # elif drone.direction == Directions.Down:
            #     dx, dy = 0, arrow_length
            # elif drone.direction == Directions.Left:
            #     dx, dy = -arrow_length, 0
            # elif drone.direction == Directions.Right:
            #     dx, dy = arrow_length, 0
            # arrow_end = (member_pos[0] + 5 + dx, member_pos[1] + 5 + dy)
            # pygame.draw.line(screen, arrow_color, (member_pos[0] + 5, member_pos[1] + 5), arrow_end, 2)

        play_pause_img = PYGAME_IMAGES['PLAY'] if self.paused else PYGAME_IMAGES['PAUSE']
        play_pause_img_rect = play_pause_img.get_rect()
        play_pause_img_rect.right = self.size * 10 - 10
        play_pause_img_rect.top = 10

        event = pygame.event.poll()
        if event is not None:
            # 检测鼠标点击
            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_pause_img_rect.collidepoint(event.pos):
                    self.paused = not self.paused  # 切换暂停状态
        screen.blit(play_pause_img, play_pause_img_rect)

        pygame.display.flip()  # Update the full display Surface to the screen


def simulate():
    render = True
    running = True
    env = FireEnvironment(50)
    if render:
        env.pygame_init()
    start_time = time.time()
    N = 5
    pbar = tqdm(total=N)
    for episode in range(N):
        env.reset()
        # pbar.set_postfix({
        #     'episode': episode,
        #     '无人机' : env.alive_drones_count,
        #     '火'     : env.alive_fires
        # })
        while (not env.done) and running:
            for platform in env.platforms:
                if platform.is_alive:
                    platform.greedy_act()
            env.update()  # Update the state of the environment
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                env.pygame_render()
                pbar.set_postfix({
                    'episode': episode,
                    '无人机' : env.alive_drones_count,
                    '火'     : env.alive_fires
                })
            if env.terminated or env.truncated:
                break
        pbar.set_postfix({
            'episode': episode,
            '无人机' : env.alive_drones_count,
            '火'     : env.alive_fires
        })
        pbar.update(1)
        time.sleep(0.05)
        print()
    cost_time = time.time() - start_time
    print(f'Cost time: {cost_time:.2f} seconds')


if __name__ == '__main__':
    simulate()
