from __future__ import annotations
import numpy as np
from astar import AStar
from constanst import *
from typing import Union, Iterable
import os
import random


class FireGridAStar(AStar):

    def __init__(self, obs: np.ndarray, goal: (int, int) = None, start: (int, int) = None):
        self.obs = obs
        self.goal = goal
        self.start = start

    def heuristic_cost_estimate(self, current, goal) -> float:
        """
        计算启发式距离，使用曼哈顿距离
        :param current:
        :param goal:
        :return:
        """
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def distance_between(self, n1, n2) -> float:
        """
        计算两个节点之间的距离, 使用曼哈顿距离
        n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
        :param n1:
        :param n2:
        :return:
        """
        distance = manhattan_distance(n1, n2)
        for pos in [n1, n2]:
            if self.obs[pos[0], pos[1]] == Objects.Obstacle:
                return float('inf')
        return distance

    def neighbors(self, node):
        """
        返回当前节点的邻居节点
        :param node:
        :return:
        """
        for dx, dy in DIRECTION_VECS:
            x2 = node[0] + dx
            y2 = node[1] + dy
            if x2 < 0 or x2 >= self.obs.shape[0] or y2 < 0 or y2 >= self.obs.shape[1]:
                continue
            yield x2, y2


def astar_find_path(obs, start: tuple[int, int], target: tuple[int, int]) -> Union[Iterable[tuple[int, int]], None]:
    """
    使用A*算法寻找路径
    :param obs:
    :param start:
    :param target:
    :return:
    """
    astar = FireGridAStar(obs=obs, goal=target, start=start)
    path = astar.astar(start, target)
    return path


def astar_find_next_direction(obs, start: tuple[int, int], target: tuple[int, int]) -> Union[tuple[int, int], None]:
    """
    使用A*算法寻找下一步移动的方向
    :param obs:
    :param start:
    :param target:
    :return:
    """
    astar = FireGridAStar(obs=obs, goal=target, start=start)
    path = astar.astar(start, target)
    if path is None:
        return None
    for i, p in enumerate(path):
        if i == 0:
            continue
        return p[0] - start[0], p[1] - start[1]
    return None


def manhattan_distance(pos1, pos2):
    return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])


def spread_init(grid, obj: int, count: int, area_size: int | tuple[int, int] = 7):
    """生长区域扩散"""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上，下，左，右
    size = grid.shape[0]
    for _ in range(count):
        for attempt in range(100):
            x, y = np.random.randint(7, size, 2)
            if grid[x, y] == 0:
                grid[x, y] = obj
                current_size = 1
                if isinstance(area_size, int):
                    asize = area_size
                else:
                    asize = np.random.randint(area_size[0], area_size[1])  # 1-area_size个格子的区域大小
                # 尝试增长易燃区域到所需大小
                growth_attempts = 0
                while current_size < asize and growth_attempts < 100:
                    growth_attempts += 1
                    # 随机选择一个方向进行扩展
                    direction = directions[np.random.randint(0, len(directions))]
                    new_x = x + direction[0]
                    new_y = y + direction[1]

                    # 检查新坐标是否有效
                    if 0 <= new_x < size and 0 <= new_y < size:
                        if grid[new_x, new_y] == 0:
                            grid[new_x, new_y] = obj
                            current_size += 1
                            x, y = new_x, new_y  # 更新当前位置
                    # 如果新坐标不可用或已达到预定大小，则结束扩展
                    if current_size >= asize:
                        break
                if current_size >= asize:  # 成功创建易燃区域
                    break


def folder_run_id(folder: str):
    os.makedirs(folder, exist_ok=True)
    id_path = os.path.join(folder, "run_id.txt")
    if os.path.exists(id_path):
        with open(id_path, "r") as f:
            run_id = int(f.read())
    else:
        run_id = 0
    run_id += 1
    with open(id_path, mode="w") as f:
        f.write('{}'.format(run_id))
    return str(run_id)


def calculate_direction_position(pos: tuple[int, int], direction: Directions) -> tuple[int, int]:
    """根据方向计算新的位置"""
    direction %= len(Directions)
    if direction == Directions.Up:
        return pos[0], pos[1] - 1
    elif direction == Directions.Down:
        return pos[0], pos[1] + 1
    elif direction == Directions.Left:
        return pos[0] - 1, pos[1]
    elif direction == Directions.Right:
        return pos[0] + 1, pos[1]


def is_in_rect(pos: tuple[int, int], rect: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """
    :param pos:
    :param rect: lt_pos, rb_pos
    :return:
    """
    lt_pos, rb_pos = rect
    return lt_pos[0] <= pos[0] < rb_pos[0] and lt_pos[1] <= pos[1] < rb_pos[1]


def rect_center(rect: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    """
    :param rect: lt_pos, rb_pos
    :return: rect center
    """
    lt_pos, rb_pos = rect
    return int((lt_pos[0] + rb_pos[0]) / 2), int((lt_pos[1] + rb_pos[1]) / 2)


def rect_nearest(rect: tuple[tuple[int, int], tuple[int, int]], pos: tuple[int, int]) -> tuple[int, int]:
    """
    返回一个区域里离pos最近的点
    :param rect: (lt_pos, rb_pos) 两个元组表示矩形的左上和右下角的位置
    :param pos: (x, y) 需要找到最近点的位置
    :return: 矩形区域内离pos最近的点
    """
    (lt_pos, rb_pos) = rect
    (x, y) = pos

    # lt_pos[0], lt_pos[1] 分别是矩形左上角的 x 和 y 坐标
    # rb_pos[0], rb_pos[1] 分别是矩形右下角的 x 和 y 坐标

    # 计算距离给定点最近的 x 坐标
    nearest_x = max(lt_pos[0], min(x, rb_pos[0]))
    # 计算距离给定点最近的 y 坐标
    nearest_y = max(lt_pos[1], min(y, rb_pos[1]))

    return (nearest_x, nearest_y)


def build_rect_from_center(center: tuple[int, int], size: tuple[int, int], max_size: int) -> tuple[
    tuple[int, int], tuple[int, int]]:
    left_x = center[0] - size[0] / 2
    left_y = center[1] - size[1] / 2
    right_x = center[0] + size[0]
    right_y = center[1] + size[1]

    left_x = max(left_x, 0)
    left_y = max(left_y, 0)
    right_x = min(right_x, max_size)
    right_y = min(right_y, max_size)

    return (int(left_x), int(left_y)), (int(right_x), int(right_y))
# class RectPointCache:
#     def __init__(self, rect: tuple[tuple[int, int], tuple[int, int]]):
#         self.rect = rect
#         self.picked_points = set()
#
#     def pick_point(self, pos: tuple[int, int]) -> tuple[int, int]:
#         left_x, left_y = self.rect[0]
#         right_x, right_y = self.rect[1]
#
#         point = random.randint(left_x, right_x), random.randint(left_y, right_y)
