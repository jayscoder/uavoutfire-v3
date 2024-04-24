from enum import IntEnum
import gymnasium as gym
import pygame


class HomeActions(IntEnum):
    KEEP = 0
    RECHARGE = 1  # 给无人机恢复状态（充电、补充灭火剂）


class DroneActions(IntEnum):
    """
    无人机执行动作 'move_up', 'move_down', 'move_left', 'move_right', 'extinguish'
    """
    KEEP = 0

    MOVE_UP = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    MOVE_LEFT = 4

    EXTINGUISH = 5

    MOVE_FORWARD = 6
    TURN_LEFT = 7
    TURN_RIGHT = 8

    @classmethod
    def get_action_list(cls):
        """返回所有可能动作的列表，方便其他模块引用和索引动作。"""
        return [action for action in DroneActions]

    @classmethod
    def move_action_from_direction_vec(cls, vec: tuple[int, int]):
        if vec is None:
            return DroneActions.KEEP
        dx, dy = vec
        if dx > 0:
            return DroneActions.MOVE_RIGHT
        if dy > 0:
            return DroneActions.MOVE_DOWN
        if dx < 0:
            return DroneActions.MOVE_LEFT
        if dy < 0:
            return DroneActions.MOVE_UP
        return DroneActions.KEEP

    @classmethod
    def move_action_from_direction(cls, direction):
        direction %= len(Directions)
        if direction == Directions.Up:
            return DroneActions.MOVE_UP
        if direction == Directions.Right:
            return DroneActions.MOVE_RIGHT
        if direction == Directions.Down:
            return DroneActions.MOVE_DOWN
        if direction == Directions.Left:
            return DroneActions.MOVE_LEFT
        return DroneActions.KEEP


DRONE_ACTIONS_SPACE = gym.spaces.Discrete(len(DroneActions))

PYGAME_IMAGES = {
    'BASE'     : pygame.image.load('imgs/home.svg'),
    'LEADER'   : pygame.image.load('imgs/uav_leader.svg'),
    'MEMBER'   : pygame.image.load('imgs/uav_member.svg'),
    'EXPLOSION': pygame.image.load('imgs/explosion.svg')
}

DIRECTION_VECS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class Directions(IntEnum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3

    @classmethod
    def calculate(cls, vec: tuple[int, int]):
        if vec is None:
            return Directions.Up
        dx, dy = vec
        if dx > 0:
            return Directions.Down
        if dy > 0:
            return Directions.Down
        if dx < 0:
            return Directions.Left
        if dy < 0:
            return Directions.Up
        return Directions.Down


class Objects(IntEnum):
    Empty = 0
    Fire = 1
    Obstacle = 2
    Flammable = 3
    Unseen = 4  # 未知区域
