from collections import deque
import random
import numpy as np
from typing import TypeVar


class BaseMessage:
    def __init__(self):
        self.sender_id: int = -1
        self.to_id = -1
        self.time = -1  # 消息创建的时间

    def __str__(self):
        return f"Message {self.sender_id} => {self.to_id}: (Type: {self.__class__.__name__})"


MSG_TYPE = TypeVar('MSG_TYPE', bound=BaseMessage)


class DroneViewUpdateMessage(BaseMessage):
    """无人机观测更新"""

    def __init__(self, rect: ((int, int), (int, int)), obs: np.array):
        """
        :param rect:  left_top: (int, int), right_bottom
        :param obs:
        """
        super().__init__()
        self.rect = rect
        self.obs = obs


class DroneUnreachableUpdateMessage(BaseMessage):
    """无法到达点更新"""

    def __init__(self, point: tuple[int, int]):
        super().__init__()
        self.point = point


class MoveToAreaMessage(BaseMessage):
    """
    移动到指定区域
    """

    def __init__(self, rect: ((int, int), (int, int))):
        """
        :param rect:  left_top: (int, int), right_bottom
        """
        super().__init__()
        self.rect = rect


class MoveToPointMessage(BaseMessage):
    """
    移动到指定点
    """

    def __init__(self, point: tuple[int, int]):
        """
        :param point
        """
        super().__init__()
        self.point = point
