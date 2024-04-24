from __future__ import annotations
from pybts import *
from envs import *
from constanst import *
from rl.tree import RLTree

class BaseBTNode(Node):
    @property
    def env(self) -> FireEnvironment:
        return self.context['env']


    @property
    def cache(self) -> dict:
        return self.context['cache']

    @property
    def platform(self) -> Platform:
        return self.env.platforms[self.context['platform_id']]

    def put_action(self, action):
        self.platform.put_action(action)

    @property
    def platform_id(self) -> int:
        return self.context['platform_id']

    @property
    def memory_grid(self) -> np.array:
        return self.platform.memory_grid


class BaseHomeNode(BaseBTNode):
    @property
    def platform(self) -> Home:
        return self.env.platforms[self.context['platform_id']]


class BaseDroneNode(BaseBTNode):
    @property
    def platform(self) -> Drone:
        return self.env.platforms[self.context['platform_id']]

    @property
    def move_to_area_task(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        return self.platform.move_to_area_task

    @move_to_area_task.setter
    def move_to_area_task(self, value):
        self.platform.move_to_area_task = value

    # @property
    # def move_to_point_task(self) -> tuple[tuple[int, int]] | None:
    #     return self.cache.get('move_to_point_task', None)
    #
    # @move_to_point_task.setter
    # def move_to_point_task(self, value):
    #     self.cache['move_to_point_task'] = value
