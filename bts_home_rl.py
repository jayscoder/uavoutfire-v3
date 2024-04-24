import numpy as np

from rl import *
from bts_base import *
from bts_builder import *
from features import *
from rl.logger import TensorboardLogger
from stable_baselines3 import *
from rl.common import is_off_policy_algo
from bts_rl import *


class BaseHomeRLNode(RLNode, BaseHomeNode):
    def __init__(self, **kwargs):
        BaseHomeNode.__init__(self, **kwargs)
        super().__init__(**kwargs)

@FIRE_BT_BUILDER.register_node
class HomeRLAssignFireExplorationAreas(BaseHomeRLNode):
    """基地RL分配灭火区域给各无人机。"""

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0, high=self.env.size, shape=(len(self.env.extinguish_drones), 4), dtype=np.float32)

    def update(self) -> Status:
        areas = list(self.take_action())
        # 对areas进行排序，按照每个区域的中心点到左上角的距离排序
        if not self.obs_uav:
            # 如果不观测无人机位置的话，就给输出的区域做个排序
            areas.sort(key=lambda area: area[0] + area[1])
        for i in range(len(areas)):
            x, y, w, h = areas[i]
            # print('HomeRLAssignFireExplorationAreas', x, y, size)
            rects = []
            rect = build_rect_from_center((x, y), (w, h), max_size=self.env.size)
            rects.append(rect)
            area_message = MoveToAreaMessage(rect=rect)
            self.platform.send_message(message=area_message, to_platform=self.env.extinguish_drones[i])
        return Status.SUCCESS


@FIRE_BT_BUILDER.register_node
class HomeRLAreaReward(BaseHomeNode, Reward):
    def __init__(self, **kwargs):
        BaseHomeNode.__init__(self, **kwargs)
        Reward.__init__(self, reward=0, **kwargs)

    def cal_reward(self):

        ideal_area = 50  # 假设理想区域面积为50个单位
        area_tolerance = 20  # 面积容忍度

        reward = 0
        for drone in self.env.drones:
            if drone.move_to_area_task is None:
                continue
            lt_pos, rb_pos = drone.move_to_area_task
            area_width = rb_pos[0] - lt_pos[0]
            area_height = rb_pos[1] - lt_pos[1]
            # area_size = area_width * area_height

            # Count fire and unseen areas
            fire_count = np.sum(drone.memory_grid[lt_pos[0]:rb_pos[0], lt_pos[1]:rb_pos[1]] == Objects.Fire)
            unseen_count = np.sum(drone.memory_grid[lt_pos[0]:rb_pos[0], lt_pos[1]:rb_pos[1]] == Objects.Unseen)

            # Fire count reward or penalty
            if fire_count == 0 and unseen_count == 0:
                reward -= 1

            # distance = manhattan_distance(drone.pos, self.env.home.pos)
            # max_area = ideal_area * (2 - distance / (self.env.size * 2))  # Distance factor reduces ideal area
            #
            # # Area size reward or penalty
            # if area_size < max_area - area_tolerance or area_size > max_area + area_tolerance:
            #     reward -= 1  # Penalty for inappropriate area size
        return reward / 10
