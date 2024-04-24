import numpy as np
from rl import *
from bts_base import *
from bts_builder import *
from features import *
from rl.logger import TensorboardLogger
from stable_baselines3 import *
from rl.common import is_off_policy_algo
from bts_rl import RLNode


@FIRE_BT_BUILDER.register_node
class DroneExtinguishFire(RLNode):
    """无人机灭火"""

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0, high=self.env.size, shape=(len(self.env.drones), 4), dtype=np.float32)

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
            self.platform.send_message(message=area_message, to_platform=self.env.platforms[i])
        return Status.SUCCESS

