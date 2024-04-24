from rl import RLBuilder
from pybts import *


class FireBTBuilder(RLBuilder):
    def register_default(self):
        super().register_default()
        # self.register_node(
        #         HomeGreedyAction
        # )
        #
        # self.register_node(
        #         DroneGreedyMove,
        #         MoveUp,
        #         MoveDown,
        #         MoveRight,
        #         MoveLeft,
        #         DroneSendViewUpdate
        # )


FIRE_BT_BUILDER = FireBTBuilder(folders='scripts')
