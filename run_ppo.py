import os.path

from envs import *
from bts_builder import *
from bts_tree import *
from utils import folder_run_id
import bts_home
import bts_drone
import bts_home_rl


class EnvWrapper(gym.Env):

    def __init__(self, env: FireEnvironment):
        self.env = env



def main():
    env = FireEnvironment(50)
    sim = BTSimulator(
            title=title or folder,
            env=env,
            home_tree_file=os.path.join(folder, home_file),
            explore_drone_tree_file=os.path.join(folder, explore_drone_file),
            extinguish_drone_tree_file=os.path.join(folder, extinguish_drone_file),
            render=render,
            context={ 'outdated_time': outdated_time }
    )
    start_time = time.time()
    sim.simulate(N, track=track, train=train)
    cost_time = time.time() - start_time


if __name__ == '__main__':
    run_sim(N=10000, folder='PPO分配')
