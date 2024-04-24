import os.path

from envs import *
from bts_builder import *
from bts_tree import *
from utils import folder_run_id
import bts_home
import bts_drone
import bts_home_rl


def main(N: int, folder: str, render: bool,
         track: int,
         train: bool,
         home_file='home.xml',
         explore_drone_file='../子树/探索UAV.xml', extinguish_drone_file='../子树/灭火UAV.xml'):
    env = FireEnvironment(50)
    sim = BTSimulator(
            title=folder,
            env=env,
            home_tree_file=os.path.join(folder, home_file),
            explore_drone_tree_file=os.path.join(folder, explore_drone_file),
            extinguish_drone_tree_file=os.path.join(folder, extinguish_drone_file),
            render=render,
    )
    start_time = time.time()
    sim.simulate(N, track=track, train=train)
    cost_time = time.time() - start_time
    print(f'Cost time: {cost_time:.2f} seconds')


if __name__ == '__main__':
    main(N=10000, folder='RL分配-知道无人机', render=True, track=0, train=True)
