import os.path

from envs import *
from bts_builder import *
from bts_tree import *
from utils import folder_run_id
import bts_home
import bts_drone
import bts_home_rl


def main(N: int, folder: str, render: bool, track: int, train: bool):
    env = FireEnvironment(50)
    sim = BTSimulator(
            title=folder,
            env=env,
            home_tree_file=os.path.join(folder, 'home.xml'),
            drone_tree_file=os.path.join(folder, 'drone.xml'),
            render=render,
    )
    start_time = time.time()
    sim.simulate(N, track=track, train=train)
    cost_time = time.time() - start_time
    print(f'Cost time: {cost_time:.2f} seconds')


if __name__ == '__main__':
    main(N=5000, folder='RLSwitcher决定任务分配-知道无人机', render=False, track=0, train=True)
