import os.path

from envs import *
from bts_builder import *
from bts_tree import *
from utils import folder_run_id
import bts_home
import bts_drone
import bts_home_rl
from main import run_sim

if __name__ == '__main__':
    # run_sim(N=200, folder='平均分配', outdated_time=300, title=f'平均分配-outdated-time-300')
    for t in [50, 100, 200, 300, 400, 500, 600, 1000, 10000]:
        run_sim(N=200, folder='平均分配', outdated_time=t, title=f'平均分配-outdated-time-{t}')
