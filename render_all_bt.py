import pybts
import os
from pybts.display import render_node
import os.path

from envs import *
from bts_builder import *
from bts_tree import *
from utils import folder_run_id
import bts_home
import bts_drone
import bts_home_rl

"""
生成scripts中所有行为树的图到scripts/images中
"""

SCRIPTS_PATH = 'scripts/子树'
IMAGES_PATH = 'scripts/images'

if __name__ == '__main__':
    for filename in os.listdir(SCRIPTS_PATH):
        if not filename.endswith('.xml'):
            continue
        path = os.path.join(SCRIPTS_PATH, filename)
        tree = FIRE_BT_BUILDER.build_from_file(path)
        render_node(tree, os.path.join(IMAGES_PATH, '{}.png'.format(filename.replace('.xml', ''))))
