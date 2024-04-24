import os

from envs import *
from bts_builder import *
from bts_tree import *
from utils import folder_run_id
import bts_home
import bts_drone
import bts_home_rl
from stable_baselines3 import PPO, SAC

from features import CNNFeaturesExtractor


def run_sim(N: int, render: bool = False,
            title: str = '', ):
    sim = BTSimulator(
            title=title,
            env=FireEnvironment(50),
            home_tree_file='scripts/子树/Stable-基地.xml',
            explore_drone_tree_file='scripts/子树/探索UAV.xml',
            extinguish_drone_tree_file='scripts/子树/灭火UAV.xml',
            render=render,
            context={ 'outdated_time': 300 }
    )
    model = PPO('CnnPolicy', env=sim, verbose=1, tensorboard_log='./logs/Stable-PPO', use_sde=False, policy_kwargs={
        'normalize_images'         : False,
        'features_extractor_class' : CNNFeaturesExtractor,
        'features_extractor_kwargs': dict(features_dim=256),
    }, device='cpu')
    model.learn(total_timesteps=N, progress_bar=True)
    model.save("models/Stable-PPO")


if __name__ == '__main__':
    run_sim(N=2500000, render=False, title='Stable-PPO')
