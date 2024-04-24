import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.sac import CnnPolicy, MlpPolicy, MultiInputPolicy


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, (2, 2), padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, (2, 2), padding=1),
                nn.ReLU(),
                nn.Flatten(),
        )

        # # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(in_features=n_flatten, out_features=features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_output = self.cnn(observations)
        print('cnn_output', cnn_output.shape)
        return self.linear(cnn_output)


class RLBTFeaturesExtractor(BaseFeaturesExtractor):
    """RLBT节点用到的特征提取器"""

    def __init__(self, observation_space: gym.Space, features_dim: int = 128, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, (3, 3)),
                nn.ReLU(),
                nn.Conv2d(8, 16, (3, 3)),
                nn.ReLU(),
                nn.Flatten(),
        )

        # # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]

        input_features = n_flatten
        # input_features += observation_space['children_status_count'].sample().shape[0]
        # input_features += observation_space['status_count'].sample().shape[0]
        # try:
        #     input_features += observation_space['others'].sample().shape[0]
        # except:
        #     pass
        self.linear = nn.Sequential(
                nn.Linear(in_features=input_features,
                          out_features=features_dim),
                nn.ReLU())

    def forward(self, obs) -> torch.Tensor:
        features = self.cnn(obs['image'])
        # print('RLBTFeaturesExtractor', obs['image'].shape)
        # features = torch.cat((obs['children_status_count'], features), dim=1)
        # features = torch.cat((obs['status_count'], features), dim=1)
        # if 'others' in obs:
        #     features = torch.cat((obs['others'], features), dim=1)
        return self.linear(features)

#
# class CompositeFeaturesExtractor(BaseFeaturesExtractor):
#     """组合节点用到的特征提取器"""
#
#     def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
#         super().__init__(observation_space, features_dim)
#         n_input_channels = observation_space['image'].shape[0]
#         self.cnn = nn.Sequential(
#                 nn.Conv2d(n_input_channels, 16, (2, 2)),
#                 nn.ReLU(),
#                 nn.Conv2d(16, 32, (2, 2)),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, (2, 2)),
#                 nn.ReLU(),
#                 nn.Flatten(),
#         )
#
#         # # Compute shape by doing one forward pass
#         with torch.no_grad():
#             n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]
#
#         self.linear = nn.Sequential(
#                 nn.Linear(in_features=n_flatten + observation_space['status'].sample().shape[0] * 4 + 4,
#                           out_features=features_dim),
#                 nn.ReLU())
#
#     def forward(self, observations) -> torch.Tensor:
#         features = self.cnn(observations['image'])
#         status = observations['status']
#         status_count = observations['status_count']
#         return self.linear(torch.cat((status, features, status_count), dim=1))
