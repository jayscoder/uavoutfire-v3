import json

import numpy as np
from collections import defaultdict
from constanst import *
import gymnasium as gym
import time


def time_str():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


# class QLearningAgent:
#     def __init__(self, alpha=0.5, gamma=0.99, epsilon=0.1):
#         self.Q = defaultdict(lambda: defaultdict(float))
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#
#     def select_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(['move', 'command_extinguish', 'return_base'])
#         else:
#             return max(self.Q[state], key=self.Q[state].get)
#
#     def update(self, state, action, reward, next_state):
#         future_rewards = max(self.Q[next_state].values()) if self.Q[next_state] else 0
#         self.Q[state][action] += self.alpha * (reward + self.gamma * future_rewards - self.Q[state][action])
#
#     def learn(self, state, action, reward, next_state):
#         self.update(state, action, reward, next_state)

class QLearningAgent:
    Q_MAP = { }

    # 同样的学习域内动作空间相同

    def __init__(self, action_space: gym.spaces.Space, domain: str = 'default', alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = self.Q_MAP.get(domain, defaultdict(lambda: defaultdict(float)))
        self.Q_MAP[domain] = self.Q
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state = None
        self.action = None
        self.action_space = action_space

    def observe(self, observation):
        self.state = observation

    def decide(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(DroneActions.get_action_list())
        else:
            return max(self.Q[self.state], key=self.Q[self.state].get)

    def learn(self, reward, next_state):
        future_rewards = max(self.Q[next_state].values()) if self.Q[next_state] else 0
        self.Q[self.state][self.action] += self.alpha * (
                reward + self.gamma * future_rewards - self.Q[self.state][self.action])

    def update(self, action, reward, new_state):
        self.learn(reward, new_state)
        self.state = new_state
        self.action = action

    def dump(self):
        with open(f'Q-{time_str()}.json', 'w') as fp:
            json.dump(self.Q, ensure_ascii=False, indent=4)

# class CommunicativeQLearningAgent(SharedQLearningAgent):
#     def __init__(self, peers, **kwargs):
#         super().__init__(**kwargs)
#         self.peers = peers  # 其他智能体的引用列表
#
#     def decide(self):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.action_space)
#         else:
#             # 可能会考虑从其他智能体那里获取最佳策略
#             peer_actions = [peer.get_best_action(self.state) for peer in self.peers]
#             # 实际决策可以结合本地和获取的策略
#             return np.random.choice(peer_actions + [max(self.Q[self.state], key=self.Q[self.state].get, default=np.random.choice(self.action_space))])
#
#     def get_best_action(self, state):
#         return max(self.Q[state], key=self.Q[state].get, default=np.random.choice(self.action_space))
