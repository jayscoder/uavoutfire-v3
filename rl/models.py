
class StepResult:
    def __init__(self, action, obs: dict, reward: float = 0, terminated: bool = False, truncated: bool = False,
                 info: dict = None):
        self.action = action
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
