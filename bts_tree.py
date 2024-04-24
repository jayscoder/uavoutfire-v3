import pybts

from envs import *
from rl import RLTree
from rl.logger import TensorboardLogger
from pybts import Node
from bts_home_rl import HomeRLNode


class BTSimulator:
    def __init__(self,
                 title: str,
                 env: FireEnvironment,
                 home_tree_file: str,
                 drone_tree_file: str,
                 render: bool = False):
        self.title = title
        self.env = env
        self.render = render
        from bts_builder import FIRE_BT_BUILDER
        self.trees = []

        self.run_id = folder_run_id(os.path.join('scripts', self.title))
        self.logs_dir = os.path.join('logs', title, self.run_id)
        self.models_dir = os.path.join('models', title, self.run_id)

        for platform in self.env.platforms:
            if isinstance(platform, Home):
                root = FIRE_BT_BUILDER.build_from_file(home_tree_file)
            else:
                root = FIRE_BT_BUILDER.build_from_file(drone_tree_file)
            self.trees.append(PlatformTree(root=root, env=env, sim=self, platform_id=platform.id))

        for tree in self.trees:
            tree.context['logs_dir'] = self.logs_dir
            tree.context['models_dir'] = self.models_dir
            tree.setup()

        if render:
            env.pygame_init()

        self.logger = TensorboardLogger(folder=self.logs_dir, verbose=1)

    def act(self):
        for tree in self.trees:
            tree.tick()

    def reset(self):
        if self.env.time > 0:
            self.log_end()

        self.env.reset()
        for tree in self.trees:
            tree.reset()
        self.logger.dump(step=self.env.episode)

    def log_update(self, pbar: tqdm):
        env = self.env
        pbar.set_postfix({
            'episode': env.episode,
            '无人机' : env.alive_drones_count,
            '火'     : env.alive_fires
        })

    def log_end(self):
        env = self.env
        avg_n = 50

        for tree in self.trees:
            # 设置环境默认奖励
            if 'reward' in tree.context:
                accum_reward = sum(tree.context['reward'].values())
                self.logger.record_and_mean_n_episodes(f'{tree.name}-奖励', accum_reward, n=avg_n)
        self.logger.record_and_mean_n_episodes('步数', env.time, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/火量', env.alive_fires, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/火量比例', env.alive_fires_ratio, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/无人机', env.alive_drones_count, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/无人机比例', env.alive_drones_ratio, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/草地', env.alive_flammables, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/草地比例', env.alive_flammables_ratio, n=avg_n)
        self.logger.record_and_mean_n_episodes('剩余/无人机编队长', len(env.alive_squad_leaders), n=avg_n)
        self.logger.record_and_mean_n_episodes('消灭/火量', env.extinguish_fire_count, n=avg_n)

        unseen_count = np.sum(self.env.home.memory_grid == Objects.Unseen)
        unseen_count_ratio = unseen_count / (self.env.size ** 2)
        self.logger.record_and_mean_n_episodes(key='未知区域', value=unseen_count, n=50)
        self.logger.record_and_mean_n_episodes(key='未知区域占比', value=unseen_count_ratio, n=50)

    def should_update(self):
        if self.render:
            return time.time() - self.env.last_update_time > 0.01
        else:
            return True

    def should_render(self):
        if self.render:
            return time.time() - self.env.last_render_time > 0.02
        else:
            return False

    def simulate(self, episodes: int, track: int = 0, train: bool = False):
        env = self.env
        running = True
        pbar = tqdm(total=episodes, desc=f'[{self.title} train={train}]')

        for tree in self.trees:
            tree.context['train'] = train

        boards = [pybts.Board(tree=t, log_dir=self.logs_dir) for t in self.trees[:2]]
        for board in boards:
            board.clear()
        for episode in range(episodes):
            self.reset()
            while (not env.done) and running:
                if self.should_update():
                    self.act()
                    _, reward, terminated, truncated, _ = env.update()  # Update the state of the environment
                    for tree in self.trees:
                        # 设置环境默认奖励
                        if 'reward' in tree.context:
                            tree.context['reward']['default'] += reward

                if track > 0 and env.time % track == 0:
                    for board in boards:
                        board.track()
                if self.render and self.should_render():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                    env.pygame_render()
                    self.log_update(pbar)
                if env.terminated or env.truncated:
                    break
            for tree in self.trees:
                tree.terminate()
            self.log_update(pbar=pbar)
            pbar.update(1)
            print()


class PlatformTree(RLTree):
    def __init__(self, root: Node, env: FireEnvironment, sim: BTSimulator, platform_id: int):
        platform = env.platforms[platform_id]
        super().__init__(root, name=f'{platform.__class__.__name__}-{platform_id}')
        self.context.update({
            'platform_id': platform_id,
            'env'        : env,
            'sim'        : sim,
            'time'       : env.time,
            'train'      : False,
            'cache'      : { }  # 缓存会在每次reset的时候清除
        })
        self.platform_id = platform_id
        self.env: FireEnvironment = env

    def reset(self):
        super().reset()
        self.context['time'] = self.env.time
        self.context['cache'].clear()

    @property
    def platform(self) -> Platform:
        return self.env.platforms[self.platform_id]

    def tick(
            self,
            pre_tick_handler: typing.Optional[
                typing.Callable[[RLTree], None]
            ] = None,
            post_tick_handler: typing.Optional[
                typing.Callable[[RLTree], None]
            ] = None,
    ) -> None:
        self.context['time'] = self.env.time
        super().tick()

    def terminate(self):
        for node in self.root.iterate():
            if isinstance(node, HomeRLNode):
                node.take_action()
