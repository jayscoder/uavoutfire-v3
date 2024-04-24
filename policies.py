from __future__ import annotations
from envs import *


def home_greedy_action(self: Home) -> HomeActions:
    for drone in self.env.drones:
        if drone.pos == self.pos:
            return HomeActions.RECHARGE
    return HomeActions.KEEP


def drone_greedy_action(self: Drone) -> DroneActions:
    # if (self.x, self.y) == (self.env.home.x, self.env.home.y):
    #     self.env.home.recharge(self)

    # 检查电量或灭火剂是否需要回基地补充
    if self.battery <= (manhattan_distance(self.pos, self.env.home.pos) + 10) or self.fire_extinguisher == 0:
        # 返回基地（移向基地）
        next_move_vec = astar_find_next_direction(self.env.grid, start=self.pos, target=self.env.home.pos)
        return DroneActions.move_action_from_direction_vec(next_move_vec)
    else:
        # 常规行动逻辑
        if self.env.grid[self.x, self.y] == Objects.Fire and self.fire_extinguisher > 0:
            return DroneActions.EXTINGUISH
        else:
            # 例子：找到最近的已知火源
            # fire_positions = np.where(self.env.memory_matrix == 1)  # 假设火源在记忆矩阵中标记为1
            # # 如果观测区域内有火，则朝着观测区域的火靠近，否则随机移动
            lt_pos, rb_pos = self.visible_area()
            fire_positions = [(i, j) for i in range(lt_pos[0], rb_pos[0]) for j in
                              range(lt_pos[1], rb_pos[1])
                              if self.env.grid[i, j] == 1]

            next_move_dir: tuple[int, int] | None = None
            if fire_positions:
                # 如果存在火源，找到最近的火源
                closest_fire = min(fire_positions, key=lambda pos: manhattan_distance(pos, self.pos))
                next_move_dir = astar_find_next_direction(self.env.grid, start=self.pos, target=closest_fire)

            if next_move_dir is None:
                # 选择一个随机方向移动，避开障碍物
                move_directions = DIRECTION_VECS.copy()
                random.shuffle(move_directions)
                for (dx, dy) in move_directions:
                    new_x = max(0, min(self.env.size - 1, self.x + dx))
                    new_y = max(0, min(self.env.size - 1, self.y + dy))
                    if self.env.grid[new_x, new_y] != Objects.Obstacle:  # 确保不是障碍物
                        next_move_dir = dx, dy
                        break
            if next_move_dir is not None:
                return DroneActions.move_action_from_direction_vec(next_move_dir)
    return DroneActions.KEEP


def drone_greedy_action_with_memory(self: Drone, grid: np.array) -> DroneActions:
    # 检查电量或灭火剂是否需要回基地补充
    if self.battery <= (manhattan_distance(self.pos, self.env.home.pos) + 10) or self.fire_extinguisher == 0:
        # 返回基地（移向基地）
        next_move_vec = astar_find_next_direction(self.env.grid, start=self.pos, target=self.env.home.pos)
        return DroneActions.move_action_from_direction_vec(next_move_vec)
    else:
        # 常规行动逻辑
        if self.env.grid[self.x, self.y] == Objects.Fire and self.fire_extinguisher > 0:
            return DroneActions.EXTINGUISH
        else:
            # 例子：找到最近的已知火源
            fire_positions = np.where(grid == Objects.Fire)  # 假设火源在记忆矩阵中标记为1

            # # 如果观测区域内有火，则朝着观测区域的火靠近，否则随机移动
            next_move_dir: tuple[int, int] | None = None
            if len(fire_positions[0]) > 0:
                fire_positions = list(zip(fire_positions[0], fire_positions[1]))
                # 如果存在火源，找到最近的火源
                # 计算与当前位置的曼哈顿距离并找到最近的火源位置
                closest_fire = min(fire_positions, key=lambda pos: manhattan_distance(self.pos, pos))
                next_move_dir = astar_find_next_direction(grid, start=self.pos, target=closest_fire)

            if next_move_dir is None:
                # 选择一个随机方向移动，避开障碍物
                move_directions = DIRECTION_VECS.copy()
                random.shuffle(move_directions)
                for (dx, dy) in move_directions:
                    new_x = max(0, min(self.env.size - 1, self.x + dx))
                    new_y = max(0, min(self.env.size - 1, self.y + dy))
                    if self.env.grid[new_x, new_y] != Objects.Obstacle:  # 确保不是障碍物
                        next_move_dir = dx, dy
                        break
            if next_move_dir is not None:
                return DroneActions.move_action_from_direction_vec(next_move_dir)
    return DroneActions.KEEP
