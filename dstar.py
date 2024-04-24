import numpy as np
from dstar_lite import OccupancyGridMap, DStarLite, SLAM
from dstar_lite.grid import Vertex, Vertices
from envs import Platform, Drone
from constanst import *


class DStarLiteManager:
    def __init__(self, platform: Drone, goal: tuple[int, int]):
        self.goal = goal
        self.map = OccupancyGridMap(x_dim=platform.memory_grid.shape[0], y_dim=platform.memory_grid.shape[1],
                                    exploration_setting='4N')

        self.dstar = DStarLite(map=self.map, s_start=platform.pos, s_goal=goal)
        self.platform = platform
        self.update_all()

    def move_and_replan(self):
        try:
            path, g, rhs = self.dstar.move_and_replan(robot_position=self.platform.pos)
            return path
        except:
            return None

    def c(self, u: (int, int), v: (int, int)) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if not self.map.is_unoccupied(u) or not self.map.is_unoccupied(v):
            return float('inf')
        else:
            return self.heuristic(u, v)

    def heuristic(self, p: (int, int), q: (int, int)) -> float:
        """
        Helper function to compute distance between two points.
        :param p: (x,y)
        :param q: (x,y)
        :return: manhattan distance
        """
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    def update_all(self):
        self.update_view(((0, 0), self.platform.memory_grid.shape))

    def update_view(self, rect: tuple[tuple[int, int], tuple[int, int]] = None):
        vertices = Vertices()
        if rect is None:
            rect = self.platform.visible_area()
        lt_pos, rb_pos = rect
        for y in range(lt_pos[1], rb_pos[1]):
            for x in range(lt_pos[0], rb_pos[1]):
                node = (x, y)
                if self.map.is_unoccupied((x, y)) and (
                        self.platform.memory_grid[x, y] == Objects.Obstacle or self.platform.unreachable_grid[
                    x, y] == 1):
                    v = Vertex(pos=node)
                    succ = self.map.succ(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                    vertices.add_vertex(v)
                    self.map.set_obstacle(node)
        self.dstar.new_edges_and_old_costs = vertices
        self.dstar.sensed_map = self.map
        return vertices
