"""Env with random maze layouts."""

import gym
import numpy as np
import tensorflow.compat.v2 as tf

from gym import spaces
from gym.utils import seeding

import dice_rl.utils.common as common_utils
from dice_rl.environments.gridworld.navigation import Navigation


class Maze(Navigation):
  """A subclass of Navigation with randomly generated maze layout.

  The ASCII representation of the mazes include the following objects:

  - `<SPACE>`: empty
  - `x`: wall
  - `S`: the start location (optional)
  - `T`: the target location.
  """
  KEY_EMPTY = 0
  KEY_WALL = 1
  KEY_TARGET = 2
  KEY_START = 3
  ASCII_MAP = {
      KEY_EMPTY: ' ',
      KEY_WALL: 'x',
      KEY_TARGET: 'T',
      KEY_START: 'S',
  }

  def __init__(self, size, wall_type, maze_seed=0, random_start=True):
    self._size = size
    self._random_start = random_start
    self._maze = self.generate_maze(size, maze_seed, wall_type)
    self._num_maze_keys = len(Maze.ASCII_MAP.keys())
    nav_map = self.maze_to_ascii(self._maze)
    super(Maze, self).__init__(
        nav_map,
        tabular_obs=False,
        done_fn=lambda x, y, tx, ty: x == tx and y == ty)

  @property
  def num_maze_keys(self):
    return self._num_maze_keys

  @property
  def size(self):
    return self._size

  def get_maze_map(self, stacked=True):
    if not stacked:
      return self._maze.copy()
    wall = self._maze.copy()
    target_x, target_y = self.target_location
    assert wall[target_x][target_y] == Maze.KEY_TARGET
    wall[target_x][target_y] = 0
    target = np.zeros((self._size, self._size))
    target[target_x][target_y] = 1
    if not self._random_start:
      assert wall[self._start_x][self._start_y] == Maze.KEY_START
      wall[self._start_x][self._start_y] = 0
    return np.stack([wall, target], axis=-1)

  def generate_maze(self, size, seed, wall_type):
    rng, _ = seeding.np_random(seed)
    maze = np.full((size, size), fill_value=Maze.KEY_EMPTY, dtype=int)

    if wall_type == 'none':
      maze[[0, -1], :] = Maze.KEY_WALL
      maze[:, [0, -1]] = Maze.KEY_WALL
    elif wall_type == 'tunnel':
      self.sample_wall(maze, rng)
    elif wall_type.startswith('blocks:'):
      maze[[0, -1], :] = Maze.KEY_WALL
      maze[:, [0, -1]] = Maze.KEY_WALL
      self.sample_blocks(maze, rng, int(wall_type.split(':')[-1]))
    else:
      raise ValueError('Unknown wall type: %s' % wall_type)

    loc_target = self.sample_location(maze, rng)
    maze[loc_target] = Maze.KEY_TARGET

    if not self._random_start:
      loc_start = self.sample_location(maze, rng)
      maze[loc_start] = Maze.KEY_START
      self._start_x, self._start_y = loc_start

    return maze

  def sample_blocks(self, maze, rng, num_blocks):
    """Sample single-block 'wall' or 'obstacles'."""
    for _ in range(num_blocks):
      loc = self.sample_location(maze, rng)
      maze[loc] = Maze.KEY_WALL

  def sample_wall(self,
                  maze,
                  rng,
                  shortcut_prob=0.1,
                  inner_wall_thickness=1,
                  outer_wall_thickness=1,
                  corridor_thickness=2):
    room = maze

    # step 1: fill everything as wall
    room[:] = Maze.KEY_WALL

    # step 2: prepare
    # we move two pixels at a time, because the walls are also occupying pixels
    delta = inner_wall_thickness + corridor_thickness
    dx = [delta, -delta, 0, 0]
    dy = [0, 0, delta, -delta]

    def get_loc_type(y, x):
      # remember there is a outside wall of 1 pixel surrounding the room
      if (y < outer_wall_thickness or
          y + corridor_thickness - 1 >= room.shape[0] - outer_wall_thickness):
        return 'invalid'
      if (x < outer_wall_thickness or
          x + corridor_thickness - 1 >= room.shape[1] - outer_wall_thickness):
        return 'invalid'
      # already visited
      if room[y, x] == Maze.KEY_EMPTY:
        return 'occupied'
      return 'valid'

    def connect_pixel(y, x, ny, nx):
      pixel = Maze.KEY_EMPTY
      if ny == y:
        room[y:y + corridor_thickness,
             min(x, nx):max(x, nx) + corridor_thickness] = pixel
      else:
        room[min(y, ny):max(y, ny) + corridor_thickness,
             x:x + corridor_thickness] = pixel

    def carve_passage_from(y, x):
      room[y, x] = Maze.KEY_EMPTY
      for direction in rng.permutation(len(dx)):
        ny = y + dy[direction]
        nx = x + dx[direction]

        loc_type = get_loc_type(ny, nx)
        if loc_type == 'invalid':
          continue
        elif loc_type == 'valid':
          connect_pixel(y, x, ny, nx)
          # recursion
          carve_passage_from(ny, nx)
        else:
          # occupied
          # we create shortcut with some probability, this is because
          # we do not want to restrict to only one feasible path.
          if rng.rand() < shortcut_prob:
            connect_pixel(y, x, ny, nx)

    carve_passage_from(outer_wall_thickness, outer_wall_thickness)

  def sample_location(self, maze, rng):
    for _ in range(1000):
      x, y = rng.randint(low=1, high=self._size, size=2)
      if maze[x, y] == Maze.KEY_EMPTY:
        return (x, y)
    raise ValueError('Cannot sample empty location, make maze bigger?')

  @staticmethod
  def key_to_ascii(key):
    if key in Maze.ASCII_MAP:
      return Maze.ASCII_MAP[key]
    assert (key >= Maze.KEY_OBJ and key < Maze.KEY_OBJ + Maze.MAX_OBJ_TYPES)
    return chr(ord('1') + key - Maze.KEY_OBJ)

  def maze_to_ascii(self, maze):
    return [[Maze.key_to_ascii(x) for x in row] for row in maze]

  def tabular_obs_action(self, status_obs, action, include_maze_layout=False):
    tabular_obs = self.get_tabular_obs(status_obs)
    multiplier = self._n_action
    if include_maze_layout:
      multiplier += self._num_maze_keys
    return multiplier * tabular_obs + action


def get_value_map(env):
  """Returns [W, W, A] one-hot VI actions."""
  target_location = env.target_location
  nav_map = env.nav_map
  current_points = [target_location]
  chosen_actions = {target_location: 0}
  visited_points = {target_location: True}

  while current_points:
    next_points = []
    for point_x, point_y in current_points:
      for (action, (next_point_x,
                    next_point_y)) in [(0, (point_x - 1, point_y)),
                                       (1, (point_x, point_y - 1)),
                                       (2, (point_x + 1, point_y)),
                                       (3, (point_x, point_y + 1))]:

        if (next_point_x, next_point_y) in visited_points:
          continue

        if not (next_point_x >= 0 and next_point_y >= 0 and
                next_point_x < len(nav_map) and
                next_point_y < len(nav_map[next_point_x])):
          continue

        if nav_map[next_point_x][next_point_y] == 'x':
          continue

        next_points.append((next_point_x, next_point_y))
        visited_points[(next_point_x, next_point_y)] = True
        chosen_actions[(next_point_x, next_point_y)] = action
    current_points = next_points

  value_map = np.zeros([env.size, env.size, env.n_action])
  for (x, y), action in chosen_actions.items():
    value_map[x][y][action] = 1
  return value_map


def get_bfs_sequence(env,
                     observation,
                     backtrack=True,
                     include_maze_layout=False):
  """Returns a sequence of tabular BFS search and backtrack."""
  start_x, start_y = observation
  target_x, target_y = env.target_location
  nav_map = env.nav_map

  bfs_sequence = []
  visited_points = [[None for _ in range(env.size)] for _ in range(env.size)]
  visited_points[start_x][start_y] = (start_x, start_y, 0)
  current_points = [(start_x, start_y)]

  found_target = False
  while current_points and not found_target:
    next_points = []
    for point_x, point_y in current_points:
      for (action, (next_point_x, next_point_y)) in [
          (3, (point_x, point_y - 1)),
          (2, (point_x - 1, point_y)),
          (1, (point_x, point_y + 1)),
          (0, (point_x + 1, point_y)),
      ]:

        if visited_points[next_point_x][next_point_y]:
          continue

        if not (next_point_x >= 0 and next_point_y >= 0 and
                next_point_x < len(nav_map) and
                next_point_y < len(nav_map[next_point_x])):
          continue

        if nav_map[next_point_x][next_point_y] == 'x':
          continue

        visited_points[next_point_x][next_point_y] = (point_x, point_y, action)
        xya = env.tabular_obs_action(
            np.array([next_point_x, next_point_y]),
            action,
            include_maze_layout=include_maze_layout)
        bfs_sequence.append(xya)
        next_points.append((next_point_x, next_point_y))

        if next_point_x == target_x and next_point_y == target_y:
          found_target = True

    current_points = next_points

  if backtrack:
    point_x, point_y = env.target_location
    while point_x != start_x or point_y != start_y:
      previous_x, previous_y, action = visited_points[point_x][point_y]
      xya = env.tabular_obs_action(
          np.array([previous_x, previous_y]),
          action,
          include_maze_layout=include_maze_layout)
      bfs_sequence.append(xya)
      point_x = previous_x
      point_y = previous_y

  return np.array(bfs_sequence)
