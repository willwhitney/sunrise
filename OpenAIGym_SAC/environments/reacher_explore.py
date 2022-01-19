# Modified from dm_control. Original license below.
#
# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Reacher domain."""

import collections
import pathlib

from dm_control import mujoco
from dm_control.rl import control
from dm_control import suite
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  current_dir = pathlib.Path(__file__).parent.absolute()
  return (resources.GetResource(f'{current_dir}/reacher_explore.xml'),
          common.ASSETS)

@SUITE.add('exploration')
def hard_fixed_init(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                    environment_kwargs=None):
  """Returns reacher with sparse reward with 1e-2 tol and randomized target,
  but no random resets."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = ReacherExplore(target_size=_SMALL_TARGET, random_resets=False,
                        random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def hard_narrow_init(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                    environment_kwargs=None):
  """Returns reacher with sparse reward with 1e-2 tol and randomized target,
  but no random resets."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = ReacherExplore(target_size=_SMALL_TARGET, random_resets=True,
                        random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinates."""
    return (self.named.data.geom_xpos['target', :2] -
            self.named.data.geom_xpos['finger', :2])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class ReacherExplore(base.Task):
  """A reacher `Task` to reach the target."""

  def __init__(self, target_size, random_resets=True, random=None):
    """Initialize an instance of `Reacher`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._target_size = target_size
    self._random_resets = random_resets
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = self._target_size

    if self._random_resets:
      arm_angles = np.random.uniform(-np.pi / 4, np.pi / 4, size=(2,))
      physics.data.qpos[:] = arm_angles
    else:
      physics.data.qpos[:] = 0

    # Randomize target position
    angle = self.random.uniform(1.25 * np.pi, 1.75 * np.pi)
    radius = self.random.uniform(.05, .20)
    physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
    physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state and the target position."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.finger_to_target()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
    return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))


if __name__ == '__main__':
  env = suite.load('reacher_explore', 'hard_fixed_init')
  print(env.reset())
#   for i in range(100):
#     print(env.step(np.ones(2)).observation)
