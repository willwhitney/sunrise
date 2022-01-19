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

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pathlib

from dm_control import mujoco, suite
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  current_dir = pathlib.Path(__file__).parent.absolute()
  return (resources.GetResource(f'{current_dir}/point.xml'),
          common.ASSETS)


@SUITE.add()
def velocity(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Point(velocity=True, vel_gain=1.0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def velocity_slow(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                  environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Point(velocity=True, vel_gain=0.2, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def mass(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Point(velocity=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

  def mass_to_target(self):
    """Returns the vector from mass to target in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['pointmass'])

  def mass_to_target_dist(self):
    """Returns the distance from mass to the target."""
    return np.linalg.norm(self.mass_to_target())


class Point(base.Task):
  """A point `Task` to reach target."""

  def __init__(self, velocity=True, vel_gain=1, random=None):
    """Initialize an instance of `Point`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self.velocity = velocity
    self.vel_gain = vel_gain
    super(Point, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    physics.named.data.qpos[:] = [-0.25, -0.25]
    super(Point, self).initialize_episode(physics)

  def before_step(self, action, physics):
    super().before_step(action, physics)
    if self.velocity:
      physics.set_control(np.zeros(2))
      physics.named.data.qvel[:] = action / self.vel_gain

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = np.array(physics.named.data.qpos)
    if not self.velocity:
      obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    target_size = physics.named.model.geom_size['target', 0]
    near_target = rewards.tolerance(physics.mass_to_target_dist(),
                                    bounds=(0, target_size), margin=target_size)
    control_reward = rewards.tolerance(physics.control(), margin=1,
                                       value_at_margin=0,
                                       sigmoid='quadratic').mean()
    small_control = (control_reward + 4) / 5
    return near_target * small_control


if __name__ == '__main__':
  env = suite.load('point', 'velocity')
  print(env.reset())
  for i in range(100):
    print(env.step(np.ones(2)).observation)
  # step = env.step(np.zeros(2))
  # pass
