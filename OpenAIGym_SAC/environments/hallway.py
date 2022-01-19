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

"""Hallway domain."""

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
  return (resources.GetResource(f'{current_dir}/hallway.xml'),
          common.ASSETS)


@SUITE.add('exploration')
def velocity_1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=1, distractor_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, distractor_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_3(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=3, distractor_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_4(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=4, distractor_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_1_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=1,
                 distractor_size=0.05, distractor_scale=1,
                 target_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2,
                 distractor_size=0.05, distractor_scale=1,
                 target_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_3_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=3,
                 distractor_size=0.05, distractor_scale=1,
                 target_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_4_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=4,
                 distractor_size=0.05, distractor_scale=1,
                 target_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_5_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=5,
                 distractor_size=0.05, distractor_scale=1,
                 target_scale=0,
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_6_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=6,
                 distractor_size=0.05, distractor_scale=1,
                 target_scale=0,
                 random=random)
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

  def mass_to_distractor(self):
    """Returns the vector from mass to distractor in global coordinate."""
    return (self.named.data.geom_xpos['distractor'] -
            self.named.data.geom_xpos['pointmass'])

  def mass_to_distractor_dist(self):
    """Returns the distance from mass to the distractor."""
    return np.linalg.norm(self.mass_to_distractor())


class Hallway(base.Task):
  """A point `Task` to reach target."""

  def __init__(self, velocity=True, vel_gain=1, length=6,
               target_size=0.05, target_scale=1,
               distractor_size=0.1, distractor_scale=0, distractor_offset=0.0,
               sigmoid='gaussian', random=None):
    """Initialize an instance of `Hallway`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self.velocity = velocity
    self.vel_gain = vel_gain
    self.max_x = length / 2
    self.target_size = target_size
    self.target_scale = target_scale
    self.distractor_size = distractor_size
    self.distractor_scale = distractor_scale
    self.distractor_offset = distractor_offset
    self.sigmoid = sigmoid
    super(Hallway, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    physics.named.data.qpos[:] = [-self.max_x + 0.1, 0]
    physics.named.model.geom_pos['wall_x'][0] = -self.max_x
    physics.named.model.geom_pos['wall_neg_x'][0] = self.max_x
    physics.named.model.geom_pos['target'][0] = self.max_x - 0.2
    physics.named.model.geom_pos['distractor'][0] = self.distractor_offset
    physics.named.model.jnt_range['root_x'] = [-self.max_x, self.max_x]
    physics.named.model.geom_size['target', 0] = self.target_size
    physics.named.model.geom_size['distractor', 0] = self.distractor_size
    super(Hallway, self).initialize_episode(physics)

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
                                    bounds=(0, target_size),
                                    margin=target_size,
                                    sigmoid=self.sigmoid)

    distractor_size = physics.named.model.geom_size['distractor', 0]
    near_distractor = rewards.tolerance(physics.mass_to_distractor_dist(),
                                        bounds=(0, distractor_size),
                                        margin=distractor_size,
                                        sigmoid=self.sigmoid)

    proximity_reward = (near_target * self.target_scale +
                        near_distractor * self.distractor_scale)

    control_reward = rewards.tolerance(physics.control(), margin=1,
                                       value_at_margin=0,
                                       sigmoid='quadratic').mean()
    small_control = (control_reward + 4) / 5
    return (proximity_reward) * small_control


if __name__ == '__main__':
  env = suite.load('hallway', 'velocity_4')
  print(env.reset())

#   for i in range(100):
#     print(env.step(np.ones(2)).observation)
  # step = env.step(np.zeros(2))
  # pass
