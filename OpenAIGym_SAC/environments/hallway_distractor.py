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

from .hallway import Physics, Hallway

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  current_dir = pathlib.Path(__file__).parent.absolute()
  return (resources.GetResource(f'{current_dir}/hallway.xml'),
          common.ASSETS)


# @SUITE.add('exploration')
# def velocity_1_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#              environment_kwargs=None):
#   """Returns the easy point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Hallway(velocity=True, vel_gain=1.0, length=1, target_size=0.2,
#                  distractor_size=0.2, distractor_scale=0.1, sigmoid='gaussian',
#                  random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, **environment_kwargs)

# @SUITE.add('exploration')
# def velocity_2_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#              environment_kwargs=None):
#   """Returns the easy point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.2,
#                  distractor_size=0.4, distractor_scale=0.1, sigmoid='gaussian',
#                  random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, **environment_kwargs)

# @SUITE.add('exploration')
# def velocity_4_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#              environment_kwargs=None):
#   """Returns the easy point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Hallway(velocity=True, vel_gain=1.0, length=4, target_size=0.2,
#                  distractor_size=1.0, distractor_scale=0.1, sigmoid='gaussian',
#                  random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, **environment_kwargs)


# @SUITE.add('exploration')
# def velocity_4_inverse_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#              environment_kwargs=None):
#   """Returns the easy point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Hallway(velocity=True, vel_gain=1.0, length=4, target_scale=0.0,
#                  distractor_size=0.01, distractor_scale=1.0,
#                  distractor_offset=-1.5, sigmoid='gaussian',
#                  random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
      # physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('exploration')
def velocity_2_distractor_p01(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.1,
                 distractor_size=0.1, distractor_scale=0.01, sigmoid='gaussian',
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_distractor_p03(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.1,
                 distractor_size=0.1, distractor_scale=0.03, sigmoid='gaussian',
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_distractor_p1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.1,
                 distractor_size=0.1, distractor_scale=0.1, sigmoid='gaussian',
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_distractor_p3(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.1,
                 distractor_size=0.1, distractor_scale=0.3, sigmoid='gaussian',
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_distractor_1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.1,
                 distractor_size=0.1, distractor_scale=1, sigmoid='gaussian',
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_distractor_3(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2, target_size=0.1,
                 distractor_size=0.1, distractor_scale=3, sigmoid='gaussian',
                 random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('exploration')
def velocity_1_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=1,
                 target_size=0.1, target_scale=1.0,
                 distractor_size=0.1, distractor_scale=0.1,
                 sigmoid='gaussian', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_1_inverse_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=1,
                 target_size=0.1, target_scale=0.1,
                 distractor_size=0.1, distractor_scale=1,
                 sigmoid='gaussian', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2,
                 target_size=0.1, target_scale=1.0,
                 distractor_offset=-0.5, distractor_size=0.1, distractor_scale=0.1,
                 sigmoid='gaussian', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_2_inverse_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=2,
                 target_size=0.1, target_scale=0.1,
                 distractor_offset=-0.5, distractor_size=0.1, distractor_scale=1,
                 sigmoid='gaussian', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_4_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=4,
                 target_size=0.2, target_scale=1.0,
                 distractor_offset=-1.0, distractor_size=0.1, distractor_scale=0.1,
                 sigmoid='gaussian', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('exploration')
def velocity_4_inverse_distractor(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hallway(velocity=True, vel_gain=1.0, length=4,
                 target_size=0.1, target_scale=0.1,
                 distractor_offset=-1.5, distractor_size=0.01, distractor_scale=1,
                 sigmoid='gaussian', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

