import numpy as np
from collections import OrderedDict

import dm_env

import jax
from jax import numpy as jnp

import utils


DTYPE = jnp.float32

ACTION_MAP = jnp.stack([
    jnp.array((1, 0), dtype=DTYPE),   # visually DOWN
    jnp.array((0, 1), dtype=DTYPE),   # visually RIGHT
    jnp.array((-1, 0), dtype=DTYPE),  # visually UP
    jnp.array((0, -1), dtype=DTYPE),  # visually LEFT
])


class GridWorld:
    def __init__(self, size, duration):
        self.size = size
        self.duration = duration
        self.agent = jnp.ndarray = jnp.array((0, 0), dtype=DTYPE)
        self.timestep = 0
        self.actions = jnp.arange(4)

    def goal(self):
        return jnp.array((self.size - 1, self.size - 1), dtype=DTYPE)

    def reset(self):
        self.agent = jnp.array((0, 0), dtype=DTYPE)
        self.timestep = 0
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST,
                               reward=0.0,
                               discount=1.0,
                               observation=self.render())

    def render(self):
        observation = OrderedDict({'position': self.agent})
        return observation

    def step(self, action):
        action = int(action)
        self.agent = self.agent + ACTION_MAP[action]
        self.agent = jnp.clip(self.agent, 0, self.size - 1)
        reward = float((self.agent == self.goal()).all())
        self.timestep += 1
        if self.timestep < self.duration:
            step_type = dm_env.StepType.MID
            discount = 1
        else:
            step_type = dm_env.StepType.LAST
            discount = 0

        return dm_env.TimeStep(step_type=step_type,
                               reward=reward,
                               discount=discount,
                               observation=self.render())

    def action_spec(self):
        return dm_env.specs.DiscreteArray(4)

    def observation_spec(self):
        ex_state = self.render()['position']
        return OrderedDict({
            'position': dm_env.specs.BoundedArray(
                name='position',
                shape=ex_state.shape,
                dtype=ex_state.dtype,
                minimum=np.zeros((2,)),
                maximum=np.full((2,), self.size - 1))})


def all_coords(size):
    grid = jnp.stack(jnp.meshgrid(jnp.linspace(0, size - 1, size),
                                  jnp.linspace(0, size - 1, size)))
    return grid.transpose().reshape(-1, 2).astype(DTYPE)


def render_function(fn, env, reduction=jnp.max):
    """Renders a given function at every state in the gridworld.

    Arguments:
    - fn: a function that takes (batch of states, batch of actions) as its
        arguments
    - env: a GridWorld instance
    - reduction: a function mapping from jnp.ndarray -> float. maps from the
        vector of values for each action at a particular state to a single
        value which will represent that state.
    """
    locations = all_coords(env.size)
    repeated_locations = locations.repeat(len(env.actions), axis=0)

    actions = env.actions
    tiled_actions = jnp.tile(actions, (len(locations),)).reshape((-1, 1))

    values = fn(repeated_locations, tiled_actions)
    sa_values = values.reshape((len(locations), len(actions)))

    rendered_values = np.zeros((env.size, env.size))
    for (location, action_values) in zip(locations.astype(int), sa_values):
        rendered_values[location[0], location[1]] = reduction(action_values)
    return rendered_values
