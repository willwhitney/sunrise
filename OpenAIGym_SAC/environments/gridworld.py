import types
import numpy as np

import jax
from jax import numpy as jnp

from flax import struct

import utils


DTYPE = jnp.int16
SIZE = 10
one_hot_10 = jax.partial(utils.one_hot, k=SIZE)

ACTION_MAP = jnp.stack([
    jnp.array((1, 0), dtype=DTYPE),   # visually DOWN
    jnp.array((0, 1), dtype=DTYPE),   # visually RIGHT
    jnp.array((-1, 0), dtype=DTYPE),  # visually UP
    jnp.array((0, -1), dtype=DTYPE),  # visually LEFT
])


@struct.dataclass
class GridWorld:
    size: int
    render: types.FunctionType = struct.field(pytree_node=False)
    agent: jnp.ndarray = jnp.array((0, 0), dtype=DTYPE)
    actions: jnp.ndarray = jnp.arange(4)


def goal(s):
    return jnp.array((s - 1, s - 1), dtype=DTYPE)


def new(size, render_onehot=True):
    if render_onehot:
        _render = jax.partial(utils.one_hot, k=size)
    else:
        _render = lambda x: x
    return GridWorld(size, _render)


def new_batch(n, size):
    return utils.tree_stack(new(size) for _ in range(n))


def reset(env):
    return env.replace(agent=jnp.array((0, 0), dtype=DTYPE))
reset_batch = jax.vmap(reset)  # noqa: E305


# @jax.profiler.trace_function
def render(env):
    return env.render(env.agent)
render_batch = jax.vmap(render)  # noqa: E305


# @jax.profiler.trace_function
def step(env, action):
    new_agent = env.agent + ACTION_MAP[action]
    new_agent = jnp.clip(new_agent, 0, env.size - 1)
    env = env.replace(agent=new_agent)
    reward = (env.agent == goal(env.size)).all()
    return env, render(env), reward
step = jax.jit(step, static_argnums=(1,))  # noqa: E305
step_batch = jax.vmap(step)


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
    render_locations = jax.vmap(env.render)
    states = render_locations(locations)
    repeated_states = states.repeat(len(env.actions), axis=0)

    actions = env.actions
    tiled_actions = jnp.tile(actions, (len(states),)).reshape((-1, 1))

    values = fn(repeated_states, tiled_actions)
    sa_values = values.reshape((len(states), len(actions)))

    rendered_values = np.zeros((env.size, env.size))
    for (location, action_values) in zip(locations, sa_values):
        rendered_values[location[0], location[1]] = reduction(action_values)
    return rendered_values


if __name__ == "__main__":
    import random
    env = new(10)
    for i in range(10):
        env, obs, r = step(env, random.randint(0, 3))
        print(obs)

    envs = new_batch(3, 10)
    print(envs)
    envs, obss, rs = step_batch(envs, jnp.array(range(3)))
