# Tools for making dm_env specs work with JAX operations

# import numpy as np
from jax import numpy as jnp, tree_util
from flax import struct

import dm_env


@struct.dataclass
class Array():
    shape: tuple

    def __eq__(self, other):
        # return self.shape == other.shape
        return self.__key() == other.__key()

    def __key(self):
        return self.shape

    def __hash__(self):
        return hash(self.__key())


@struct.dataclass
class BoundedArray(Array):
    shape: tuple
    minimum: jnp.ndarray
    maximum: jnp.ndarray

    # def __eq__(self, other):
    #     return all((super().__eq__(other),
    #                 jnp.allclose(self.minimum, other.minimum),
    #                 jnp.allclose(self.maximum, other.maximum)))

    def __key(self):
        return (self.shape,
                tuple(self.minimum.flatten()),
                tuple(self.maximum.flatten()))

@struct.dataclass
class DiscreteArray(BoundedArray):
    shape: tuple
    minimum: jnp.ndarray
    maximum: jnp.ndarray
    num_values: int

    # def __eq__(self, other):
    #     return all((super().__eq__(other),
    #                 self.num_values == other.num_values))

    def __key(self):
        return (self.shape,
                tuple(self.minimum.flatten()),
                tuple(self.maximum.flatten()),
                self.num_values)


def convert_dm_spec_single(spec):
    if isinstance(spec, Array):
        return spec
    elif isinstance(spec, dm_env.specs.DiscreteArray):
        return DiscreteArray(spec.shape,
                             spec.minimum,
                             spec.maximum,
                             spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return BoundedArray(spec.shape,
                            spec.minimum,
                            spec.maximum)
    else:
        # this case covers the fact that tree_map will apply this function
        # to the *properties* of a jax_specs.Array
        return spec


def convert_dm_spec(spec):
    return tree_util.tree_map(convert_dm_spec_single, spec)
