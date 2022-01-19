import inspect

from environments import point
from environments import hallway
from environments import hallway_distractor
from environments import hallway_midstart
from environments import reacher_explore
from environments import ball_in_cup_explore
from environments import manipulator_explore
from environments import walker_explore
from environments import finger_explore

NEW_DOMAINS = {name: module for name, module in locals().items()
               if inspect.ismodule(module) and hasattr(module, 'SUITE')}


from dm_control import suite
suite._DOMAINS.update(NEW_DOMAINS)