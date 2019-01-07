from collections import namedtuple
import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

"""Inspired by Simon Meister's https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/pre_processing.py#L14"""

FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])

NUM_FUNCTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale

FLAT_FEATURES = [
    FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),
    FlatFeature(1,  features.FeatureType.SCALAR, 1, 'minerals'),
    FlatFeature(2,  features.FeatureType.SCALAR, 1, 'vespene'),
    FlatFeature(3,  features.FeatureType.SCALAR, 1, 'food_used'),
    FlatFeature(4,  features.FeatureType.SCALAR, 1, 'food_cap'),
    FlatFeature(5,  features.FeatureType.SCALAR, 1, 'food_army'),
    FlatFeature(6,  features.FeatureType.SCALAR, 1, 'food_workers'),
    FlatFeature(7,  features.FeatureType.SCALAR, 1, 'idle_worker_count'),
    FlatFeature(8,  features.FeatureType.SCALAR, 1, 'army_count'),
    FlatFeature(9,  features.FeatureType.SCALAR, 1, 'warp_gate_count'),
    FlatFeature(10, features.FeatureType.SCALAR, 1, 'larva_count'),
]

is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
    # HACK: we should infer the point type automatically
    is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']

