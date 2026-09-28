"""The five properties of a motion that MoReFlow can be asked to keep, plus a null entry.

A request is passed to the model as a number. Zero means no request at all, which is used
during training to teach the model what it would do unprompted; at generation time that
answer is weighed against the requested one, which is how a request can be given more or
less influence. The other five name the measurements defined in
methods/common/motion_descriptors.py.

Pairing source clips with target clips never uses the null entry: a measurement that is the
same for every clip says nothing about which clip goes with which, so the pairing it
produces would be arbitrary.
"""
from __future__ import annotations

COND_TYPE_TO_INT = {'null': 0, 'root_vel': 1, 'EE_local': 2, 'EE_world': 3,
                    'root_XY': 4, 'root_Z': 5}

COUPLING_CONDITIONS = ['root_vel', 'EE_local', 'EE_world', 'root_XY', 'root_Z']

# Generation ensembles these conditions, averaging their velocity fields at every step.
DEFAULT_INFERENCE_CONDS = ['null', 'root_vel', 'root_XY', 'root_Z']
DEFAULT_CFG_GAMMA = 2.0
DEFAULT_HEUN_STEPS = 25
