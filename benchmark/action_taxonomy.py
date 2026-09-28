"""Action taxonomy for the Truebones benchmark.

Truebones clip filenames carry an action name. This module normalizes that name and maps
it to one of ten coarse action clusters; clips whose action maps to nothing are excluded
from the benchmark.
"""
from __future__ import annotations
import re
from typing import Optional


# The ten action clusters. Anything that maps to none of them is left out.
ACTION_CLUSTERS = {
    'locomotion': {
        'walk', 'walkforward', 'walkright', 'walkleft', 'slowwalk', 'walkloop', 'walkcall',
        'run', 'running', 'runloop', 'runs', 'dash',
        'trot', 'trotleft', 'trotright',
        'backup', 'backwards', 'backwalk',
        'turn', 'turnleft', 'turnright',
        'march', 'sneaky',  # stealth locomotion
        'charge',  # locomotion leading to combat
    },
    'combat': {
        'attack', 'attackleft', 'attackright',
        'bite', 'bitehard',
        'fight', 'hit',
        'roar', 'growl',
        'jumpbite',
        'hoofscrape',
        'throw', 'thrown',
        'tailwhip',
        'rearing',
        'strike',
        'headbutt',
        'hit', 'hithead',
        'sting',
        'ready',  # combat ready pose
        'claw', 'clawattack',
    },
    'idle': {
        'idle', 'idleloop',
        'slowidle', 'restless',
        'rest', 'sit', 'stand',
        'sniff', 'eat', 'yawn', 'shake',
        'laydown', 'lie', 'liedown',
        'eatfish', 'grazing', 'drink', 'drinking',
        'cud',  # cud chewing
        'sleepup', 'sleep', 'sleeping',
        'stretchyawnidle', 'lickcleanidle', 'idlepurr', 'idleenergetic', 'idlelaydown',
    },
    'death': {
        'die', 'dieloop', 'death', 'dies', 'deathloop', 'longdeath',
        'fall', 'falling',
        'knockedback', 'knockedbacka',
        'shot', 'dying',
    },
    'fly': {
        'fly', 'flyloop', 'slowfly', 'glide', 'flying', 'flapping',
    },
    'takeoff': {
        'takeoff', 'land', 'landing', 'circleland', 'lander', 'downloop',
    },
    'jump': {
        'jump', 'jumpforward', 'jumping', 'jumps',
        'rise', 'rising',
        'getup', 'getupagain', 'getupfromside',
        'rear',
    },
    'agitated': {
        'agitated', 'scared', 'restless2', 'spin',
    },
    'swim': {
        'swim', 'swims', 'swimming',
    },
    'reaction': {
        'bleat', 'cry', 'vocalize', 'crys', 'yell', 'neigh', 'bark', 'meow',
        'take', 'taken',  # reaction to being taken
    },
}

# Action name to its cluster
_ACTION_TO_CLUSTER = {}
for cluster, actions in ACTION_CLUSTERS.items():
    for a in actions:
        _ACTION_TO_CLUSTER[a] = cluster


# Species prefixes that should be stripped from action names
# E.g., "cat_idlepurr" → strip "cat" → "idlepurr"
SPECIES_PREFIXES = {
    'cat', 'dog', 'horse', 'puppy', 'raptor', 'eagle', 'pigeon', 'parrot',
    'deer', 'bear', 'lion', 'tiger', 'hound', 'wolf', 'fox', 'monkey',
    'elephant', 'rhino', 'goat', 'buffalo', 'cow', 'camel', 'gazelle',
    'spider', 'crab', 'snake', 'anaconda', 'alligator', 'turtle', 'frog',
    'fish', 'pirrana', 'shark', 'trex', 'tyranno', 'stego', 'raptor3',
    'chicken', 'ostrich', 'flamingo', 'tukan', 'buzzard', 'pteranodon',
    'bird', 'rat', 'bee', 'giantbee', 'ant', 'fireant', 'roach', 'scorpion',
    'centipede', 'isopetra', 'mantis', 'hermitcrab', 'cricket', 'comodoa',
    'brownbear', 'polarbear', 'polarbearb', 'sabretoothtiger', 'jaguar',
    'coyote', 'raindeer', 'crocodile', 'tricera', 'tukan',
    'kingcobra', 'dragon', 'pegasus', 'bearpolar',
}


def _strip_prefix(s: str, prefix_set=SPECIES_PREFIXES) -> str:
    """Strip species prefix if present. 'cat_idlepurr' → 'idlepurr'."""
    for pfx in sorted(prefix_set, key=len, reverse=True):
        if s.startswith(pfx) and len(s) > len(pfx):
            rest = s[len(pfx):]
            # Require separator or immediate action continuation
            if rest.startswith('_'):
                return rest[1:]
            # Also accept no separator if remaining is valid action name
            if rest[0].isalpha():
                return rest
    return s


def normalize_action(raw: str) -> str:
    """Normalize: 'Attack2' → 'attack', 'Cat_IdlePurr' → 'idlepurr',
    '-_Attack' → 'attack', 'hit_head' → 'hithead'."""
    raw = raw.lower().strip()
    # Drop any leading dash or underscore
    raw = re.sub(r'^[-_]+', '', raw)
    # Strip trailing digits and underscores
    stripped = re.sub(r'[_\d]+$', '', raw)
    stripped = stripped if stripped else raw
    # Strip species prefix if present
    stripped = _strip_prefix(stripped)
    # Collapse remaining underscores (hit_head → hithead, long_death → longdeath)
    stripped = stripped.replace('_', '')
    return stripped


def parse_action_from_filename(fname: str) -> str:
    """Extract action label from filename like 'Skel___Attack2_1039.npy'."""
    stem = fname.rsplit('.', 1)[0] if '.' in fname else fname
    if '___' in stem:
        parts = stem.split('___')
        action_part = parts[1]
    elif '_' in stem:
        parts = stem.split('_', 1)
        action_part = parts[1]
    else:
        action_part = stem
    # Drop trailing _NN
    action_part = re.sub(r'_\d+$', '', action_part)
    return normalize_action(action_part)


def action_to_cluster(action: str) -> Optional[str]:
    """Map normalized action → cluster, or None if unrecognized."""
    return _ACTION_TO_CLUSTER.get(normalize_action(action))


def is_other_label(action: str) -> bool:
    """True if the action is 'other' or unmappable to any cluster."""
    norm = normalize_action(action)
    return norm == 'other' or action_to_cluster(norm) is None
