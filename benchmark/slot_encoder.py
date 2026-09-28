"""Rewrite a motion in terms of body parts, so that different animals can be compared.

Two skeletons have different numbers of joints arranged in different ways, so their motions
cannot be compared joint by joint. This module instead assigns every joint of a skeleton to
one of 32 named body parts, called slots: the root, the head, the four feet, wings, claws,
segments of the tail, and numbered middle legs for animals with many of them. Whatever a
skeleton has goes into the matching slot, and slots it does not have stay empty.

For every frame and every slot the result records where that part is, whether it is touching
the ground, how fast it is moving, and where it is in its own step cycle, counted from one
footfall to the next. The action-level AUC evaluator uses this to compare the timing of two
motions.

The slot names are listed in benchmark/slot_vocabulary.json. Joints reach their slot in two
steps: the named contact groups of benchmark/contact_groups.json are translated into slot
names, and whatever joints are left over are placed by which side of the body they are on and
how far they sit from the root.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# --------------------------------------------------------------------------------------
# Slot vocabulary
# --------------------------------------------------------------------------------------
_VOCAB_PATH = Path(__file__).resolve().parent / "slot_vocabulary.json"


@lru_cache(maxsize=1)
def load_slot_vocabulary() -> Dict:
    with open(_VOCAB_PATH) as f:
        return json.load(f)


_vocab = load_slot_vocabulary()
SLOT_COUNT: int = _vocab["slot_count"]
ALL_SLOT_TYPES: List[str] = [s["type"] for s in _vocab["slots"]]
_TYPE_TO_IDX: Dict[str, int] = {s["type"]: s["idx"] for s in _vocab["slots"]}


def slot_type_to_idx(slot_type: str) -> int:
    if slot_type not in _TYPE_TO_IDX:
        raise KeyError(f"Unknown slot type {slot_type!r}; known: {list(_TYPE_TO_IDX)}")
    return _TYPE_TO_IDX[slot_type]


# --------------------------------------------------------------------------------------
# Translating the per-skeleton contact groups into slot names
# --------------------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_CONTACT_GROUPS_PATH = _ROOT / "benchmark/contact_groups.json"
_COND_PATH = Path("dataset/truebones/zoo/truebones_processed/cond.npy")

# Group names that are already slot names and need no translation.
_PASSTHROUGH = {
    "root", "head", "LF", "RF", "LH", "RH", "LW", "RW",
    "claw_L", "claw_R",
    "tail_0", "tail_1", "tail_2", "tail_3",
}

# Group names that always mean the same body part, whatever the animal.
_SIMPLE_RENAMES = {
    "L_hand": "claw_L",     # SpiderG manipulator
    "R_hand": "claw_R",
    "L_leg1": "LH",         # SpiderG rear leg
    "R_leg1": "RH",
}

# Animals with numbered legs, such as spiders and crabs. The lowest number is the front
# leg and the highest the rear one; the rest become middle legs.
def _resolve_numeric_legs(raw_keys):
    L_nums = sorted(int(k[1:]) for k in raw_keys
                    if len(k) >= 2 and k[0] == "L" and k[1:].isdigit())
    R_nums = sorted(int(k[1:]) for k in raw_keys
                    if len(k) >= 2 and k[0] == "R" and k[1:].isdigit())
    out = {}
    if L_nums:
        out[f"L{L_nums[0]}"] = "LF"
        if len(L_nums) >= 2:
            out[f"L{L_nums[-1]}"] = "LH"
        for i, n in enumerate(L_nums[1:-1]):
            out[f"L{n}"] = f"mid_leg_{2 * i}"
    if R_nums:
        out[f"R{R_nums[0]}"] = "RF"
        if len(R_nums) >= 2:
            out[f"R{R_nums[-1]}"] = "RH"
        for i, n in enumerate(R_nums[1:-1]):
            out[f"R{n}"] = f"mid_leg_{2 * i + 1}"
    return out

# Named middle legs, as insects and centipedes have.
_MID_PAIRS = {
    "L_mid": "mid_leg_0",   "R_mid": "mid_leg_1",
    "L_mid1": "mid_leg_2",  "R_mid1": "mid_leg_3",
    "L_mid2": "mid_leg_4",  "R_mid2": "mid_leg_5",
    "L_mid3": "mid_leg_6",  "R_mid3": "mid_leg_7",
}

# Animals that stand on two legs and have arms in front.
_ARM_PAIRS = {
    "L_arm": "LF", "R_arm": "RF",
}

# Animals with a single pair of legs: count them as the hind pair.
_BIPED_PAIRS = {
    "L": "LH", "R": "RH",
}

# Snakes, whose body is described in segments from head to tail. The segment named tail
# has to land in the last slot so that it cannot collide with the one nearest the head.
_SNAKE_SEGMENTS = {
    "front": "tail_0",
    "mid_front": "tail_1",
    "mid": "tail_2",
    "mid_back": "tail_2",
    "tail": "tail_3",
}

# Three skeletons have a single undifferentiated contact group in the source file. Their
# slots are written out explicitly, read off the joint names:
#   Pigeon (9j): 0=Hips, 1=RightArm, 2=RightForeArm, 3=RightLeg, 4=LeftLeg,
#                5=Tail01, 6=LeftArm, 7=LeftForeArm, 8=Spine
#   Tukan (18j): 0=Hips, 3=kosi(waist), 4=R_momo(thigh), 6=L_momo, 8=mune(chest),
#                9=R_kata(shoulder), 10=R_hiji(elbow), 11=L_kata, 12=L_hiji,
#                13=kao(face)/14=ago(jaw)
#   Pirrana (21j, fish): 0=N_ALL, 3=atama(head), 4=munabireR, 5=munabireL,
#                15=obire, 16=obireB, 17=obireA, 18=sebire, 19=harabireR, 20=harabireL
_UNRESOLVED_OVERRIDES: Dict[str, Dict[str, List[int]]] = {
    "Pigeon": {
        "root": [0],
        "head": [8],          # Spine -- Pigeon has no explicit head joint
        "LW": [6, 7],         # LeftArm + LeftForeArm = left wing
        "RW": [1, 2],         # RightArm + RightForeArm = right wing
        "LH": [4],            # LeftLeg
        "RH": [3],            # RightLeg
        "tail_0": [5],        # Tail01
    },
    "Tukan": {
        "root": [0],
        "head": [13, 14],     # kao + ago (face + jaw)
        "LW": [11, 12],       # L_kata + L_hiji
        "RW": [9, 10],        # R_kata + R_hiji
        "LH": [6],            # L_momo
        "RH": [4],            # R_momo
        "tail_0": [3, 5],     # kosi (waist) + o (mid-pelvis joint between thighs)
        "tail_1": [8],        # mune (chest)
    },
    "Pirrana": {
        # Pirrana's raw `all` group = [4, 5, 15, 16, 17, 18, 19, 20]
        # = munabireR/L (pectoral fins), obire/B/A (caudal-fin segments),
        # sebire (dorsal fin), harabireR/L (pelvic fins). All 8 covered:
        "root": [0],          # N_ALL
        "head": [3],          # atama (head)
        "LW": [5],            # munabireL (left pectoral fin)
        "RW": [4],            # munabireR (right pectoral fin)
        "LH": [20],           # harabireL (left pelvic fin)
        "RH": [19],           # harabireR (right pelvic fin)
        "tail_0": [18],       # sebire (dorsal fin)
        "tail_1": [16],       # obireB
        "tail_2": [17],       # obireA
        "tail_3": [15],       # obire (main caudal fin)
    },
}


def _build_canonical_for_skel(skel_object_type: str, raw_groups: Dict) -> Dict[str, List[int]]:
    """Turn one skeleton's group names into slot names."""
    out: Dict[str, List[int]] = {}

    # The three skeletons written out above use those slots directly.
    if skel_object_type in _UNRESOLVED_OVERRIDES:
        for k, v in _UNRESOLVED_OVERRIDES[skel_object_type].items():
            out[k] = list(v)
        return out

    src = raw_groups.get(skel_object_type, {})
    raw_keys = [k for k in src if not str(k).startswith("_") and k != "all"]
    numeric_resolution = _resolve_numeric_legs(raw_keys)

    for raw_key, joint_list in src.items():
        if str(raw_key).startswith("_"):
            continue
        if raw_key == "all":
            continue
        canonical = (
            raw_key if raw_key in _PASSTHROUGH else
            _SIMPLE_RENAMES.get(raw_key) or
            numeric_resolution.get(raw_key) or
            _MID_PAIRS.get(raw_key) or
            _ARM_PAIRS.get(raw_key) or
            _BIPED_PAIRS.get(raw_key) or
            _SNAKE_SEGMENTS.get(raw_key)
        )
        if canonical is None:
            # A group name we have no slot for; leave it out.
            continue
        out.setdefault(canonical, []).extend(int(j) for j in joint_list)
    # One entry per joint, in order.
    for k in out:
        out[k] = sorted(set(out[k]))
    # Every produced name must be a real slot.
    for k in out:
        assert k in ALL_SLOT_TYPES, f"{skel_object_type}: produced non-vocab key {k!r}"
    return out


@lru_cache(maxsize=1)
def _build_canonical_contact_groups_cached() -> Dict[str, Dict[str, List[int]]]:
    """Build the table once and keep it. Use load_canonical_contact_groups instead, which
    hands back a copy that callers are free to change.
    """
    with open(_CONTACT_GROUPS_PATH) as f:
        raw = json.load(f)
    cond = np.load(_COND_PATH, allow_pickle=True).item()

    out: Dict[str, Dict[str, List[int]]] = {}
    for skel_name, skel_cond in cond.items():
        ot = skel_cond["object_type"]
        out[ot] = _build_canonical_for_skel(ot, raw)
        if not out[ot]:
            raise RuntimeError(
                f"{ot}: empty canonical groups after translation. "
                f"Raw entry: {raw.get(ot, {})}"
            )
    return out


def load_canonical_contact_groups() -> Dict[str, Dict[str, List[int]]]:
    """The slots of every skeleton, as a copy that the caller may change freely."""
    return copy.deepcopy(_build_canonical_contact_groups_cached())


def canonical_groups_for(object_type: str) -> Dict[str, List[int]]:
    """The slots of one skeleton."""
    g = load_canonical_contact_groups()
    if object_type not in g:
        raise KeyError(f"{object_type!r} not in canonical contact groups")
    return g[object_type]


# --------------------------------------------------------------------------------------
# Assigning joints to slots
# --------------------------------------------------------------------------------------
@dataclass
class AssignmentResult:
    joint_to_slot: Dict[int, int] = field(default_factory=dict)
    slot_to_joints: Dict[int, List[int]] = field(default_factory=dict)


def _assign_mid_legs(
    uncovered_joints: List[int], parents: List[int], joint_names: List[str]
) -> Dict[int, int]:
    """Place the joints that no named body part claimed into the numbered middle-leg slots.

    Each joint is sorted into left, right or neither by its name, and then ordered by how far
    it sits from the root. Left and right are filled in turn. Joints that are neither, and
    any that run out of slots, are left out.
    """
    depth_by_joint: Dict[int, int] = {0: 0}
    for j in range(1, len(parents)):
        p = int(parents[j])
        depth_by_joint[j] = depth_by_joint.get(p, 0) + 1

    left_candidates = []
    right_candidates = []
    mid_candidates = []
    for j in uncovered_joints:
        name = str(joint_names[j]).lower()
        is_left = (
            "_l_" in name
            or name.endswith("_l")
            or name.startswith("l_")
            or "left" in name
        )
        is_right = (
            "_r_" in name
            or name.endswith("_r")
            or name.startswith("r_")
            or "right" in name
        )
        if is_left and not is_right:
            left_candidates.append(j)
        elif is_right and not is_left:
            right_candidates.append(j)
        else:
            mid_candidates.append(j)

    left_candidates.sort(key=lambda j: depth_by_joint.get(j, 99))
    right_candidates.sort(key=lambda j: depth_by_joint.get(j, 99))
    mid_candidates.sort(key=lambda j: depth_by_joint.get(j, 99))

    mapping: Dict[int, int] = {}
    mid_leg_pairs = [(f"mid_leg_{2*i}", f"mid_leg_{2*i+1}") for i in range(8)]
    for i, (l_slot, r_slot) in enumerate(mid_leg_pairs):
        if i < len(left_candidates):
            mapping[left_candidates[i]] = slot_type_to_idx(l_slot)
        if i < len(right_candidates):
            mapping[right_candidates[i]] = slot_type_to_idx(r_slot)

    null_idx = slot_type_to_idx("null")
    for j in mid_candidates:
        mapping[j] = null_idx
    for j in left_candidates[len(mid_leg_pairs):]:
        mapping[j] = null_idx
    for j in right_candidates[len(mid_leg_pairs):]:
        mapping[j] = null_idx
    return mapping


_HEAD_NAME_TOKENS = ("head", "skull", "cranium", "kao")  # 'kao' = face in Japanese (Tukan)


def _find_head_joint(uncovered: List[int], joint_names: List[str]) -> int | None:
    """The first unclaimed joint whose name suggests it is the head, if there is one."""
    for j in uncovered:
        name = str(joint_names[j]).lower()
        if any(tok in name for tok in _HEAD_NAME_TOKENS):
            return j
    return None


def assign_joints_to_slots(skel_cond: Dict[str, Any]) -> AssignmentResult:
    """Give every joint of a skeleton the body part it belongs to."""
    joint_names = list(skel_cond["joints_names"])
    parents = list(skel_cond["parents"])
    if "object_type" not in skel_cond:
        raise ValueError(
            "skel_cond must carry an 'object_type' key (Truebones cond convention)"
        )
    object_type = skel_cond["object_type"]

    groups = canonical_groups_for(object_type)

    result = AssignmentResult()
    covered_joints = set()
    for slot_type, joint_list in groups.items():
        try:
            slot_idx = slot_type_to_idx(slot_type)
        except KeyError:
            continue
        for j in joint_list:
            j = int(j)
            if j < 0 or j >= len(joint_names):
                continue
            if j in covered_joints:
                continue
            result.joint_to_slot[j] = slot_idx
            result.slot_to_joints.setdefault(slot_idx, []).append(j)
            covered_joints.add(j)

    if 0 not in result.joint_to_slot:
        root_idx = slot_type_to_idx("root")
        result.joint_to_slot[0] = root_idx
        result.slot_to_joints.setdefault(root_idx, []).append(0)
        covered_joints.add(0)

    # Most skeletons list only ground-contact groups, so the head has to be found by name.
    head_slot_idx = slot_type_to_idx("head")
    if head_slot_idx not in result.slot_to_joints:
        uncovered_so_far = [j for j in range(len(joint_names)) if j not in covered_joints]
        head_j = _find_head_joint(uncovered_so_far, joint_names)
        if head_j is not None:
            result.joint_to_slot[head_j] = head_slot_idx
            result.slot_to_joints.setdefault(head_slot_idx, []).append(head_j)
            covered_joints.add(head_j)

    uncovered = [j for j in range(len(joint_names)) if j not in covered_joints]
    mid_assignments = _assign_mid_legs(uncovered, parents, joint_names)
    for j, slot in mid_assignments.items():
        result.joint_to_slot[j] = slot
        result.slot_to_joints.setdefault(slot, []).append(j)

    assert len(result.joint_to_slot) == len(joint_names)
    return result


# --------------------------------------------------------------------------------------
# Encoding a motion into slots
# --------------------------------------------------------------------------------------
POS_SLICE_IN = slice(0, 3)  # columns in the input [T,J,13] tensor


def encode_positions(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Where each body part is, taken as the average position of the joints inside it.

    Body parts this skeleton does not have come out as zero.
    """
    assert motion.ndim == 3 and motion.shape[2] >= 3
    T, J, _ = motion.shape
    pos_joint = motion[:, :, POS_SLICE_IN].astype(np.float32)

    asg = assign_joints_to_slots(skel_cond)
    out = np.zeros((T, SLOT_COUNT, 3), dtype=np.float32)
    null_idx = slot_type_to_idx("null")

    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        idxs = [j for j in joint_list if j < J]
        if not idxs:
            continue
        out[:, slot_idx, :] = pos_joint[:, idxs, :].mean(axis=1)
    return out


CONTACT_CH_IN = 12  # channel index in the input [T,J,13] tensor


def encode_contacts(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Whether each body part is touching the ground. It counts if any of its joints does."""
    assert motion.ndim == 3 and motion.shape[2] >= 13
    T, J, _ = motion.shape
    contact_joint = motion[:, :, CONTACT_CH_IN].astype(np.float32)

    asg = assign_joints_to_slots(skel_cond)
    out = np.zeros((T, SLOT_COUNT, 1), dtype=np.float32)
    null_idx = slot_type_to_idx("null")

    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        idxs = [j for j in joint_list if j < J]
        if not idxs:
            continue
        out[:, slot_idx, 0] = contact_joint[:, idxs].max(axis=1)
    return out


VEL_SLICE_IN = slice(9, 12)


def encode_velocity_phase(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """How fast each body part moves, and where it is in its own step cycle.

    A step cycle runs from one moment the part lands to the next. Within a cycle the position
    in the cycle rises evenly from the start to the end, so two animals stepping at different
    speeds can still be compared at the same point of their stride. A part that lands fewer
    than twice in the clip has no cycle to speak of, so it is given an even count across the
    whole clip instead.
    """
    assert motion.ndim == 3 and motion.shape[2] >= 13
    T, J, _ = motion.shape
    vel_joint = motion[:, :, VEL_SLICE_IN].astype(np.float32)
    contact_joint = motion[:, :, CONTACT_CH_IN].astype(np.float32)

    asg = assign_joints_to_slots(skel_cond)
    out = np.zeros((T, SLOT_COUNT, 4), dtype=np.float32)
    null_idx = slot_type_to_idx("null")

    linear_phase = np.linspace(0.0, 2 * np.pi, T, endpoint=False, dtype=np.float32)

    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        idxs = [j for j in joint_list if j < J]
        if not idxs:
            continue
        out[:, slot_idx, 0:3] = vel_joint[:, idxs, :].mean(axis=1)

        # The frames where this part lands on the ground
        slot_contact = contact_joint[:, idxs].max(axis=1)
        is_contact = (slot_contact > 0.5).astype(np.int8)
        rising = np.where(np.diff(is_contact, prepend=0) > 0)[0]  # frame indices

        if rising.size < 2:
            out[:, slot_idx, 3] = linear_phase
            continue

        # Rise evenly from one landing to the next
        phase = np.zeros(T, dtype=np.float32)
        # Before the first landing, count backwards at the rate of the first cycle
        first_cycle = rising[1] - rising[0]
        for t in range(rising[0]):
            frac = (t - rising[0]) / max(first_cycle, 1)  # negative
            phase[t] = np.mod(2 * np.pi * frac, 2 * np.pi)
        # Between landings
        for k in range(len(rising) - 1):
            s, e = rising[k], rising[k + 1]
            cycle_len = max(e - s, 1)
            for t in range(s, e):
                phase[t] = 2 * np.pi * (t - s) / cycle_len
        # After the last landing, carry on at the rate of the last cycle
        last_cycle = rising[-1] - rising[-2]
        for t in range(rising[-1], T):
            phase[t] = np.mod(2 * np.pi * (t - rising[-1]) / max(last_cycle, 1), 2 * np.pi)
        out[:, slot_idx, 3] = phase
    return out


CHANNEL_COUNT = 8  # pos(3) + contact(1) + vel(3) + phase(1)


def encode_motion_to_invariant(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Describe a motion by body part: position, ground contact, speed and step cycle."""
    pos = encode_positions(motion, skel_cond)
    con = encode_contacts(motion, skel_cond)
    vp = encode_velocity_phase(motion, skel_cond)
    out = np.concatenate([pos, con, vp[:, :, 0:3], vp[:, :, 3:4]], axis=-1)
    assert out.shape[-1] == CHANNEL_COUNT
    return out
