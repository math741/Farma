"""The world interface (spec §23-25, §58, §244-245).

This is the *only* coupling between an organism and the universe: a vector of
raw physical signals in, a vector of motor/effector drives out. No labels, no
classes, no text, no semantic categories ("food", "enemy", "human") ever
cross this boundary. Organisms have no other channel to anything — no
filesystem, no network, no code.
"""

from __future__ import annotations

DIRECTIONS = ("front", "right", "back", "left")
DIRECTION_ANGLES = (0.0, -1.5707963267948966, 3.141592653589793, 1.5707963267948966)

#: per-direction raw signals (sampled at a point ahead/right/behind/left)
DIR_SIGNALS = ("light", "resource", "temperature", "scent0", "scent1", "vib0", "vib1", "vib2", "vib3",
               "structure", "mark0", "mark1", "water")
#: which body sense modality gates each directional signal
DIR_MODALITY = {"light": 0, "resource": 1, "temperature": 3, "scent0": 1, "scent1": 1, "vib0": 2, "vib1": 2,
                "vib2": 2, "vib3": 2, "structure": 2, "mark0": 1, "mark1": 1, "water": 2}
MODALITIES = ("photo", "chemo", "mechano", "thermo")

LOCAL_SIGNALS = ("resource_here", "store_here", "carried_amount", "carried_hardness", "carried_insulation",
                 "carried_reactivity", "crowd", "contact_damage", "received_transfer", "energy", "integrity",
                 "thermal_deviation", "speed", "bump", "bias")

INPUT_NAMES = tuple(f"{d}.{s}" for d in DIRECTIONS for s in DIR_SIGNALS) + LOCAL_SIGNALS
N_INPUTS = len(INPUT_NAMES)
N_DIR_SIGNALS = len(DIR_SIGNALS)

OUTPUT_NAMES = ("turn", "thrust", "emit0", "emit1", "emit2", "emit3", "eat", "dig", "build", "process",
                "strike", "give", "store", "mark0", "mark1", "reproduce")
N_OUTPUTS = len(OUTPUT_NAMES)
O = {n: i for i, n in enumerate(OUTPUT_NAMES)}
I = {n: i for i, n in enumerate(INPUT_NAMES)}

#: bit positions used to pack discrete actions for the viewer / ring buffer
ACTION_BITS = {"eat": 0, "dig": 1, "build": 2, "process": 3, "strike": 4, "give": 5, "store": 6, "mark": 7,
               "reproduce": 8, "emit": 9, "water": 10, "carrying": 11, "sheltered": 12, "moving": 13}

BODY_CAPS = ("mass", "land_speed", "water_speed", "turn", "grasp", "strike", "armor", "insulation", "storage",
             "emit_power", "bite", "cells", "units", "aquatic")
CAP = {n: i for i, n in enumerate(BODY_CAPS)}
N_CAPS = len(BODY_CAPS)

LIFE_STATS = ("ticks", "water_ticks", "distance", "emitted", "builds", "processes", "strikes", "gives", "kills",
              "offspring", "eaten", "stored", "dug", "marks", "sheltered_ticks")
ST = {n: i for i, n in enumerate(LIFE_STATS)}
N_STATS = len(LIFE_STATS)
