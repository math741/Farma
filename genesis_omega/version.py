"""Version identifiers for every layer that can change scientific meaning.

Rule (spec §422-430): any change to the laws of the artificial universe bumps
``PHYSICS_VERSION``. A checkpoint written under a different physics version is
never silently continued on the prime timeline; it requires an explicit,
recorded migration.
"""

ENGINE_VERSION = "0.1.0"
#: Laws of the artificial universe (world physics, chemistry, body physics).
PHYSICS_VERSION = "phys-1"
#: Procedural generation of galaxies / systems / planets / terrain.
WORLDGEN_VERSION = "wgen-1"
#: On-disk checkpoint layout.
CHECKPOINT_SCHEMA = 1
#: Scientist / validation statistics and thresholds.
VALIDATOR_VERSION = "val-1"
#: Server <-> viewer protocol.
PROTOCOL_VERSION = 1


def version_info() -> dict:
    return {
        "engine": ENGINE_VERSION,
        "physics": PHYSICS_VERSION,
        "worldgen": WORLDGEN_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "validator": VALIDATOR_VERSION,
        "protocol": PROTOCOL_VERSION,
    }
