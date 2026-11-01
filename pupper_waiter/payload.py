"""Rigid stand-in for a glass of liquid on the tray: mass and centre-of-mass height from the volume.

The payload link of the ``_tray`` URDF variant represents a light cup (5 g) holding ``volume_ml`` of beer
(1 g/ml), with the centre of mass rising with the fill level. The evaluation of the walking policies uses it
as a fixed weight on the tray.
"""

from __future__ import annotations

import torch

CUP_MASS_KG = 0.005
COM_Z_EMPTY, COM_Z_FULL = 0.02, 0.07  # centre-of-mass height above the tray [m] at 0 and 300 ml
LIQUID_RADIUS = 0.02  # radius of the liquid column, for the payload inertia [m]


def volume_to_payload(volume_ml: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mass [kg], centre-of-mass height [m]) for the given volumes."""
    mass = CUP_MASS_KG + volume_ml / 1000.0
    com_z = COM_Z_EMPTY + (COM_Z_FULL - COM_Z_EMPTY) * volume_ml / 300.0
    return mass, com_z
