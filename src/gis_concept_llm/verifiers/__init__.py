from .base import Verification
from .connectivity_verifier import ConnectivityVerifier
from .direction_verifier import DirectionVerifier
from .topology_verifier import TopologyVerifier


def verifier_for(task_type: str):
    mapping = {"direction": DirectionVerifier(), "topology": TopologyVerifier(), "connectivity": ConnectivityVerifier()}
    try:
        return mapping[task_type]
    except KeyError as exc:
        raise ValueError(f"no GIS verifier for task type {task_type!r}") from exc


__all__ = ["Verification", "DirectionVerifier", "TopologyVerifier", "ConnectivityVerifier", "verifier_for"]
