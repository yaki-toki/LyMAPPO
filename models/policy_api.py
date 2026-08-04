"""Policy plug-in API for the ns-3 evaluation driver (``drivers/feddrl.py``).

Third parties implement a link-selection policy without touching the driver:

    # mypkg/my_policy.py
    from models.policy_api import Policy, PolicyAction

    class MyPolicy(Policy):
        def act(self, obs, link_masks):
            links = ...  # (n_ap, n_sta_per_ap) int, values in [0, n_links)
            return PolicyAction(selected_links=links, map_mode=0)

and run it with ``--policy mypkg.my_policy.MyPolicy`` (the driver puts
``--repo-root`` on ``sys.path``, so any package importable from the repo root
works). See ``docs/TUTORIAL.md`` for the full walkthrough.

The obs each slot is the same per-AP dict list the LyMAPPO actor sees
(``drivers/feddrl.py:_ns3_obs_to_python``): ``csi (n_sta, n_links)``,
``queue (n_sta, n_links, 4)``, ``hol_age (n_sta, n_links)`` in seconds,
``cbr (n_links,)`` in [0, 1], ``Z_99``/``Z_99_9 (n_sta,)`` reconstructed
Lyapunov duals, ``link_mask (n_links,)`` multi-hot allowed-link set.

This module depends on numpy only — a custom policy does not need torch.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

import numpy as np

# ActMsg mapMode values understood by the ns-3 scenario:
# 0 = no coordination, 1 = Co-SR, 2 = Co-TDMA.
N_MAP_MODES = 3


@dataclass(frozen=True)
class PolicyAction:
    """One macro-slot decision: per-STA link choice + one MAPC mode scalar.

    ``selected_links``: array-like of shape ``(n_ap, n_sta_per_ap)`` with link
    indices in ``[0, n_links)``. ``map_mode``: single int in ``[0, 3)`` (the
    ns-3 ActMsg carries one mapMode for the whole network).
    """

    selected_links: Any
    map_mode: int = 0


class Policy:
    """Base class for pluggable evaluation policies.

    Subclass and implement :meth:`act`. The driver constructs the policy with
    the network dimensions, the run seed, and the ``--ckpt`` path (``None``
    unless given), so a learned custom policy can load its own weights.
    """

    def __init__(
        self,
        n_ap: int,
        n_sta_per_ap: int,
        n_links: int,
        seed: int = 0,
        ckpt: str | None = None,
        **_unused: Any,
    ) -> None:
        self.n_ap = n_ap
        self.n_sta_per_ap = n_sta_per_ap
        self.n_links = n_links
        self.ckpt = ckpt
        self.rng = np.random.default_rng(seed)

    def act(self, obs: List[dict], link_masks: np.ndarray) -> PolicyAction:
        """Return the decision for this macro slot.

        ``obs``: per-AP dict list (schema in the module docstring).
        ``link_masks``: ``(n_ap, n_links)`` multi-hot allowed-link sets
        (same content as the per-AP ``obs[ap]["link_mask"]``, provided
        stacked for convenience). Selecting a link outside the mask is not
        rejected by the driver (the round-robin diagnostic does it on
        purpose) but breaks the asymmetric-topology assumption — stay inside
        the mask unless that is the experiment.
        """
        raise NotImplementedError


_REGISTRY: Dict[str, Type[Policy]] = {}


def register_policy(name: str) -> Callable[[Type[Policy]], Type[Policy]]:
    """Class decorator: make a Policy loadable as ``--policy <name>``.

    The registry only helps once the defining module has been imported, so
    dotted paths are the primary loading route for external code.
    """

    def _decorator(cls: Type[Policy]) -> Type[Policy]:
        if not (isinstance(cls, type) and issubclass(cls, Policy)):
            raise ValueError(f"register_policy({name!r}): {cls!r} is not a Policy subclass")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def load_policy(spec: str, **kwargs: Any) -> Policy:
    """Instantiate a policy from a registry name or a dotted import path.

    ``spec`` is first looked up in the registry; otherwise it is treated as
    ``package.module.ClassName``. ``kwargs`` are forwarded to the constructor
    (the driver passes ``n_ap``, ``n_sta_per_ap``, ``n_links``, ``seed``,
    ``ckpt``).
    """
    cls: Type[Policy]
    if spec in _REGISTRY:
        cls = _REGISTRY[spec]
    elif "." in spec:
        module_name, _, class_name = spec.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except (ImportError, TypeError) as exc:
            # TypeError covers relative-import specs (leading dot), which
            # importlib rejects without a package argument.
            raise ValueError(
                f"--policy {spec!r}: cannot import module {module_name!r} "
                f"({exc}); is the repo root (or your package) on sys.path?"
            ) from exc
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"--policy {spec!r}: module {module_name!r} has no attribute "
                f"{class_name!r}"
            ) from exc
    else:
        raise ValueError(
            f"--policy {spec!r}: not a registered policy name "
            f"(known: {sorted(_REGISTRY)}) and not a dotted import path"
        )
    if not (isinstance(cls, type) and issubclass(cls, Policy)):
        raise ValueError(f"--policy {spec!r}: {cls!r} is not a Policy subclass")
    return cls(**kwargs)


def validate_action(
    action: PolicyAction, n_ap: int, n_sta_per_ap: int, n_links: int
) -> PolicyAction:
    """Structurally validate a policy's output; return a normalized copy.

    Raises ``ValueError`` with a specific message on any contract breach so a
    buggy custom policy fails fast on slot 0 instead of silently corrupting
    an episode.
    """
    if not isinstance(action, PolicyAction):
        raise ValueError(
            f"Policy.act must return a PolicyAction, got {type(action).__name__}"
        )
    links = np.asarray(action.selected_links)
    if links.shape != (n_ap, n_sta_per_ap):
        raise ValueError(
            f"selected_links shape {links.shape} != ({n_ap}, {n_sta_per_ap})"
        )
    if links.dtype.kind not in "iu":
        if not np.all(links == np.floor(links)):
            raise ValueError("selected_links must contain integer link indices")
    links = links.astype(np.int64)
    if links.min() < 0 or links.max() >= n_links:
        raise ValueError(
            f"selected_links contains a link index outside [0, {n_links}): "
            f"min={links.min()}, max={links.max()}"
        )
    try:
        mode = int(action.map_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"map_mode must be an int, got {action.map_mode!r}") from exc
    if not 0 <= mode < N_MAP_MODES:
        raise ValueError(f"map_mode {mode} outside [0, {N_MAP_MODES})")
    return PolicyAction(selected_links=links, map_mode=mode)
