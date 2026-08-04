# Tutorial: plug in your own policy

This walkthrough takes you from a working install to evaluating **your own
link-selection policy** inside ns-3, without modifying any driver code.
Prerequisites: `install.sh` and `scripts/smoke.sh` have passed
(see [INSTALL.md](INSTALL.md)).

## 1. What a policy is here

Every 20 ms macro slot, the evaluation driver (`drivers/feddrl.py`) receives
a network observation from ns-3 over shared memory and must answer with:

- **`selected_links`** — for each of the 16 APs × 5 STAs, which of the 3
  MLO links (0 = 2.4 GHz/20 MHz, 1 = 5 GHz/40 MHz, 2 = 6 GHz/40 MHz) the
  STA's traffic should use next slot;
- **`map_mode`** — one network-wide MAPC coordination mode
  (0 = none, 1 = Co-SR, 2 = Co-TDMA).

The plug-in API (`models/policy_api.py`) lets you supply that mapping as a
Python class. Your policy sees **exactly the same per-slot observation
bridge as the LyMAPPO actor**, including the reconstructed Lyapunov duals.

## 2. The interface

```python
from models.policy_api import Policy, PolicyAction

class MyPolicy(Policy):
    def act(self, obs, link_masks):
        # obs: list of 16 per-AP dicts (schema below)
        # link_masks: (16, 3) multi-hot allowed-link sets
        links = ...  # (16, 5) int array, values in [0, 3)
        return PolicyAction(selected_links=links, map_mode=0)
```

The constructor receives `n_ap`, `n_sta_per_ap`, `n_links`, `seed`, and
`ckpt` (the `--ckpt` value, `None` unless given) — override `__init__` and
call `super().__init__(**kwargs)` if you need extra parameters. A seeded
`self.rng` (`numpy` Generator) is provided for reproducible stochastic
policies. State kept on `self` persists across slots within an episode
(one process per episode).

### Observation schema (per-AP dict)

| Key | Shape | Meaning | Source in ns-3 |
|---|---|---|---|
| `csi` | `(5, 3)` | per-STA-per-link channel quality | measured by the ns-3 channel |
| `queue` | `(5, 3, 4)` | backlog by access class | single shared buffer: only `[:, 0, 0]` is filled |
| `hol_age` | `(5, 3)` | head-of-line age (seconds) | only `[:, 0]` is filled |
| `cbr` | `(3,)` | per-link channel busy ratio in [0, 1] | measured per band |
| `Z_99`, `Z_99_9` | `(5,)` | Lyapunov dual, broadcast per AP | reconstructed from violation counts with the training rule |
| `link_mask` | `(3,)` | multi-hot allowed-link set K_i | asymmetric tiling (`models/topology.py`) |

Two honesty notes (the same sim2sim caveats documented for the actor path):
the ns-3 EnvMsg carries one aggregate queue/HoL per STA, so the `queue` and
`hol_age` entries are attributed to index 0, and `csi` is the value the ns-3
channel owns — do not re-derive it.

### Contract

- `selected_links` must have shape `(n_ap, n_sta_per_ap)` with integer link
  indices in `[0, n_links)`; `map_mode` must be 0, 1, or 2. The driver
  validates on **every slot** and fails fast with a specific `ValueError` —
  a buggy policy dies on slot 0 instead of silently corrupting an episode.
- Links outside `link_mask` are *not* rejected (the round-robin diagnostic
  violates masks on purpose), but they break the asymmetric-topology
  assumption — stay inside the mask unless that is your experiment.

## 3. Write and run a policy

Create a module importable from the repo root — the driver puts
`--repo-root` on `sys.path`. The shipped example is
[`examples/policies/greedy_score.py`](../examples/policies/greedy_score.py)
(per-STA score `csi − w·cbr`, ~15 lines of logic). Run it with the standard
development protocol (11 s episode ≈ 3 min wall clock):

```bash
cd "$NS3_ROOT/contrib/ai/examples/feddrl"
python3 -B feddrl.py --seed 0 --arrival-pps 60 --episode-ms 11000 \
  --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 \
  --link2-width 40 \
  --policy examples.policies.greedy_score.GreedyScorePolicy \
  --repo-root "$REPO_ROOT" --baseline-tag my_policy
```

Replace the `--policy` value with your own `pkg.module.ClassName`. Anything
that is not a built-in name (`auto`, `rssi`, `stub`, `slci`) goes through
the plug-in loader. Alternatively, decorate your class with
`@register_policy("my-name")` and pass `--policy my-name` (the registry
only helps once your module is imported, so dotted paths are the primary
route).

What to look for in the output:

- `[feddrl.py] loaded plug-in policy: ...` — your class was found;
- `[KPI_AP]` / `[KPI]` lines at episode end — per-AP and network deadline
  KPIs (silence until then is normal);
- `[ACTDIAG] modes=[...] links=[...]` — how often each mapMode was sent and
  how STA-slots distributed over links: a quick sanity check that your
  policy actually steers (the stub shows all selections on link 0).

## 4. Compare against the baselines

Run the built-ins under the **same protocol flags and seeds** and compare
seed-level results:

```bash
--policy rssi                    # best-CSI per STA
--policy slci                    # least-CBR per BSS (published heuristic)
--policy stub --balance-links    # round-robin
--ckpt <trained.pt>              # LyMAPPO actor (see README Step 2-3)
```

Conventions (details in [REPRODUCE.md](REPRODUCE.md)): the main protocol is
`--episode-ms 110000` on held-out channel seeds {10, 12, 18}; report
mean ± std over seeds, no pooled tests; channel seed 11 is excluded
(pre-existing ns-3.40 PHY assertion).

## 5. Learned custom policies

Pass your weights file via `--ckpt` — with a plug-in policy active the path
is handed to *your* constructor (`self.ckpt`) instead of the built-in
LyMAPPO actor loader. `models/policy_api.py` depends only on numpy, so
whether you load torch, JAX, or an ONNX runtime is up to your module. To
train in the loop, see `drivers/train_ns3.py` (README Step 2) — the plug-in
API covers evaluation; a training-side hook is on the
[roadmap](ROADMAP.md).

## 6. Unit-test without ns-3

`tests/test_policy_api.py` shows the pattern: build synthetic obs dicts,
call `act()` directly, and assert on the returned `PolicyAction` — no ns-3
build needed, runs on any host with `pytest`:

```bash
python -B -m pytest tests/test_policy_api.py -q
```
