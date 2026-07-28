# Roadmap

## v1.0 — platform ergonomics (planned)

- **Policy plug-in API**: `Policy.act(obs, link_mask) -> per-STA link`
  with a registry and `--policy mypkg.MyPolicy`, so external agents do
  not need to touch `train_ns3.py`. All bundled arms re-wired through it.
- Scenario parameter presets (YAML) replacing long CLI flag lists.
- Tutorial: write, evaluate, and train a custom policy end-to-end.
- CI: host-side unit tests on every push; periodic full ns-3 build smoke.
- English pass over remaining in-code comments.
- Vendor full GPL-2.0 text in `scenario/LICENSE`.

## v2.0 — native 802.11be MLO and 802.11bn MAPC prototypes (planned)

The current environment emulates MLO with per-link single-link 802.11ax
devices. v2 will:

1. Port the scenario to a recent ns-3 release with **native 11be
   multi-link devices** (STR first, EMLSR optional), preserving the
   shared-buffer service-time routing and survivor-bias-free accounting.
2. Open the frozen coordination field of the action interface with
   **draft-inspired 802.11bn MAPC prototypes** (Coordinated Spatial
   Reuse, Co-TDMA), clearly labeled as candidate-feature prototypes
   until the amendment is finalized (expected ~2028).
3. Re-calibrate operating points and re-run the benchmark batteries in
   the new environment. v1 numbers will remain reproducible from the
   v1 tag — v1 and v2 environments are not comparable.

Blocking questions tracked for v2: ns3-ai compatibility with recent
ns-3 releases; mapping the shared-buffer queueing onto the native MLO
queue model.
