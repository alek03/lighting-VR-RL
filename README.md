# vr-nav-rl

RL pipeline for training a vision-based agent to navigate a VR environment in Unreal Engine 5.8.

## Layout
- `training/` — RL agent, vision model, training loop
- `bridge_client/` — Python side of the Unreal <-> Python bridge API
- `infra/` — Xvfb / x11vnc / noVNC scripts for remote headless-render monitoring

## Related
Unreal project lives on the shared server at `/opt/Unrealprojects/vr_nav_env` (not in this repo — binary content).
