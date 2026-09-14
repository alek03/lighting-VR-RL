# lighting-VR-RL

RL pipeline for training a vision-based agent to navigate a VR environment in Unreal Engine 5.8,
running headless on a shared GPU server.

## Layout

- `training/` — RL agent, vision model, training loop (not started)
- `bridge_client/` — Python side of the Unreal <-> Python bridge (not started)
- `infra/` — running the engine headless, plus a live viewer
- `captures/`, `logs/` — run artifacts, gitignored

The Unreal project lives outside this repo at `/opt/Unrealprojects/` — it is mostly binary
`.uasset` content, which git handles poorly.

## Host constraints

These shaped every decision here and are worth knowing before changing anything.

**No display.** The box is headless, so `-RenderOffScreen` is required: Vulkan can render on the
GPU but cannot present to a surface. Xvfb does *not* help — it is software-only with no GPU
attachment, and the engine exits on a modal "Cannot find a compatible Vulkan device that supports
surface presentation".

**Only port 22 is reachable.** WebRTC/Pixel Streaming negotiates media over UDP on dynamic ports,
so it cannot traverse an SSH tunnel, and it stalls at `IceConnectionChecking`. Hence the MJPEG
viewer, which needs exactly one TCP port. Pixel Streaming itself works fine *on* the host
(verified 1280x720, 60fps, no packet loss) and becomes usable remotely if a port is ever opened.

**No console commands at runtime.** RemoteControl's HTTP API hard-blocks `ExecuteConsoleCommand`
regardless of `bAllowAnyRemoteFunctionCall`, and Pixel Streaming's `emitConsoleCommand` is not
wired through in this build. The only runtime channel is synthetic keyboard/mouse input.

**No project authoring.** Creating a project properly needs the editor's GUI wizard. Copying a
template directory by hand silently omits `/Game/LevelPrototyping`, `/Game/Characters` and
`/Game/Input` (they live in `Engine/Templates/TemplateResources/High/`), which leaves actors
loading but rendering nothing and the PlayerController failing to spawn a pawn. Author projects on
a machine with a display; use this host for training.

Headless *editing* is possible via Python commandlets, which do work:

    UnrealEditor-Cmd <project>.uproject -ExecutePythonScript=script.py \
        -unattended -nosplash -nosound -nullrhi -nopause

## Usage

Start everything, then tunnel one port to watch:

    ./infra/start.sh /opt/Unrealprojects/render_test/render_test.uproject

    # from your own machine; the left-hand port is any free local one
    ssh -L 9847:localhost:8090 <user>@<host>
    # then open http://localhost:9847

    ./infra/stop.sh

Drive the character so there is something to watch:

    python3 infra/walk_agent.py --pattern circle --duration 300

`start.sh` runs three processes: the Pixel Streaming signalling server (used
*only* as an input channel — the camera buttons and `walk_agent` need it),
Unreal, and the MJPEG viewer. `launch_ue.sh` runs the engine alone if you want
it without the viewer.

## Status

Working: headless GPU rendering; frame extraction; MJPEG viewer over the tunnel; keyboard/mouse
input into the running game; free-camera toggle and a top-down view.

Known rough edges:

- Capture runs at ~4fps because `-dumpmovie` encodes a full PNG per frame on the game thread. The
  MJPEG transport is nowhere near saturated; replacing the frame source should lift this.
- Camera nudge buttons are approximate. There is no absolute camera positioning available, so all
  camera motion is relative; "Bird's eye" resets to a known state first (toggling the debug camera
  off and on snaps it to the player) but the nudges are open-loop.
- The control bridge holds a headless Chromium open for the session. It costs ~15s to connect and
  is a demo/debug path, not something to build a training loop on.

Not started: SceneCapture2D observation camera, the trajectory format for human VR demos, and the
training code itself.
