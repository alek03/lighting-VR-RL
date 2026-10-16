# unreal_bridge/server.py

Runs **inside** Unreal. `start.sh` loads it when Unreal starts. You shouldn't need to run or edit
it; your code talks to it through the `Unreal` class.

## Why it exists

Unreal's Python only exists inside the Unreal editor, so code outside can't call it directly.
This file opens a connection on this machine only (127.0.0.1, port from `config.json`). Outside
code sends one line of JSON (a command), and gets one line of JSON back. Only the commands below
are accepted, never arbitrary code.

## Commands

| Command | What it does |
|---|---|
| `ping` | Checks the server is alive. |
| `spawn` | Puts an object on the table using the project's own spawn code (`Content/Python/objectlab_bridge.py`), which also sets the object's lighting channel. Returns the object's name and centre. |
| `remove` | Clears the table. |
| `set_camera` | Creates the camera on first use, then sets its position, angle, zoom (FOV) and picture size. |
| `set_light` | Creates the flashlight on first use, copying the VR flashlight's settings (`BP_Torch`), then places it and points it at a target. |
| `capture` | Saves a picture from the camera to a file. |
| `capture_mask` | Saves a picture of the object alone, as depth (empty space = 0), then restores the camera. |
| `cleanup` | Removes the object, camera and flashlight. |

## How it works

- The connection is checked once per rendered frame, because Unreal's Python must run on the
  editor's main thread.
- A command that needs frames to render (`capture`, `capture_mask`) uses `yield`: it pauses
  there and continues on the next frame.
- Running the file again (e.g. after editing it) replaces the old server instead of adding one.

## Good to know

- The flashlight copies `lighting_channels` from the VR flashlight. Table objects only receive
  light on channel 1, so without it the object would be black.
- Errors are sent back as `{"ok": false, "error": ...}`, and `Unreal.call` raises them in Python.
  The full error is also in `logs/editor.log`.
- After editing this file, restart Unreal (`stop.sh`, `start.sh`).
