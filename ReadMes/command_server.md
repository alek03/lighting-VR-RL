# `unreal_scripts/command_server.py`

The only code that runs **inside** Unreal. `start_editor.sh` loads it at startup. It listens on
`127.0.0.1:<UE_COMMAND_PORT>`, reads one JSON request per line, runs the named command, and replies
with one JSON line. It accepts only the commands listed below, never arbitrary code.

Why it exists: Epic's own Python remote execution can't be discovered on Linux with the safe
localhost-only setting, and opening it to the network would allow anyone to run Python in the
editor.

## How it works

- **Runs on Unreal's frame tick.** Unreal's Python API is only safe on the main thread, so the
  server registers a function Unreal calls once per frame (`Server.tick`). Each call accepts new
  connections, reads data and runs complete requests.
- **Multi-frame commands are generators.** A command that needs frames to render (e.g. `capture`)
  `yield`s; the server resumes it on the next frame and sends its `return` value as the reply.
- **Uses the lab's modules** from the project's `Content/Python`: `objectlab_bridge` (spawn/remove
  objects via `BP_TableObjectSpawner`) and `objectlab_capture` (the photo camera).
- **Re-loading is safe:** running the script again shuts down the previous server first.

## Commands

| Command | Arguments | What it does |
|---|---|---|
| `ping` | | Alive? Returns frame count and FPS. |
| `describe` | `world`, `nearby_cm` | Table centre, lights, nearby geometry, BP_Torch's light settings, stored size of every catalogue object. |
| `status` | `world` | What's on the table. |
| `spawn` | `object` (name or number), `world` | Puts an object on the table (lighting channel 1 only, set by the lab's bridge). |
| `remove` | `world` | Clears the table. |
| `scale_object` | `scale` | Uniformly scales the object on the table (fix for mis-sized models). |
| `capture_setup` | `camera` (`location`, `rotation`, `fov`), `width`, `height` | Creates the photo camera. |
| `capture` | `path`, `settle_frames` | Renders and saves a PNG. |
| `capture_mask` | `path` | Saves the object alone, unlit: its exact silhouette. Restores the camera afterwards. |
| `capture_cleanup` | | Deletes the photo camera. |
| `light_setup` | `overrides` | Creates the flashlight: a spotlight copying BP_Torch's settings, **including its lighting channels** (objects only receive light on channel 1). |
| `set_light` | `location`, `target`, `intensity` | Moves and aims the flashlight. |
| `light_cleanup` | | Deletes the flashlight. |
| `play` | `action` (`simulate`/`stop`) | Simulate-in-Editor (not used). |
| `set_cvars` | `cvars` (`r.*` names, numbers only) | Rendering settings. |

## Editing

- **Add a command:** write `def _my_command(req): ... return {'ok': True, ...}` (use `yield` if it
  must wait for frames), add it to the `COMMANDS` table, restart the editor, and call it from Python
  with `ue.call('my_command', ...)`. Changes only take effect after a restart.
- **Different assets:** `TORCH_CLASS` (the VR flashlight Blueprint) and `OBJECT_MESHES` (where the
  object meshes live) at the top of the file.
- **Flashlight settings copied from the torch:** `TORCH_LIGHT_PROPS`.
- Errors inside a command come back as `{"ok": false, "error": ...}` and are also printed to
  `logs/editor.log` with a traceback.
