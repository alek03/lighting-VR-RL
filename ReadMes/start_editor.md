# `infra/start_editor.sh` and `infra/stop.sh`

Start and stop the headless Unreal editor that every experiment talks to.

## `start_editor.sh`

```bash
PROJECT=<copy>/tufaelz.uproject GPU_INDEX=1 UE_COMMAND_PORT=6782 ./infra/start_editor.sh
```

Starts the full Unreal **editor** (the lab's spawner code needs editor features) with no screen,
opens the `L_XRTemplate` level, and loads `unreal_scripts/command_server.py`, which listens on
`127.0.0.1:<UE_COMMAND_PORT>`. It waits until the server reports `listening` (about 1–2 minutes;
about 11 minutes the first time while shaders compile), then checks with `nvidia-smi` that the
editor is on the requested GPU and that the map loaded, and warns if not. Output goes to
`logs/editor.log`.

| Setting (environment) | Default | Meaning |
|---|---|---|
| `PROJECT` | `/opt/Unrealprojects/tufaelz/tufaelz.uproject` | The `.uproject` to open. Use your own copy. |
| `MAP` | `<project folder>/Content/XRFramework/Levels/L_XRTemplate.umap` | The level, as a **file path**. |
| `GPU_INDEX` | `1` | GPU number as shown by `nvidia-smi`. |
| `UE_COMMAND_PORT` | `6780` | Command-server port. Also export it for the Python side, which reads the same variable. |
| `UE_ROOT` | `/opt/UnrealEngine/UE-5.8.2` | Engine install. |

**Why the unusual flags** (each was needed on this host):

| Flag | Why |
|---|---|
| `-RenderOffScreen -vulkan` | No display; without it the engine exits looking for a screen. |
| `-graphicsadapter=N` | Unreal **ignores `-gpuindex`**. `-graphicsadapter` works but counts GPUs in Vulkan's order, which here is `nvidia-smi`'s shifted by one (adapter 0 = GPU 3, 1 = GPU 0, 2 = GPU 1). The script converts `GPU_INDEX`; it assumes this host's order. |
| map as a file path | A `/Game/...` path fails ("Map load failed. The filename ''…") when the project folder and `.uproject` names differ, as in copies. |
| `-dpcvars="r.TextureStreaming=0"` | Otherwise textures stay blurry (streaming follows the main viewport, which never looks at the table). |
| `-ExecCmds="py .../command_server.py"` | Loads the server after startup; `-ExecutePythonScript` would run it and **quit**. |

## `stop.sh`

```bash
./infra/stop.sh
```

Stops **your own** Unreal processes (and the old live-viewer stack's processes, if running), then
prints GPU usage so you can confirm nothing is left holding memory. It never touches other users'
processes.

## Editing

- Different host or GPU order: change the `ADAPTER=$(( (GPU_INDEX + 1) % GPU_COUNT ))` line; the
  placement check after startup tells you if the mapping is wrong.
- Different level: set `MAP`. The command server needs exactly one `BP_TableObjectSpawner` in it.
