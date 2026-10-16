# unreal_bridge/start.sh

Starts Unreal with no screen (headless) on this server, with `server.py` loaded, and returns when
it is ready to take commands.

```
cd /home/axm6766/lighting-rl
./unreal_bridge/start.sh
```

Prints `ready on port 6780` after about 1 minute (the first run of a new project copy: ~10
minutes, while it prepares shaders). Unreal keeps running in the background after the script ends,
until `stop.sh`. Unreal's own log is `logs/editor.log`.

Don't press Ctrl+C while it is starting: that stops Unreal too. Use a second terminal if you need
one meanwhile.

## Settings used (config.json)

`unreal.engine`, `unreal.project`, `unreal.map`, `unreal.port`, `gpu`.

## What it does, and why

| Step | Why |
|---|---|
| Waits up to 30 s if the port is still in use | A just-stopped Unreal can hold the port for a moment. |
| `umask 000` | Files Unreal creates in the project stay writable for labmates. Without it, one person's run can crash the next person's. |
| `-graphicsadapter` from `gpu` | Unreal ignores `-gpuindex`, and counts GPUs in a different order from `nvidia-smi` on this server (adapter 0 = GPU 3, 1 = GPU 0, 2 = GPU 1). The script converts, and warns if Unreal lands on the wrong GPU. |
| `-RenderOffScreen -vulkan` | The server has no screen. |
| `r.TextureStreaming=0` | Otherwise textures stay blurry: Unreal picks texture detail from its unused main view, not our camera. |
| `-ExecCmds="py .../server.py"` | Loads the server. (`-ExecutePythonScript` would quit Unreal after the script runs.) |
