# `bridge_client/unreal_client.py`

The Python side of the connection to `command_server.py`: open a socket to
`127.0.0.1:<UE_COMMAND_PORT>`, send one JSON line per command, read one JSON line back.

## As a library

```python
from bridge_client.unreal_client import UnrealClient, wait_for_file

with UnrealClient() as ue:                         # port from UE_COMMAND_PORT
    ue.call('spawn', object='Mug', world='editor')
    ue.call('capture', path='/dev/shm/frame.png')
    wait_for_file('/dev/shm/frame.png')            # the PNG can land just after the reply
```

`call(command, **arguments)` accepts any command the server knows (see
[command_server.md](command_server.md)) and raises `UnrealError` if the reply says
`"ok": false`. `training/scene.py` builds on this.

## From the command line (quick checks)

```bash
export UE_COMMAND_PORT=6782
.venv/bin/python -m bridge_client.unreal_client ping               # alive? frame rate
.venv/bin/python -m bridge_client.unreal_client describe           # lights, table, object sizes
.venv/bin/python -m bridge_client.unreal_client spawn Apple        # by name or catalogue number
.venv/bin/python -m bridge_client.unreal_client status
.venv/bin/python -m bridge_client.unreal_client remove
.venv/bin/python -m bridge_client.unreal_client check --object Mug # spawn, one photo into captures/, clean up
```

`--world {editor,pie,auto}` goes before the command (default `editor`). `check` uses the camera
pose from `VRNAV_POSE` (default: the live project's `Scripts/pose.json`) without zooming.
