# unreal_bridge/stop.sh

Stops your Unreal and shows how much memory each GPU is using, so you can check nothing is left
running on this shared machine.

```
./unreal_bridge/stop.sh
```

- Only stops Unreal processes that belong to you, never labmates'.
- Waits up to 30 s for Unreal to shut down properly, then forces it if needed.
- Prints `Unreal stopped`, `Unreal force-stopped`, or `Unreal was not running`, then the GPU list
  (a few MiB per GPU = nothing running).

Run it whenever you are done: a running Unreal holds a GPU even when idle.
