"""Client for unreal_scripts/command_server.py, running inside the Unreal editor.

Library use:

    from bridge_client.unreal_client import UnrealClient

    with UnrealClient() as ue:
        ue.call('spawn', object='Mug', world='editor')
        ue.call('capture', path='/tmp/frame.png')

Command line (needs ./infra/start_editor.sh running):

    .venv/bin/python -m bridge_client.unreal_client check --object Mug
    .venv/bin/python -m bridge_client.unreal_client describe
    .venv/bin/python -m bridge_client.unreal_client spawn Apple
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POSE = Path('/opt/Unrealprojects/tufaelz/Scripts/pose.json')


class UnrealError(RuntimeError):
    pass


class UnrealClient:
    def __init__(self, port=None, timeout=60.0):
        self.port = port or int(os.environ.get('UE_COMMAND_PORT', '6780'))
        self.timeout = timeout
        self.sock = None
        self._buf = b''

    def __enter__(self):
        self.sock = socket.create_connection(('127.0.0.1', self.port), timeout=self.timeout)
        return self

    def __exit__(self, *exc):
        if self.sock:
            self.sock.close()

    def call(self, command, **args):
        self.sock.sendall(json.dumps(dict(command=command, **args)).encode() + b'\n')
        while b'\n' not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise UnrealError('Unreal closed the connection')
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        reply = json.loads(line)
        if not reply.get('ok'):
            raise UnrealError(reply.get('error', 'command failed'))
        return reply


def wait_for_file(path, timeout=30.0):
    """Render-target export can land a moment after the command returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        time.sleep(0.05)
    raise UnrealError('timed out waiting for ' + path)


def check(ue, object_name, width, height):
    """End to end: spawn, capture from the fixed camera, clean up."""
    camera = json.loads(POSE.read_text())['camera']
    target = REPO / 'captures' / ('check_%s.png' % object_name.lower())
    target.parent.mkdir(exist_ok=True)
    target.unlink(missing_ok=True)
    spawned = ue.call('spawn', object=object_name, world='editor')
    ue.call('capture_setup', world='editor', camera=camera, width=width, height=height)
    started = time.monotonic()
    ue.call('capture', path=str(target))
    wait_for_file(str(target))
    elapsed = time.monotonic() - started
    ue.call('capture_cleanup')
    ue.call('remove', world='editor')
    return {'object': spawned['object_name'], 'centre_cm': spawned.get('object_bounds_center_cm'),
            'frame': str(target), 'capture_s': round(elapsed, 3)}


def main():
    ap = argparse.ArgumentParser(description='Talk to the Unreal command server.')
    ap.add_argument('--world', default='editor', choices=['editor', 'pie', 'auto'])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('ping', 'status', 'remove', 'describe'):
        sub.add_parser(name)
    sp = sub.add_parser('spawn')
    sp.add_argument('object', help='1-10 or a name such as Mug')
    ck = sub.add_parser('check', help='spawn, capture a frame into captures/, clean up')
    ck.add_argument('--object', default='Mug')
    ck.add_argument('--width', type=int, default=1280)
    ck.add_argument('--height', type=int, default=720)
    args = ap.parse_args()

    try:
        with UnrealClient() as ue:
            if args.cmd == 'check':
                out = check(ue, args.object, args.width, args.height)
            elif args.cmd == 'spawn':
                out = ue.call('spawn', object=args.object, world=args.world)
            elif args.cmd == 'ping':
                out = ue.call('ping')
            else:
                out = ue.call(args.cmd, world=args.world)
    except (OSError, UnrealError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
