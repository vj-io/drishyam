"""A minimal MCP client that lets a model drive `drishyam mcp` one tool call at a time, from outside the process.

The recorded sessions in this folder were made with it: `claude -p` was not available on the recording machine, so
Claude (Opus 5.5, inside Claude Code) issued each call by writing a request file, read the reply (and looked at
the crop JPEGs it saved) and only then chose the next call. The bridge never chooses anything.

It is ../mcp-session/bridge.py with one change: the MCP stdio client starts the server with only HOME, LOGNAME,
PATH, SHELL, TERM and USER, so the human's DRISHYAM_* settings (DRISHYAM_ONNX selects the v1 segmenter) are passed
through here and listed in session.json.

    python docs/mcp-session-v1/bridge.py DIR   # serves DIR/req-NNN.json -> DIR/rep-NNN.json

A request is {"tool": name, "args": {...}}; {"tool": "_quit"} ends the session. Image content in a reply is saved
as DIR/img-NNN-K.<ext> and replaced by its file name, so the reply stays readable.
"""

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXT = {"image/jpeg": "jpg", "image/png": "png"}


def _content(blocks, out: Path, n: int) -> list:
    items = []
    for k, b in enumerate(blocks):
        d = b.model_dump(mode="json", by_alias=True, exclude_none=True)
        if d.get("type") == "image":
            name = f"img-{n:03d}-{k}.{EXT.get(d.get('mimeType'), 'bin')}"
            (out / name).write_bytes(base64.b64decode(d["data"]))
            d = {"type": "image", "file": name}
        elif d.get("type") == "text":
            try:
                d = {"type": "json", "value": json.loads(d["text"])}
            except ValueError:
                pass
        items.append(d)
    return items


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k.startswith("DRISHYAM_")}
    server = StdioServerParameters(command="uv", args=["run", "drishyam", "mcp"], env=env)
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        tools = await session.list_tools()
        info = init.model_dump(mode="json", by_alias=True)
        (out / "session.json").write_text(json.dumps({
            "server": info.get("serverInfo"), "instructions": info.get("instructions"),
            "tools": [t.name for t in tools.tools], "env": env}, indent=2))
        n = 0
        while True:
            req = out / f"req-{n:03d}.json"
            if not req.exists():
                await asyncio.sleep(0.2)
                continue
            call = json.loads(req.read_text())
            if call["tool"] == "_quit":
                break
            t0 = time.time()
            res = await session.call_tool(call["tool"], call.get("args", {}))
            rep = {"n": n, "tool": call["tool"], "args": call.get("args", {}),
                   "is_error": bool(getattr(res, "is_error", False)),
                   "content": _content(getattr(res, "content", []), out, n),
                   "seconds": round(time.time() - t0, 3)}
            (out / f"rep-{n:03d}.json").write_text(json.dumps(rep, indent=2))
            n += 1


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
