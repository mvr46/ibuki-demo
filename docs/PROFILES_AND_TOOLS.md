# Profiles And Tools

## Profiles

Profiles control the assistant instructions, voice preference, and enabled core tools. Production profiles live in:

```text
src/reachy_mini_conversation_app/profiles/
```

A profile directory contains:

```text
src/reachy_mini_conversation_app/profiles/default/
  instructions.txt
  tools.txt
  voice.txt        # optional
```

`ProfileStore` is the only persistence interface for profile operations. It supports listing profiles, loading one profile, saving a new profile, overwriting an existing profile, and resolving the startup profile with a default fallback.

The dashboard can edit `instructions.txt`, `tools.txt`, and `voice.txt`. Prompt and voice changes apply live. Tool-list changes are persisted and take effect after restart because `ToolRegistry` is built during startup.

With the local backend, `voice.txt` normally contains the logical voice label `local`. The actual audible speaker comes from the Piper `.onnx` file configured by `PIPER_VOICE` at app startup.

## Tool Loading

Profiles list enabled tool module names in `tools.txt`, one per line. Blank lines and lines starting with `#` are ignored.

`ToolRegistry` imports only selected core tool modules from:

```text
src/reachy_mini_conversation_app/tools/
```

System tools from `SystemTool` are appended automatically. Profile-local Python tools and external tool autoloading are intentionally removed from the production path.

## Tool Interface

Tools subclass `Tool` from `src/reachy_mini_conversation_app/tools/core_tools.py`.

Minimal shape:

```python
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class ExampleTool(Tool):
    name = "example_tool"
    description = "Do one small, concrete thing."
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
        "required": ["message"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        message = str(kwargs.get("message", ""))
        return {"ok": True, "message": message}
```

## Practical Rules

- Keep tool names stable. They are part of model-facing behavior and profile configuration.
- Keep schemas narrow and explicit.
- Return structured dictionaries, not prose-only strings.
- Use `BackgroundToolManager` for long-running or cancellable routines.
- Do not call `ReachyMini.set_target` directly from multiple places. Prefer `movement_manager` for coordinated motion.
