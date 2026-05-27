# Profiles And Tools

## Profiles

Profiles control the assistant personality and the tools available to that personality. Built-in profiles live in `profiles/`. Starter external examples live in `external_content/external_profiles/`.

A typical profile directory contains:

```text
profiles/example/
  instructions.txt
  tools.txt
  optional_custom_tool.py
  voice.txt
```

`instructions.txt` is the prompt loaded for the session. It may include shared prompt fragments from `src/reachy_mini_conversation_app/prompts/` by placing an include on its own line:

```text
[identities/basic_info]
[behaviors/silent_robot]
```

`tools.txt` lists one tool name per line. Blank lines and lines starting with `#` are ignored.

`voice.txt` is optional. If present, it is treated as a preferred voice for the active backend. Unsupported voices are ignored and the backend default is used.

## Tool loading order

For each name in the selected profile's `tools.txt`, the loader tries:

1. A profile-local Python file named `<tool_name>.py`.
2. A core tool module at `reachy_mini_conversation_app.tools.<tool_name>`.
3. An external tools directory, when configured.

System tools from `SystemTool` are appended automatically.

If external tool autoloading is enabled, valid `*.py` files from the external tools directory can be added even when they are not listed in the profile.

## Tool interface

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

The class name does not need to match the tool name, but the module filename must match the entry in `tools.txt` unless the tool is imported through another loaded module.

## Available dependencies

Tools receive a `ToolDependencies` object:

| Dependency | Use |
| --- | --- |
| `reachy_mini` | Direct access to the Reachy Mini SDK. Prefer higher-level managers when possible. |
| `movement_manager` | Queue motion and set listening/speech/tracking offsets. |
| `camera_worker` | Read the latest camera frame or toggle head tracking. May be `None` with `--no-camera`. |
| `vision_processor` | Local vision processor when `--local-vision` is active. May be `None`. |
| `head_wobbler` | Speech-reactive motion helper. May be `None`. |
| `motion_duration_s` | Default motion duration for simple movement tools. |

## Practical rules

- Keep tool names stable. They are part of model-facing behavior and profile configuration.
- Keep schemas narrow and explicit. Realtime models behave better with small parameter surfaces.
- Return structured dictionaries, not prose-only strings.
- Avoid blocking work inside a tool call. Use `BackgroundToolManager` for long-running or cancellable routines.
- Treat camera frames and base64 image payloads as transport details. Avoid sending bulky data back into model context unless the handler explicitly needs it.
- Do not call `ReachyMini.set_target` directly from multiple places. Prefer `movement_manager` for coordinated motion.

## Built-in profile examples

The built-in `profiles/` directory contains examples with different instruction styles and tool selections:

- `default`
- `example`
- `chess_coach`
- `nature_documentarian`
- `victorian_butler`
- `mars_rover`
- `hype_bot`
- `noir_detective`
- `cosmic_kitchen`

Use those as reference material for tone, tool selection, and prompt size.

