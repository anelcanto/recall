---
name: recall:update
description: Check for a newer version of recall-cli on PyPI and optionally upgrade
---

Check if a newer version of `recall-cli` is available on PyPI and offer to upgrade.

## Steps

1. **Get the currently installed version:**

```bash
pip show recall-cli 2>/dev/null | grep ^Version
```

Or with uv:

```bash
uv pip show recall-cli 2>/dev/null | grep ^Version
```

2. **Get the latest version from PyPI:**

```bash
curl -s https://pypi.org/pypi/recall-cli/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
```

3. **Compare versions** and report to the user:
   - If already on the latest: "recall-cli is up to date (vX.Y.Z)"
   - If an update is available: show current vs latest versions

4. **If an update is available**, ask the user:

```
A new version of recall-cli is available: vX.Y.Z → vA.B.C

Upgrade now? (pip install --upgrade recall-cli)
```

5. **If the user confirms**, run the appropriate upgrade command:

   - With pip: `pip install --upgrade recall-cli`
   - With uv: `uv pip install --upgrade recall-cli`
   - With brew: `brew upgrade recall-cli` (if installed via Homebrew)

6. After upgrading, confirm the new version is active:

```bash
recall --version
```

> **Note:** After upgrading, restart any running `recall serve` process and reload the MCP server in Claude Code (`/mcp` → reconnect) to pick up changes.
