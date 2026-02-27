# Memory Management Skill

You have access to a personal semantic memory system via the `recall` MCP server. Use it proactively to build a persistent understanding of the user across sessions.

## When to search memories

Search **before** answering questions that could benefit from prior context:
- "What are we working on?" → `search_memories("current project")`
- "How do I prefer X?" → `search_memories("user preference X")`
- "What did we decide about Y?" → `search_memories("decision Y")`
- When you encounter an unfamiliar project name, person, or concept the user mentions casually

Use `top_k=3` for quick lookups, `top_k=10` for comprehensive research.

## When to store memories

Store proactively after interactions that contain durable, reusable information:

**Always store:**
- User preferences: tools, workflows, coding style, communication style
- Project context: tech stack, architecture decisions, goals, constraints
- Key decisions with reasoning: "decided to use X because Y"
- Debugging insights: root causes, non-obvious solutions, gotchas
- Recurring patterns: things the user corrects or emphasises repeatedly

**Use consistent tags:**
- `project:<name>` — e.g. `project:recall`
- `preference` — user style/workflow preferences
- `decision` — architectural or design decisions
- `debugging` — bug fixes and root causes
- `context` — general project background

**Example store call:**
```
store_memory(
    text="User prefers uv over pip for Python package management",
    tags=["preference", "python", "tooling"],
    source="claude",
    dedupe_key="preference:package-manager"
)
```

Use `dedupe_key` when the memory represents a single updatable fact (e.g. a preference or a project's current state). Omit it for discrete events or decisions.

## What NOT to store

- Trivial exchanges ("thanks", "ok", "got it")
- Temporary debugging state that won't apply next session
- Information already in the codebase or docs
- Duplicates — search first with a relevant query before storing

## Listing and deleting

- Use `list_memories()` when the user asks "what do you remember?" or "show my memories"
- Use `delete_memory(id)` when the user says "forget that" or asks to remove a specific memory
- Use `check_health()` to diagnose connectivity issues with the recall service

## How to call MCP tools

Always call recall MCP tools **directly** in the main conversation — never via the Task tool or a subagent. Wrapping MCP calls in a subagent causes multiple authorization prompts; calling them directly causes at most one.

Correct:
```
list_memories()   ← called directly as an MCP tool
```

Incorrect:
```
Task(subagent_type="general-purpose", prompt="call list_memories()")  ← causes extra auth prompts
```

## Transparency

When you use a memory tool, briefly mention it:
- "Searching my memory for your React preferences…"
- "I'll remember that for next time."

Don't over-explain — one short line is enough.
