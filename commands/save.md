---
name: recall:save
description: Review the current conversation and store relevant memories to recall
---

Review this conversation and save anything worth remembering to long-term memory using the recall MCP tools.

## Steps

1. **Search for existing memories** to avoid duplicates:
   - Use `search_memories()` with relevant keywords from the conversation
   - Note any existing memories that would overlap

2. **Identify saveable content** from the conversation. Look for:
   - **Preferences**: Tools, workflows, formatting styles, communication preferences
   - **Decisions**: Architecture choices, technology selections, naming conventions
   - **Project context**: Goals, constraints, key file paths, team conventions
   - **Debugging insights**: Root causes found, solutions that worked, things to avoid
   - **Personal facts**: Information the user shared about their setup, projects, or situation

   Skip: greetings, one-off questions, information already captured in code/docs, trivial exchanges.

3. **Store each memory** via `store_memory()`:
   - Write a clear, standalone text that will be useful out of context
   - Add relevant tags (e.g., `["preference", "python"]`, `["project", "recall"]`)
   - Set a `dedupe_key` to prevent future duplicates (e.g., `"python-formatter-preference"`)

4. **Report what was saved**:
   - List each memory stored with its tags
   - If nothing worth saving was found, say so briefly
