# Agents

How AI agents cooperate on this project. Each agent has a role, a trigger, and boundaries.

## Active agents

### Claude Code (CLI)
- **Where:** Local terminal, runs on Rolf's machine
- **Trigger:** Manual — Rolf starts a session
- **Can do:** Read/write code, run commands, create branches, commit, push, create PRs
- **Cannot do:** Merge to master (branch protection), bypass CI
- **Role:** Primary implementation partner. Writes features, fixes bugs, refactors. Has full codebase context via CLAUDE.md and memory files.
- **Config:** `CLAUDE.md` (project instructions), `.claude/` (memory, settings)

### Claude Code Review (GitHub Action)
- **Where:** GitHub Actions (`claude-code-review.yml`)
- **Trigger:** Auto on every PR (opened, updated, reopened)
- **Can do:** Read code, post review comments
- **Cannot do:** Write code, push commits, approve PRs
- **Role:** Automated reviewer. Catches issues, suggests improvements. Runs the `/code-review` plugin.
- **Config:** `.github/workflows/claude-code-review.yml`

### Claude Code (GitHub Action)
- **Where:** GitHub Actions (`claude.yml`)
- **Trigger:** `@claude` mention in PR comments, issues, or reviews
- **Can do:** Read code, read CI results, post comments, push commits to PR branches
- **Cannot do:** Merge to master, approve PRs
- **Role:** On-demand assistant. Responds to specific questions, implements requested changes directly on PR branches.
- **Config:** `.github/workflows/claude.yml`

### Greptile
- **Where:** GitHub integration (MCP plugin)
- **Trigger:** Auto on PR (configured externally)
- **Can do:** Read code with deep codebase understanding, post review comments, track custom patterns
- **Cannot do:** Write code, push commits
- **Role:** Codebase-aware reviewer. Understands project patterns and conventions. Tracks whether review comments are addressed.
- **Config:** Managed via Greptile dashboard

### OpenAI Codex
- **Where:** GitHub (dynamic workflow)
- **Trigger:** Configured externally
- **Can do:** Read code, suggest changes
- **Cannot do:** Push commits, approve PRs
- **Role:** Additional review perspective. Provides independent analysis from a different model family.

## Cooperation model

```
Developer (Rolf)
  │
  ├─ Works with Claude Code (CLI) locally
  │   └─ Creates branch, implements, commits, pushes, opens PR
  │
  └─ PR opened on GitHub
      │
      ├─ Tests workflow ──────────► lint + unit-tests (required)
      ├─ Claude Code Review ──────► auto-review comments
      ├─ Greptile ────────────────► auto-review comments
      ├─ Codex ───────────────────► auto-review comments
      │
      ├─ @claude in comment ──────► Claude Code Action responds / pushes fixes
      │
      └─ Rolf reads all feedback ─► approves ─► merge
```

## Rules of engagement

1. **Agents don't override each other.** Each posts its own review independently. No agent dismisses another's comments.
2. **Human decides conflicts.** If agents disagree, Rolf makes the call.
3. **Claude Code (CLI) is the builder.** It creates implementations. The other agents review.
4. **No agent merges to master.** Branch protection enforces this — only human-approved, CI-passing PRs merge.
5. **Agents read CLAUDE.md.** Project conventions (embedding model, CSS rules, architecture) apply to all agents equally.

## File ownership awareness

| Files | Primary reviewer concern |
|-------|------------------------|
| `GOVERNANCE.md`, `AGENTS.md` | Always require human review |
| `.github/workflows/` | CI/CD changes — human must verify |
| `src/research_agent/ui/app.py` | Large file — agents should focus on changed sections |
| `src/research_agent/explorer/` | D3/iframe architecture — check CLAUDE.md patterns |
| `configs/`, `.env*` | Security-sensitive — no secrets committed |

## Future: structured verdicts

When the tally workflow is built (see GOVERNANCE.md), agents will include a machine-readable verdict in their comments:

```
<!-- AI_VERDICT: APPROVE -->
<!-- AI_VERDICT: REQUEST_CHANGES -->
<!-- AI_VERDICT: ABSTAIN -->
```

A GitHub Action will parse these and set a status check (`ai-review: N/M approve`), which branch protection can require. This enables the transition from "human always approves" to "human as tiebreaker."
