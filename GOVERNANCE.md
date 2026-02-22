# Governance

Rules for merging code into `master` in the research_agent project.

## Current: Human + AI approval

Every pull request requires **both**:

1. **Human approval** — Rolf reviews and approves via GitHub
2. **AI review** — At least one AI reviewer posts a review (Greptile, Claude Code Review, or Codex)

A PR is mergeable when:
- CI passes (lint + unit tests)
- Human has approved
- At least one AI review is present and does not have unresolved blocking issues

### AI reviewers in use

| Reviewer | Trigger | Role |
|----------|---------|------|
| Greptile | Auto on PR | Automated code review with codebase context |
| Claude Code Review | Auto on PR | Code review via `claude-code-action` |
| Claude Code (CLI) | `@claude` mention | On-demand review, implementation, Q&A |

### Branch protection (active)

Currently enforced on `master`:
- Require pull request before merging
- Require at least 1 human approval
- Require status checks to pass (`lint`, `unit-tests`)
- Dismiss stale reviews on new pushes
- Force push blocked

> Note: AI agents post review comments but cannot submit formal GitHub approvals (requires collaborator status). The human reads agent feedback and decides whether to approve. See `AGENTS.md` for how agents cooperate.

## Future: Multi-AI voting

> Status: **planned, not yet implemented**

Goal: replace the single "at least one AI" requirement with a structured voting system.

### Concept

- Each AI reviewer posts a structured verdict: `APPROVE`, `REQUEST_CHANGES`, or `ABSTAIN`
- A GitHub Action tallies votes and sets a status check
- Merge requires: human approval + majority AI approval (e.g., 2 of 3)
- Human retains veto/override power at all times

### Implementation path

The key insight: GitHub only counts approvals from collaborators, not bot comments. To bridge this gap:

1. Each agent posts a comment containing a structured verdict tag (e.g., `<!-- AI_VERDICT: APPROVE -->`)
2. A **tally workflow** runs after agent workflows complete, scans for verdict tags, and sets a status check (`ai-review: 2/3 approve`)
3. Add `ai-review` as a required status check on `master`
4. Gradually relax human approval from "always required" to "tiebreaker" to "spot check"
5. Define escalation rules (e.g., any `REQUEST_CHANGES` blocks merge until addressed)

### Open questions

- How to weight different reviewers (equal votes vs. specialization-based)?
- Should certain file paths require human approval regardless (e.g., `GOVERNANCE.md`, CI config)?
- What's the right quorum — all reviewers must vote, or a subset?
