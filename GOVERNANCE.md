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

### Branch protection (recommended)

On GitHub, set these for `master`:
- Require pull request before merging
- Require at least 1 approval
- Require status checks to pass (lint, unit-tests)
- Do not allow bypassing the above settings

## Future: Multi-AI voting

> Status: **planned, not yet implemented**

Goal: replace the single "at least one AI" requirement with a structured voting system.

### Concept

- Each AI reviewer posts a structured verdict: `APPROVE`, `REQUEST_CHANGES`, or `ABSTAIN`
- A GitHub Action tallies votes and sets a status check
- Merge requires: human approval + majority AI approval (e.g., 2 of 3)
- Human retains veto/override power at all times

### Steps to get there

1. Standardize AI review output format (structured comment with verdict tag)
2. Build a GitHub Action that parses verdicts and reports a combined status
3. Add the combined status as a required check on `master`
4. Gradually relax human approval from "always required" to "tiebreaker" to "spot check"
5. Define escalation rules (e.g., any `REQUEST_CHANGES` blocks merge until addressed)

### Open questions

- How to weight different reviewers (equal votes vs. specialization-based)?
- Should certain file paths require human approval regardless (e.g., `GOVERNANCE.md`, CI config)?
- What's the right quorum — all reviewers must vote, or a subset?
