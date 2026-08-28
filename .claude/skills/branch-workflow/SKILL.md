---
name: branch-workflow
description: The git workflow for this repo — branch off current main, one unit of work per branch, PR, merge, delete, repeat. Use BEFORE starting any new milestone, feature, fix, or experiment that will produce commits; and AFTER a PR merges. Triggers on "start M5", "let's build X", "get started", "next thing", "open the PR", "I merged it", or any request that begins a new body of work.
---

# Branch workflow

The rule this exists to enforce: **one branch, one reviewable unit of work, one PR, then delete the branch.** Work is never committed to `main`, and a branch is never reused after its PR merges.

This was written after three failures on this repo, all avoidable:

- Ten commits spanning M3, M4 and M6 were piled onto a single branch, producing a +6,823-line PR that nobody can review as one thing.
- Three fully-merged branches (`m0`, `m1`, `m2`) sat undeleted, local and remote, for weeks.
- Local `main` drifted 21 commits behind `origin/main` because nobody pulled after a merge.

## Before starting any new work

Run this first. Do not start editing, and do not assume the current branch is the right one.

```bash
git status --short                  # must be clean; stop and ask if it is not
git branch --show-current
gh pr list --state open
```

Then:

```bash
git checkout main
git pull --ff-only
git checkout -b <type>/<short-name>
```

`--ff-only` is deliberate. If it refuses, local `main` has diverged from the remote and that is a thing to understand rather than paper over with a merge commit.

**Branch names** follow the work, not the milestone alone: `m5/ablation-matrix`, `fix/tool-arg-coercion`, `evals/embedder-swap`, `docs/adr-19`. A branch called `m5` invites the same pile-up that produced PR #4.

## Scoping a branch

One branch should be one thing a reviewer can hold in their head and approve or reject as a unit.

**Split when** the work spans two milestones; or the diff passes roughly 800 lines; or the commit messages start needing "also"; or one part could ship while another is still being argued about.

**Do not split** a change and its tests, a change and the doc section it invalidates, or a measurement and the code that produced it. Those belong together — a PR that changes behaviour without its test is not smaller, it is incomplete.

When work turns out to be two things mid-flight, say so and branch again from `main` rather than continuing. A second branch is cheap; an unreviewable PR is not.

## While working

Commit at each coherent step rather than once at the end. This repo's commit messages carry the *reasoning*, not just the change — what was measured, what was rejected and why, what is still unverified. Read `git log` before writing one; the existing style is the spec.

Never commit or push without being asked, unless the task was explicitly "commit this" or "open a PR".

## Before opening the PR

All four, every time. A red build on a PR you opened is worse than a slow one.

```bash
pytest -q
ruff check app evals tests scripts
git status --short                  # must be empty
git log --oneline origin/main..HEAD
```

Then `gh pr create --base main`. The PR body states what changed, what was measured, what is *not* done, and what a reviewer should be sceptical of. Known gaps go in the PR, not in a follow-up conversation.

## After the PR merges

This is the step that keeps getting skipped. Run it as soon as a merge is confirmed — including when the user says they merged it.

```bash
git checkout main
git pull --ff-only
git branch -d <branch>              # -d not -D: refuses if unmerged, which is the point
git push origin --delete <branch>
git branch -a                       # confirm both refs are gone
```

Then, before any further work, start again from **Before starting any new work**. Do not carry on in the deleted branch's working tree.

## Checking for drift

Cheap, and worth running whenever the repo state is unclear:

```bash
git fetch origin --prune
git rev-list --count main..origin/main            # 0, or local main is stale
for b in $(git branch --format='%(refname:short)' | grep -v '^main$'); do
  printf '%-34s ahead of main: %s\n' "$b" "$(git rev-list --count origin/main..$b)"
done
```

A branch showing `ahead of main: 0` is fully merged and should have been deleted.

## This repo specifically

- **Baselines under `evals/baselines/` change only in a PR that says why.** A regression must never be absorbable by regenerating the baseline in the commit that caused it — that rule is in TRD §14.3 and it is the whole reason the gate is worth having.
- **A paid eval run belongs in its own commit**, with the numbers and the spend in the message. It is not reproducible from the diff, so the message is the only record.
- **Amendments to `PRD.md` / `TRD.md` ship with the code that motivated them**, not afterwards. A spec that describes a system nobody built is the failure both documents are structured to avoid.
