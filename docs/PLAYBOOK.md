# Playbook: how this course was built

Read this first if you are starting a new interactive course (for example on databases) based on this one. It records what worked, so a new project does not have to rediscover it.

## The learner and the goal

- The learner is from Taiwan, has **no background in the subject**, and reads English and Traditional Chinese. Every lesson exists in both languages.
- Teaching style, taken from an interactive system-design course: **one diagram per lesson that evolves step by step** (7–11 steps). Steps are tagged Problem → Technique → Solution → (Case: a problem that breaks the idea) → Recap. Short text beside the diagram: at most two short paragraphs, one callout, a "What changed" list, live metric tiles above the diagram.
- The recap's "Coming up" callout names the next lesson in the outline.

## The rule that made the course trustworthy

**Every number shown is computed, never typed in.** Numbers come from a small real system running in the browser (here: a 3,115-weight transformer that learns first names), or from real experiment runs saved to a JSON file by a script. Illustrations that are not computed are labelled as illustrations. Outside facts (real-world models, papers) are few, well known, and checked with a web search; their sources go in the pull request.

When an experiment shows little or no effect at small scale, the lesson says so honestly. Several claims were dropped or reworded because the real numbers did not support them (see the history of the attention and MoE lessons).

For a databases course, the equivalent would be a small real storage engine, log, index, replication or consensus simulation running in the browser, with real measurements.

## Architecture (reusable as is)

| Part | Files | Reuse for a new course? |
|---|---|---|
| Page shell, routing (`#/lesson-id/step`), progress in `localStorage`, language switch, dark mode, keyboard | `src/main.ts`, `src/style.css`, `index.html` | Yes, change the brand text |
| Bilingual strings | `src/i18n.ts` | Yes |
| Diagram engine: keyed SVG items that animate between steps (token, number strip, heat grid, text, arrow, box, chart, bars, path, dot) | `src/engine/scene.ts` | Yes; add item types the subject needs |
| Interactive controls (slider, text, toggle, choice) passed to `scene(step, input)` | `src/lessons/types.ts`, `src/main.ts` | Yes |
| Lesson format and auto-registration | `src/lessons/types.ts`, `src/course.ts` | Yes; replace the outline |
| Tests: every step of every lesson drawn in both languages, every control at its extremes | `tests/lessons.test.ts` | Yes |
| Browser walk-through with screenshots | `scripts/walk.cjs` | Yes |
| Deploy to GitHub Pages on every merge to `main` | `.github/workflows/deploy.yml`, `vite.config.ts` (`base: './'`) | Yes |
| The subject's real system | `src/model/`, `scripts/train.ts`, `scripts/experiments.ts` | No, build the new subject's equivalent |
| Lessons | `src/lessons/*.ts` | No, but copy their style |

## Process that worked

1. **Study a reference** for teaching style (not content), then plan the outline.
2. **Build the engine and the real system first**, with tests (gradient checks here; for databases, e.g. crash-recovery or replication invariants).
3. **Write 2–3 pilot lessons yourself** to fix the format, then hand the rest to parallel writers with one shared brief: `docs/process/writing-brief.md`. Give each writer a few lessons, a dev-server port and a screenshot folder. Writers only add their own lesson files; shared files stay with the lead.
4. **Independent review**: fresh reviewers who are not shown the lead's opinion, each covering about nine lessons, reading as a beginner, checking facts and the Chinese, proposing interactive illustrations: `docs/process/review-brief.md`.
5. **Fix pass**: fixers re-check each finding against the real system before changing anything, then build the interactives: `docs/process/fix-brief.md`.
6. **The lead verifies everything** in a real browser before each pull request: tests, build, every step in both languages, every control at its extremes, then opens the PR. Merge → GitHub Pages redeploys → check the live site.

Lessons learned along the way:
- A dataset that is too small makes the model memorize; this broke the overfitting and fine-tuning lessons until the name list grew from 379 to about 3,000.
- Name the different runs/systems consistently ("the course model", "the logged run", "comparison runs"); mixing them made numbers look contradictory.
- Define every term at first use; reviewers found "block" and "held-out loss" used before being taught.
- Reviewers' strongest catch: a comparison that changed two things at once (attention lesson). Check that every "A vs B" changes only one thing.

## Traditional Chinese conventions

Natural Taiwan Mandarin, not Mainland usage: 資料 (not 數據), 程式, 記憶體, 位元, 網路, 預設, 品質; 「」 for quotes everywhere, including diagram labels. Keep a glossary per course and use it consistently; the transformer glossary is in `docs/process/writing-brief.md`. The Chinese has not yet been checked by a native speaker.
