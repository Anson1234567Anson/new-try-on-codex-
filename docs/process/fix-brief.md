# Fix brief: act on the review of "Transformers, step by step"

You are fixing lessons after an independent review. First read, in this order:
1. `docs/process/writing-brief.md`: the original writing brief (audience, style, honesty rules, glossary, technical rules, checks). It still applies, with the updates below.
2. Your review report (path in your assignment). Treat each finding as a claim to check, not a fact: re-check it against the code and the real model before changing anything, and skip it if it is wrong (say so in your report).
3. The lessons you own.

## Decisions that apply to every lesson (follow them, so all fixers stay consistent)

**Terms (EN / ZH):**
- The 615 names the model never trains on: EN "held-out names", their loss "held-out loss". ZH: 驗證集 (validation set), 驗證損失 (held-out loss); the first time in a lesson, explain it in one line, e.g. "held-out loss: the loss on 615 names the model never trained on" / 「驗證損失：模型在訓練時從沒看過的 615 個名字上的損失」. Replace 保留集, 保留資料, 保留資料損失 everywhere in your lessons.
- activation → 活化值 (replace 激活值). Neuron "fires" → define once per lesson with ONE threshold and use it consistently.
- Replace 數據 with 資料 (data) or 數字 (numbers); replace 偏差值 and 外掛 with natural wording.
- Chinese text, including metric labels and values, uses 「」 for quotes, never “ ” or " ".
- Terms used before they are taught: define them in one short line where first used (e.g. "weights: the numbers inside the model that training adjusts", "block: one attention layer plus one MLP layer, with their adding").

**Which trained model is which.** Several runs appear across lessons. Whenever you quote a number, say which run it comes from, with these names:
- **the course model** 課程模型: `src/model/weights.json`, made by `scripts/train.ts` (6,000 steps, Adam, batch 32, learning rate 0.01 decaying to 0.001, trained on the 2,456 training names). Held-out loss ≈ 2.207. Lessons that run "the model" live use this.
- **the logged run** 紀錄的訓練: `experiments.main` (4,000 steps, warmup + cosine, peak 0.02). Used by Evaluation, Real Run, Compute.
- **comparison runs** 比較實驗: the 1,500-step runs in `experiments.optimizers`, `.schedules`, `.variants` and `.moe`. Note: `variants.base` (constant learning rate 0.01) ends at 2.218, `moe.dense` (warmup + cosine) at 2.203: different schedules, so say which one you compare against and why.
Never present numbers from two different runs side by side as if they were one model.

**"Pure guessing"** is ln 27 ≈ 3.30. An untrained model's loss (≈3.45) is a little worse than pure guessing; do not call that number pure guessing.

## New: interactive controls (available now)

A step can return `controls` in its `Scene`, and receives the learner's values in `scene(step, input)`. Read `Control` / `Input` in `src/lessons/types.ts` and `renderScene` in `src/main.ts`. Kinds: `slider` (min/max/step/value/format), `text` (maxLength, `pattern` like `'a-z'` filters characters), `toggle`, `choice` (segmented buttons). Values are kept per lesson step while the learner stays. Lessons with `tick` (live training) receive `input` in `tick(step, input)`, and a control change restarts the loop: compare the input with the settings your current run used, and `reset` the run yourself when they differ.

New diagram items in `src/engine/scene.ts`: `path` (free line through points, optional fill/dashed/closed) and `dot` (circle with optional label).

Rules for interactives:
- Add them only where they make a hard idea clearer than stepping can. Each interactive step's text should start with an instruction like "Try it: drag … and watch …" / 「動手試試：拖動……，看看……」, and say what to notice.
- Start values must reproduce the step's teaching point; the text must stay true for the starting values and must not claim specific numbers that change when the learner moves a control (refer to the diagram/metrics instead, or phrase generally).
- Each control change must redraw in under ~50 ms: memoize expensive work by input value; anything slower (training) must go through `tick` in ≤10 ms slices.
- The lesson test now draws every control at its edges (slider min/max, toggles, each choice, sample texts including empty text): handle empty or odd input gracefully (e.g. empty name → show a hint, do not crash).
- Test controls in the browser yourself with Playwright: the control elements have ids `ctl-<key>`; set a slider with `page.$eval('#ctl-x', e => { e.value = '0.5'; e.dispatchEvent(new Event('input', { bubbles: true })) })`, type with `page.fill('#ctl-word', 'anna')`, and screenshot. Write such scripts to a FILE and run with `node file.cjs` (inline `node -e "..."` breaks on `$`).

## Scope and files

- Edit only the lesson files you own (listed in your assignment) and, if needed, new helper files named after your lessons. Do not edit shared files (`src/model/*`, `src/engine/*`, `src/main.ts`, `src/course.ts`, `src/i18n.ts`, `src/style.css`, tests, scripts). If you need a shared change, describe it in your report.
- Fix every **[error]**, **[confusing]**, **[chinese]** and **[layout]** finding you confirm. Fix **[polish]** findings when cheap.
- Keep lessons to 7–11 steps; you may add or merge steps. The last step stays the recap; keep each recap's "next" callout pointing to the next lesson in the outline.
- Do not run `scripts/train.ts` or `scripts/experiments.ts`.

## Checks (required)

1. `npx vitest run tests/lessons.test.ts` passes for your lessons; `npx tsc --noEmit` has no errors in your files.
2. Dev server on your port; walk every changed lesson in en and zh with `node scripts/walk.cjs` and LOOK at the screenshots; exercise every control with your own Playwright script at its start value and at both extremes, in both languages, and look at those screenshots too.
3. Final report: per lesson, which findings you fixed, which you rejected and why, which interactives you built (step, controls, what they show); any outside fact you added with its source; any shared change you need.
