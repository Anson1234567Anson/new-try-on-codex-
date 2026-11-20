# Transformers, step by step

An interactive course on how transformers work and how AI models are trained,
in English and Traditional Chinese (繁體中文). It assumes no machine-learning background.

Each lesson is one diagram that you step through. The numbers in the diagrams are
not scripted: a real transformer with 3,115 weights runs (and trains) in the browser tab,
and the larger experiments shown in later lessons are real training runs saved by
`scripts/experiments.ts`.

Live site: https://anson1234567anson.github.io/new-try-on-codex-/

Starting a similar course? Read [`docs/PLAYBOOK.md`](docs/PLAYBOOK.md): the teaching style, architecture, honesty rules and the write → review → fix process, with the briefs used in [`docs/process/`](docs/process/).

## Run it

```bash
npm install
```

```bash
npm run dev
```

Other commands: `npm test` (model gradient checks, and every step of every lesson drawn
in both languages), `npm run build` (static site in `dist/`), `npm run train` (retrain the
shipped weights), `npm run experiments` (re-run the saved training experiments, a few minutes;
`ONLY=finetune npm run experiments` re-runs one section).

## Data

The model learns to write first names. The list is about 3,000 English first names
(2–7 letters) from the [human-names](https://www.npmjs.com/package/human-names) package
by Alessandro Minoccheri (MIT licence, see `THIRD_PARTY_NOTICES.md`), lower-cased, de-duplicated and split into
2,456 training and 615 held-out names.

## How it is organised

| Path | What it holds |
|---|---|
| `src/model/` | The tiny transformer: autograd (`tensor.ts`), model, optimizers and LoRA (`gpt.ts`), names dataset (`data.ts`, `names.json`), trained weights (`weights.json`, `weights-2h.json`), saved experiment results (`experiments.json`) |
| `src/engine/scene.ts` | The diagram engine: a step is a list of keyed items, and items that persist between steps animate to their new place |
| `src/lessons/` | One file per lesson: bilingual step text plus a `scene(step)` function |
| `src/course.ts` | The course outline; lessons in `src/lessons/` register themselves |
| `src/main.ts`, `src/style.css` | Page shell, routing (`#/lesson-id/step`), progress saved in `localStorage` |

## Add a lesson

1. Create `src/lessons/<name>.ts` exporting a `Lesson` (see `src/lessons/types.ts`; `tokens.ts` is the simplest example).
2. Give every string both `en` and `zh`.
3. Use the id listed for it in `MODULES` in `src/course.ts`; it is picked up automatically.
4. Run `npm test`, then `node scripts/walk.cjs http://localhost:5173/ <id> <folder>` with the dev server running to screenshot every step.

## If you retrain

Most numbers in the lessons are computed when the page runs, but some lesson text quotes
results directly (for example the byte-pair merge counts, the fine-tuning results and the
compute totals). If you re-run `npm run train` or `npm run experiments`, walk the lessons
again and update any text that no longer matches the diagrams.
