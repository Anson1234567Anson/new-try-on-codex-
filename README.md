# Transformers, step by step

An interactive course on how transformers work and how AI models are trained,
in English and Traditional Chinese (繁體中文). It assumes no machine-learning background.

Each lesson is one diagram that you step through. The numbers in the diagrams are
not scripted: a real transformer with 3,115 weights runs (and trains) in the browser tab.

## Run it

```bash
npm install
```

```bash
npm run dev
```

Other commands: `npm test` (model tests, including a gradient check),
`npm run build` (static site in `dist/`), `npm run train` (retrain the shipped weights).

## Status

3 of the 25 planned lessons are written: **Tokens**, **Self-Attention, Step by Step**
and **The Training Loop**. The rest appear in the course outline marked "soon".

## How it is organised

| Path | What it holds |
|---|---|
| `src/model/` | The tiny transformer: autograd (`tensor.ts`), model and optimizers (`gpt.ts`), names dataset (`data.ts`), trained weights (`weights.json`) |
| `src/engine/scene.ts` | The diagram engine: a step is a list of keyed items, and items that persist between steps animate to their new place |
| `src/lessons/` | One file per lesson: bilingual step text plus a `scene(step)` function |
| `src/course.ts` | The course outline and the list of written lessons |
| `src/main.ts`, `src/style.css` | Page shell, routing (`#/lesson-id/step`), progress saved in `localStorage` |

## Add a lesson

1. Create `src/lessons/<name>.ts` exporting a `Lesson` (see `src/lessons/types.ts`; `tokens.ts` is the simplest example).
2. Give every string both `en` and `zh`.
3. Register it in `LESSONS` in `src/course.ts`, using the id already listed in `MODULES`.
