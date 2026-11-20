# Review brief: "Transformers, step by step"

You are an independent reviewer. Do NOT edit any file in the repository. Your only output is a written report file (path given in your assignment) plus a short summary in your final message.

## What the course is

A free web course, in English and Traditional Chinese (Taiwan), that teaches how transformers work and how AI models are trained. The intended learner is a Taiwanese adult with **no machine-learning background** who can read basic code. Each lesson is one diagram the learner steps through (7–10 steps); every step has a short text panel on the right. The numbers in diagrams come from a tiny real transformer (it learns to write English first names) that runs in the browser, or from real training runs saved in `src/model/experiments.json`. The goal of this review: **make sure the course can really explain hard concepts to that learner**, and find where interactive illustrations (things the learner can drag, type or toggle, not just step through) would explain a hard idea far better.

Repository: /home/user/new-try-on-codex- (Vite + TypeScript). Useful files:
- `src/lessons/<id>.ts`: one lesson each (text in `en`/`zh`, and `scene(step)` builds the diagram).
- `src/course.ts`: the outline, in order. `src/engine/scene.ts`: the diagram primitives available today (token chip, number strip, heat grid, text, arrow, box, line chart, bar chart). There are currently **no** sliders, inputs or draggable elements: the only interaction is stepping (Back/Next, arrow keys), Replay, Pause, a language switch.
- `src/model/`: the real model and data.

## How to review

1. Start a dev server on your assigned port: `cd /home/user/new-try-on-codex- && npx vite --port <PORT> --strictPort` (in the background).
2. For each of your lessons, screenshot every step in both languages:
   `node scripts/walk.cjs http://localhost:<PORT>/ <lesson-id> <your-scratch-dir> en` (and `zh`). Lessons with live training (`training-loop`, `train-live`) need a longer wait: add `9000` as the 5th argument.
3. **Look at the screenshots** with the Read tool AND read the lesson source, step by step, as the beginner would. Ask yourself at every step: would someone with no ML background understand this? Is a term used before it is explained? Is the diagram actually showing what the text says? Is there a leap the learner cannot make? Is the "why" there, not just the "what"?
4. Check correctness: technical claims, numbers, and outside facts (verify any claim about real models with WebSearch if in doubt).
5. Check the Traditional Chinese: natural Taiwan usage, consistent terms, no Mainland vocabulary, no mistranslations that change meaning.
6. Stop your dev server when done.

## Report format (Markdown, in English)

Start with a one-paragraph overall verdict for your lessons. Then, for each lesson:

- **Clarity for a beginner**: score 1–5, one sentence why.
- **Problems**: a numbered list. Each item names the step number, quotes the exact text or describes the diagram element, says what is wrong (error / confusing / jargon before definition / diagram does not match text / missing intuition / Chinese issue / layout), and gives a concrete fix (the replacement wording, or the diagram change). Order by severity, and mark each as **[error]**, **[confusing]**, **[chinese]**, **[layout]** or **[polish]**.
- **Interactive illustration idea** (only where it would clearly help a hard concept): what the learner manipulates, what updates live, which real model function computes it, and why it beats stepping. Be concrete enough that a developer could build it. Skip this for lessons that do not need it.

End with a section **Top 5 across your lessons**: the five changes that would most improve understanding, in priority order.

Be specific and honest. Do not pad with praise. If a lesson is good, say so in one line and move on. Do not invent problems.
