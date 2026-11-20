# Brief: writing lessons for "Transformers, step by step"

Repository: /home/user/new-try-on-codex- (Vite + TypeScript, no framework). Read these first:
- `src/lessons/types.ts`: the `Lesson` / `Step` / `Scene` shapes.
- `src/lessons/tokens.ts` (simplest), `src/lessons/attention.ts` (reads the real model), `src/lessons/training.ts` (trains live with `tick`/`reset`). Copy their style closely.
- `src/engine/scene.ts`: the diagram item types you can draw (token, vec, heat, text, arrow, box, chart, bars).
- `src/model/gpt.ts`, `src/model/data.ts`, `src/model/tensor.ts`: the real model you compute with.
- `src/course.ts`: the outline (`MODULES`). Your lesson id MUST equal the id listed there. Lessons register themselves: any exported `Lesson` in `src/lessons/*.ts` is picked up, so you never edit `course.ts`.

## Who it is for, and the teaching style

The learner is from Taiwan, has **no machine-learning background**, and reads both English and Traditional Chinese. The course copies the teaching style of an interactive system-design course:
- One lesson = one diagram that **evolves** across 7–10 steps. Items keep their key `k` between steps, so they animate to new places instead of being redrawn. Build the picture up a piece at a time.
- Each step has a chapter tag. Open with a concrete **problem** (1 step), add **technique** steps one idea at a time, reach the **solution**, often add one or two **problem** "Case:" steps that break or stress the idea, and end with a **recap** (the last step MUST be `chapter: 'recap'`).
- Per step: a short title, **at most 2 short paragraphs** (2–3 sentences each), optional one callout (`insight` = Key idea, `note` = Good to know, `next` = Coming up, used only in the recap), and an optional "What changed" list (`+` added, `~` changed).
- 3–5 metric tiles above the diagram that update from step to step and show the consequence of each idea with a real number.
- One running example per lesson with concrete numbers. Explain every new term the first time, in plain words. No equations beyond simple arithmetic unless shown with real numbers. Prefer "what it does and why" over notation.
- The recap's `next` callout names the next lesson in the outline order (given in your assignment) and teases it in one sentence.
- 4–5 "You'll learn" items, each ticked at the step where it is taught (`step` is 0-based).

## Honesty rules (important)

- Numbers shown must be **computed**: from the shipped trained model, from live computation in the browser, or from `src/model/experiments.json` (real training runs). Never type in a number you did not compute, except well-established outside facts.
- If you state facts about real-world models (vocabulary sizes, parameter counts, formats, rules of thumb), keep them few and well-known, and **verify each with WebSearch** before using it. List them with their source in your final report. If you cannot verify a fact, leave it out or phrase it generally ("tens of thousands").
- Anything illustrative (not computed) must say so in the text ("illustration", 「示意」).
- When an experiment shows a small or no effect on our tiny task, say so honestly and explain why (tiny model, tiny task), rather than overselling.

## Traditional Chinese (Taiwan)

Write natural Taiwan Mandarin, not a word-for-word translation and not Mainland usage (use 資料 not 数据/數據 for data, 程式, 記憶體, 位元, 網路, 預設, 品質). Use 「」 for quotes in Chinese text. Use this glossary consistently:

token 詞元 · vocabulary 詞彙表 · id 編號 · embedding 嵌入 · vector 向量 · matrix 矩陣 · weight 權重 · parameter 參數 · attention 注意力 · query 查詢 · key 鍵 · value 值 · head 頭（多頭注意力 multi-head attention）· position 位置 · residual 殘差（residual stream 殘差流）· layer normalization 層正規化 · MLP 多層感知器（MLP）· neuron 神經元 · activation 激活值 · logits logits · probability 機率 · softmax softmax · temperature 溫度 · sampling 取樣 · loss 損失 · gradient 梯度 · backpropagation 反向傳播 · chain rule 連鎖律 · optimizer 最佳化器 · learning rate 學習率 · warmup 暖身 · schedule 排程 · batch 批次 · iteration 迭代 · training set 訓練集 · validation set 驗證集 · overfitting 過度擬合 · checkpoint 檢查點 · perplexity 困惑度 · pretraining 預訓練 · fine-tuning 微調 · dataset 資料集 · tokenizer 分詞器 · floating point 浮點數 · mixed precision 混合精度 · GPU GPU · rank 秩 · layer 層 · block 區塊 · causal mask 因果遮罩 · dot product 內積

## The model and data you can use

- `src/model/weights.json`: the trained 1-layer, 1-head model (3,115 weights, d=16, MLP hidden 32, context 8). Load with `GPT.fromJSON(weights as Weights)`. Trained only on `TRAIN_NAMES`. Held-out loss ≈ 2.21.
- `src/model/weights-2h.json`: the same size but **2 heads** (for the multi-head lesson).
- `GPT` API: `forward(ids)` returns a `Trace` with every intermediate (`tokEmb`, `posEmb`, `x`, `q/k/v/scores/att/mixed` per head, `attOut`, `x1`, `hidden` (MLP after ReLU), `mlpOut`, `x2`, `final`, `logits`, plus `blocks[]`). `next(ids, temperature)` gives next-token probabilities. `sample(rand, temperature, topK)`. `loss(batch, train)`, `gradNorm()`, `paramCount()`, `clone()`, `addLoRA(names, rank, rand)`, `p` = weights by name (`tok`, `pos`, `q0`, `k0`, `v0`, `o`, `w1`, `b1`, `w2`, `b2`, `u`, `bu`; 2-head adds `q1`, `k1`, `v1`). Optimizers `sgdStep`, `Adam`, schedule `warmupCosine`. Matrices are `Mat` with `.d` (values), `.g` (gradients), `.row(i)`.
- `src/model/data.ts`: `NAMES` (3,071), `TRAIN_NAMES` (2,456), `VAL_NAMES` (615), `CHARS` (`.` then a–z, `.` is id 0 = start/end), `encode`, `decode`, `example(name)` → `{x, y}`, `rng(seed)` (always use seeded randomness so screens are repeatable).
- `src/model/experiments.json` (real runs; arrays are logged every `every` steps, index 0 = before training; `train` = average training loss over that interval, `val` = held-out loss; `samples` = 8 names written at that step):
  - `optimizers.{sgd (lr 0.5), adam (lr 0.01)}`: 1,500 steps, batch 32.
  - `schedules.{low (0.003), high (0.05, diverges), warmupCosine (peak 0.05, 200 warmup steps)}`: 2,000 steps, with `lr` arrays.
  - `variants.{base, noPos, heads2, layers2, layers2norm}`: 1,500 steps each, Adam 0.01.
  - `overfit`: trained on just the 30 names in `overfit.names` (MLP hidden 64), 1,500 steps logged every 10; memorizes them (train → 0.55, val → 12).
  - `main`: 4,000 steps, Adam with warmup (200) + cosine decay (peak 0.02), batch 32, logged every 50, with `samples` at steps 0, 100, 250, 500, 1000, 2000, 4000 and `gradNorm`.
  - `finetune`: start from the shipped model, fine-tune on 40 names ending in "a" (`trainNames`), check on 122 held-out names ending in "a". `full` (all 3,115 weights, lr 0.003), `lora` (rank 2 on q0,k0,v0,o,w1,w2,u: 534 trainable weights), `scratch` (new model trained only on those 40). 80 steps logged every 4. `endsA` = share of 400 samples ending in "a" (base model: `baseEndsA` 0.198). Best val: full ≈ 1.88 at step 12, LoRA ≈ 1.91, scratch ≈ 2.12; all rise later (overfitting). `split` gives counts.
- Do NOT rerun `scripts/train.ts` or `scripts/experiments.ts`, and do not edit shared files (`src/model/*`, `src/engine/*`, `src/course.ts`, `src/main.ts`, `src/style.css`, `src/i18n.ts`, tests, existing lessons). If you truly need a new helper, put it in a new file named after your lesson (e.g. `src/lessons/data-pipeline.bpe.ts` must NOT export a Lesson-shaped object; or put helpers inside your lesson file). If you believe a shared change is needed, describe it in your report instead.

## Technical rules

- Diagram area is **900 × 520** units. Keep every item inside it (the walk script flags anything outside). Token chips are 44×44 (`TOKEN`), vec cells default 11 wide, heat cells default 52. Text sizes 13–17 for labels; 20+ only for headline numbers. Leave margins of ~20.
- Item keys must be unique within a step. Reuse the same key across steps for "the same thing" so it animates.
- Tones: `ink` (default), `muted`, `a` (teal: the thing we follow / correct answer), `b` (amber: special/secondary), `good`, `bad` (red: failure). Legend tones: `pos`, `neg`, `heat`, `a`, `b`, `bad`.
- Charts: `{ t: 'chart', x, y, w, h, series: [{pts, tone}], ymin, ymax, xmax, every, xlabel, ylabel, guide? }`. Point i is drawn at x = i × every. Non-finite values are clamped to the top.
- All visible strings go through `t({ en, zh })` (diagram labels too). Step text may use `<b>`, `<i>`, `<code>`, `<br>`.
- Compute lazily and memoize (e.g. `let cache; const data = () => cache ??= compute();`) so the page does not do heavy work at start-up for lessons the learner has not opened. Module-level work must stay tiny.
- A lesson that computes over time (live training) uses `tick(step)` (do ≤10 ms of work, return true while more remains) and `reset()`; follow `training.ts`.
- Metric `value` strings: keep under ~14 characters.

## Check your work (required)

1. `cd /home/user/new-try-on-codex- && npx vitest run tests/lessons.test.ts` must pass for your lessons (it draws every step in both languages, checks keys are unique, text exists in both languages, last step is a recap).
2. `npx tsc --noEmit`: fix every error in YOUR files (others may be mid-edit; ignore errors in files you do not own).
3. Start a dev server on YOUR assigned port: `npx vite --port <PORT> --strictPort` (run it in the background), then for each lesson and each language:
   `node scripts/walk.cjs http://localhost:<PORT>/ <lesson-id> <scratch-dir>/shots en` and `... zh`.
   Open the screenshots with the Read tool and actually look at them: overlapping text, items outside the diagram, unreadable or empty diagrams, Chinese text that overflows its box. Fix and re-check. Stop your dev server when done.
4. Final report (keep it short): lessons written (id, step count), every outside fact used with its source URL, anything you could not do, and any shared-file change you think is needed.
