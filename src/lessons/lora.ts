import type { Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { rng } from '../model/data';
import weights from '../model/weights.json';
import experiments from '../model/experiments.json';
import { ADAPTED, FT_SETS, finish, newRun, slice, type FtRun } from './fine-tuning.run'; // ADAPTED: the weights the LoRA run adapted

// LoRA on the course model: the real shapes of its matrices, and the real fine-tuning runs
// (scripts/experiments.ts) with LoRA rank 2 versus changing every weight.

const FT = experiments.finetune;
const RANK = FT.lora.rank;
const STEPS = (FT.full.val.length - 1) * FT.full.every;

interface Shapes { rows: { name: string; r: number; c: number }[]; total: number; lora: number; B0: number[][] }
function measure(): Shapes {
  const m = GPT.fromJSON(weights as Weights);
  const total = m.paramCount();
  m.addLoRA(ADAPTED, RANK, rng(3));
  const B = m.lora.u.B;
  return {
    rows: ADAPTED.map((name) => ({ name, r: m.p[name].r, c: m.p[name].c })),
    total, lora: m.loraCount(),
    B0: Array.from({ length: B.r }, (_, i) => B.row(i)),
  };
}
let cache: Shapes | undefined;
const shapes = () => (cache ??= measure());

const fmt = (n: number) => n.toLocaleString('en-US');
const pct = (v: number) => (v * 100).toFixed(1) + '%';
const best = (v: number[]) => Math.min(...v);
const BIG = 4096, BIG_R = 8; // an example size, not our model
const U = { r: 16, c: 27 }; // the output matrix u, checked against the model in the shapes table
const PX = 10; // pixels per row/column when drawing u
const LR = FT.lora.lr[0];
const bestOf = (v: number[], every: number) => { const m = Math.min(...v); return { loss: m, step: v.indexOf(m) * every }; };
const LBEST = bestOf(FT.lora.val, FT.lora.every), FBEST = bestOf(FT.full.val, FT.full.every);

// The experiment's rank-2 adapter at its best step, replayed in the browser: the real A, B and A·B of u.
interface Real { A: number[][]; B: number[][]; dW: number[][]; hi: number }
let realCache: Real | undefined;
const REAL = () => realCache ??= (() => {
  const r = finish(newRun(RANK, LR), LBEST.step, FT.lora.every);
  const { A, B } = r.model.lora.u;
  const rowsOf = (m: { r: number; row(i: number): number[] }) => Array.from({ length: m.r }, (_, i) => m.row(i));
  const a = rowsOf(A), b = rowsOf(B);
  const dW = a.map((ar) => b[0].map((_, j) => ar.reduce((s, v, k) => s + v * b[k][j], 0)));
  const norms = dW.map((row) => Math.hypot(...row));
  return { A: a, B: b, dW, hi: norms.indexOf(Math.max(...norms)) };
})();

// The rank slider: adapters of any rank, fine-tuned live on the 40 names (same setup as the experiment).
const LIVE_STEPS = 20, LIVE_EVERY = 4, TRY_STEP = 7;
const runs = new Map<number, FtRun>();
const liveRun = (rank: number) => { let r = runs.get(rank); if (!r) { r = newRun(rank, LR); runs.set(rank, r); } return r; };
const rankOf = (v: unknown) => Math.min(8, Math.max(1, Math.round(Number(v) || RANK)));
const loraSize = (rank: number) => rank * ADAPTED.reduce((s, n) => s + (FT_SETS().base.p[n].r + FT_SETS().base.p[n].c), 0);

export const lora: Lesson = {
  id: 'lora',
  title: { en: 'LoRA', zh: 'LoRA（低秩調適）' },
  blurb: {
    en: 'Fine-tune by learning a small correction made of two thin matrices, and leave the big model untouched.',
    zh: '只學一個由兩個細長矩陣組成的小修正來微調，大模型本身完全不動。',
  },
  minutes: 11,
  level: 'intermediate',
  learn: [
    { text: { en: 'Why changing every weight is costly', zh: '為什麼改動每個權重代價很高' }, step: 0 },
    { text: { en: 'How A·B makes a low-rank correction', zh: 'A·B 如何組成低秩修正' }, step: 3 },
    { text: { en: 'Why the adapter starts at zero', zh: '為什麼轉接器一開始是零' }, step: 4 },
    { text: { en: 'How close LoRA gets to full fine-tuning', zh: 'LoRA 和完整微調的效果有多接近' }, step: 6 },
    { text: { en: 'What the rank changes', zh: '秩改變了什麼' }, step: 7 },
    { text: { en: 'Why it matters at large scale', zh: '為什麼它在大規模時特別重要' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Fine-tuning changes every weight', zh: '微調會改動每一個權重' },
      body: [
        { en: 'In the last lesson, fine-tuning on 40 names ending in “a” changed all 3,115 weights. The result is a whole second model.', zh: '上一課用 40 個以「a」結尾的名字做微調，改動了全部 3,115 個權重，結果就是一整個新模型。' },
        { en: 'For our tiny model that is fine. For a model with billions of weights, every task would need its own full copy, plus gradients and Adam’s averages for every weight while training.', zh: '對我們的小模型來說沒問題。但如果模型有數十億個權重，每個任務都得有自己的一整份副本，訓練時每個權重還要存梯度和 Adam 的平均值。' },
      ],
      callout: { kind: 'insight', text: { en: 'Idea: the change needed for one small task is probably simple. Turning “names in general” into “names ending in a” is mostly one nudge (raise “a” before the end mark), not 3,115 independent changes. Maybe we can describe it with far fewer numbers.', zh: '想法：一個小任務所需要的改變，可能很單純。把「一般的名字」變成「以 a 結尾的名字」，主要就是一個方向的推動（在結尾前提高「a」），而不是 3,115 個各自獨立的改變。也許可以用少得多的數字來描述它。' } },
    },
    {
      short: { en: 'One matrix', zh: '一個矩陣' }, chapter: 'technique',
      title: { en: 'Look at one weight matrix', zh: '看其中一個權重矩陣' },
      body: [
        { en: 'Take the model’s last matrix, <code>u</code>. It turns each 16-number vector into 27 scores, one per token. So it has 16 rows and 27 columns: 16 × 27 = <b>432</b> weights.', zh: '拿模型的最後一個矩陣 <code>u</code> 來看。它把每個 16 個數字的向量變成 27 個分數，每個詞元一個。所以它有 16 列、27 欄：16 × 27 = <b>432</b> 個權重。' },
        { en: 'Full fine-tuning nudges all 432 of them, and does the same for every other matrix.', zh: '完整微調會調整這全部 432 個權重，其他每個矩陣也一樣。' },
      ],
    },
    {
      short: { en: 'Correction', zh: '修正量' }, chapter: 'technique',
      title: { en: 'Freeze W, learn a correction', zh: '凍結 W，學一個修正量' },
      body: [
        { en: 'Write the fine-tuned matrix as the original plus a change: <b>W + ΔW</b> (ΔW, “delta W”, means “the change in W”). We freeze W: it is never modified.', zh: '把微調後的矩陣寫成「原本的加上一個改變」：<b>W + ΔW</b>（ΔW 唸作 delta W，意思是 W 的改變量）。我們把 W 凍結，永遠不修改它。' },
        { en: 'So far nothing is saved: ΔW is also 16 × 27, 432 numbers to learn.', zh: '但到目前為止什麼都沒省到：ΔW 也是 16 × 27，還是要學 432 個數字。' },
      ],
      changed: [{ sign: '~', text: { en: 'W frozen', zh: 'W 凍結' } }, { sign: '+', text: { en: 'ΔW: 432 numbers', zh: 'ΔW：432 個數字' } }],
    },
    {
      short: { en: 'A · B', zh: 'A · B' }, chapter: 'technique',
      title: { en: 'Build the correction from two thin matrices', zh: '用兩個細長矩陣組出修正量' },
      body: [
        { en: `<b>LoRA</b> (low-rank adaptation) builds ΔW as a product <b>A · B</b>. A is 16 × ${RANK} and B is ${RANK} × 27; multiplied, they give a full 16 × 27 matrix, made from only 16 × ${RANK} + ${RANK} × 27 = <b>${16 * RANK + RANK * 27}</b> numbers instead of 432.`, zh: `<b>LoRA</b>（low-rank adaptation，低秩調適）把 ΔW 做成乘積 <b>A · B</b>。A 是 16 × ${RANK}，B 是 ${RANK} × 27；兩者相乘，就得到完整的 16 × 27 矩陣，卻只用了 16 × ${RANK} + ${RANK} × 27 = <b>${16 * RANK + RANK * 27}</b> 個數字，而不是 432 個。` },
        { en: `These are the real A, B and A · B of <code>u</code> after ${LBEST.step} steps of the LoRA run, replayed in your browser. Every row of ΔW is a mix of just ${RANK} patterns, the rows of B: the highlighted row is a₁ × (B row 1) + a₂ × (B row 2), with a₁, a₂ from the same row of A. That middle size, ${RANK}, is the <b>rank</b>.`, zh: `這是 LoRA 訓練跑了 ${LBEST.step} 步之後，<code>u</code> 真實的 A、B 和 A · B（就在你的瀏覽器裡重跑）。ΔW 的每一列，都只是 ${RANK} 個圖樣（B 的各列）的混合：標出的那一列 = a₁ × （B 的第 1 列）+ a₂ × （B 的第 2 列），a₁、a₂ 來自 A 的同一列。中間那個尺寸 ${RANK}，就是<b>秩（rank）</b>。` },
      ],
      callout: { kind: 'note', text: { en: 'A low rank limits which changes are possible. The bet is that a small task needs only a simple change.', zh: '秩很低，代表能做的改變有限制。LoRA 賭的是：小任務只需要簡單的改變。' } },
      changed: [{ sign: '~', text: { en: `ΔW = A · B: 432 → ${16 * RANK + RANK * 27} numbers`, zh: `ΔW = A · B：432 → ${16 * RANK + RANK * 27} 個數字` } }],
    },
    {
      short: { en: 'Start at zero', zh: '從零開始' }, chapter: 'technique',
      title: { en: 'Start with no change at all', zh: '一開始完全不改變' },
      body: [
        { en: 'A starts with small random numbers, but B starts at exactly <b>zero</b>. Anything times zero is zero, so at first A · B = 0 and the model is exactly the pretrained one.', zh: 'A 一開始是小小的隨機數，B 則剛好從<b>零</b>開始。任何數乘以零都是零，所以一開始 A · B = 0，模型就和預訓練的一模一樣。' },
        { en: `You can check it in the real runs: before any training, the LoRA model’s held-out loss (on the held-out names ending in “a”) is ${FT.lora.val[0].toFixed(3)}, identical to the pretrained model’s ${FT.baseValLoss.toFixed(3)}. Training then grows the correction from nothing.`, zh: `真實的訓練紀錄可以驗證：訓練之前，LoRA 模型的驗證損失（在驗證集中以「a」結尾的名字上的損失）是 ${FT.lora.val[0].toFixed(3)}，和預訓練模型的 ${FT.baseValLoss.toFixed(3)} 完全相同。接著，訓練會讓修正量從無到有慢慢長出來。` },
      ],
      callout: { kind: 'insight', text: { en: 'Why not start both at zero? A’s gradient is multiplied by B, and B’s by A. With both at zero, both gradients would be zero and neither could ever move. One random, one zero: the output starts unchanged, but training can still get going.', zh: '為什麼不兩個都從零開始？A 的梯度要乘上 B，B 的梯度要乘上 A。如果兩個都是零，兩邊的梯度都是零，誰都動不了。一個隨機、一個為零：輸出一開始完全不變，訓練卻還是能啟動。' } },
      changed: [{ sign: '+', text: { en: 'B = 0 at the start', zh: '一開始 B = 0' } }],
    },
    {
      short: { en: 'Whole model', zh: '整個模型' }, chapter: 'technique',
      title: { en: 'An adapter on seven matrices', zh: '在七個矩陣上加轉接器' },
      body: [
        { en: `Our LoRA run adds a rank-${RANK} adapter to seven matrices: the attention matrices <code>q0</code>, <code>k0</code>, <code>v0</code>, <code>o</code>, the MLP’s <code>w1</code>, <code>w2</code>, and the output <code>u</code>. The embeddings and biases stay frozen.`, zh: `我們的 LoRA 訓練在七個矩陣上加了秩 ${RANK} 的轉接器（adapter）：注意力的 <code>q0</code>、<code>k0</code>、<code>v0</code>、<code>o</code>，MLP 的 <code>w1</code>、<code>w2</code>，以及輸出的 <code>u</code>。嵌入和偏差都保持凍結。` },
        { en: 'Adding up the table: <b>534</b> trainable numbers instead of 3,115, about 17% of the model.', zh: '把表格加總：要訓練的數字只有 <b>534</b> 個，而不是 3,115 個，大約是模型的 17%。' },
      ],
      callout: { kind: 'note', text: { en: 'In real LoRA code you will see W + A · B × (α / r), where α is a fixed number chosen by hand. Our runs set α = r, so the factor is just 1.', zh: '在真實的 LoRA 程式碼裡，你會看到 W + A · B × (α / r)，其中 α 是人工選定的固定數字。我們的訓練設 α = r，所以這個係數就是 1。' } },
      changed: [{ sign: '~', text: { en: 'Trainable: 3,115 → 534', zh: '要訓練的：3,115 → 534' } }],
    },
    {
      short: { en: 'Result', zh: '結果' }, chapter: 'solution',
      title: { en: 'Nearly the same result', zh: '結果幾乎一樣' },
      body: [
        { en: `On the same 40 names, LoRA reaches its best held-out loss, <b>${LBEST.loss.toFixed(2)}</b>, at step ${LBEST.step}; changing every weight reaches ${FBEST.loss.toFixed(2)} at step ${FBEST.step}. A tiny bit worse, with 17% of the trainable numbers, and both far better than training from scratch (${best(FT.scratch.val).toFixed(2)}).`, zh: `用同樣的 40 個名字，LoRA 在第 ${LBEST.step} 步達到最好的驗證損失 <b>${LBEST.loss.toFixed(2)}</b>；改動全部權重則在第 ${FBEST.step} 步達到 ${FBEST.loss.toFixed(2)}。只差一點點，而要訓練的數字只有 17%，兩者也都遠比從頭訓練（${best(FT.scratch.val).toFixed(2)}）好。` },
        { en: `Both runs went on to step ${STEPS} and both over-trained: the lines climb back above the dashed pretrained line. At that point ${pct(FT.lora.endsA)} of LoRA’s names end in “a” (full fine-tuning: ${pct(FT.full.endsA)}), partly by copying the 40 names.`, zh: `兩次訓練都繼續跑到第 ${STEPS} 步，也都練過頭了：線又爬回預訓練的虛線之上。那時 LoRA 寫的名字有 ${pct(FT.lora.endsA)} 以「a」結尾（完整微調：${pct(FT.full.endsA)}），有一部分是照抄那 40 個名字。` },
      ],
      callout: { kind: 'note', text: { en: 'LoRA used a higher learning rate (0.02 vs 0.003), which is common: the adapter starts from zero and has fewer numbers to move.', zh: 'LoRA 用了較高的學習率（0.02 對 0.003），這很常見：轉接器從零開始，要移動的數字也比較少。' } },
    },
    {
      short: { en: 'Try a rank', zh: '試試秩' }, chapter: 'technique',
      title: { en: 'Try it: pick the rank yourself', zh: '動手試試：自己挑秩' },
      body: [
        { en: `Try it: drag the rank from 1 to 8 and watch the trainable count, the held-out curve and ΔW of <code>u</code>. Each new rank trains fresh adapters on the same seven matrices, right here: ${LIVE_STEPS} steps on the 40 names, learning rate ${LR}, like the experiment.`, zh: `動手試試：把秩從 1 拖到 8，看看要訓練的數字、驗證損失曲線和 <code>u</code> 的 ΔW 怎麼變。每換一個秩，就在同樣七個矩陣上、就在這裡訓練一組新的轉接器：用那 40 個名字跑 ${LIVE_STEPS} 步，學習率 ${LR}，和實驗一樣。` },
        { en: 'A higher rank means more numbers to train and more kinds of change ΔW can make. Notice that on this small task even rank 1 gets close to full fine-tuning, and high ranks overfit sooner: their curve turns up earlier. At rank 1, every row of ΔW is the same pattern, only scaled.', zh: '秩越高，要訓練的數字越多，ΔW 能做的改變也越多樣。注意：在這個小任務上，連秩 1 都很接近完整微調；秩很高時反而更早過度擬合，曲線更早往上翹。秩 1 時，ΔW 的每一列都是同一個圖樣，只是放大縮小。' },
      ],
      callout: { kind: 'note', text: { en: 'The gray line is full fine-tuning from the experiment; the dashed line is the pretrained model. Ranks you already tried are kept, so switching back is instant.', zh: '灰線是實驗中的完整微調；虛線是預訓練模型。試過的秩會保留下來，切回去馬上就有結果。' } },
    },
    {
      short: { en: 'Case: scale', zh: '案例：規模' }, chapter: 'problem',
      title: { en: 'Case: the saving grows with size', zh: '案例：規模越大，省得越多' },
      body: [
        { en: `In our tiny model the saving is modest (534 vs 3,115). But a full matrix grows like rows × columns, while A and B grow like rank × (rows + columns). Take an example ${BIG} × ${BIG} matrix: ${fmt(BIG * BIG)} weights. Rank ${BIG_R}: ${fmt(BIG_R * (BIG + BIG))}, just ${(100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2)}%.`, zh: `在我們的小模型裡，省下的不算多（534 對 3,115）。但完整矩陣的大小是「列 × 欄」，A 和 B 則是「秩 × (列 + 欄)」。舉一個 ${BIG} × ${BIG} 的矩陣為例：${fmt(BIG * BIG)} 個權重。秩 ${BIG_R} 只要 ${fmt(BIG_R * (BIG + BIG))} 個，僅佔 ${(100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2)}%。` },
        { en: 'The LoRA paper (Hu and colleagues, 2021) reports that for GPT-3 with 175 billion weights it cut the trainable numbers by 10,000 times and GPU memory by 3 times, with quality on par with full fine-tuning.', zh: 'LoRA 論文（Hu 等人，2021）指出：對 1,750 億權重的 GPT-3，它讓要訓練的數字減少一萬倍，GPU 記憶體減少為三分之一，品質則和完整微調相當。' },
      ],
    },
    {
      short: { en: 'Adapters', zh: '轉接器' }, chapter: 'technique',
      title: { en: 'One base model, many small adapters', zh: '一個基礎模型，許多小轉接器' },
      body: [
        { en: 'Because W never changes, the base model is stored once. Each task keeps only its own A and B. Our “ends in a” adapter is 534 numbers; others can be swapped in when needed.', zh: '因為 W 從不改變，基礎模型只需要存一份，每個任務只保留自己的 A 和 B。我們的「以 a 結尾」轉接器只有 534 個數字；其他轉接器需要時再換上去即可。' },
        { en: 'Training is lighter too: gradients and Adam’s two averages are needed only for trainable numbers. With 4 bytes each: full fine-tuning needs 49,840 bytes, LoRA 21,004.', zh: '訓練也比較輕：只有要訓練的數字才需要梯度和 Adam 的兩個平均值。每個 4 個位元組來算：完整微調需要 49,840 個位元組，LoRA 只要 21,004。' },
      ],
      callout: { kind: 'note', text: { en: 'The other adapters drawn here are an illustration; only the “ends in a” one was trained.', zh: '圖中其他的轉接器只是示意；真正訓練過的只有「以 a 結尾」這一個。' } },
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'A small correction instead of a new model', zh: '用小修正取代新模型' },
      body: [
        { en: `<b>Freeze</b> W, learn ΔW = <b>A · B</b> with a small rank r.<br><b>B starts at 0</b>: training begins from the pretrained model.<br><b>Our model</b>: 534 trainable vs 3,115; best loss ${best(FT.lora.val).toFixed(2)} vs ${best(FT.full.val).toFixed(2)}.<br><b>At scale</b>: ${BIG} × ${BIG} with rank ${BIG_R} is ${(100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2)}% of the weights; one base, many adapters.`, zh: `<b>凍結</b> W，用小的秩 r 學 ΔW = <b>A · B</b>。<br><b>B 從 0 開始</b>：訓練從預訓練模型出發。<br><b>我們的模型</b>：要訓練的 534 個對 3,115 個；最佳驗證損失 ${best(FT.lora.val).toFixed(2)} 對 ${best(FT.full.val).toFixed(2)}。<br><b>大規模時</b>：${BIG} × ${BIG}、秩 ${BIG_R} 只佔權重的 ${(100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2)}%；一個基礎模型，許多轉接器。` },
      ],
      callout: { kind: 'next', text: { en: 'Next: Training Mixture-of-Experts Models. Keeping many experts busy, balanced and spread across machines.', zh: '下一課：訓練混合專家模型。如何讓許多專家都有事做、分工平均，並分散到多台機器上。' } },
    },
  ],

  scene(step, input = {}) {
    const items: Item[] = [];
    const S = shapes();
    let trainable = fmt(S.total), share = '100%', lossV = '—', ends = '—';

    // matrix u drawn to scale, and the pieces of LoRA around it
    const W = { x: 120, y: 150, w: U.c * PX, h: U.r * PX };
    const drawW = (frozen: boolean, x = W.x) => {
      items.push({ k: 'W', t: 'box', x, y: W.y, w: W.w, h: W.h, label: t({ en: frozen ? 'W (frozen)' : 'W = u', zh: frozen ? 'W（凍結）' : 'W = u' }), sub: `16 × 27 = ${U.r * U.c}`, tone: frozen ? 'muted' : 'ink' });
      items.push({ k: 'Wr', t: 'text', x: x - 10, y: W.y + W.h / 2 + 5, text: t({ en: '16 rows', zh: '16 列' }), anchor: 'end', size: 13, tone: 'muted' });
      items.push({ k: 'Wc', t: 'text', x: x + W.w / 2, y: W.y - 10, text: t({ en: '27 columns', zh: '27 欄' }), anchor: 'middle', size: 13, tone: 'muted' });
    };

    if (step === 0) {
      items.push({ k: 'base', t: 'box', x: 80, y: 120, w: 300, h: 120, label: t({ en: 'pretrained model', zh: '預訓練模型' }), sub: t({ en: `${fmt(S.total)} weights`, zh: `${fmt(S.total)} 個權重` }), tone: 'ink' });
      items.push({ k: 'fa', t: 'arrow', x1: 386, y1: 180, x2: 514, y2: 180, tone: 'b' });
      items.push({ k: 'fal', t: 'text', x: 450, y: 166, text: t({ en: 'fine-tune', zh: '微調' }), anchor: 'middle', size: 13, tone: 'muted' });
      items.push({ k: 'copy', t: 'box', x: 520, y: 120, w: 300, h: 120, label: t({ en: 'a whole new model', zh: '一整個新模型' }), sub: t({ en: `all ${fmt(S.total)} weights changed`, zh: `${fmt(S.total)} 個權重全部改變` }), tone: 'b' });
      items.push({ k: 'cost', t: 'text', x: 450, y: 330, text: t({ en: 'every task = one more full copy', zh: '每多一個任務 = 多一整份副本' }), anchor: 'middle', size: 18, bold: true, tone: 'bad' });
      items.push({ k: 'cost2', t: 'text', x: 450, y: 362, text: t({ en: 'and, while training, a gradient and two Adam averages per weight', zh: '而且訓練時，每個權重還要一個梯度和兩個 Adam 平均值' }), anchor: 'middle', size: 14, tone: 'muted' });
    }

    if (step >= 1 && step <= 4 && step !== 3) {
      drawW(step >= 2);
      const r = RANK * PX;
      if (step >= 2) {
        items.push({ k: 'plus', t: 'text', x: W.x + W.w + 30, y: W.y + W.h / 2 + 10, text: '+', size: 30, anchor: 'middle', bold: true });
        const dx = W.x + W.w + 60;
        items.push({ k: 'dW', t: 'box', x: dx + (step >= 3 ? r + 10 : 0), y: W.y, w: W.w, h: W.h, label: step >= 3 ? 'A · B' : 'ΔW', sub: step >= 3 ? t({ en: `16 × 27, from ${16 * RANK + RANK * 27} numbers`, zh: `16 × 27，只用 ${16 * RANK + RANK * 27} 個數字` }) : `16 × 27 = 432`, tone: step >= 3 ? 'muted' : 'a' });
        if (step >= 3) {
          items.push({ k: 'A', t: 'box', x: dx, y: W.y, w: r, h: W.h, tone: 'a' });
          items.push({ k: 'Al', t: 'text', x: dx + r / 2, y: W.y + W.h + 22, text: `A 16×${RANK}`, anchor: 'middle', size: 14, bold: true, tone: 'a' });
          items.push({ k: 'B', t: 'box', x: dx + r + 10, y: W.y - r - 10, w: W.w, h: r, tone: step === 4 ? 'muted' : 'a' });
          items.push({ k: 'Bl', t: 'text', x: dx + r + 10 + W.w + 8, y: W.y - r / 2 - 4, text: `B ${RANK}×27`, size: 14, bold: true, tone: 'a' });
          if (step === 3) items.push({ k: 'cnt', t: 'text', x: 450, y: 420, text: `16 × ${RANK} + ${RANK} × 27 = ${16 * RANK + RANK * 27}   vs   16 × 27 = ${U.r * U.c}`, anchor: 'middle', size: 18, bold: true, mono: true, tone: 'a' });
        }
      }
      if (step === 1) items.push({ k: 'cnt', t: 'text', x: 450, y: 420, text: t({ en: 'turns a 16-number vector into 27 token scores', zh: '把 16 個數字的向量變成 27 個詞元分數' }), anchor: 'middle', size: 16, tone: 'muted' });
      if (step === 2) items.push({ k: 'cnt', t: 'text', x: 450, y: 420, text: t({ en: 'the model uses W + ΔW · only ΔW is trained', zh: '模型使用 W + ΔW · 只訓練 ΔW' }), anchor: 'middle', size: 16, bold: true });
      if (step === 4) {
        S.B0.forEach((row, i) => items.push({ k: 'b0' + i, t: 'vec', x: 120, y: 455 + i * 30, v: row, cell: 12, max: 1, label: i === 0 ? t({ en: 'the real B of u at the start: all zeros', zh: 'u 的 B 一開始的真實數值：全部是 0' }) : undefined }));
        items.push({ k: 'z', t: 'text', x: 450, y: 410, anchor: 'middle', text: t({ en: `loss before training: ${FT.lora.val[0].toFixed(3)} = pretrained ${FT.baseValLoss.toFixed(3)}`, zh: `訓練前的損失：${FT.lora.val[0].toFixed(3)} = 預訓練 ${FT.baseValLoss.toFixed(3)}` }), size: 14, bold: true, tone: 'a' });
      }
      if (step >= 3) { trainable = String(16 * RANK + RANK * 27); share = pct((16 * RANK + RANK * 27) / (U.r * U.c)); }
      else if (step === 2) { trainable = '432'; share = '100%'; }
      else { trainable = '432'; share = '100%'; }
      if (step === 4) lossV = FT.lora.val[0].toFixed(3);
    }

    if (step === 3) {
      // the real rank-2 adapter of u: A (16 × 2), B (2 × 27), and their product ΔW (16 × 27)
      const R = REAL(), C = 15, X = 260, BY = 62, DY = 140, RH = 23;
      const mx = (m: number[][]) => Math.max(1e-9, ...m.flat().map(Math.abs));
      const mA = mx(R.A), mB = mx(R.B), mD = mx(R.dW);
      items.push({ k: 'hiBox', t: 'box', x: 192, y: DY + R.hi * RH - 4, w: X + 27 * C - 192 + 6, h: 30, tone: 'a' });
      [...'.abcdefghijklmnopqrstuvwxyz'].forEach((ch, j) => items.push({ k: 'col' + j, t: 'text', x: X + j * C + 6, y: BY - 12, text: ch, size: 11, mono: true, anchor: 'middle', tone: ch === 'a' || ch === '.' ? 'a' : 'muted', bold: ch === 'a' || ch === '.' }));
      R.B.forEach((row, k) => {
        items.push({ k: 'Bv' + k, t: 'vec', x: X, y: BY + k * 28, v: row, cell: C, max: mB });
        items.push({ k: 'Bn' + k, t: 'text', x: X - 10, y: BY + k * 28 + 16, text: t({ en: `B row ${k + 1}`, zh: `B 第 ${k + 1} 列` }), size: 13, anchor: 'end', tone: 'b', bold: true });
      });
      items.push({ k: 'Al', t: 'text', x: 215, y: DY - 12, text: `A 16×${RANK}`, size: 13, anchor: 'middle', bold: true, tone: 'a' });
      items.push({ k: 'dWl', t: 'text', x: X, y: DY - 12, text: t({ en: 'ΔW = A · B (16 × 27), the change to u', zh: 'ΔW = A · B（16 × 27），u 的改變量' }), size: 13, bold: true });
      R.dW.forEach((row, i) => {
        items.push({ k: 'Av' + i, t: 'vec', x: 200, y: DY + i * RH, v: R.A[i], cell: C, max: mA });
        items.push({ k: 'Dv' + i, t: 'vec', x: X, y: DY + i * RH, v: row, cell: C, max: mD });
      });
      const a = R.A[R.hi], sg = (v: number) => (v < 0 ? `(−${Math.abs(v).toFixed(2)})` : v.toFixed(2));
      const fx = X + 27 * C + 22, fy = DY + R.hi * RH;
      items.push({ k: 'f0', t: 'text', x: fx, y: Math.min(fy, 400) - 4, text: t({ en: `row ${R.hi + 1} of ΔW`, zh: `ΔW 第 ${R.hi + 1} 列` }), size: 13, bold: true, tone: 'a' });
      items.push({ k: 'f1', t: 'text', x: fx, y: Math.min(fy, 400) + 20, text: t({ en: `= ${sg(a[0])} × B row 1`, zh: `= ${sg(a[0])} × B 第 1 列` }), size: 13, mono: true });
      items.push({ k: 'f2', t: 'text', x: fx, y: Math.min(fy, 400) + 42, text: t({ en: `+ ${sg(a[1])} × B row 2`, zh: `+ ${sg(a[1])} × B 第 2 列` }), size: 13, mono: true });
      items.push({ k: 'f3', t: 'text', x: fx, y: 470, text: t({ en: 'colour = sign, shade = size', zh: '顏色 = 正負，深淺 = 大小' }), size: 12, tone: 'muted' });
      trainable = String(16 * RANK + RANK * 27); share = pct((16 * RANK + RANK * 27) / (U.r * U.c));
    }

    if (step === TRY_STEP) {
      const rank = rankOf(input.rank), r = liveRun(rank);
      const n = loraSize(rank), uN = rank * (U.r + U.c);
      const pts = r.val, b = pts.length ? bestOf(pts, LIVE_EVERY) : undefined;
      const ymin = 1.8, ymax = 2.5;
      items.push({
        k: 'chart', t: 'chart', x: 80, y: 50, w: 400, h: 220, ymin, ymax, xmax: LIVE_STEPS, every: LIVE_EVERY,
        series: [{ pts: FT.full.val.slice(0, LIVE_STEPS / FT.full.every + 1), tone: 'muted' }, { pts, tone: 'a' }],
        xlabel: t({ en: 'fine-tuning steps', zh: '微調步數' }), ylabel: t({ en: 'held-out loss, names ending in “a”', zh: '驗證損失（以「a」結尾的名字）' }),
        guide: { y: FT.baseValLoss, label: t({ en: `pretrained · ${FT.baseValLoss.toFixed(2)}`, zh: `預訓練 · ${FT.baseValLoss.toFixed(2)}` }) },
      });
      if (b) {
        const x = 80 + (b.step / LIVE_STEPS) * 400, y = 50 + 220 - ((b.loss - ymin) / (ymax - ymin)) * 220;
        items.push({ k: 'bdot', t: 'dot', x, y, r: 6, tone: 'a' });
      }
      const lines: [string, Tone, boolean][] = [
        [t({ en: `rank ${rank} · for u: 16×${rank} + ${rank}×27 = ${uN}`, zh: `秩 ${rank} · u 的部分：16×${rank} + ${rank}×27 = ${uN}` }), 'ink', false],
        [t({ en: `all 7 matrices: ${fmt(n)} trainable of ${fmt(S.total)} (${pct(n / S.total)})`, zh: `七個矩陣合計：${fmt(S.total)} 個中要訓練 ${fmt(n)} 個（${pct(n / S.total)}）` }), 'a', true],
        [b ? t({ en: `best held-out so far: ${b.loss.toFixed(3)} at step ${b.step}`, zh: `目前最好的驗證損失：第 ${b.step} 步的 ${b.loss.toFixed(3)}` }) : t({ en: 'training…', zh: '訓練中…' }), 'a', true],
        [t({ en: `full fine-tuning: ${FBEST.loss.toFixed(3)} at step ${FBEST.step}, ${fmt(S.total)} trainable`, zh: `完整微調：第 ${FBEST.step} 步 ${FBEST.loss.toFixed(3)}，要訓練 ${fmt(S.total)} 個` }), 'muted', false],
      ];
      lines.forEach(([text, tone, bold], i) => items.push({ k: 'tl' + i, t: 'text', x: 40, y: 345 + i * 30, text, size: 15, tone, bold }));
      const { A, B } = r.model.lora.u;
      const dW = Array.from({ length: A.r }, (_, i) => Array.from({ length: B.c }, (_, j) => { let v = 0; for (let k = 0; k < A.c; k++) v += A.d[i * A.c + k] * B.d[k * B.c + j]; return v; }));
      const mD = Math.max(0.05, ...dW.flat().map(Math.abs));
      items.push({ k: 'dWl', t: 'text', x: 540, y: 44, text: t({ en: `ΔW of u at rank ${rank} (16 × 27)`, zh: `秩 ${rank} 時 u 的 ΔW（16 × 27）` }), size: 13, bold: true });
      [...'.abcdefghijklmnopqrstuvwxyz'].forEach((ch, j) => (ch === '.' || ch === 'a') && items.push({ k: 'col' + j, t: 'text', x: 540 + j * 11 + 5, y: 64, text: ch, size: 12, mono: true, anchor: 'middle', tone: 'a', bold: true }));
      dW.forEach((row, i) => items.push({ k: 'Dv' + i, t: 'vec', x: 540, y: 74 + i * 24, v: row, max: mD }));
      trainable = fmt(n); share = pct(n / S.total); lossV = b ? b.loss.toFixed(3) : '—';
      ends = `${r.step} / ${LIVE_STEPS}`;
    }

    if (step === 5) {
      const cols = [60, 220, 380, 560, 740];
      const head = [t({ en: 'matrix', zh: '矩陣' }), t({ en: 'shape', zh: '形狀' }), t({ en: 'full', zh: '完整' }), t({ en: `rank ${RANK}: A + B`, zh: `秩 ${RANK}：A + B` }), t({ en: 'share', zh: '比例' })];
      head.forEach((h, i) => items.push({ k: 'h' + i, t: 'text', x: cols[i], y: 50, text: h, size: 14, bold: true, tone: 'muted' }));
      let sumFull = 0, sumL = 0;
      S.rows.forEach((r, i) => {
        const full = r.r * r.c, l = RANK * (r.r + r.c);
        sumFull += full; sumL += l;
        const y = 82 + i * 40;
        const cells = [r.name, `${r.r} × ${r.c}`, fmt(full), `${r.r}×${RANK} + ${RANK}×${r.c} = ${l}`, pct(l / full)];
        cells.forEach((c, j) => items.push({ k: `c${i}-${j}`, t: 'text', x: cols[j], y, text: c, size: 15, mono: true, tone: j === 3 ? 'a' : 'ink', bold: j === 0 }));
      });
      const y = 82 + S.rows.length * 40 + 20;
      items.push({ k: 'sumbox', t: 'box', x: 40, y: y - 30, w: 820, h: 44, tone: 'a' });
      items.push({ k: 'sum0', t: 'text', x: 60, y, text: t({ en: 'adapters', zh: '轉接器合計' }), size: 15, bold: true });
      items.push({ k: 'sum1', t: 'text', x: 380, y, text: fmt(sumFull), size: 15, mono: true, bold: true });
      items.push({ k: 'sum2', t: 'text', x: 560, y, text: fmt(sumL), size: 15, mono: true, bold: true, tone: 'a' });
      items.push({ k: 'sum3', t: 'text', x: 740, y, text: pct(sumL / sumFull), size: 15, mono: true, bold: true });
      items.push({ k: 'tot', t: 'text', x: 450, y: y + 50, text: t({ en: `trainable ${fmt(S.lora)} of ${fmt(S.total)} weights (the rest stay frozen)`, zh: `要訓練的：${fmt(S.total)} 個權重中的 ${fmt(S.lora)} 個（其餘都凍結）` }), anchor: 'middle', size: 16, bold: true });
      trainable = fmt(S.lora); share = pct(S.lora / S.total);
    }

    if (step === 6) {
      const ymin = 1.5, ymax = 2.7;
      items.push({
        k: 'chart', t: 'chart', x: 80, y: 50, w: 480, h: 250, ymin, ymax, xmax: STEPS, every: FT.full.every,
        series: [{ pts: FT.full.val, tone: 'muted' }, { pts: FT.lora.val, tone: 'a' }],
        xlabel: t({ en: 'fine-tuning steps', zh: '微調步數' }), ylabel: t({ en: 'loss on held-out names ending in “a”', zh: '驗證集中以「a」結尾的名字上的損失' }),
        guide: { y: FT.baseValLoss, label: t({ en: `pretrained · ${FT.baseValLoss.toFixed(2)}`, zh: `預訓練 · ${FT.baseValLoss.toFixed(2)}` }) },
      });
      items.push({ k: 'r1', t: 'box', x: 620, y: 60, w: 250, h: 66, label: t({ en: `LoRA · best ${best(FT.lora.val).toFixed(2)}`, zh: `LoRA · 最佳 ${best(FT.lora.val).toFixed(2)}` }), sub: t({ en: `${fmt(S.lora)} trainable · ${pct(FT.lora.endsA)} end in “a”`, zh: `訓練 ${fmt(S.lora)} 個 · ${pct(FT.lora.endsA)} 以「a」結尾` }), tone: 'a' });
      items.push({ k: 'r2', t: 'box', x: 620, y: 146, w: 250, h: 66, label: t({ en: `full · best ${best(FT.full.val).toFixed(2)}`, zh: `完整微調 · 最佳 ${best(FT.full.val).toFixed(2)}` }), sub: t({ en: `${fmt(S.total)} trainable · ${pct(FT.full.endsA)} end in “a”`, zh: `訓練 ${fmt(S.total)} 個 · ${pct(FT.full.endsA)} 以「a」結尾` }) });
      items.push({ k: 'r3', t: 'box', x: 620, y: 232, w: 250, h: 66, label: t({ en: `scratch · best ${best(FT.scratch.val).toFixed(2)}`, zh: `從頭訓練 · 最佳 ${best(FT.scratch.val).toFixed(2)}` }), sub: t({ en: 'for comparison', zh: '對照用' }), tone: 'b' });
      items.push({ k: 'sl', t: 'text', x: 40, y: 400, text: t({ en: `LoRA model writes (after ${STEPS} steps):`, zh: `LoRA 模型寫出（跑完 ${STEPS} 步）：` }), size: 14, tone: 'muted' });
      FT.lora.samples['80'].forEach((n, i) => items.push({ k: 's' + i, t: 'box', x: 40 + i * 104, y: 410, w: 96, h: 40, label: n, tone: n.endsWith('a') ? 'a' : 'ink' }));
      trainable = fmt(S.lora); share = pct(S.lora / S.total); lossV = best(FT.lora.val).toFixed(2); ends = `${pct(FT.lora.endsA)} @${STEPS}`;
    }

    if (step === 8) {
      // drawn to scale: a 300-pixel square stands for 4096 × 4096
      const side = 300, thin = Math.max(3, (BIG_R / BIG) * side);
      items.push({ k: 'W', t: 'box', x: 80, y: 120, w: side, h: side, label: `${BIG} × ${BIG}`, sub: fmt(BIG * BIG), tone: 'muted' });
      items.push({ k: 'Wc', t: 'text', x: 230, y: 100, text: t({ en: 'example size (not our model)', zh: '舉例的尺寸（不是我們的模型）' }), anchor: 'middle', size: 13, tone: 'muted' });
      items.push({ k: 'A', t: 'box', x: 460, y: 120, w: thin, h: side, tone: 'a' });
      items.push({ k: 'B', t: 'box', x: 480, y: 120, w: side, h: thin, tone: 'a' });
      items.push({ k: 'Al', t: 'text', x: 470, y: 445, text: `A ${BIG}×${BIG_R}`, anchor: 'middle', size: 14, bold: true, tone: 'a' });
      items.push({ k: 'Bl', t: 'text', x: 630, y: 150, text: `B ${BIG_R}×${BIG}`, anchor: 'middle', size: 14, bold: true, tone: 'a' });
      items.push({ k: 'cnt', t: 'text', x: 630, y: 260, text: `${BIG_R} × (${BIG} + ${BIG})`, anchor: 'middle', size: 17, mono: true });
      items.push({ k: 'cnt2', t: 'text', x: 630, y: 296, text: `= ${fmt(BIG_R * 2 * BIG)}`, anchor: 'middle', size: 24, bold: true, tone: 'a' });
      items.push({ k: 'cnt3', t: 'text', x: 630, y: 330, text: t({ en: `${(100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2)}% of ${fmt(BIG * BIG)}`, zh: `只佔 ${fmt(BIG * BIG)} 的 ${(100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2)}%` }), anchor: 'middle', size: 15, tone: 'muted' });
      items.push({ k: 'our', t: 'text', x: 630, y: 400, text: t({ en: `ours, u 16 × 27 at rank 2: ${pct(86 / 432)}`, zh: `我們的 u（16 × 27，秩 2）：${pct(86 / 432)}` }), anchor: 'middle', size: 14, tone: 'muted' });
      trainable = fmt(BIG_R * 2 * BIG); share = (100 * BIG_R * 2 * BIG / (BIG * BIG)).toFixed(2) + '%';
    }

    if (step === 9) {
      const full = S.total * 16; // weight + gradient + two Adam averages, 4 bytes each
      const lo = S.total * 4 + S.lora * 16; // frozen weights, plus adapters with their gradients and averages
      items.push({ k: 'base', t: 'box', x: 330, y: 40, w: 240, h: 80, label: t({ en: 'pretrained model', zh: '預訓練模型' }), sub: t({ en: `stored once · ${fmt(S.total)} weights`, zh: `只存一份 · ${fmt(S.total)} 個權重` }) });
      const ads: { label: string; sub: string; tone: Tone }[] = [
        { label: t({ en: 'ends in “a”', zh: '以「a」結尾' }), sub: t({ en: `${fmt(S.lora)} numbers · trained`, zh: `${fmt(S.lora)} 個數字 · 已訓練` }), tone: 'a' },
        { label: t({ en: 'style 2', zh: '風格 2' }), sub: t({ en: 'illustration', zh: '示意' }), tone: 'muted' },
        { label: t({ en: 'style 3', zh: '風格 3' }), sub: t({ en: 'illustration', zh: '示意' }), tone: 'muted' },
      ];
      ads.forEach((a, i) => {
        items.push({ k: 'ad' + i, t: 'box', x: 90 + i * 260, y: 180, w: 200, h: 64, label: a.label, sub: a.sub, tone: a.tone });
        items.push({ k: 'ada' + i, t: 'arrow', x1: 450, y1: 124, x2: 190 + i * 260, y2: 176, tone: i === 0 ? 'a' : 'muted', dashed: i > 0 });
      });
      items.push({ k: 'mt', t: 'text', x: 90, y: 300, text: t({ en: 'training memory for our model (4 bytes per number)', zh: '我們模型的訓練記憶體（每個數字 4 個位元組）' }), size: 14, tone: 'muted' });
      const scale = 700 / full;
      items.push({ k: 'mf', t: 'box', x: 90, y: 315, w: full * scale, h: 50, label: t({ en: `full: ${fmt(S.total)} × 16 = ${fmt(full)} B`, zh: `完整微調：${fmt(S.total)} × 16 = ${fmt(full)} B` }), tone: 'b' });
      items.push({ k: 'ml', t: 'box', x: 90, y: 385, w: lo * scale, h: 50, label: t({ en: `LoRA: ${fmt(lo)} B`, zh: `LoRA：${fmt(lo)} B` }), tone: 'a' });
      items.push({ k: 'mlx', t: 'text', x: 90 + lo * scale + 12, y: 415, text: t({ en: `${fmt(S.total)} × 4 frozen + ${fmt(S.lora)} × 16`, zh: `凍結的 ${fmt(S.total)} × 4 + ${fmt(S.lora)} × 16` }), size: 14, mono: true, tone: 'muted' });
      trainable = fmt(S.lora); share = pct(S.lora / S.total); lossV = best(FT.lora.val).toFixed(2); ends = `${pct(FT.lora.endsA)} @${STEPS}`;
    }

    if (step === 10) {
      drawW(true, 60);
      const dx = 60 + W.w + 60, r = RANK * PX;
      items.push({ k: 'plus', t: 'text', x: 60 + W.w + 30, y: W.y + W.h / 2 + 10, text: '+', size: 30, anchor: 'middle', bold: true });
      items.push({ k: 'A', t: 'box', x: dx, y: W.y, w: r, h: W.h, tone: 'a' });
      items.push({ k: 'B', t: 'box', x: dx + r + 10, y: W.y - r - 10, w: W.w, h: r, tone: 'a' });
      items.push({ k: 'dW', t: 'box', x: dx + r + 10, y: W.y, w: W.w, h: W.h, label: 'A · B', sub: t({ en: 'learned, starts at 0', zh: '學出來的，從 0 開始' }), tone: 'muted' });
      items.push({ k: 'cnt', t: 'text', x: 450, y: 420, text: t({ en: `${fmt(S.lora)} trainable of ${fmt(S.total)} · best loss ${best(FT.lora.val).toFixed(2)} vs ${best(FT.full.val).toFixed(2)}`, zh: `訓練 ${fmt(S.total)} 個中的 ${fmt(S.lora)} 個 · 最佳損失 ${best(FT.lora.val).toFixed(2)} 對 ${best(FT.full.val).toFixed(2)}` }), anchor: 'middle', size: 16, bold: true, tone: 'a' });
      trainable = fmt(S.lora); share = pct(S.lora / S.total); lossV = best(FT.lora.val).toFixed(2); ends = `${pct(FT.lora.endsA)} @${STEPS}`;
    }

    return {
      metrics: [
        { label: { en: 'Trainable', zh: '要訓練的數字' }, value: trainable },
        { label: { en: 'Share', zh: '比例' }, value: share },
        { label: { en: 'Held-out loss', zh: '驗證損失' }, value: lossV },
        step === TRY_STEP ? { label: { en: 'Steps', zh: '步數' }, value: ends } : { label: { en: 'Ends in “a”', zh: '以「a」結尾' }, value: ends },
      ],
      items,
      legend: step === 6 ? [{ tone: 'a', text: { en: 'LoRA, rank 2 (the gray line is full fine-tuning)', zh: 'LoRA，秩 2（灰線是完整微調）' } }]
        : step === TRY_STEP ? [{ tone: 'a', text: { en: 'LoRA at your rank, trained live (dot = best so far)', zh: '你選的秩，現場訓練的 LoRA（圓點 = 目前最佳）' } }, { tone: 'pos', text: { en: 'ΔW positive', zh: 'ΔW 正數' } }, { tone: 'neg', text: { en: 'negative', zh: '負數' } }]
        : step === 3 ? [{ tone: 'pos', text: { en: 'positive', zh: '正數' } }, { tone: 'neg', text: { en: 'negative', zh: '負數' } }] : undefined,
      controls: step === TRY_STEP ? [{ key: 'rank', kind: 'slider', label: { en: 'Rank r', zh: '秩 r' }, min: 1, max: 8, step: 1, value: RANK }] : undefined,
    };
  },

  tick(step, input) {
    if (step !== TRY_STEP) return false;
    const r = liveRun(rankOf(input.rank));
    const until = performance.now() + 10;
    let more = true;
    while (more && performance.now() < until) more = slice(r, LIVE_STEPS, LIVE_EVERY);
    return more;
  },

  reset() { runs.clear(); },
};
