import type { Lesson, Step } from './types';
import type { Item, Tone } from '../engine/scene';
import { t, type L } from '../i18n';
import { NAMES } from '../model/data';
import experiments from '../model/experiments.json';

// Reading the logs of real training runs (src/model/experiments.json):
// `main` is a healthy 4,000-step run; `schedules.high` is a run that breaks.

interface Log { every: number; train: (number | null)[]; val: (number | null)[]; lr: number[]; gradNorm: number[]; samples: Record<string, string[]>; seconds: number }

const fin = (a: (number | null)[]) => a.map((v) => (v === null ? NaN : v));
const n0 = (v: number) => v.toLocaleString('en-US');
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : 'NaN');

let cache: ReturnType<typeof compute> | undefined;
function compute() {
  const E = experiments as unknown as { main: Log & { steps: number; peakLr: number; warmup: number; batch: number; params: number }; schedules: { high: Log }; overfit: Log };
  const M = E.main, H = E.schedules.high, O = E.overfit;
  const mt = fin(M.train), mv = fin(M.val);
  const at = (a: number[], step: number) => a[step / M.every];
  // gradNorm[i] is the average over the interval that ends at step (i + 1) × every
  const gnAt = (log: Log, step: number) => log.gradNorm[step / log.every - 1];
  const best = Math.min(...mv);
  const bestStep = mv.indexOf(best) * M.every;
  const last = mv.length - 1;
  const lastStep = last * M.every;
  const dropBy500 = (mv[0] - at(mv, 500)) / (mv[0] - mv[last]);
  const gnMed = [...M.gradNorm].sort((a, b) => a - b)[Math.floor(M.gradNorm.length / 2)];
  const mainSpikes = M.gradNorm.filter((g) => g > 3 * gnMed).length;

  const ht = fin(H.train), hv = fin(H.val);
  // the loss "spike": the first log where the training loss is not a number or more than doubles
  const spikeIdx = ht.findIndex((v, i) => i > 0 && (!Number.isFinite(v) || v > 2 * ht[i - 1]));
  const spikeStep = spikeIdx * H.every;
  const hMed = [...H.gradNorm].sort((a, b) => a - b)[Math.floor(H.gradNorm.length / 2)];
  const gnJumpIdx = H.gradNorm.findIndex((g) => g > 10 * Math.min(hMed, H.gradNorm[0]));
  const gnJumpStep = (gnJumpIdx + 1) * H.every;
  const gnBefore = Math.max(...H.gradNorm.slice(0, gnJumpIdx));
  const gnPeak = Math.max(...H.gradNorm);
  // compare both runs at the last step before the spike that both logged
  const cmpStep = Math.floor(((spikeIdx - 1) * H.every) / M.every) * M.every;
  const hvBefore = hv[cmpStep / H.every];
  const mvSame = mv[cmpStep / M.every];
  // a small early hint: the largest gradient norm before the jump (after the first few logs), and its typical size
  const pre = H.gradNorm.slice(4, gnJumpIdx);
  const bumpIdx = 4 + pre.indexOf(Math.max(...pre));
  const bumpStep = (bumpIdx + 1) * H.every;
  const gnTypical = [...pre].sort((a, b) => a - b)[Math.floor(pre.length / 2)];
  const hvBump = hv[bumpIdx + 1];
  const hFinal = hv[hv.length - 1];

  const ot = fin(O.train), ov = fin(O.val);
  return {
    M, H, O, mt, mv, ht, hv, ot, ov, at, gnAt, best, bestStep, last, lastStep, dropBy500, mainSpikes,
    gap: (step: number) => at(mv, step) - at(mt, step),
    spikeIdx, spikeStep, gnJumpStep, gnBefore, gnPeak, hvBefore, mvSame, hFinal, cmpStep, bumpStep, gnBump: H.gradNorm[bumpIdx], gnTypical, hvBump,
    gnMin: Math.min(...M.gradNorm), gnMax: Math.max(...M.gradNorm),
    oBest: Math.min(...ov), oBestStep: ov.indexOf(Math.min(...ov)) * O.every, oTrainEnd: ot[ot.length - 1], oValEnd: ov[ov.length - 1], oN: (O as unknown as { names: string[] }).names.length,
    checkpoints: Object.keys(M.samples).map(Number).sort((a, b) => a - b),
  };
}
const D = () => cache ??= compute();

// chart helpers
const lrChart = (k: string, x: number, y: number, w: number, h: number, series: { pts: number[]; tone: Tone }[], ymax = 20): Item => ({
  k, t: 'chart', x, y, w, h, ymin: 0, ymax, xmax: D().M.steps, every: D().M.every, series,
  xlabel: t({ en: 'step', zh: '步數' }), ylabel: w < 200 ? t({ en: 'lr (× 0.001)', zh: '學習率 ×0.001' }) : t({ en: 'learning rate (× 0.001)', zh: '學習率（× 0.001）' }),
});
const lossChart = (k: string, x: number, y: number, w: number, h: number, series: { pts: number[]; tone: Tone }[], ymin = 1.6, ymax = 3.6, every = D().M.every, xmax = D().M.steps): Item => ({
  k, t: 'chart', x, y, w, h, ymin, ymax, xmax, every, series,
  xlabel: t({ en: 'step', zh: '步數' }), ylabel: t({ en: 'loss', zh: '損失' }),
});
const gnChart = (k: string, x: number, y: number, w: number, h: number, series: { pts: number[]; tone: Tone }[]): Item => ({
  k, t: 'chart', x, y, w, h, ymin: 0, ymax: 0.8, xmax: D().M.steps, every: D().M.every, series,
  xlabel: t({ en: 'step', zh: '步數' }), ylabel: t({ en: 'gradient norm', zh: '梯度範數' }),
});
const milli = (a: number[]) => a.map((v) => v * 1000);

const MODS: { title: L; take: L }[] = [
  { title: { en: 'Foundations', zh: '基礎' }, take: { en: 'text → ids → vectors → a guess', zh: '文字 → 編號 → 向量 → 一個猜測' } },
  { title: { en: 'Attention', zh: '注意力' }, take: { en: 'each token gathers from earlier ones', zh: '每個詞元從前面的詞元收集資訊' } },
  { title: { en: 'The Transformer Block', zh: 'Transformer 區塊' }, take: { en: 'attention + MLP, added to the stream', zh: '注意力＋MLP，加回殘差流' } },
  { title: { en: 'Training', zh: '訓練' }, take: { en: 'loss, gradients, Adam, learning rate', zh: '損失、梯度、Adam、學習率' } },
  { title: { en: 'Training at Scale', zh: '大規模訓練' }, take: { en: 'data, compute, checkpoints, fine-tuning', zh: '資料、運算、檢查點、微調' } },
  { title: { en: 'Whole Systems', zh: '完整系統' }, take: { en: 'build it, train it, read its charts', zh: '組起來、訓練它、讀懂圖表' } },
];

let stepsCache: Step[] | undefined;
function buildSteps(): Step[] {
  const d = D();
  const M = d.M;
  return [
    {
      short: { en: 'The log', zh: '紀錄' }, chapter: 'problem',
      title: { en: 'A training run leaves a log of numbers', zh: '一次訓練會留下一大串數字' },
      body: [
        { en: `This is a real log: our tiny GPT trained for ${n0(M.steps)} steps with batches of ${M.batch} names. Every ${M.every} steps it wrote down the learning rate, the training loss, the held-out loss and the gradient norm.`, zh: `這是一份真實的紀錄：我們的迷你 GPT 訓練了 ${n0(M.steps)} 步，每個批次 ${M.batch} 個名字。每 ${M.every} 步，它記下學習率、訓練損失、驗證損失與梯度範數。` },
        { en: `That is ${n0(M.train.length)} rows. Nobody reads them as a table. In this lesson you learn to read them as charts, and to spot a healthy run and a broken one.`, zh: `一共 ${n0(M.train.length)} 列。沒有人會把它當表格讀。這一課要學的，是把它畫成圖來讀，並且分辨出健康的訓練和出問題的訓練。` },
      ],
      callout: { kind: 'insight', text: { en: 'Big labs watch exactly these kinds of curves for runs that last weeks. The skills are the same at any size.', zh: '大型實驗室在長達數週的訓練裡，盯著的也正是這類曲線。不論規模大小，看圖的功夫都一樣。' } },
    },
    {
      short: { en: 'Learning rate', zh: '學習率' }, chapter: 'technique',
      title: { en: 'First check the plan: the learning rate', zh: '先看計畫：學習率' },
      body: [
        { en: `The learning rate rises for ${M.warmup} steps (warmup), peaks at ${M.peakLr}, then follows a cosine curve down to ${M.lr[M.lr.length - 1]} (Learning Rate and Warmup lesson).`, zh: `學習率先在前 ${M.warmup} 步上升（暖身），在 ${M.peakLr} 達到最高，然後沿著餘弦曲線降到 ${M.lr[M.lr.length - 1]}（「學習率與暖身」那一課）。` },
        { en: 'This chart is the plan we chose, not something the model learned. Reading it first tells you how big the steps were at every moment, which explains much of the other curves.', zh: '這張圖是我們自己訂的計畫，不是模型學出來的。先讀它，就知道每個時刻的步伐有多大，這能解釋其他曲線的很多現象。' },
      ],
      changed: [{ sign: '+', text: { en: 'Learning rate chart', zh: '學習率圖' } }],
    },
    {
      short: { en: 'Loss', zh: '損失' }, chapter: 'technique',
      title: { en: 'The loss: a fast drop, then a long slow slope', zh: '損失：先快速下降，再緩緩下坡' },
      body: [
        { en: `Held-out loss starts at ${f3(d.mv[0])} (pure guessing) and is already ${f3(d.at(d.mv, 500))} at step 500. That is ${Math.round(d.dropBy500 * 100)}% of the whole run’s improvement in its first eighth.`, zh: `驗證損失從 ${f3(d.mv[0])}（純粹亂猜）開始，到第 500 步已經是 ${f3(d.at(d.mv, 500))}。整次訓練的進步，有 ${Math.round(d.dropBy500 * 100)}% 發生在最前面的八分之一。` },
        { en: 'This shape is normal and healthy: easy patterns are learned first, hard ones slowly. The training line is a bit jagged because each point averages only a few batches.', zh: '這種形狀很正常，也很健康：簡單的規律先學會，難的慢慢來。訓練損失的線有點鋸齒，因為每個點只是幾個批次的平均。' },
      ],
      changed: [{ sign: '+', text: { en: 'Training and held-out loss', zh: '訓練損失與驗證損失' } }],
    },
    {
      short: { en: 'The gap', zh: '差距' }, chapter: 'technique',
      title: { en: 'Watch the gap between the two lines', zh: '注意兩條線之間的差距' },
      body: [
        { en: `Zoomed in: the gap grows from ${f3(d.gap(500))} at step 500 to ${f3(d.gap(d.lastStep))} at the end. Training loss keeps falling, but held-out loss was best at step ${n0(d.bestStep)} (${f3(d.best)}) and then stopped improving.`, zh: `放大來看：差距從第 500 步的 ${f3(d.gap(500))}，擴大到最後的 ${f3(d.gap(d.lastStep))}。訓練損失還在下降，但驗證損失在第 ${n0(d.bestStep)} 步最好（${f3(d.best)}），之後就不再進步。` },
        { en: 'The model is starting to learn its training names a little too specifically (Overfitting and Validation lesson). Here the effect is tiny, because the model is too small to memorize much.', zh: '模型開始把訓練用的名字學得有點太「專門」了（「過度擬合與驗證」那一課）。在這裡影響非常小，因為模型太小，背不了多少東西。' },
      ],
      callout: { kind: 'note', text: { en: 'This is why you save checkpoints and keep the one with the best held-out loss, not simply the last one (Checkpoints and Evaluation lesson).', zh: '這就是為什麼要存檢查點，並保留驗證損失最好的那一個，而不是直接用最後一個（「檢查點與評估」那一課）。' } },
      changed: [{ sign: '~', text: { en: 'Zoom in on the gap', zh: '放大看差距' } }],
    },
    {
      short: { en: 'Grad norm', zh: '梯度範數' }, chapter: 'technique',
      title: { en: 'The gradient norm: how hard the model is pushed', zh: '梯度範數：模型被推得多用力' },
      body: [
        { en: 'The <b>gradient norm</b> is the length of the whole gradient: all 3,115 gradients treated as one arrow (Gradients and Backpropagation lesson). Big means the loss is very steep where the model stands.', zh: '<b>梯度範數</b>是整個梯度的長度：把 3,115 個梯度當成同一支箭頭（「梯度與反向傳播」那一課）。數值大，代表模型所在的地方坡度非常陡。' },
        { en: `Here it stays between ${d.gnMin.toFixed(2)} and ${d.gnMax.toFixed(2)} for the whole run, drifting up slowly with no sudden jumps. Calm and boring is exactly what you want.`, zh: `在這次訓練裡，它全程都在 ${d.gnMin.toFixed(2)} 到 ${d.gnMax.toFixed(2)} 之間，緩慢往上飄，沒有任何突然的跳動。平靜又無聊，正是你想看到的。` },
      ],
      changed: [{ sign: '+', text: { en: 'Gradient norm chart', zh: '梯度範數圖' } }],
    },
    {
      short: { en: 'Samples', zh: '樣本' }, chapter: 'technique',
      title: { en: 'Read the samples, not only the numbers', zh: '讀樣本，不只讀數字' },
      body: [
        { en: `At ${d.checkpoints.length} checkpoints the run wrote 8 names. A loss of ${f3(d.at(d.mv, 250))} versus ${f3(d.at(d.mv, 4000))} is hard to feel; the names make it concrete.`, zh: `這次訓練在 ${d.checkpoints.length} 個檢查點各寫了 8 個名字。損失 ${f3(d.at(d.mv, 250))} 和 ${f3(d.at(d.mv, 4000))} 的差別很難體會，但名字一看就懂。` },
        { en: 'Notice how most of the change happens early, just like the loss curve. Later samples are only a little better: that is the long slow slope, in words.', zh: '注意大部分的改變都發生在早期，就和損失曲線一樣。後面的樣本只稍微好一點：這就是那段「緩緩下坡」，換成文字的樣子。' },
      ],
      callout: { kind: 'note', text: { en: 'Samples catch problems a single number hides, such as a model that writes the same name again and again.', zh: '樣本能抓到單一數字看不出的問題，例如模型一直重複寫同一個名字。' } },
      changed: [{ sign: '+', text: { en: 'Names at each checkpoint', zh: '每個檢查點的名字' } }],
    },
    {
      short: { en: 'Healthy', zh: '健康' }, chapter: 'solution',
      title: { en: 'What a healthy run looks like', zh: '健康的訓練長什麼樣子' },
      body: [
        { en: 'Put the charts side by side and check four things: the learning rate did what you planned, both losses fell, the gap stayed small, and the gradient norm stayed calm.', zh: '把圖並排放在一起，檢查四件事：學習率照計畫走、兩條損失都下降、差距維持很小、梯度範數保持平穩。' },
        { en: 'This run passes all four. That is the baseline you compare every other run against.', zh: '這次訓練四項全部通過。之後的每一次訓練，都拿它當比較的基準。' },
      ],
      changed: [{ sign: '+', text: { en: 'A four-point health check', zh: '四項健康檢查' } }],
    },
    {
      short: { en: 'Broken', zh: '壞掉' }, chapter: 'problem',
      title: { en: 'Case: a run that blows up', zh: '案例：一次爆掉的訓練' },
      body: [
        { en: `Now a different real run: a flat learning rate of ${d.H.lr[0]} with no warmup, ${d.H.lr[0] / M.peakLr}× our peak. It trains, but its held-out loss sits higher than the healthy run (${f3(d.hvBefore)} vs ${f3(d.mvSame)} at step ${n0(d.cmpStep)}).`, zh: `換一次真實的訓練：學習率固定在 ${d.H.lr[0]}，沒有暖身，是我們最高值的 ${d.H.lr[0] / M.peakLr} 倍。它還是在學，但驗證損失比健康的那次高（第 ${n0(d.cmpStep)} 步時是 ${f3(d.hvBefore)}，健康的是 ${f3(d.mvSame)}）。` },
        { en: `Then, in the stretch ending at step ${n0(d.spikeStep)}, the loss spikes and becomes NaN (not a number). In the same stretch the gradient norm leaps from under ${d.gnBefore.toFixed(1)} to hundreds, peaking at ${n0(Math.round(d.gnPeak))}. It never recovers: the final held-out loss is ${f3(d.hFinal)}.`, zh: `接著，在截至第 ${n0(d.spikeStep)} 步的那一段，損失暴衝，變成 NaN（不是一個數字）。同一段裡，梯度範數從不到 ${d.gnBefore.toFixed(1)} 一口氣跳到好幾百，最高達 ${n0(Math.round(d.gnPeak))}。之後再也回不來：最後的驗證損失是 ${f3(d.hFinal)}。` },
      ],
      callout: { kind: 'insight', text: { en: `There was a quiet hint first: around step ${n0(d.bumpStep)} the gradient norm briefly rose to ${d.gnBump.toFixed(2)} (typical: ${d.gnTypical.toFixed(2)}) and the held-out loss bumped to ${f3(d.hvBump)}. Bumps like that at a high learning rate are worth taking seriously.`, zh: `其實之前有個小小的徵兆：大約第 ${n0(d.bumpStep)} 步，梯度範數短暫升到 ${d.gnBump.toFixed(2)}（平常約 ${d.gnTypical.toFixed(2)}），驗證損失也跳到 ${f3(d.hvBump)}。在高學習率下出現這種小起伏，就值得認真看待。` } },
      changed: [{ sign: '~', text: { en: `Learning rate ${d.H.lr[0]}, no warmup`, zh: `學習率 ${d.H.lr[0]}，沒有暖身` } }],
    },
    {
      short: { en: 'What to try', zh: '怎麼辦' }, chapter: 'technique',
      title: { en: 'Warning signs, and what to try', zh: '警訊，以及該試什麼' },
      body: [
        { en: 'Each row is a real run from this course. <b>Spikes</b>: lower the peak learning rate, add warmup, cap the gradient’s length (gradient clipping), and restart from a checkpoint saved before the spike.', zh: '每一列都是本課程裡真實的訓練。<b>暴衝</b>：調低最高學習率、加上暖身、限制梯度的長度（梯度裁剪），並從暴衝前存好的檢查點重新開始。' },
        { en: `<b>Growing gap</b>: the model memorizes (here ${d.oN} names: training ${f3(d.oTrainEnd)}, held-out ${f3(d.oValEnd)}); use more data, a smaller model or a shorter run. <b>Flat</b>: normal near the end of a run; early on, check the learning rate, then try a bigger model.`, zh: `<b>差距擴大</b>：模型在背答案（這裡是 ${d.oN} 個名字：訓練 ${f3(d.oTrainEnd)}、驗證 ${f3(d.oValEnd)}），可以加資料、縮小模型或縮短訓練。<b>持平</b>：在訓練尾聲很正常；若一開始就這樣，先檢查學習率，再試更大的模型。` },
      ],
      changed: [{ sign: '+', text: { en: 'Three symptoms, three fixes', zh: '三種症狀，三種對策' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'From one letter to reading a whole run', zh: '從一個字母，到讀懂整次訓練' },
      body: [
        { en: '<b>This lesson</b>: read the learning rate first, then both losses and their gap, then the gradient norm and samples. Calm curves are healthy; spikes and a growing gap are warnings.', zh: '<b>這一課</b>：先讀學習率，再看兩條損失和它們的差距，然後看梯度範數和樣本。平穩的曲線代表健康；暴衝和越來越大的差距是警訊。' },
        { en: `<b>The course</b>: text became ids, ids became vectors, attention and the MLP refined them, and a loop of small corrections found ${n0(M.params)} good numbers. Real language models are this same recipe at a vastly larger scale.`, zh: `<b>整門課</b>：文字變成編號，編號變成向量，注意力與 MLP 一層層加工，再由一個不斷小幅修正的迴圈，找出 ${n0(M.params)} 個好數字。真實的語言模型，就是同一套配方放大到極大的規模。` },
      ],
      callout: { kind: 'next', text: { en: 'Where next: try reading the code of a small open-source GPT, train one on text you care about, or read the paper that introduced the transformer. You now know every part they talk about.', zh: '接下來：可以試著讀一個小型開源 GPT 的程式碼、用你在乎的文字訓練一個模型，或去讀提出 Transformer 的那篇論文。它們談到的每個零件，你現在都認識了。' } },
    },
  ];
}

export const realRun: Lesson = {
  id: 'real-run',
  title: { en: 'Read a Real Training Run', zh: '讀懂一次真實的訓練' },
  blurb: {
    en: 'Learn to read the charts of a real training run: what healthy looks like, what breaking looks like, and what to do about it.',
    zh: '學會讀一次真實訓練的圖表：健康的樣子、壞掉的樣子，以及該怎麼處理。',
  },
  minutes: 12,
  level: 'intermediate',
  learn: [
    { text: { en: 'Read a learning-rate schedule', zh: '讀懂學習率排程' }, step: 1 },
    { text: { en: 'Read training vs held-out loss and their gap', zh: '讀懂訓練損失、驗證損失與兩者的差距' }, step: 3 },
    { text: { en: 'What the gradient norm tells you', zh: '梯度範數告訴你什麼' }, step: 4 },
    { text: { en: 'Spot a broken run early', zh: '及早發現壞掉的訓練' }, step: 7 },
    { text: { en: 'What to try when a run goes wrong', zh: '訓練出錯時可以試什麼' }, step: 8 },
  ],
  get steps() { return stepsCache ??= buildSteps(); },

  scene(step) {
    const d = D();
    const M = d.M;
    const items: Item[] = [];
    // layout: one big chart on the left, small charts in a column on the right
    const BIG = { x: 80, y: 50, w: 560, h: 300 };
    const SM = (i: number) => ({ x: 725, y: 50 + i * 130, w: 145, h: 70 });

    if (step === 0) {
      const cols = [
        { x: 150, h: { en: 'step', zh: '步數' } }, { x: 330, h: { en: 'learning rate', zh: '學習率' } },
        { x: 500, h: { en: 'training loss', zh: '訓練損失' } }, { x: 660, h: { en: 'held-out loss', zh: '驗證損失' } },
        { x: 820, h: { en: 'gradient norm', zh: '梯度範數' } },
      ];
      cols.forEach((c, j) => items.push({ k: 'th' + j, t: 'text', x: c.x, y: 70, anchor: 'end', size: 13.5, bold: true, tone: 'muted', text: t(c.h) }));
      [0, 100, 200, 500, 1000, 2000, 4000].forEach((s, i) => {
        const y = 112 + i * 40;
        const vals = [n0(s), d.at(M.lr, s).toFixed(4), f3(d.at(d.mt, s)), f3(d.at(d.mv, s)), s === 0 ? '—' : d.gnAt(M, s).toFixed(3)];
        vals.forEach((v, j) => items.push({ k: `td${i}_${j}`, t: 'text', x: cols[j].x, y, anchor: 'end', size: 15, mono: true, tone: j === 0 ? 'muted' : 'ink', text: v }));
      });
      items.push({ k: 'tn', t: 'text', x: 450, y: 430, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: `7 of ${M.train.length} rows · one row every ${M.every} steps`, zh: `${M.train.length} 列中的 7 列 · 每 ${M.every} 步一列` }) });
    }

    const lrSeries = [{ pts: milli(M.lr), tone: 'a' as Tone }];
    const lossSeries = [{ pts: d.mt, tone: 'a' as Tone }, { pts: d.mv, tone: 'b' as Tone }];
    const gnSeries = [{ pts: M.gradNorm, tone: 'a' as Tone }];

    if (step === 1) {
      items.push(lrChart('lr', BIG.x, BIG.y, BIG.w, BIG.h, lrSeries));
      const wx = BIG.x + (M.warmup / M.steps) * BIG.w;
      items.push({ k: 'lw', t: 'arrow', x1: wx + 70, y1: BIG.y + 120, x2: wx + 4, y2: BIG.y + 6, tone: 'b' });
      items.push({ k: 'lwt', t: 'text', x: wx + 74, y: BIG.y + 136, size: 14, tone: 'b', bold: true, text: t({ en: `warmup ends at step ${M.warmup}: peak ${M.peakLr}`, zh: `第 ${M.warmup} 步暖身結束：最高 ${M.peakLr}` }) });
      items.push({ k: 'lft', t: 'text', x: BIG.x + BIG.w, y: BIG.y + BIG.h - 22, anchor: 'end', size: 14, tone: 'muted', text: t({ en: `ends at ${M.lr[M.lr.length - 1]} (10% of peak)`, zh: `最後是 ${M.lr[M.lr.length - 1]}（最高值的 10%）` }) });
    }

    if (step === 2 || step === 3) {
      const zoom = step === 3;
      items.push(lossChart('loss', BIG.x, BIG.y, BIG.w, BIG.h, lossSeries, zoom ? 1.9 : 1.6, zoom ? 2.5 : 3.6));
      items.push(lrChart('lr', SM(0).x, SM(0).y, SM(0).w, SM(0).h, lrSeries));
      if (zoom) {
        [500, 2000, d.lastStep].forEach((s, i) => items.push({ k: 'gp' + i, t: 'text', x: BIG.x + (i * BIG.w) / 2.6, y: 420, size: 14.5, mono: true, tone: i === 2 ? 'bad' : 'ink', text: t({ en: `step ${n0(s)}: gap ${f3(d.gap(s))}`, zh: `第 ${n0(s)} 步：差距 ${f3(d.gap(s))}` }) }));
        items.push({ k: 'gb', t: 'text', x: BIG.x, y: 450, size: 14, tone: 'b', text: t({ en: `best held-out loss: ${f3(d.best)} at step ${n0(d.bestStep)} · last: ${f3(d.mv[d.last])}`, zh: `最好的驗證損失：第 ${n0(d.bestStep)} 步的 ${f3(d.best)} · 最後：${f3(d.mv[d.last])}` }) });
      } else {
        items.push({ k: 'gb', t: 'text', x: BIG.x, y: 420, size: 14, tone: 'muted', text: t({ en: `held-out: ${f3(d.mv[0])} → ${f3(d.at(d.mv, 500))} by step 500 → ${f3(d.mv[d.last])} at step ${n0(d.lastStep)}`, zh: `驗證損失：${f3(d.mv[0])} → 第 500 步 ${f3(d.at(d.mv, 500))} → 第 ${n0(d.lastStep)} 步 ${f3(d.mv[d.last])}` }) });
      }
    }

    if (step === 4) {
      items.push(gnChart('gn', BIG.x, BIG.y, BIG.w, BIG.h, gnSeries));
      items.push(lrChart('lr', SM(0).x, SM(0).y, SM(0).w, SM(0).h, lrSeries));
      items.push(lossChart('loss', SM(1).x, SM(1).y, SM(1).w, SM(1).h, lossSeries));
      items.push({ k: 'gb', t: 'text', x: BIG.x, y: 420, size: 14, tone: 'muted', text: t({ en: `range ${d.gnMin.toFixed(2)} – ${d.gnMax.toFixed(2)} · sudden jumps: ${d.mainSpikes}`, zh: `範圍 ${d.gnMin.toFixed(2)} – ${d.gnMax.toFixed(2)} · 突然跳動：${d.mainSpikes} 次` }) });
    }

    if (step === 5) {
      d.checkpoints.forEach((s, i) => {
        const y = 70 + i * 52;
        items.push({ k: 'ss' + i, t: 'text', x: 60, y, size: 14, mono: true, bold: true, tone: 'muted', text: t({ en: `step ${n0(s)}`, zh: `第 ${n0(s)} 步` }) });
        items.push({ k: 'sv' + i, t: 'text', x: 60, y: y + 20, size: 13, mono: true, tone: 'b', text: t({ en: `held-out ${f3(d.at(d.mv, s))}`, zh: `驗證 ${f3(d.at(d.mv, s))}` }) });
        items.push({ k: 'sn' + i, t: 'text', x: 200, y: y + 9, size: 15, mono: true, tone: s === 0 ? 'bad' : 'ink', text: M.samples[String(s)].join('  ') });
      });
    }

    if (step === 6) {
      const sm = { w: 330, h: 120 };
      items.push(lrChart('lr', 80, 50, sm.w, sm.h, lrSeries));
      items.push(lossChart('loss', 520, 50, sm.w, sm.h, lossSeries));
      items.push(gnChart('gn', 80, 250, sm.w, sm.h, gnSeries));
      const checks: L[] = [
        { en: `learning rate followed the plan`, zh: '學習率照計畫走' },
        { en: `both losses fell: held-out ${f3(d.mv[0])} → ${f3(d.mv[d.last])}`, zh: `兩條損失都下降：驗證 ${f3(d.mv[0])} → ${f3(d.mv[d.last])}` },
        { en: `gap stayed small: ${f3(d.gap(d.lastStep))} at the end`, zh: `差距維持很小：最後 ${f3(d.gap(d.lastStep))}` },
        { en: `gradient norm calm: ${d.gnMin.toFixed(2)} – ${d.gnMax.toFixed(2)}`, zh: `梯度範數平穩：${d.gnMin.toFixed(2)} – ${d.gnMax.toFixed(2)}` },
      ];
      checks.forEach((c, i) => items.push({ k: 'ck' + i, t: 'text', x: 500, y: 270 + i * 34, size: 14.5, tone: 'good', text: '✓ ' + t(c) }));
    }

    if (step === 7) {
      const H = d.H;
      // the broken run was logged every 25 steps; keep every other point so both runs share the 50-step grid
      const even = <T,>(a: T[]) => a.filter((_, i) => i % 2 === 0);
      const span = 2000, n = span / M.every + 1;
      items.push({
        k: 'loss', t: 'chart', x: BIG.x, y: BIG.y, w: BIG.w, h: BIG.h, ymin: 1.6, ymax: 3.6, xmax: span, every: M.every,
        series: [{ pts: d.mv.slice(0, n), tone: 'muted' }, { pts: even(d.hv), tone: 'bad' }],
        xlabel: t({ en: 'step', zh: '步數' }), ylabel: t({ en: 'held-out loss (off the top = huge or NaN)', zh: '驗證損失（超出上緣 = 極大或 NaN）' }),
      });
      const sx = BIG.x + (d.spikeStep / span) * BIG.w;
      items.push({ k: 'spk', t: 'arrow', x1: sx - 90, y1: BIG.y + 120, x2: sx - 3, y2: BIG.y + 8, tone: 'bad' });
      items.push({ k: 'spt', t: 'text', x: sx - 95, y: BIG.y + 138, anchor: 'end', size: 14, bold: true, tone: 'bad', text: t({ en: `step ${n0(d.spikeStep)}: NaN`, zh: `第 ${n0(d.spikeStep)} 步：NaN` }) });
      items.push({ k: 'lr', t: 'chart', x: SM(0).x, y: SM(0).y, w: SM(0).w, h: SM(0).h, ymin: 0, ymax: 60, xmax: span, every: M.every, series: [{ pts: milli(M.lr.slice(0, n)), tone: 'muted' }, { pts: milli(even(H.lr)), tone: 'bad' }], xlabel: t({ en: 'step', zh: '步數' }), ylabel: t({ en: 'lr (× 0.001)', zh: '學習率 ×0.001' }) });
      items.push({ k: 'gn', t: 'chart', x: SM(1).x, y: SM(1).y, w: SM(1).w, h: SM(1).h, ymin: 0, ymax: 0.8, xmax: span, every: H.every, series: [{ pts: H.gradNorm, tone: 'bad' }], xlabel: t({ en: 'step', zh: '步數' }), ylabel: t({ en: 'gradient norm', zh: '梯度範數' }) });
      items.push({ k: 'gnt', t: 'text', x: SM(1).x, y: SM(1).y + SM(1).h + 42, size: 13, tone: 'bad', text: t({ en: `peak ${n0(Math.round(d.gnPeak))} (off the chart)`, zh: `最高 ${n0(Math.round(d.gnPeak))}（超出圖表）` }) });
      items.push({ k: 'gb', t: 'text', x: BIG.x, y: 420, size: 14, tone: 'muted', text: t({ en: `grey: the healthy run · red: flat learning rate ${H.lr[0]}`, zh: `灰色：健康的訓練 · 紅色：固定學習率 ${H.lr[0]}` }) });
    }

    if (step === 8) {
      const rows: { chart: Item; sym: [L, L]; fix: [L, L]; tone: Tone }[] = [
        {
          chart: { k: 'm0', t: 'chart', x: 80, y: 40, w: 220, h: 90, ymin: 1.6, ymax: 3.6, xmax: 2000, every: d.H.every, series: [{ pts: d.ht, tone: 'bad' }], xlabel: '', ylabel: t({ en: 'loss · flat lr 0.05', zh: '損失 · 固定學習率 0.05' }) },
          sym: [{ en: 'Loss spikes or NaN', zh: '損失暴衝或 NaN' }, { en: 'gradient norm leaps too', zh: '梯度範數同時暴增' }],
          fix: [{ en: 'Lower lr · warmup · clipping', zh: '降學習率 · 暖身 · 梯度裁剪' }, { en: 'resume from a checkpoint before it', zh: '從暴衝前的檢查點重來' }],
          tone: 'bad',
        },
        {
          chart: { k: 'm1', t: 'chart', x: 80, y: 195, w: 220, h: 90, ymin: 0, ymax: 13, xmax: (d.ot.length - 1) * d.O.every, every: d.O.every, series: [{ pts: d.ot, tone: 'a' }, { pts: d.ov, tone: 'b' }], xlabel: '', ylabel: t({ en: `loss · only ${d.oN} names`, zh: `損失 · 只有 ${d.oN} 個名字` }) },
          sym: [{ en: 'Gap keeps growing', zh: '差距一直擴大' }, { en: 'held-out up, training down', zh: '驗證上升、訓練下降' }],
          fix: [{ en: 'More data · smaller model', zh: '更多資料 · 更小的模型' }, { en: 'stop early, keep the best checkpoint', zh: '提早停，保留最好的檢查點' }],
          tone: 'b',
        },
        {
          chart: { k: 'm2', t: 'chart', x: 80, y: 350, w: 220, h: 90, ymin: 1.9, ymax: 2.5, xmax: M.steps, every: M.every, series: [{ pts: d.mt, tone: 'a' }, { pts: d.mv, tone: 'b' }], xlabel: '', ylabel: t({ en: 'loss · healthy run, zoomed', zh: '損失 · 健康的訓練（放大）' }) },
          sym: [{ en: 'Held-out loss flat', zh: '驗證損失持平' }, { en: 'normal near the end', zh: '在尾聲很正常' }],
          fix: [{ en: 'Early on? check the lr', zh: '一開始就平？檢查學習率' }, { en: 'then try a bigger model or more data', zh: '再試更大的模型或更多資料' }],
          tone: 'muted',
        },
      ];
      rows.forEach((r, i) => {
        const y = (r.chart as { y: number }).y;
        items.push(r.chart);
        items.push({ k: 'sy' + i, t: 'box', x: 340, y: y + 10, w: 230, h: 64, label: t(r.sym[0]), sub: t(r.sym[1]), tone: r.tone });
        items.push({ k: 'sa' + i, t: 'arrow', x1: 576, y1: y + 42, x2: 612, y2: y + 42 });
        items.push({ k: 'fx' + i, t: 'box', x: 618, y: y + 10, w: 262, h: 64, label: t(r.fix[0]), sub: t(r.fix[1]), tone: 'good' });
      });
    }

    if (step === 9) {
      MODS.forEach((m, i) => {
        const x = 30 + (i % 3) * 290, y = 90 + Math.floor(i / 3) * 110;
        items.push({ k: 'mod' + i, t: 'box', x, y, w: 260, h: 70, label: t(m.title), sub: t(m.take), tone: i === 5 ? 'a' : 'ink' });
        if (i % 3 < 2) items.push({ k: 'ma' + i, t: 'arrow', x1: x + 262, y1: y + 35, x2: x + 288, y2: y + 35 });
      });
      items.push({ k: 'ma2b', t: 'arrow', x1: 740, y1: 162, x2: 160, y2: 198, tone: 'muted', dashed: true });
      items.push({ k: 'rs', t: 'text', x: 450, y: 370, anchor: 'middle', size: 24, bold: true, tone: 'a', text: t({ en: `${n0(M.params)} weights · ${n0(NAMES.length)} names · one loop`, zh: `${n0(M.params)} 個權重 · ${n0(NAMES.length)} 個名字 · 一個迴圈` }) });
      items.push({ k: 'rs2', t: 'text', x: 450, y: 400, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: 'the same recipe, scaled up, is how real language models are made', zh: '同一套配方放大規模，就是真實語言模型的做法' }) });
    }

    const broken = step === 7 || step === 8;
    const metrics = broken
      ? [
        { label: { en: 'Learning rate', zh: '學習率' }, value: t({ en: `${d.H.lr[0]} flat`, zh: `固定 ${d.H.lr[0]}` }) },
        { label: { en: 'Loss spike at', zh: '損失暴衝於' }, value: t({ en: `step ${n0(d.spikeStep)}`, zh: `第 ${n0(d.spikeStep)} 步` }) },
        { label: { en: 'Peak grad norm', zh: '梯度範數最高' }, value: n0(Math.round(d.gnPeak)) },
        { label: { en: 'Final held-out', zh: '最後驗證損失' }, value: f3(d.hFinal) },
      ]
      : step === 9
        ? [
          { label: { en: 'Weights', zh: '權重' }, value: n0(M.params) },
          { label: { en: 'Names', zh: '名字' }, value: n0(NAMES.length) },
          { label: { en: 'Training steps', zh: '訓練步數' }, value: n0(M.steps) },
          { label: { en: 'Best held-out', zh: '最佳驗證損失' }, value: f3(d.best) },
        ]
        : [
          { label: { en: 'Steps', zh: '步數' }, value: n0(M.steps) },
          { label: { en: 'Peak lr', zh: '最高學習率' }, value: String(M.peakLr) },
          { label: { en: 'Best held-out', zh: '最佳驗證損失' }, value: step >= 2 ? f3(d.best) : '—' },
          { label: { en: 'Grad norm max', zh: '梯度範數最高' }, value: step >= 4 ? d.gnMax.toFixed(2) : '—' },
          { label: { en: 'Run time', zh: '花費時間' }, value: `${M.seconds} s` },
        ];
    return {
      metrics,
      items,
      legend: step >= 2 && step <= 6 || step === 8
        ? [{ tone: 'a', text: { en: 'training loss', zh: '訓練損失' } }, { tone: 'b', text: { en: 'held-out loss', zh: '驗證損失' } }]
        : step === 7 ? [{ tone: 'bad', text: { en: 'broken run', zh: '壞掉的訓練' } }] : undefined,
    };
  },
};
