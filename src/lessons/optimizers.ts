import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, DEFAULT_CONFIG } from '../model/gpt';
import { TRAIN_NAMES, example, rng } from '../model/data';
import exp from '../model/experiments.json';

// A toy problem with two weights, computed live, shows why one step size for every weight
// is awkward and how Adam fixes it. Then the real runs from experiments.json.

// Toy loss: K[0]·A² + K[1]·B². Best when both weights are 0. A sits in a steep valley, B in a gentle one.
const K = [10, 0.1];
const STEPS = 60;
const LR_SGD = 0.08, LR_HOT = 0.11, LR_ADAM = 0.05;

function toy(opt: 'sgd' | 'adam', lr: number) {
  const w = [1, 1], m = [0, 0], v = [0, 0];
  const hist = [[1], [1]];
  const first: number[] = [];
  for (let s = 1; s <= STEPS; s++) {
    const g = w.map((x, i) => 2 * K[i] * x);
    for (let i = 0; i < 2; i++) {
      const before = w[i];
      if (opt === 'sgd') w[i] -= lr * g[i];
      else {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        const mh = m[i] / (1 - 0.9 ** s), vh = v[i] / (1 - 0.999 ** s);
        w[i] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
      }
      if (s === 1) first.push(Math.abs(w[i] - before));
      hist[i].push(w[i]);
    }
  }
  return { hist, first, end: [hist[0][STEPS], hist[1][STEPS]] };
}
let toyCache: { sgd: ReturnType<typeof toy>; hot: ReturnType<typeof toy>; adam: ReturnType<typeof toy>; g0: number[] } | undefined;
const toys = () => (toyCache ??= { sgd: toy('sgd', LR_SGD), hot: toy('sgd', LR_HOT), adam: toy('adam', LR_ADAM), g0: K.map((k) => 2 * k) });

// The real model at the start of training (same random start as the runs below).
let realCache: ReturnType<typeof makeReal> | undefined;
function makeReal() {
  const model = new GPT(DEFAULT_CONFIG, rng(1));
  const r = rng(3);
  const pick = (n: number) => Array.from({ length: n }, () => example(TRAIN_NAMES[Math.floor(r() * TRAIN_NAMES.length)]));
  model.loss(pick(32), true);
  const sizes = Object.entries(model.p).map(([name, m]) => {
    let s = 0;
    for (const g of m.g) s += Math.abs(g);
    return { name, v: s / m.g.length };
  }).sort((a, b) => b.v - a.v);
  // One weight's gradient on 40 different batches of 16 names (model held fixed), and its running average.
  const raw: number[] = [], avg: number[] = [];
  let m = 0;
  for (let i = 0; i < 40; i++) {
    model.loss(pick(16), true);
    const g = model.p.u.g[27 + 1];
    raw.push(g);
    m = 0.9 * m + 0.1 * g;
    avg.push(m / (1 - 0.9 ** (i + 1)));
  }
  return { sizes, ratio: sizes[0].v / sizes[sizes.length - 1].v, raw, avg };
}
const real = () => (realCache ??= makeReal());

const O = exp.optimizers;
const at = (arr: (number | null)[], step: number, every: number) => arr[step / every] as number;
const f2 = (v: number) => v.toFixed(2);
const fw = (v: number) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : v.toFixed(2));
const f3 = (v: number) => v.toFixed(3);

export const optimizers: Lesson = {
  id: 'optimizers',
  title: { en: 'Optimizers: SGD and Adam', zh: '最佳化器：SGD 與 Adam' },
  blurb: {
    en: 'Every weight has a gradient. How far should each one move? Two answers, tested on a toy and on the real model.',
    zh: '每個權重都有了梯度，那每個該移動多遠？兩種答案，在小玩具和真實模型上各試一次。',
  },
  minutes: 12,
  level: 'beginner',
  learn: [
    { text: { en: 'SGD: step against the gradient, one learning rate for all', zh: 'SGD：往梯度反方向走，所有權重共用一個學習率' }, step: 1 },
    { text: { en: 'Why one step size cannot suit every weight', zh: '為什麼一種步伐大小無法適合所有權重' }, step: 2 },
    { text: { en: 'Momentum: a running average of gradients', zh: '動量：梯度的移動平均' }, step: 3 },
    { text: { en: 'Adam: scale each weight’s step by its typical gradient', zh: 'Adam：依各權重平常的梯度大小調整步伐' }, step: 4 },
    { text: { en: 'How SGD and Adam compare on the real model', zh: 'SGD 與 Adam 在真實模型上的比較' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Gradients come in very different sizes', zh: '梯度的大小差很多' },
      body: [
        { en: 'Backpropagation gave every weight a gradient. An <b>optimizer</b> is the rule that turns those gradients into actual changes to the weights.', zh: '反向傳播替每個權重算出了梯度。<b>最佳化器（optimizer）</b>就是把這些梯度變成實際權重改變的規則。' },
        { en: 'Here is the real model at the start of training, on one batch of 32 names. The bars show the average gradient size in each of its weight matrices. Some are dozens of times larger than others.', zh: '這是真實模型在訓練一開始、用一個 32 個名字的批次算出來的結果。長條是每個權重矩陣中梯度的平均大小。有些比其他大了好幾十倍。' },
      ],
      callout: { kind: 'insight', text: { en: 'If every weight moves by “learning rate × gradient”, the weights with small gradients barely move at all.', zh: '如果每個權重都移動「學習率 × 梯度」，那些梯度小的權重幾乎不會動。' } },
    },
    {
      short: { en: 'SGD', zh: 'SGD' }, chapter: 'technique',
      title: { en: 'SGD on a toy with two weights', zh: '在兩個權重的小玩具上跑 SGD' },
      body: [
        { en: `To see this clearly, take a toy problem with just two weights, A and B, both starting at 1. The loss is lowest when both are 0. A sits in a steep valley (gradient ${toys().g0[0]} at the start), B in a gentle one (gradient ${toys().g0[1]}).`, zh: `為了看清楚，拿一個只有兩個權重 A 和 B 的小玩具，兩者都從 1 出發，兩個都是 0 時損失最低。A 在陡峭的山谷裡（一開始梯度是 ${toys().g0[0]}），B 在平緩的山谷裡（梯度 ${toys().g0[1]}）。` },
        { en: `<b>SGD</b> (plain gradient descent) moves each weight by learning rate × gradient, here ${LR_SGD}. A drops to 0 almost at once. After ${STEPS} steps B has only reached ${f2(toys().sgd.end[1])}.`, zh: `<b>SGD</b>（最基本的梯度下降）讓每個權重移動「學習率 × 梯度」，這裡學習率是 ${LR_SGD}。A 幾乎立刻掉到 0，但走了 ${STEPS} 步之後，B 才到 ${f2(toys().sgd.end[1])}。` },
      ],
      callout: { kind: 'note', text: { en: '“Stochastic” in SGD means each step uses a random batch, not all the data. The toy has no batches, but the update rule is the same.', zh: 'SGD 中的「隨機（stochastic）」指每一步只用一個隨機批次，而不是全部資料。這個小玩具沒有批次，但更新規則是一樣的。' } },
      changed: [{ sign: '+', text: { en: `SGD, learning rate ${LR_SGD}`, zh: `SGD，學習率 ${LR_SGD}` } }],
    },
    {
      short: { en: 'Push harder', zh: '加大力道' }, chapter: 'problem',
      title: { en: 'Case: raise the learning rate to help B', zh: '案例：調高學習率來幫 B' },
      body: [
        { en: `B is too slow, so raise the learning rate from ${LR_SGD} to ${LR_HOT}. It is a small change, and B does speed up a little.`, zh: `B 太慢了，那就把學習率從 ${LR_SGD} 調到 ${LR_HOT}。只是小小的改變，B 也確實快了一點。` },
        { en: `But A, with its 100× larger gradient, now overshoots 0 by more each time and swings further and further out: ${fw(toys().hot.end[0])} after ${STEPS} steps. One learning rate cannot be right for both.`, zh: `但梯度大了 100 倍的 A，現在每次都衝過 0 更多，越擺越遠：${STEPS} 步後變成 ${fw(toys().hot.end[0])}。一個學習率不可能同時適合兩者。` },
      ],
      changed: [{ sign: '~', text: { en: `Learning rate ${LR_SGD} → ${LR_HOT}: A blows up`, zh: `學習率 ${LR_SGD} → ${LR_HOT}：A 爆掉` } }],
    },
    {
      short: { en: 'Momentum', zh: '動量' }, chapter: 'technique',
      title: { en: 'Idea 1: average the gradients over time', zh: '第一招：把梯度隨時間平均' },
      body: [
        { en: 'Back to the real model. Each batch is a different random handful of names, so a weight’s gradient jumps around from batch to batch (grey). Here we keep the model fixed and just draw 40 batches.', zh: '回到真實模型。每個批次都是隨機抓的一把名字，所以同一個權重的梯度每批都在跳動（灰線）。這裡讓模型保持不動，只是抽了 40 個批次。' },
        { en: 'A <b>running average</b> keeps 90% of the old average and mixes in 10% of the new gradient (teal). The noise mostly cancels and the steady direction remains. This is called <b>momentum</b>.', zh: '<b>移動平均</b>每次保留 90% 的舊平均，再混入 10% 的新梯度（青線）。雜訊大多互相抵消，留下穩定的方向。這叫做<b>動量（momentum）</b>。' },
      ],
      changed: [{ sign: '+', text: { en: 'Running average of gradients', zh: '梯度的移動平均' } }],
    },
    {
      short: { en: 'Scaling', zh: '縮放' }, chapter: 'technique',
      title: { en: 'Idea 2: divide by the typical gradient size', zh: '第二招：除以平常的梯度大小' },
      body: [
        { en: 'Adam also keeps a running average of each weight’s <i>squared</i> gradient, which tells it how big that weight’s gradients usually are. It divides the step by that size.', zh: 'Adam 另外為每個權重記錄<i>梯度平方</i>的移動平均，由此知道這個權重的梯度通常有多大，再用這個大小去除步伐。' },
        { en: `So a weight with big gradients and a weight with small gradients both take steps of about the learning rate. In the toy, SGD’s first steps differ by 100×; Adam’s are both ${f3(toys().adam.first[0])}.`, zh: `所以梯度大的權重和梯度小的權重，步伐都差不多是學習率那麼大。在小玩具裡，SGD 第一步的步伐差了 100 倍；Adam 的兩步都是 ${f3(toys().adam.first[0])}。` },
      ],
      callout: { kind: 'note', text: { en: 'Adam = momentum (direction) + per-weight scaling (step size). The original 2017 Transformer was trained with Adam, and Adam-style optimizers are still a common choice.', zh: 'Adam = 動量（決定方向）+ 每個權重各自縮放（決定步伐）。2017 年最早的 Transformer 就是用 Adam 訓練的，Adam 這一類最佳化器至今仍很常用。' } },
      changed: [{ sign: '+', text: { en: 'A step size for each weight', zh: '每個權重各自的步伐' } }],
    },
    {
      short: { en: 'Adam', zh: 'Adam' }, chapter: 'solution',
      title: { en: 'Adam on the same toy', zh: '在同一個小玩具上跑 Adam' },
      body: [
        { en: `With Adam (learning rate ${LR_ADAM}) A and B move in lockstep, even though their gradients differ by 100×. Both are near 0 after about 25 steps.`, zh: `用 Adam（學習率 ${LR_ADAM}）時，A 和 B 步調一致地前進，即使它們的梯度差了 100 倍。大約 25 步後兩者都接近 0。` },
        { en: 'They overshoot a little and settle back: that is momentum carrying them past the bottom. The lines lie exactly on top of each other, because Adam ignores how big a gradient is and only keeps its direction and consistency.', zh: '它們會稍微衝過頭再回來，那是動量把它們帶過了谷底。兩條線完全重疊，因為 Adam 不管梯度有多大，只看它的方向與穩定程度。' },
      ],
      changed: [{ sign: '~', text: { en: `SGD → Adam, learning rate ${LR_ADAM}`, zh: `SGD → Adam，學習率 ${LR_ADAM}` } }],
    },
    {
      short: { en: 'Real runs', zh: '真實訓練' }, chapter: 'solution',
      title: { en: 'SGD vs Adam on the real model', zh: '真實模型上的 SGD 與 Adam' },
      body: [
        { en: `Two real training runs from the same random start: ${((O.sgd.train.length - 1) * O.sgd.every).toLocaleString()} steps, batch 32. The chart shows the loss on held-out names. Adam is ahead early (${f3(at(O.adam.val, 100, O.adam.every))} vs ${f3(at(O.sgd.val, 100, O.sgd.every))} at step 100) and finishes slightly lower.`, zh: `兩次真實訓練，從同一個隨機起點出發：${((O.sgd.train.length - 1) * O.sgd.every).toLocaleString()} 步，批次 32。圖中是模型在沒練過的名字上的損失。Adam 一開始領先（第 100 步：${f3(at(O.adam.val, 100, O.adam.every))} 對 ${f3(at(O.sgd.val, 100, O.sgd.every))}），最後也略低一些。` },
        { en: `To be fair: each run uses its own learning rate (SGD ${O.sgd.lr[0]}, Adam ${O.adam.lr[0]}), each set to a reasonable value. On a model this tiny the gap is small, and we should not read more into it than that.`, zh: `說句公道話：兩次訓練各用自己的學習率（SGD ${O.sgd.lr[0]}、Adam ${O.adam.lr[0]}），都設在合理的數值。在這麼小的模型上差距不大，不宜過度解讀。` },
      ],
      changed: [{ sign: '+', text: { en: 'Two real 1,500-step runs', zh: '兩次真實的 1,500 步訓練' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Turning gradients into steps', zh: '把梯度變成步伐' },
      body: [
        { en: '<b>Optimizer</b>: the rule that turns gradients into weight changes.<br><b>SGD</b>: step = learning rate × gradient, the same rate for every weight.<br><b>Momentum</b>: average gradients over time to cancel batch noise.<br><b>Adam</b>: momentum plus a per-weight step size, so small-gradient weights still move.', zh: '<b>最佳化器</b>：把梯度變成權重改變的規則。<br><b>SGD</b>：步伐 = 學習率 × 梯度，所有權重同一個學習率。<br><b>動量</b>：把梯度隨時間平均，抵消批次帶來的雜訊。<br><b>Adam</b>：動量再加上每個權重各自的步伐，讓梯度小的權重也動得了。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Learning Rate and Warmup. Even Adam needs a learning rate. What happens when it is too high, and why change it during training?', zh: '下一課：學習率與暖身。就算是 Adam 也需要學習率。學習率太高會怎樣？又為什麼要在訓練中途調整它？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const T = step >= 1 && step <= 5 && step !== 3 ? toys() : null;
    const R = step === 0 || step === 3 ? real() : null;

    if (step === 0 && R) {
      const max = R.sizes[0].v;
      items.push({ k: 'szT', t: 'text', x: 40, y: 40, size: 14.5, tone: 'muted', text: t({ en: 'average gradient size in each weight matrix (× 0.001), real model, start of training', zh: '各權重矩陣的平均梯度大小（× 0.001），真實模型，訓練剛開始時' }) });
      R.sizes.forEach((s, i) => {
        const y = 62 + i * 36;
        items.push({ k: 'szn' + i, t: 'text', x: 110, y: y + 20, anchor: 'end', size: 14.5, mono: true, text: s.name });
        items.push({ k: 'szb' + i, t: 'box', x: 125, y, w: Math.max(4, (s.v / max) * 600), h: 28, tone: i === 0 ? 'b' : i === R.sizes.length - 1 ? 'a' : 'ink' });
        items.push({ k: 'szv' + i, t: 'text', x: 135 + Math.max(4, (s.v / max) * 600), y: y + 20, size: 14, mono: true, tone: 'muted', text: (s.v * 1000).toFixed(2) });
      });
    }

    if (T) {
      const toyChart = (k: string, x: number, title: string, run: ReturnType<typeof toy>, bad = false): void => {
        items.push({
          k, t: 'chart', x, y: 70, w: 340, h: 250, ymin: -1, ymax: 1, xmax: STEPS, every: 1,
          series: [{ pts: run.hist[0], tone: bad ? 'bad' : 'a' }, { pts: run.hist[1], tone: 'b' }],
          xlabel: t({ en: 'steps', zh: '步數' }), ylabel: title,
          guide: { y: 0, label: t({ en: 'goal: 0', zh: '目標：0' }) },
        });
      };
      if (step === 1 || step === 4) toyChart('tc1', 90, t({ en: `SGD, learning rate ${LR_SGD}`, zh: `SGD，學習率 ${LR_SGD}` }), T.sgd);
      if (step === 2) {
        toyChart('tc1', 90, t({ en: `SGD, learning rate ${LR_SGD}`, zh: `SGD，學習率 ${LR_SGD}` }), T.sgd);
        toyChart('tc2', 520, t({ en: `SGD, learning rate ${LR_HOT}`, zh: `SGD，學習率 ${LR_HOT}` }), T.hot, true);
      }
      if (step === 4 || step === 5) toyChart('tc2', 520, t({ en: `Adam, learning rate ${LR_ADAM}`, zh: `Adam，學習率 ${LR_ADAM}` }), T.adam);
      if (step === 5) toyChart('tc1', 90, t({ en: `SGD, learning rate ${LR_SGD}`, zh: `SGD，學習率 ${LR_SGD}` }), T.sgd);
      if (step === 4) {
        const rows = [
          [t({ en: 'first step of A', zh: 'A 的第一步' }), f3(T.sgd.first[0]), f3(T.adam.first[0])],
          [t({ en: 'first step of B', zh: 'B 的第一步' }), f3(T.sgd.first[1]), f3(T.adam.first[1])],
        ];
        items.push({ k: 'fsH1', t: 'text', x: 400, y: 400, anchor: 'middle', size: 14.5, bold: true, tone: 'muted', text: 'SGD' });
        items.push({ k: 'fsH2', t: 'text', x: 600, y: 400, anchor: 'middle', size: 14.5, bold: true, tone: 'muted', text: 'Adam' });
        rows.forEach((r, i) => {
          const y = 432 + i * 30;
          items.push({ k: 'fsl' + i, t: 'text', x: 260, y, anchor: 'end', size: 14.5, text: r[0] });
          items.push({ k: 'fss' + i, t: 'text', x: 400, y, anchor: 'middle', size: 15, mono: true, tone: i === 0 ? 'a' : 'b', text: r[1] });
          items.push({ k: 'fsa' + i, t: 'text', x: 600, y, anchor: 'middle', size: 15, mono: true, bold: true, tone: i === 0 ? 'a' : 'b', text: r[2] });
        });
      }
      if (step === 1 || step === 2 || step === 5) {
        const runs = step === 1 ? [['SGD', T.sgd]] : step === 2 ? [[`SGD ${LR_SGD}`, T.sgd], [`SGD ${LR_HOT}`, T.hot]] : [['SGD', T.sgd], ['Adam', T.adam]];
        (runs as [string, ReturnType<typeof toy>][]).forEach(([name, run], i) => {
          const x = i === 0 ? 260 : 690;
          items.push({ k: 'end' + i, t: 'text', x, y: 390, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: `${name} after ${STEPS} steps:`, zh: `${name} 走 ${STEPS} 步後：` }) });
          items.push({ k: 'endv' + i, t: 'text', x, y: 418, anchor: 'middle', size: 16, mono: true, bold: true, tone: Math.abs(run.end[0]) > 2 ? 'bad' : 'ink', text: `A = ${fw(run.end[0])} · B = ${f2(run.end[1])}` });
        });
      }
    }

    if (step === 3 && R) {
      const sc = 100;
      const vals = [...R.raw, ...R.avg].map((v) => v * sc);
      const lo = Math.floor(Math.min(...vals)), hiRaw = Math.ceil(Math.max(...vals));
      const hi = lo + Math.ceil((hiRaw - lo) / 4) * 4;
      items.push({
        k: 'tc1', t: 'chart', x: 110, y: 60, w: 700, h: 300, ymin: lo, ymax: hi, xmax: R.raw.length - 1, every: 1,
        series: [{ pts: R.raw.map((v) => v * sc), tone: 'muted' }, { pts: R.avg.map((v) => v * sc), tone: 'a' }],
        xlabel: t({ en: 'batch number', zh: '第幾個批次' }), ylabel: t({ en: 'gradient of one output weight (× 0.01)', zh: '一個輸出層權重的梯度（× 0.01）' }),
      });
      const pos = R.raw.filter((g) => g > 0).length;
      items.push({ k: 'mn', t: 'text', x: 460, y: 430, anchor: 'middle', size: 14.5, tone: 'muted',
        text: t({ en: `${pos} of ${R.raw.length} batches say “raise it”, ${R.raw.length - pos} say “lower it”; the average says “raise it”`, zh: `${R.raw.length} 個批次中，${pos} 個說「調高」，${R.raw.length - pos} 個說「調低」；平均起來是「調高」` }) });
    }

    if (step >= 6) {
      items.push({
        k: 'rc', t: 'chart', x: 90, y: 50, w: 760, h: 320, ymin: 2.1, ymax: 2.9, xmax: (O.sgd.val.length - 1) * O.sgd.every, every: O.sgd.every,
        series: [{ pts: O.sgd.val as number[], tone: 'b' }, { pts: O.adam.val as number[], tone: 'a' }],
        xlabel: t({ en: 'training steps', zh: '訓練步數' }), ylabel: t({ en: 'loss on held-out names (lower is better)', zh: '沒練過的名字上的損失（越低越好）' }),
      });
      const fin = (a: (number | null)[]) => a[a.length - 1] as number;
      items.push({ k: 'rs', t: 'text', x: 470, y: 440, anchor: 'middle', size: 15, mono: true,
        text: t({ en: `final: SGD ${f3(fin(O.sgd.val))} · Adam ${f3(fin(O.adam.val))}`, zh: `最後：SGD ${f3(fin(O.sgd.val))} · Adam ${f3(fin(O.adam.val))}` }) });
    }

    const metrics = step === 0
      ? [
        { label: { en: 'Weight matrices', zh: '權重矩陣' }, value: String(R!.sizes.length) },
        { label: { en: 'Biggest ÷ smallest', zh: '最大 ÷ 最小' }, value: `${Math.round(R!.ratio)}×` },
        { label: { en: 'Learning rates', zh: '學習率個數' }, value: '1' },
      ]
      : step === 3
        ? [
          { label: { en: 'Batches drawn', zh: '抽出的批次' }, value: String(real().raw.length) },
          { label: { en: 'Last gradient', zh: '最後一個梯度' }, value: f3(real().raw[39]) },
          { label: { en: 'Running average', zh: '移動平均' }, value: f3(real().avg[39]) },
        ]
        : step >= 6
          ? [
            { label: { en: 'SGD @ step 100', zh: 'SGD 第 100 步' }, value: f3(at(O.sgd.val, 100, O.sgd.every)) },
            { label: { en: 'Adam @ step 100', zh: 'Adam 第 100 步' }, value: f3(at(O.adam.val, 100, O.adam.every)) },
            { label: { en: 'SGD final', zh: 'SGD 最後' }, value: f3(O.sgd.val[O.sgd.val.length - 1] as number) },
            { label: { en: 'Adam final', zh: 'Adam 最後' }, value: f3(O.adam.val[O.adam.val.length - 1] as number) },
          ]
          : (() => {
            const run = step === 2 ? T!.hot : step === 5 ? T!.adam : T!.sgd;
            return [
              { label: { en: 'Optimizer', zh: '最佳化器' }, value: step === 5 ? 'Adam' : 'SGD' },
              { label: { en: 'Learning rate', zh: '學習率' }, value: String(step === 2 ? LR_HOT : step === 5 ? LR_ADAM : LR_SGD) },
              { label: { en: `A after ${STEPS}`, zh: `A（${STEPS} 步後）` }, value: fw(run.end[0]) },
              { label: { en: `B after ${STEPS}`, zh: `B（${STEPS} 步後）` }, value: f2(run.end[1]) },
            ];
          })();

    const toyLegend = [{ tone: 'a' as const, text: { en: 'weight A (steep)', zh: '權重 A（陡）' } }, { tone: 'b' as const, text: { en: 'weight B (gentle)', zh: '權重 B（緩）' } }];
    return {
      metrics,
      items,
      legend: step === 2 ? [...toyLegend, { tone: 'bad', text: { en: 'weight A, too-high learning rate', zh: '權重 A，學習率太高' } }]
        : step === 3 ? [{ tone: 'a', text: { en: 'running average (momentum)', zh: '移動平均（動量）' } }]
          : step >= 1 && step <= 5 ? toyLegend
            : step >= 6 ? [{ tone: 'b', text: { en: `SGD · learning rate ${O.sgd.lr[0]}`, zh: `SGD · 學習率 ${O.sgd.lr[0]}` } }, { tone: 'a', text: { en: `Adam · learning rate ${O.adam.lr[0]}`, zh: `Adam · 學習率 ${O.adam.lr[0]}` } }]
              : undefined,
    };
  },
};
