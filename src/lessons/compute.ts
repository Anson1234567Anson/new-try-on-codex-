import type { Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { TRAIN_NAMES, VAL_NAMES, example, rng } from '../model/data';
import weights from '../model/weights.json';
import experiments from '../model/experiments.json';
import { bits32, bitsBF16, bitsF16, toBF16, toF16 } from './compute.float';

// How much arithmetic our main training run did, and how numbers are stored while training.

const MAIN = experiments.main;
const N = MAIN.params; // 3,115 weights

interface Data {
  tokens: number; // predicted tokens in the real batches of the main run
  flops: number;
  w: number; // one real weight of the trained model
  where: string; // which one
  loss: { f64: number; f32: number; f16: number; bf16: number }; // held-out loss with every weight rounded
  lrEnd: number;
  after100: { f32: number; f16: number; bf16: number }; // the weight after 100 updates of +lrEnd
}

function measure(): Data {
  // Replay exactly how the main run drew its batches (same seeded generator), and count the
  // tokens the model had to predict: one per letter plus the end mark.
  const ex = TRAIN_NAMES.map(example);
  const rand = rng(2);
  let tokens = 0;
  for (let s = 0; s < MAIN.steps; s++) for (let b = 0; b < MAIN.batch; b++) tokens += ex[Math.floor(rand() * ex.length)].y.length;

  const model = GPT.fromJSON(weights as Weights);
  // the weight with the largest size: row 7, column 12 of the position table
  let best = { name: 'tok', i: 0, v: 0 };
  for (const [name, m] of Object.entries(model.p)) m.d.forEach((v, i) => { if (Math.abs(v) > Math.abs(best.v)) best = { name, i, v }; });
  const cols = model.p[best.name].c;

  const val = VAL_NAMES.map(example);
  const rounded = (f: (v: number) => number) => {
    const c = model.clone();
    for (const m of Object.values(c.p)) for (let i = 0; i < m.d.length; i++) m.d[i] = f(m.d[i]);
    return c.loss(val);
  };
  const lrEnd = MAIN.lr[MAIN.lr.length - 1];
  const repeat = (f: (v: number) => number) => { let x = f(best.v); for (let i = 0; i < 100; i++) x = f(x + lrEnd); return x; };
  return {
    tokens, flops: 6 * N * tokens, w: best.v,
    where: `${best.name}[${Math.floor(best.i / cols)}][${best.i % cols}]`,
    loss: { f64: model.loss(val), f32: rounded(Math.fround), f16: rounded(toF16), bf16: rounded(toBF16) },
    lrEnd,
    after100: { f32: repeat(Math.fround), f16: repeat(toF16), bf16: repeat(toBF16) },
  };
}
let cache: Data | undefined;
const data = () => (cache ??= measure());

const fmt = (n: number) => n.toLocaleString('en-US');
const sci = (v: number) => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(2)} × 10^${e}`; };
const num = (v: number, d = 7) => (Number.isFinite(v) ? (v < 0 ? '−' : '') + Math.abs(v).toFixed(d) : '∞');
const small = (v: number) => (v === 0 ? '0' : Math.abs(v) < 1e-4 ? v.toExponential(4) : fmt(+v.toPrecision(6)));

const FORMATS = [
  { k: 'f32', name: 'fp32', bits: (v: number) => bits32(v), round: Math.fround, exp: 8, frac: 23 },
  { k: 'f16', name: 'fp16', bits: bitsF16, round: toF16, exp: 5, frac: 10 },
  { k: 'bf', name: 'bf16', bits: bitsBF16, round: toBF16, exp: 8, frac: 7 },
] as const;

const CELL = 24;
const X0 = 450 - (32 * CELL) / 2;

export const computeLesson: Lesson = {
  id: 'compute',
  title: { en: 'Compute and Mixed Precision', zh: '運算資源與混合精度' },
  blurb: {
    en: 'Count the arithmetic in a real training run, and see how 16-bit numbers save memory without breaking training.',
    zh: '算一算一次真實訓練用了多少運算，再看 16 位元的數字如何省下記憶體，又不弄壞訓練。',
  },
  minutes: 12,
  level: 'intermediate',
  learn: [
    { text: { en: 'The 6 × weights × tokens rule of thumb', zh: '「6 × 權重 × 詞元」的經驗法則' }, step: 2 },
    { text: { en: 'How fp32, fp16 and bf16 store a number', zh: 'fp32、fp16、bf16 如何儲存一個數字' }, step: 4 },
    { text: { en: 'Why small updates vanish in 16 bits', zh: '為什麼小小的更新在 16 位元裡會消失' }, step: 6 },
    { text: { en: 'How mixed precision keeps a 32-bit master copy', zh: '混合精度如何保留一份 32 位元的主副本' }, step: 7 },
    { text: { en: 'What uses memory during training', zh: '訓練時記憶體花在哪裡' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'How much work is one training run?', zh: '一次訓練要做多少工作？' },
      body: [
        { en: 'Our main training run took 4,000 steps, and each step read a <b>batch</b> of 32 names: 128,000 names in all.', zh: '我們的主要訓練跑了 4,000 步，每一步讀一個 32 個名字的<b>批次</b>，總共讀了 128,000 個名字。' },
        { en: 'Large models train on vastly more text with vastly more weights. Before buying computers for that, people estimate the cost. Let us do the estimate for our own run, with real numbers.', zh: '大型模型要用多得多的文字、多得多的權重來訓練。在花錢買電腦之前，大家會先估算成本。我們就用真實的數字，替自己的這次訓練估一次。' },
      ],
      callout: { kind: 'insight', text: { en: 'Two numbers decide the cost: how many weights the model has, and how many tokens it trains on.', zh: '決定成本的是兩個數字：模型有多少權重，以及它用多少詞元來訓練。' } },
    },
    {
      short: { en: 'Tokens', zh: '詞元' }, chapter: 'technique',
      title: { en: 'Count the tokens it learned from', zh: '數一數它學過多少詞元' },
      body: [
        { en: 'Each name is a set of little quizzes: <code>emma</code> asks the model to predict e, m, m, a and the end mark, so 5 predicted tokens. A name with n letters gives n + 1.', zh: '每個名字都是一組小考題：<code>emma</code> 要模型預測 e、m、m、a 和結尾符號，一共 5 個詞元。n 個字母的名字，就有 n + 1 題。' },
        { en: 'We replayed the exact batches of the main run and counted: <b>824,179</b> predicted tokens, about 6.44 per name.', zh: '我們重播了主要訓練實際用的每一個批次，數出來是 <b>824,179</b> 個被預測的詞元，平均每個名字約 6.44 個。' },
      ],
      changed: [{ sign: '+', text: { en: 'D = 824,179 training tokens', zh: 'D = 824,179 個訓練詞元' } }],
    },
    {
      short: { en: '6 × N × D', zh: '6 × N × D' }, chapter: 'technique',
      title: { en: 'The rule of thumb: 6 × weights × tokens', zh: '經驗法則：6 × 權重 × 詞元' },
      body: [
        { en: 'For every token, the forward pass uses each weight about once: one multiply and one add, so about 2 operations per weight. The backward pass costs about twice that. Together: about <b>6 operations per weight per token</b>.', zh: '每處理一個詞元，前向計算大約會用到每個權重一次：一次乘法加一次加法，也就是每個權重約 2 次運算。反向計算大約是它的兩倍。加起來就是<b>每個權重、每個詞元約 6 次運算</b>。' },
        { en: 'This rule, from the 2020 “Scaling Laws” paper by Kaplan and colleagues, gives 6 × 3,115 × 824,179 ≈ 1.54 × 10^10 floating-point operations for our run, which took 32.8 seconds.', zh: '這條法則出自 Kaplan 等人 2020 年的〈Scaling Laws〉論文。套到我們的訓練：6 × 3,115 × 824,179 ≈ 1.54 × 10^10 次浮點數運算，實際花了 32.8 秒。' },
      ],
      callout: { kind: 'note', text: { en: 'It is an estimate. It ignores some work, such as attention between tokens, which is small for our 8-token window.', zh: '這只是估算，它忽略了一些運算，例如詞元之間的注意力計算；對我們只有 8 個詞元的視窗來說，這部分很小。' } },
      changed: [{ sign: '+', text: { en: 'Compute ≈ 1.54 × 10^10 operations', zh: '運算量 ≈ 1.54 × 10^10 次' } }],
    },
    {
      short: { en: 'fp32', zh: 'fp32' }, chapter: 'technique',
      title: { en: 'How a computer stores one weight', zh: '電腦如何儲存一個權重' },
      body: [
        { en: 'Weights are <b>floating-point</b> numbers, stored like scientific notation in binary: a <b>sign</b> (+ or −), an <b>exponent</b> (how big, roughly) and a <b>fraction</b> (the precise digits).', zh: '權重是<b>浮點數</b>，用二進位的科學記號儲存：一個<b>正負號</b>（+ 或 −）、一段<b>指數</b>（大約多大），和一段<b>小數</b>（精確的位數）。' },
        { en: 'Here is a real weight of our trained model, from its position table, in the standard 32-bit format <b>fp32</b>: 1 + 8 + 23 bits, 4 bytes.', zh: '這是我們訓練好的模型裡一個真實的權重，來自位置表，用標準的 32 位元格式 <b>fp32</b> 儲存：1 + 8 + 23 個位元，共 4 個位元組。' },
      ],
      callout: { kind: 'note', text: { en: 'Our browser code actually computes with 64-bit numbers, which is simplest in JavaScript. Large models train with 32 and 16 bits.', zh: '我們在瀏覽器裡的程式其實是用 64 位元的數字在算，這在 JavaScript 裡最簡單。大型模型則用 32 位元和 16 位元訓練。' } },
    },
    {
      short: { en: '16 bits', zh: '16 位元' }, chapter: 'technique',
      title: { en: 'Two ways to use only 16 bits', zh: '只用 16 位元的兩種方法' },
      body: [
        { en: '<b>fp16</b> keeps 5 exponent bits and 10 fraction bits. <b>bf16</b> (“brain float”) keeps all 8 exponent bits of fp32 but only 7 fraction bits: it is simply the top half of fp32.', zh: '<b>fp16</b> 保留 5 個指數位元和 10 個小數位元。<b>bf16</b>（brain float）保留 fp32 全部 8 個指數位元，小數卻只剩 7 個：它其實就是 fp32 的前半段。' },
        { en: 'Both halve the memory. Rounding our weight to them costs an error of about 0.001, which is small next to the weight itself.', zh: '兩者都讓記憶體減半。把我們的權重四捨五入到這兩種格式，誤差約為 0.001，和權重本身相比很小。' },
      ],
      changed: [{ sign: '+', text: { en: 'fp16 and bf16: 2 bytes each', zh: 'fp16 與 bf16：各 2 個位元組' } }],
    },
    {
      short: { en: 'Range', zh: '範圍' }, chapter: 'technique',
      title: { en: 'Precision versus range', zh: '精度 vs 範圍' },
      body: [
        { en: 'fp16 is more precise, but its few exponent bits limit the range: above 65,504 it overflows to infinity, and tiny numbers like 0.00000001 become 0. bf16 has the same range as fp32, so it rarely overflows, but it is less precise.', zh: 'fp16 比較精確，但指數位元太少，範圍有限：超過 65,504 就溢位成無限大，而像 0.00000001 這樣的極小數會變成 0。bf16 的範圍和 fp32 一樣，幾乎不會溢位，但精度較差。' },
        { en: 'For our trained model, rounding <i>every</i> weight barely matters: the held-out loss stays at 2.207 in all three formats.', zh: '對我們訓練好的模型來說，把<i>每一個</i>權重都四捨五入幾乎沒有影響：三種格式下，保留集上的損失都維持在 2.207。' },
      ],
      callout: { kind: 'insight', text: { en: 'Using a trained model in 16 bits is usually fine. The trouble starts during training, when weights must change by tiny amounts.', zh: '用 16 位元來「使用」訓練好的模型，通常沒問題。麻煩出在訓練過程中，因為權重每次只能改變一點點。' } },
    },
    {
      short: { en: 'Case: lost update', zh: '案例：更新消失' }, chapter: 'problem',
      title: { en: 'Case: a small update disappears', zh: '案例：小小的更新不見了' },
      body: [
        { en: 'At the end of our main run the learning rate was 0.002, so a weight moves by about that much per step. Add 0.002 to our weight stored in bf16, and the result rounds back to the very same number.', zh: '主要訓練快結束時，學習率是 0.002，所以一個權重每一步大約移動這麼多。把 0.002 加到用 bf16 儲存的權重上，結果四捨五入後又變回同一個數字。' },
        { en: 'Repeat it 100 times: in fp32 the weight moves by 0.2 as it should; in bf16 it never moves at all. fp16 keeps up only because 0.002 is just above its spacing near 2.3 (about 0.002); a smaller step would vanish there too.', zh: '重複 100 次：在 fp32 中，權重如預期移動了 0.2；在 bf16 中，它一動也不動。fp16 還撐得住，只是因為 0.002 剛好大於它在 2.3 附近的間距（約 0.002）；更小的一步在 fp16 裡也會消失。' },
      ],
      callout: { kind: 'insight', text: { en: 'Near 2.3, bf16 can only store numbers about 0.016 apart. Any change smaller than half of that is rounded away.', zh: '在 2.3 附近，bf16 能儲存的數字彼此相差約 0.016。任何小於這個間距一半的改變，都會被四捨五入掉。' } },
    },
    {
      short: { en: 'Mixed precision', zh: '混合精度' }, chapter: 'solution',
      title: { en: 'Mixed precision: 16-bit work, 32-bit memory', zh: '混合精度：用 16 位元計算，用 32 位元記住' },
      body: [
        { en: 'The fix, described by Micikevicius and colleagues (2018): keep a <b>master copy</b> of every weight in fp32. For each step, round it to a 16-bit copy and do the heavy forward and backward arithmetic with that.', zh: '解法由 Micikevicius 等人（2018）提出：每個權重都保留一份 fp32 的<b>主副本</b>。每一步先把它四捨五入成 16 位元的副本，用它來做繁重的前向和反向計算。' },
        { en: 'The update is then added to the fp32 master, where small changes survive. GPUs do 16-bit arithmetic much faster, so most of the work gets cheaper while learning stays exact enough.', zh: '接著把更新加到 fp32 主副本上，小小的改變就能保留下來。GPU 做 16 位元運算快得多，所以大部分的工作變便宜了，學習卻依然夠精確。' },
      ],
      callout: { kind: 'note', text: { en: 'With fp16, tiny gradients can also round to 0. So training multiplies the loss by a big factor first (loss scaling) and divides the gradients back afterwards.', zh: '使用 fp16 時，極小的梯度也可能被捨入成 0。所以訓練時會先把損失乘上一個大倍數（損失縮放），算完梯度再除回來。' } },
      changed: [{ sign: '+', text: { en: 'fp32 master weights', zh: 'fp32 主權重' } }, { sign: '~', text: { en: 'forward and backward in 16 bits', zh: '前向與反向改用 16 位元' } }],
    },
    {
      short: { en: 'Memory', zh: '記憶體' }, chapter: 'technique',
      title: { en: 'Memory: about 16 bytes per weight', zh: '記憶體：每個權重約 16 個位元組' },
      body: [
        { en: 'Training with the Adam optimizer keeps, for every weight: the weight, its gradient, and Adam’s two running averages. In fp32 that is 4 + 4 + 4 + 4 = 16 bytes. Mixed precision is 2 + 2 + 4 + 4 + 4 = 16 bytes too, because of the master copy.', zh: '用 Adam 最佳化器訓練時，每個權重都要存：權重本身、它的梯度，以及 Adam 的兩個移動平均。用 fp32 就是 4 + 4 + 4 + 4 = 16 個位元組。混合精度因為多了主副本，2 + 2 + 4 + 4 + 4 也是 16 個位元組。' },
        { en: 'Our model: 3,115 × 16 = 49,840 bytes, about 50 KB. The ZeRO paper’s example: a 7.5-billion-weight model needs 120 GB just for this, before counting the activations.', zh: '我們的模型：3,115 × 16 = 49,840 個位元組，大約 50 KB。ZeRO 論文舉的例子：一個 75 億權重的模型光是這些就需要 120 GB，還沒算激活值。' },
      ],
      callout: { kind: 'insight', text: { en: 'Mixed precision saves time, not this memory. The memory win comes from the 16-bit activations and from tricks like LoRA, coming up soon.', zh: '混合精度省的是時間，不是這部分的記憶體。記憶體上的好處來自 16 位元的激活值，以及像 LoRA 這樣的技巧（很快就會介紹）。' } },
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Counting the cost of training', zh: '計算訓練的成本' },
      body: [
        { en: '<b>Compute</b> ≈ 6 × weights × tokens: 1.54 × 10^10 for our run.<br><b>fp32</b> 4 bytes; <b>fp16</b> more precise, small range; <b>bf16</b> fp32’s range, less precise.<br><b>Mixed precision</b>: 16-bit arithmetic, fp32 master copy for the updates.<br><b>Memory</b> with Adam: about 16 bytes per weight.', zh: '<b>運算量</b> ≈ 6 × 權重 × 詞元：我們的訓練是 1.54 × 10^10。<br><b>fp32</b> 4 個位元組；<b>fp16</b> 較精確、範圍小；<b>bf16</b> 範圍同 fp32、精度較低。<br><b>混合精度</b>：用 16 位元運算，用 fp32 主副本做更新。<br>使用 Adam 時的<b>記憶體</b>：每個權重約 16 個位元組。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Checkpoints and Evaluation. A long run must be saved along the way, and we need fair ways to tell whether it is getting better.', zh: '下一課：檢查點與評估。長時間的訓練要邊跑邊存檔，我們也需要公平的方法，來判斷它是否越來越好。' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const D = step >= 1 ? data() : undefined;
    let tokens = '—', flops = '—', bytes = '—';

    if (step <= 2) {
      // the run: steps × batch
      items.push({ k: 'rs', t: 'box', x: 70, y: 40, w: 200, h: 64, label: fmt(MAIN.steps), sub: t({ en: 'steps', zh: '步' }) });
      items.push({ k: 'rx', t: 'text', x: 300, y: 80, text: '×', size: 26, anchor: 'middle', bold: true });
      items.push({ k: 'rb', t: 'box', x: 330, y: 40, w: 200, h: 64, label: fmt(MAIN.batch), sub: t({ en: 'names per batch', zh: '每批次的名字數' }) });
      items.push({ k: 're', t: 'text', x: 560, y: 80, text: '=', size: 26, anchor: 'middle', bold: true });
      items.push({ k: 'rn', t: 'box', x: 590, y: 40, w: 240, h: 64, label: fmt(MAIN.steps * MAIN.batch), sub: t({ en: 'names read', zh: '讀過的名字' }), tone: 'a' });
      if (step === 0) {
        items.push({ k: 'rq1', t: 'box', x: 160, y: 220, w: 260, h: 80, label: t({ en: `${fmt(N)} weights`, zh: `${fmt(N)} 個權重` }), sub: t({ en: 'N: model size', zh: 'N：模型大小' }) });
        items.push({ k: 'rq2', t: 'box', x: 480, y: 220, w: 260, h: 80, label: '?', sub: t({ en: 'D: training tokens', zh: 'D：訓練詞元數' }), tone: 'b' });
        items.push({ k: 'rq3', t: 'text', x: 450, y: 380, text: t({ en: 'cost = ?', zh: '成本 = ?' }), size: 22, anchor: 'middle', bold: true, tone: 'muted' });
      }
    }

    if (step === 1 && D) {
      const seq = ['.', 'e', 'm', 'm', 'a'], tgt = ['e', 'm', 'm', 'a', '.'];
      const x0 = 450 - (5 * 62 - 18) / 2;
      seq.forEach((c, i) => {
        items.push({ k: 'sq' + i, t: 'token', x: x0 + i * 62, y: 160, text: c, tone: c === '.' ? 'b' : 'ink' });
        items.push({ k: 'sqa' + i, t: 'arrow', x1: x0 + i * 62 + 22, y1: 208, x2: x0 + i * 62 + 22, y2: 246, tone: 'a' });
        items.push({ k: 'tg' + i, t: 'token', x: x0 + i * 62, y: 250, text: tgt[i], tone: 'a' });
      });
      items.push({ k: 'tgl', t: 'text', x: x0 - 20, y: 278, text: t({ en: '5 predictions', zh: '5 次預測' }), anchor: 'end', tone: 'muted', size: 14 });
      items.push({ k: 'tok', t: 'box', x: 250, y: 360, w: 400, h: 70, label: t({ en: `${fmt(D.tokens)} tokens`, zh: `${fmt(D.tokens)} 個詞元` }), sub: t({ en: `in the real batches · ${(D.tokens / (MAIN.steps * MAIN.batch)).toFixed(2)} per name`, zh: `實際批次中 · 平均每個名字 ${(D.tokens / (MAIN.steps * MAIN.batch)).toFixed(2)} 個` }), tone: 'a' });
    }

    if (step === 2 && D) {
      items.push({ k: 'fw', t: 'box', x: 70, y: 170, w: 230, h: 64, label: t({ en: `forward ≈ 2 × ${fmt(N)}`, zh: `前向 ≈ 2 × ${fmt(N)}` }), sub: t({ en: 'per token', zh: '每個詞元' }) });
      items.push({ k: 'bw', t: 'box', x: 335, y: 170, w: 230, h: 64, label: t({ en: `backward ≈ 4 × ${fmt(N)}`, zh: `反向 ≈ 4 × ${fmt(N)}` }), sub: t({ en: 'about twice the forward', zh: '大約是前向的兩倍' }) });
      items.push({ k: 'tw', t: 'box', x: 600, y: 170, w: 230, h: 64, label: t({ en: `total ≈ 6 × ${fmt(N)}`, zh: `合計 ≈ 6 × ${fmt(N)}` }), sub: t({ en: `= ${fmt(6 * N)} per token`, zh: `= 每個詞元 ${fmt(6 * N)} 次` }), tone: 'a' });
      items.push({ k: 'fl', t: 'text', x: 450, y: 320, text: `6 × ${fmt(N)} × ${fmt(D.tokens)}`, size: 20, anchor: 'middle', mono: true });
      items.push({ k: 'fl2', t: 'text', x: 450, y: 372, text: `≈ ${sci(D.flops)}`, size: 30, anchor: 'middle', bold: true, tone: 'a' });
      items.push({ k: 'fl3', t: 'text', x: 450, y: 420, text: t({ en: `floating-point operations (the real run took ${MAIN.seconds} s)`, zh: `次浮點數運算（實際訓練花了 ${MAIN.seconds} 秒）` }), size: 14, anchor: 'middle', tone: 'muted' });
    }

    // bit patterns of one real weight
    const bitRow = (f: (typeof FORMATS)[number], y: number) => {
      const bits = f.bits(D!.w);
      const stored = f.round(D!.w);
      const tone = (i: number): Tone => (i === 0 ? 'b' : i <= f.exp ? 'a' : 'ink');
      items.push({ k: f.k + 'name', t: 'text', x: X0, y: y - 12, text: `${f.name} · ${t({ en: `1 sign + ${f.exp} exponent + ${f.frac} fraction bits`, zh: `1 個正負號 + ${f.exp} 個指數 + ${f.frac} 個小數位元` })}`, size: 14.5, bold: true });
      [...bits].forEach((b, i) => items.push({ k: `${f.k}b${i}`, t: 'box', x: X0 + i * CELL, y, w: CELL - 3, h: 32, label: b, tone: tone(i) }));
      items.push({ k: f.k + 'val', t: 'text', x: X0, y: y + 56, text: t({ en: `stores ${num(stored)} · error ${Math.abs(stored - D!.w).toFixed(7)}`, zh: `存成 ${num(stored)} · 誤差 ${Math.abs(stored - D!.w).toFixed(7)}` }), size: 14, mono: true, tone: 'muted' });
    };

    if ((step === 3 || step === 4) && D) {
      items.push({ k: 'wv', t: 'text', x: 450, y: 40, text: t({ en: `a real weight: ${D.where} = ${num(D.w, 4)}`, zh: `一個真實的權重：${D.where} = ${num(D.w, 4)}` }), size: 17, bold: true, anchor: 'middle', tone: 'a' });
      bitRow(FORMATS[0], step === 3 ? 160 : 100);
      if (step === 3) {
        items.push({ k: 'gs', t: 'text', x: X0 + 10, y: 256, text: t({ en: 'sign', zh: '正負號' }), size: 13, tone: 'b', anchor: 'middle' });
        items.push({ k: 'ge', t: 'text', x: X0 + 5 * CELL, y: 256, text: t({ en: 'exponent: how big', zh: '指數：大約多大' }), size: 13, tone: 'a', anchor: 'middle' });
        items.push({ k: 'gf', t: 'text', x: X0 + 21 * CELL, y: 256, text: t({ en: 'fraction: the precise digits', zh: '小數：精確的位數' }), size: 13, anchor: 'middle' });
        items.push({ k: 'gb', t: 'text', x: 450, y: 350, text: t({ en: '32 bits = 4 bytes per weight', zh: '32 位元 = 每個權重 4 個位元組' }), size: 20, anchor: 'middle', bold: true });
      } else {
        bitRow(FORMATS[1], 250);
        bitRow(FORMATS[2], 400);
      }
      bytes = step === 3 ? '4' : '4 → 2';
    }

    if (step === 5 && D) {
      const vals = [D.w, 100000, 1e-8];
      const colX = [250, 470, 690];
      items.push({ k: 'th', t: 'text', x: 60, y: 60, text: t({ en: 'number', zh: '數字' }), size: 14, tone: 'muted' });
      vals.forEach((v, j) => items.push({ k: 'tc' + j, t: 'text', x: colX[j] + 90, y: 60, text: j === 0 ? t({ en: 'our weight', zh: '我們的權重' }) : j === 1 ? '100,000' : '0.00000001', size: 14, anchor: 'middle', bold: true }));
      FORMATS.forEach((f, i) => {
        const y = 80 + i * 66;
        items.push({ k: 'tr' + i, t: 'text', x: 60, y: y + 30, text: f.name, size: 16, bold: true, mono: true });
        vals.forEach((v, j) => {
          const s = f.round(v);
          const bad = !Number.isFinite(s) || s === 0;
          const label = !Number.isFinite(s) ? t({ en: '∞ overflow', zh: '∞ 溢位' }) : s === 0 ? t({ en: '0 underflow', zh: '0 下溢' }) : j === 0 ? num(s, 6) : small(s);
          items.push({ k: `tv${i}-${j}`, t: 'box', x: colX[j], y, w: 180, h: 48, label, tone: bad ? 'bad' : 'ink' });
        });
      });
      items.push({ k: 'ml', t: 'text', x: 60, y: 330, text: t({ en: 'every weight of our model rounded · held-out loss', zh: '把模型每個權重都四捨五入 · 保留集損失' }), size: 14, tone: 'muted' });
      (['f32', 'f16', 'bf16'] as const).forEach((k, i) => {
        items.push({ k: 'ml' + i, t: 'box', x: colX[i], y: 350, w: 180, h: 56, label: D.loss[k].toFixed(4), sub: ['fp32', 'fp16', 'bf16'][i], tone: 'a' });
      });
      items.push({ k: 'ml64', t: 'text', x: 450, y: 450, text: t({ en: `unrounded (64-bit): ${D.loss.f64.toFixed(4)}`, zh: `未捨入（64 位元）：${D.loss.f64.toFixed(4)}` }), size: 14, anchor: 'middle', tone: 'muted' });
      bytes = '4 / 2 / 2';
    }

    if (step === 6 && D) {
      items.push({ k: 'uh1', t: 'text', x: 200, y: 50, text: t({ en: 'stored weight', zh: '儲存的權重' }), size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'uh2', t: 'text', x: 460, y: 50, text: t({ en: `after + ${D.lrEnd}`, zh: `加上 ${D.lrEnd} 之後` }), size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'uh3', t: 'text', x: 740, y: 50, text: t({ en: `after 100 such steps`, zh: `重複 100 步之後` }), size: 14, anchor: 'middle', tone: 'muted' });
      FORMATS.forEach((f, i) => {
        const y = 80 + i * 110;
        const w0 = f.round(D.w), w1 = f.round(w0 + D.lrEnd);
        const w100 = D.after100[f.k === 'bf' ? 'bf16' : f.k];
        const stuck = w1 === w0;
        items.push({ k: 'ul' + i, t: 'text', x: 40, y: y + 34, text: f.name, size: 16, bold: true, mono: true });
        items.push({ k: 'u0' + i, t: 'box', x: 110, y, w: 180, h: 54, label: num(w0) });
        items.push({ k: 'ua' + i, t: 'arrow', x1: 296, y1: y + 27, x2: 364, y2: y + 27, tone: stuck ? 'bad' : 'a' });
        items.push({ k: 'u1' + i, t: 'box', x: 370, y, w: 180, h: 54, label: num(w1), sub: stuck ? t({ en: 'unchanged', zh: '沒有改變' }) : undefined, tone: stuck ? 'bad' : 'a' });
        items.push({ k: 'ub' + i, t: 'arrow', x1: 556, y1: y + 27, x2: 644, y2: y + 27, tone: 'muted', dashed: true });
        items.push({ k: 'u2' + i, t: 'box', x: 650, y, w: 180, h: 54, label: num(w100), sub: t({ en: `moved ${(w100 - w0).toFixed(4)}`, zh: `移動了 ${(w100 - w0).toFixed(4)}` }), tone: w100 === w0 ? 'bad' : 'a' });
      });
      bytes = '4 / 2 / 2';
      items.push({ k: 'ut', t: 'text', x: 450, y: 440, text: t({ en: `should move: 100 × ${D.lrEnd} = ${(100 * D.lrEnd).toFixed(1)}`, zh: `應該移動：100 × ${D.lrEnd} = ${(100 * D.lrEnd).toFixed(1)}` }), size: 15, anchor: 'middle', bold: true });
    }

    if (step === 7) {
      const B = (k: string, x: number, y: number, label: string, sub: string, tone: Tone = 'ink') => items.push({ k, t: 'box', x, y, w: 230, h: 70, label, sub, tone });
      B('m0', 50, 80, t({ en: 'master weights', zh: '主權重' }), t({ en: 'fp32 · 4 bytes', zh: 'fp32 · 4 個位元組' }), 'a');
      B('m1', 335, 80, t({ en: 'rounded copy', zh: '捨入後的副本' }), t({ en: '16 bits · 2 bytes', zh: '16 位元 · 2 個位元組' }), 'b');
      B('m2', 620, 80, t({ en: 'forward + backward', zh: '前向 + 反向' }), t({ en: 'heavy arithmetic in 16 bits', zh: '繁重運算用 16 位元' }), 'b');
      B('m3', 620, 300, t({ en: 'gradients', zh: '梯度' }), t({ en: '16 bits', zh: '16 位元' }), 'b');
      B('m4', 335, 300, t({ en: 'update', zh: '更新' }), t({ en: 'added in fp32: small steps survive', zh: '在 fp32 中相加：小步也保留' }), 'a');
      items.push({ k: 'ma0', t: 'arrow', x1: 284, y1: 115, x2: 331, y2: 115 });
      items.push({ k: 'ma1', t: 'arrow', x1: 569, y1: 115, x2: 616, y2: 115 });
      items.push({ k: 'ma2', t: 'arrow', x1: 735, y1: 154, x2: 735, y2: 296 });
      items.push({ k: 'ma3', t: 'arrow', x1: 616, y1: 335, x2: 569, y2: 335 });
      items.push({ k: 'ma4', t: 'arrow', x1: 335, y1: 335, x2: 165, y2: 335, tone: 'a' });
      items.push({ k: 'ma5', t: 'arrow', x1: 165, y1: 335, x2: 165, y2: 154, tone: 'a' });
      items.push({ k: 'mt', t: 'text', x: 450, y: 450, text: t({ en: 'one training step, repeated thousands of times', zh: '一個訓練步驟，重複數千次' }), size: 15, anchor: 'middle', tone: 'muted' });
      bytes = '2 + 4';
    }

    if (step === 8) {
      const SC = 40;
      const bar = (k: string, y: number, title: string, segs: { b: number; sub: string; tone: Tone }[]) => {
        items.push({ k: k + 't', t: 'text', x: 60, y: y - 12, text: title, size: 14.5, bold: true });
        let x = 60;
        segs.forEach((s, i) => { items.push({ k: `${k}${i}`, t: 'box', x, y, w: s.b * SC - 4, h: 60, label: `${s.b}`, sub: s.sub, tone: s.tone }); x += s.b * SC; });
        items.push({ k: k + 'sum', t: 'text', x: x + 12, y: y + 36, text: t({ en: '= 16 bytes', zh: '= 16 個位元組' }), size: 16, bold: true });
      };
      const W = t({ en: 'weight', zh: '權重' }), G = t({ en: 'gradient', zh: '梯度' }), A1 = t({ en: 'Adam avg 1', zh: 'Adam 平均 1' }), A2 = t({ en: 'Adam avg 2', zh: 'Adam 平均 2' });
      bar('fp', 60, t({ en: 'fp32 training (bytes per weight)', zh: 'fp32 訓練（每個權重的位元組）' }), [
        { b: 4, sub: W, tone: 'ink' }, { b: 4, sub: G, tone: 'ink' }, { b: 4, sub: A1, tone: 'muted' }, { b: 4, sub: A2, tone: 'muted' }]);
      bar('mp', 190, t({ en: 'mixed precision', zh: '混合精度' }), [
        { b: 2, sub: W, tone: 'b' }, { b: 2, sub: G, tone: 'b' }, { b: 4, sub: t({ en: 'fp32 master', zh: 'fp32 主副本' }), tone: 'a' }, { b: 4, sub: A1, tone: 'muted' }, { b: 4, sub: A2, tone: 'muted' }]);
      items.push({ k: 'mo1', t: 'box', x: 60, y: 320, w: 380, h: 70, label: `${fmt(N)} × 16 = ${fmt(N * 16)} B`, sub: t({ en: 'our model · about 50 KB', zh: '我們的模型 · 約 50 KB' }), tone: 'a' });
      items.push({ k: 'mo2', t: 'box', x: 470, y: 320, w: 380, h: 70, label: t({ en: '7.5 billion × 16 = 120 GB', zh: '75 億 × 16 = 120 GB' }), sub: t({ en: 'example from the ZeRO paper', zh: 'ZeRO 論文中的例子' }), tone: 'bad' });
      items.push({ k: 'mo3', t: 'text', x: 450, y: 440, text: t({ en: 'plus the activations, which grow with the batch size', zh: '另外還要加上激活值，它會隨批次大小增加' }), size: 14, anchor: 'middle', tone: 'muted' });
      bytes = '16';
    }

    if (step === 9 && D) {
      const cards = [
        { label: `≈ ${sci(D.flops)}`, sub: t({ en: 'compute: 6 × N × D', zh: '運算量：6 × N × D' }), tone: 'a' as Tone },
        { label: 'fp32 · fp16 · bf16', sub: t({ en: '4 · 2 · 2 bytes', zh: '4 · 2 · 2 個位元組' }), tone: 'ink' as Tone },
        { label: t({ en: 'fp32 master copy', zh: 'fp32 主副本' }), sub: t({ en: 'small updates survive', zh: '小小的更新得以保留' }), tone: 'b' as Tone },
        { label: t({ en: '16 bytes per weight', zh: '每個權重 16 個位元組' }), sub: t({ en: 'training with Adam', zh: '用 Adam 訓練時' }), tone: 'ink' as Tone },
      ];
      cards.forEach((c, i) => items.push({ k: 'rc' + i, t: 'box', x: 70 + (i % 2) * 400, y: 120 + Math.floor(i / 2) * 140, w: 360, h: 90, label: c.label, sub: c.sub, tone: c.tone }));
      bytes = '16';
    }

    if (D) { tokens = fmt(D.tokens); }
    if (step >= 2 && D) flops = sci(D.flops).replace(' × 10^', 'e');
    if (step === 1) bytes = '—';

    return {
      metrics: [
        { label: { en: 'Weights (N)', zh: '權重數（N）' }, value: fmt(N) },
        { label: { en: 'Tokens (D)', zh: '詞元數（D）' }, value: tokens },
        { label: { en: 'Operations', zh: '運算次數' }, value: flops },
        { label: { en: 'Bytes per weight', zh: '每個權重的位元組' }, value: bytes },
      ],
      items,
      legend: step === 3 || step === 4
        ? [{ tone: 'b', text: { en: 'sign', zh: '正負號' } }, { tone: 'a', text: { en: 'exponent', zh: '指數' } }]
        : undefined,
    };
  },
};
