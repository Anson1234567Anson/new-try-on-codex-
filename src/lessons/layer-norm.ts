import type { Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, DEFAULT_CONFIG, type Weights } from '../model/gpt';
import { VAL_NAMES, encode, example, rng } from '../model/data';
import { Mat, layerNorm } from '../model/tensor';
import weights from '../model/weights.json';
import experiments from '../model/experiments.json';

// We normalize a real vector from the trained model (the second "m" of ".jimm"
// after the block), and look at what happens to vector sizes in deep stacks.

const WORD = '.jimm';
const F = WORD.length - 1;
const DEPTH = 12;

const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const size = (v: ArrayLike<number>) => Math.sqrt(Array.from(v).reduce((s, x) => s + x * x, 0));
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

interface Data {
  vecs: number[][]; // x, x1, x2 of the second "m"
  v: number[]; mu: number; centered: number[]; sd: number; norm: number[];
  gain: number[]; bias: number[]; out: number[];
  normed: number[][]; // each of vecs, normalized
  avg: number[]; // average size of x, x1, x2 over the held-out names
}
let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const tr = model.forward(encode(WORD));
  const vecs = [tr.x.row(F), tr.x1.row(F), tr.x2.row(F)];
  const v = vecs[2];
  const mu = mean(v);
  const centered = v.map((a) => a - mu);
  const sd = Math.sqrt(mean(centered.map((a) => a * a)));
  // gain and bias start at 1 and 0 when a model with layer norm is created
  const gain = new Mat(1, 16); gain.d.fill(1);
  const bias = new Mat(1, 16);
  const ln = (u: number[]) => { const m = new Mat(1, 16); m.d.set(u); return layerNorm(m, gain, bias).row(0); };
  const avg = [0, 0, 0];
  let n = 0;
  for (const name of VAL_NAMES) {
    const t2 = model.forward(example(name).x);
    for (let i = 0; i < t2.x.r; i++) { avg[0] += size(t2.x.row(i)); avg[1] += size(t2.x1.row(i)); avg[2] += size(t2.x2.row(i)); n++; }
  }
  return {
    vecs, v, mu, centered, sd, norm: centered.map((a) => a / sd),
    gain: Array.from(gain.d), bias: Array.from(bias.d), out: ln(v), normed: vecs.map(ln), avg: avg.map((a) => a / n),
  };
})();

// Fresh (untrained) 12-block versions of our model, without and with pre-norm.
interface Deep { stream: number[]; seen: number }
let deepCache: { plain: Deep; norm: Deep } | undefined;
const deep = () => deepCache ??= (() => {
  const ids = encode('.olivia');
  const avgSize = (m: Mat) => { let s = 0; for (let i = 0; i < m.r; i++) s += size(m.row(i)); return s / m.r; };
  const one = (norm: boolean): Deep => {
    const tr = new GPT({ ...DEFAULT_CONFIG, layers: DEPTH, norm }, rng(1)).forward(ids);
    return { stream: [avgSize(tr.x), ...tr.blocks.map((b) => avgSize(b.x2))], seen: norm ? avgSize(tr.blocks[DEPTH - 1].ln1!) : NaN };
  };
  return { plain: one(false), norm: one(true) };
})();

const V = experiments.variants;
const last = (a: number[]) => a[a.length - 1];
const best = (a: number[]) => Math.min(...a);
const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const big = (v: number) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));

const fx = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
const lnWith = (u: number[], g: number[], b: number[]) => {
  const m = new Mat(1, u.length); m.d.set(u);
  const gm = new Mat(1, g.length); gm.d.set(g);
  const bm = new Mat(1, b.length); bm.d.set(b);
  return layerNorm(m, gm, bm).row(0);
};

// stream diagram geometry (pre-norm step)
const SY = 250;

export const layerNorm_: Lesson = {
  id: 'layer-norm',
  title: { en: 'Layer Normalization', zh: '層正規化' },
  blurb: {
    en: 'Adding again and again makes vectors grow. Layer normalization rescales them to a steady size.',
    zh: '一直往上加，向量就會越變越大。層正規化把它們調回穩定的大小。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'Why vector sizes drift in deep models', zh: '為什麼深層模型裡向量的大小會漂移' }, step: 0 },
    { text: { en: 'Mean and spread (standard deviation)', zh: '平均值與離散程度（標準差）' }, step: 2 },
    { text: { en: 'What the learned gain and bias are for', zh: '學習得來的增益與偏差有什麼用' }, step: 3 },
    { text: { en: 'Where “pre-norm” puts it in a block', zh: '「前置正規化」把它放在區塊的哪裡' }, step: 5 },
    { text: { en: 'How much it helps a tiny model', zh: '它對小模型有多少幫助' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Vectors keep growing', zh: '向量越長越大' },
      body: [
        lz(() => ({ en: `Every sub-layer <i>adds</i> onto the residual stream, so vectors grow. In our trained model the average size goes from ${f2(data().avg[0])} (input) to ${f2(data().avg[1])} (after attention) to ${f2(data().avg[2])} (after the MLP).`, zh: `每個子層都是往殘差流上<i>加</i>東西，所以向量會變大。在我們訓練好的模型裡，平均大小從 ${f2(data().avg[0])}（輸入）變成 ${f2(data().avg[1])}（注意力之後），再到 ${f2(data().avg[2])}（MLP 之後）。` })),
        lz(() => ({ en: `One <b>block</b> (one attention layer plus one MLP layer, with their two additions) copes. Adding keeps the signal alive, but nothing stops it from growing: in the Residual lesson, small MLP layers only grew it a few times over, but ${DEPTH} full blocks of our design (fresh, untrained) compound to ${big(last(deep().plain.stream))}. Later layers get inputs of wildly different sizes, which makes training unstable.`, zh: `一個<b>區塊</b>（一層注意力加一層 MLP，連同它們的兩次相加）還應付得來。相加讓訊號不會消失，但也沒有東西阻止它長大：在「殘差流」那一課，小小的 MLP 層只讓它長大了幾倍；把 ${DEPTH} 個同樣設計的完整區塊疊起來（全新、未訓練），卻一路疊加到 ${big(last(deep().plain.stream))}。後面的層收到的輸入大小天差地遠，訓練就會很不穩定。` })),
      ],
      callout: { kind: 'insight', text: { en: 'Each layer learns to expect inputs of a certain size. If the size keeps drifting, the layers are aiming at a moving target.', zh: '每一層都會習慣某個大小的輸入。如果大小一直漂移，這些層就像在瞄準一個會動的靶。' } },
    },
    {
      short: { en: 'Mean', zh: '平均' }, chapter: 'technique',
      title: { en: 'Step 1: find the average', zh: '第一步：找出平均值' },
      body: [
        lz(() => ({ en: `Take one real vector: the second “m” of <code>.jimm</code> after the block, size ${f2(size(data().v))}. Add up its 16 numbers and divide by 16. That is the <b>mean</b>: ${f3(data().mu)}.`, zh: `拿一個真實的向量來看：<code>.jimm</code> 第二個「m」通過區塊後的向量，大小 ${f2(size(data().v))}。把 16 個數字加起來再除以 16，就是<b>平均值（mean）</b>：${f3(data().mu)}。` })),
        { en: 'Subtract the mean from every number. Now the numbers are centred on zero: the second row.', zh: '每個數字都減掉平均值，數字就以 0 為中心了，也就是第二列。' },
      ],
      changed: [{ sign: '~', text: { en: 'Every number − mean', zh: '每個數字 − 平均值' } }],
    },
    {
      short: { en: 'Spread', zh: '標準差' }, chapter: 'technique',
      title: { en: 'Step 2: measure the spread', zh: '第二步：量出分散程度' },
      body: [
        lz(() => ({ en: `How far are the numbers from zero, typically? Square each one, average the squares, take the square root. This is the <b>standard deviation</b>, the spread: ${f3(data().sd)}.`, zh: `這些數字通常離 0 多遠？把每個數字平方、取平均、再開根號。這就是<b>標準差（standard deviation）</b>，也就是分散程度：${f3(data().sd)}。` })),
        { en: 'Divide every number by the spread. The result has mean 0 and spread 1, whatever the vector looked like before. With 16 numbers its size is always √16 = 4.', zh: '每個數字都除以標準差。結果的平均是 0、標準差是 1，不管原本的向量長什麼樣子都一樣。16 個數字的話，大小一定是 √16 = 4。' },
      ],
      changed: [{ sign: '~', text: { en: 'Every number ÷ spread', zh: '每個數字 ÷ 標準差' } }],
    },
    {
      short: { en: 'Gain & bias', zh: '增益與偏差' }, chapter: 'technique',
      title: { en: 'Step 3: learned gain and bias', zh: '第三步：學習得來的增益與偏差' },
      body: [
        { en: 'Try it: drag the gain of cell 6 and the bias of cell 10, and watch the output row. Forcing every vector into exactly the same shape could throw away something useful, so each of the 16 cells (numbers) of the vector gets a learned <b>gain</b> (multiply) and <b>bias</b> (add).', zh: '動手試試：拖動第 6 格的增益和第 10 格的偏差，看看輸出那一列。硬把每個向量壓成完全一樣的形狀，可能會丟掉有用的資訊，所以向量的 16 個格子（數字）各有一個學習得來的<b>增益（gain，用乘的）</b>和<b>偏差（bias，用加的）</b>。' },
        { en: 'They start at 1 and 0, so at first the output equals the normalized vector. Training then adjusts them, letting the model stretch or shift any cell it needs; the sliders do by hand what training might do.', zh: '它們一開始是 1 和 0，所以一開始輸出就等於正規化後的向量。之後訓練會調整它們，讓模型在需要的格子上拉伸或平移；滑桿就是用手做訓練可能會做的事。' },
      ],
      callout: { kind: 'note', text: { en: 'All three steps together are layer normalization. The last row is computed with the same layerNorm function the model uses.', zh: '這三步合起來就是層正規化（layer normalization）。最後一列是用模型本身的 layerNorm 函式算出來的。' } },
      changed: [{ sign: '+', text: { en: '× gain + bias (32 learned numbers)', zh: '× 增益 + 偏差（32 個學習得來的數字）' } }],
    },
    {
      short: { en: 'Same size', zh: '同樣大小' }, chapter: 'technique',
      title: { en: 'Any size in, the same size out', zh: '不管輸入多大，輸出都一樣大' },
      body: [
        { en: 'Try it: drag “multiply the input by” to shrink or blow up the bottom vector, and watch its size after layer norm. The pattern (which cells are high or low) is kept; only the overall scale and offset are removed.', zh: '動手試試：拖動「輸入乘以」，把最下面那個向量縮小或放大，看看它經過層正規化後的大小。格子的高低模式被保留下來，被拿掉的只有整體的尺度和偏移。' },
        lz(() => ({ en: `Here are three real vectors of the same token, at three points in the stream, with sizes ${data().vecs.map((v) => f2(size(v))).join(', ')}. After layer normalization all three have size ${f2(size(data().normed[0]))}.`, zh: `這是同一個詞元在殘差流三個地方的真實向量，大小分別是 ${data().vecs.map((v) => f2(size(v))).join('、')}。經過層正規化後，三個的大小都是 ${f2(size(data().normed[0]))}。` })),
      ],
      changed: [{ sign: '~', text: { en: 'Three sizes → one size', zh: '三種大小 → 同一個大小' } }],
    },
    {
      short: { en: 'Pre-norm', zh: '前置正規化' }, chapter: 'solution',
      title: { en: 'Where it goes: pre-norm', zh: '放在哪裡：前置正規化' },
      body: [
        { en: 'A common placement is <b>pre-norm</b>: normalize the vector just before it enters attention, and again before the MLP. One more normalization sits right before the output layer.', zh: '一種常見的放法是<b>前置正規化（pre-norm）</b>：向量進入注意力之前先正規化，進入 MLP 之前再正規化一次。輸出層前面還會再放一個。' },
        { en: 'The stream itself is never normalized; only the copy each part reads. This is what our model’s optional <code>norm</code> setting does, and the placement GPT-2 used.', zh: '殘差流本身從不被正規化，被正規化的只是每個部分讀進去的那一份。我們模型的選用設定 <code>norm</code> 就是這樣做的，GPT-2 也是用這種放法。' },
      ],
      callout: { kind: 'note', text: { en: 'The shipped model in this course does not use layer normalization. It is one block deep, so it did not need it.', zh: '這門課實際使用的模型並沒有用層正規化。它只有一個區塊，所以不需要。' } },
      changed: [{ sign: '+', text: { en: 'Norm before attention, MLP and output', zh: '在注意力、MLP 和輸出前加上正規化' } }],
    },
    {
      short: { en: 'Deep stack', zh: '深層堆疊' }, chapter: 'solution',
      title: { en: 'The deep stack, tamed', zh: '馴服深層堆疊' },
      body: [
        lz(() => ({ en: `Same ${DEPTH} fresh blocks, now with pre-norm. The stream still grows a little (to ${big(last(deep().norm.stream))}), but instead of ${big(last(deep().plain.stream))}.`, zh: `同樣 ${DEPTH} 個全新的區塊，這次加上前置正規化。殘差流還是會長大一點（到 ${big(last(deep().norm.stream))}），而不是 ${big(last(deep().plain.stream))}。` })),
        lz(() => ({ en: `More importantly, what each attention and MLP actually reads always has size ${f2(deep().norm.seen)}, in block 1 or block ${DEPTH}. Every layer gets inputs on the same scale.`, zh: `更重要的是，每個注意力和 MLP 實際讀到的輸入，不論在第 1 個還是第 ${DEPTH} 個區塊，大小都是 ${f2(deep().norm.seen)}。每一層拿到的輸入都在同樣的尺度上。` })),
      ],
      changed: [{ sign: '~', text: { en: 'Stream growth tamed', zh: '殘差流的成長被控制住' } }],
    },
    {
      short: { en: 'Tiny model', zh: '小模型' }, chapter: 'problem',
      title: { en: 'Case: does it help our tiny model?', zh: '案例：它對我們的小模型有幫助嗎？' },
      body: [
        { en: `We really trained two 2-block models (comparison runs) for ${V.steps.toLocaleString('en-US')} steps, one without and one with layer norm. Held-out loss (the loss on 615 names the model never trained on) at the end: ${f3(last(V.layers2.val))} without, ${f3(last(V.layers2norm.val))} with. Best along the way: ${f3(best(V.layers2.val))} vs ${f3(best(V.layers2norm.val))}.`, zh: `我們真的訓練了兩個 2 區塊的模型（比較實驗），各 ${V.steps.toLocaleString('en-US')} 步，一個沒有、一個有層正規化。最後的驗證損失（模型在訓練時從沒看過的 615 個名字上的損失）：沒有的是 ${f3(last(V.layers2.val))}，有的是 ${f3(last(V.layers2norm.val))}；過程中最好的分別是 ${f3(best(V.layers2.val))} 和 ${f3(best(V.layers2norm.val))}。` },
        { en: 'The difference is tiny, about the size of run-to-run noise. Two blocks are too shallow for sizes to drift far. Layer norm earns its place in models with dozens of blocks.', zh: '差距非常小，大概就跟每次訓練的隨機誤差差不多。兩個區塊太淺，向量大小還漂不了多遠。層正規化真正的價值，是在有幾十個區塊的模型裡。' },
      ],
      changed: [{ sign: '+', text: { en: 'Real runs: 2 blocks, with vs without', zh: '真實訓練：2 個區塊，有 vs 沒有' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Layer normalization in one line', zh: '一句話說完層正規化' },
      body: [
        { en: 'For each token’s vector: subtract the <b>mean</b>, divide by the <b>spread</b>, then multiply by a learned <b>gain</b> and add a learned <b>bias</b>.', zh: '對每個詞元的向量：減掉<b>平均值</b>，除以<b>標準差</b>，再乘上學習得來的<b>增益</b>、加上學習得來的<b>偏差</b>。' },
        { en: 'With <b>pre-norm</b>, every attention and MLP reads a vector of the same scale, however deep the stack. Tiny models barely notice; deep models depend on it.', zh: '用了<b>前置正規化</b>，不管堆得多深，每個注意力和 MLP 讀到的向量尺度都一樣。小模型幾乎感覺不到差別，深層模型卻少不了它。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Stacking Blocks. With residuals and normalization in place, we can stack many identical blocks, and see what depth buys.', zh: '下一課：堆疊區塊。有了殘差和正規化，就能把許多相同的區塊疊起來，看看深度能帶來什麼。' } },
    },
  ],

  scene(step, input = {}) {
    const d = data();
    const items: Item[] = [];
    // learner's gain (cell 6) and bias (cell 10) on step 4, and input scale on step 5
    const g6 = step === 3 ? Math.max(0, Math.min(3, Number(input.gain ?? 1))) : 1;
    const b10 = step === 3 ? Math.max(-2, Math.min(2, Number(input.bias ?? 0))) : 0;
    const gainV = d.gain.map((v, i) => (i === 5 ? g6 : v));
    const biasV = d.bias.map((v, i) => (i === 9 ? b10 : v));
    const outV = g6 === 1 && b10 === 0 ? d.out : lnWith(d.v, gainV, biasV);

    if (step === 0 || step === 6) {
      const dp = deep();
      const series: { pts: number[]; tone: Tone }[] = [{ pts: dp.plain.stream, tone: 'bad' }];
      if (step === 6) series.push({ pts: dp.norm.stream, tone: 'a' });
      items.push({
        k: 'deep', t: 'chart', x: 80, y: 60, w: 470, h: 360, series, ymin: 0, ymax: 1100, xmax: DEPTH, every: 1,
        xlabel: t({ en: 'after block …', zh: '第幾個區塊之後' }), ylabel: t({ en: `vector size (fresh ${DEPTH}-block model)`, zh: `向量大小（全新的 ${DEPTH} 區塊模型）` }),
      });
      items.push({ k: 'dl1', t: 'text', x: 600, y: 90, text: t({ en: 'without layer norm', zh: '沒有層正規化' }), size: 15, bold: true, tone: 'bad' });
      items.push({ k: 'dv1', t: 'text', x: 600, y: 124, text: `${big(dp.plain.stream[0])} → ${big(last(dp.plain.stream))}`, size: 24, bold: true, mono: true, tone: 'bad' });
      if (step === 0) {
        items.push({ k: 'tl', t: 'text', x: 600, y: 220, text: t({ en: 'our trained 1-block model', zh: '我們訓練好的 1 區塊模型' }), size: 15, bold: true });
        const names = [t({ en: 'input', zh: '輸入' }), t({ en: 'after attention', zh: '注意力之後' }), t({ en: 'after MLP', zh: 'MLP 之後' })];
        d.avg.forEach((a, i) => {
          items.push({ k: 'ta' + i, t: 'text', x: 600, y: 256 + i * 30, text: names[i], size: 14, tone: 'muted' });
          items.push({ k: 'tv' + i, t: 'text', x: 860, y: 256 + i * 30, text: f2(a), size: 15, mono: true, bold: true, anchor: 'end' });
        });
        items.push({ k: 'tn', t: 'text', x: 600, y: 360, text: t({ en: 'average size over 615 held-out names', zh: '615 個驗證集名字上的平均大小' }), size: 13, tone: 'muted' });
      } else {
        items.push({ k: 'dl2', t: 'text', x: 600, y: 200, text: t({ en: 'with pre-norm', zh: '有前置正規化' }), size: 15, bold: true, tone: 'a' });
        items.push({ k: 'dv2', t: 'text', x: 600, y: 234, text: `${big(dp.norm.stream[0])} → ${big(last(dp.norm.stream))}`, size: 24, bold: true, mono: true, tone: 'a' });
        items.push({ k: 'dl3', t: 'text', x: 600, y: 300, text: t({ en: 'what each part reads:', zh: '每個部分讀到的輸入：' }), size: 14, tone: 'muted' });
        items.push({ k: 'dv3', t: 'text', x: 600, y: 334, text: t({ en: `size ${f2(dp.norm.seen)}, every block`, zh: `大小 ${f2(dp.norm.seen)}，每個區塊都一樣` }), size: 18, bold: true, mono: true, tone: 'a' });
        const top = Math.ceil(Math.max(...dp.norm.stream) / 5) * 5;
        items.push({ k: 'dzoom', t: 'chart', x: 640, y: 390, w: 220, h: 80, series: [{ pts: dp.norm.stream, tone: 'a' }], ymin: 0, ymax: top, xmax: DEPTH, every: 1,
          xlabel: '', ylabel: t({ en: 'with pre-norm, zoomed in', zh: '有前置正規化，放大來看' }) });
      }
    }

    if (step >= 1 && step <= 3) {
      const r = (k: L, v: number[]): [string, number[]] => [t(k), v];
      const rows: [string, number[]][] = [r({ en: 'vector', zh: '向量' }, d.v)];
      if (step >= 1) rows.push(r({ en: '− mean', zh: '− 平均' }, d.centered));
      if (step >= 2) rows.push(r({ en: '÷ spread', zh: '÷ 標準差' }, d.norm));
      if (step === 3) rows.push(r({ en: 'gain', zh: '增益' }, gainV), r({ en: 'bias', zh: '偏差' }, biasV), r({ en: 'output', zh: '輸出' }, outV));
      const cell = step === 3 ? 44 : 46;
      items.push({
        k: 'ln', t: 'heat', x: 130, y: 60, cell, fmt: 'num', max: 3,
        m: rows.map((x) => x[1]), rows: rows.map((x) => x[0]), cols: d.v.map((_, j) => String(j + 1)), hi: rows.length - 1,
      });
      const ny = 60 + rows.length * cell + 30;
      const facts = [
        t({ en: `mean ${f3(mean(d.v))}`, zh: `平均 ${f3(mean(d.v))}` }),
        step >= 2 ? t({ en: `spread ${f3(d.sd)}`, zh: `標準差 ${f3(d.sd)}` }) : '',
        t({ en: `size ${f2(size(d.v))} → ${f2(size(step === 1 ? d.centered : step === 2 ? d.norm : outV))}`, zh: `大小 ${f2(size(d.v))} → ${f2(size(step === 1 ? d.centered : step === 2 ? d.norm : outV))}` }),
      ].filter(Boolean);
      facts.forEach((f, i) => items.push({ k: 'lf' + i, t: 'text', x: 130 + i * 240, y: ny, text: f, size: 17, bold: true, mono: true, tone: i === facts.length - 1 ? 'a' : 'ink' }));
      if (step === 3) items.push({ k: 'lfn', t: 'text', x: 130, y: ny + 28, text: (g6 === 1 && b10 === 0)
        ? t({ en: 'gain and bias at their starting values (1 and 0): the output equals the row “÷ spread”', zh: '增益與偏差是起始值（1 和 0）：輸出和「÷ 標準差」那一列一樣' })
        : t({ en: `illustration of what training might do: gain ${g6.toFixed(1)} on cell 6, bias ${b10 >= 0 ? '+' : '−'}${Math.abs(b10).toFixed(1)} on cell 10`, zh: `示意訓練可能會做的事：第 6 格增益 ${g6.toFixed(1)}，第 10 格偏差 ${b10 >= 0 ? '+' : '−'}${Math.abs(b10).toFixed(1)}` }), size: 13, tone: 'muted' });
    }

    if (step === 4) {
      const names = ['x', 'x1', 'x2'];
      d.vecs.forEach((v, i) => {
        const y = 70 + i * 100;
        items.push({ k: 'bv' + i, t: 'vec', x: 60, y, v, cell: 18, max: 3, label: t({ en: `${names[i]} · size ${f2(size(v))}`, zh: `${names[i]} · 大小 ${f2(size(v))}` }) });
        items.push({ k: 'ba' + i, t: 'arrow', x1: 360, y1: y + 11, x2: 470, y2: y + 11, tone: 'a' });
        items.push({ k: 'bl' + i, t: 'text', x: 415, y: y - 2, text: 'layer norm', size: 13, tone: 'a', anchor: 'middle' });
        items.push({ k: 'bn' + i, t: 'vec', x: 490, y, v: d.normed[i], cell: 18, max: 3, label: t({ en: `size ${f2(size(d.normed[i]))}`, zh: `大小 ${f2(size(d.normed[i]))}` }) });
      });
      const sc = 10 ** Math.max(-1, Math.min(1, Number(input.scale ?? 0)));
      const scaled = d.v.map((a) => a * sc), scaledOut = lnWith(scaled, d.gain, d.bias);
      const y = 380;
      items.push({ k: 'bsv', t: 'vec', x: 60, y, v: scaled, cell: 18, max: 3, label: t({ en: `x2 × ${fx(sc)} · size ${big(size(scaled))}`, zh: `x2 × ${fx(sc)} · 大小 ${big(size(scaled))}` }) });
      items.push({ k: 'bsa', t: 'arrow', x1: 360, y1: y + 11, x2: 470, y2: y + 11, tone: 'b' });
      items.push({ k: 'bsl', t: 'text', x: 415, y: y - 2, text: 'layer norm', size: 13, tone: 'b', anchor: 'middle' });
      items.push({ k: 'bsn', t: 'vec', x: 490, y, v: scaledOut, cell: 18, max: 3, label: t({ en: `size ${f2(size(scaledOut))}`, zh: `大小 ${f2(size(scaledOut))}` }) });
      items.push({ k: 'bsh', t: 'text', x: 60, y: y - 32, text: t({ en: 'your vector (slider):', zh: '你的向量（滑桿）：' }), size: 13.5, bold: true, tone: 'b' });
      items.push({ k: 'bnote', t: 'text', x: 450, y: 480, text: t({ en: 'same colour scale everywhere · the second “m” of “.jimm” at three points in the stream', zh: '全部用同樣的顏色比例 · 「.jimm」第二個「m」在殘差流三個位置的向量' }), size: 13, tone: 'muted', anchor: 'middle' });
    }

    if (step === 5) {
      items.push({ k: 'stream', t: 'arrow', x1: 30, y1: SY, x2: 880, y2: SY, tone: 'a' });
      items.push({ k: 'sl', t: 'text', x: 30, y: SY + 26, text: t({ en: 'residual stream (never normalized)', zh: '殘差流（本身不正規化）' }), size: 13, tone: 'a' });
      const parts: [string, L, number][] = [['a', { en: 'attention', zh: '注意力' }, 120], ['m', { en: 'MLP', zh: 'MLP' }, 420]];
      for (const [k, label, x] of parts) {
        items.push({ k: 'n' + k, t: 'box', x, y: 150, w: 110, h: 44, label: t({ en: 'norm', zh: '正規化' }), tone: 'b' });
        items.push({ k: 'p' + k, t: 'box', x: x + 130, y: 150, w: 120, h: 44, label: t(label) });
        items.push({ k: 'pi' + k, t: 'arrow', x1: x + 20, y1: SY - 2, x2: x + 20, y2: 198, tone: 'muted' });
        items.push({ k: 'pn' + k, t: 'arrow', x1: x + 112, y1: 172, x2: x + 128, y2: 172, tone: 'muted' });
        items.push({ k: 'po' + k, t: 'arrow', x1: x + 230, y1: 198, x2: x + 230, y2: SY - 4, tone: 'a' });
        items.push({ k: 'pp' + k, t: 'text', x: x + 240, y: SY + 22, text: '+', size: 20, bold: true, tone: 'a' });
      }
      items.push({ k: 'nf', t: 'box', x: 720, y: 290, w: 110, h: 44, label: t({ en: 'norm', zh: '正規化' }), tone: 'b' });
      items.push({ k: 'nfi', t: 'arrow', x1: 775, y1: SY + 2, x2: 775, y2: 288, tone: 'muted' });
      items.push({ k: 'out', t: 'box', x: 720, y: 370, w: 110, h: 44, label: t({ en: 'output', zh: '輸出層' }) });
      items.push({ k: 'outa', t: 'arrow', x1: 775, y1: 336, x2: 775, y2: 368, tone: 'muted' });
      items.push({ k: 'cfg', t: 'text', x: 30, y: 440, text: t({ en: 'our model with { norm: true } · one block shown; every block repeats this', zh: '我們的模型設定 { norm: true } · 圖中是一個區塊，每個區塊都重複這個樣子' }), size: 13, tone: 'muted', mono: true });
    }

    if (step === 7) {
      items.push({
        k: 'runs', t: 'chart', x: 80, y: 60, w: 500, h: 360, ymin: 2.1, ymax: 2.7, xmax: V.steps, every: V.layers2.every,
        series: [{ pts: V.layers2.val, tone: 'bad' }, { pts: V.layers2norm.val, tone: 'a' }],
        xlabel: t({ en: 'training step', zh: '訓練步數' }), ylabel: t({ en: 'held-out loss (2 blocks)', zh: '驗證損失（2 個區塊）' }),
      });
      items.push({ k: 'r1', t: 'text', x: 620, y: 120, text: t({ en: 'without layer norm', zh: '沒有層正規化' }), size: 15, bold: true, tone: 'bad' });
      items.push({ k: 'r1v', t: 'text', x: 620, y: 152, text: t({ en: `end ${f3(last(V.layers2.val))} · best ${f3(best(V.layers2.val))}`, zh: `最後 ${f3(last(V.layers2.val))} · 最佳 ${f3(best(V.layers2.val))}` }), size: 15, mono: true });
      items.push({ k: 'r2', t: 'text', x: 620, y: 210, text: t({ en: 'with pre-norm', zh: '有前置正規化' }), size: 15, bold: true, tone: 'a' });
      items.push({ k: 'r2v', t: 'text', x: 620, y: 242, text: t({ en: `end ${f3(last(V.layers2norm.val))} · best ${f3(best(V.layers2norm.val))}`, zh: `最後 ${f3(last(V.layers2norm.val))} · 最佳 ${f3(best(V.layers2norm.val))}` }), size: 15, mono: true });
      items.push({ k: 'r3', t: 'text', x: 620, y: 310, text: t({ en: 'difference: tiny', zh: '差距：非常小' }), size: 18, bold: true });
    }

    if (step === 8) {
      const boxes: L[] = [
        { en: 'vector', zh: '向量' }, { en: '− mean', zh: '− 平均' }, { en: '÷ spread', zh: '÷ 標準差' },
        { en: '× gain + bias', zh: '× 增益 + 偏差' }, { en: 'size ≈ 4, steady', zh: '大小 ≈ 4，穩定' },
      ];
      boxes.forEach((b, i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 210, w: 150, h: 60, label: t(b), tone: i === 4 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 240, x2: bx + 170, y2: 240 });
      });
      items.push({ k: 'rt', t: 'text', x: 450, y: 320, text: t({ en: 'pre-norm: before attention, before the MLP, and before the output', zh: '前置正規化：放在注意力前、MLP 前，以及輸出層前' }), size: 14, tone: 'muted', anchor: 'middle' });
    }

    const curSize = step === 1 ? size(d.centered) : step === 2 ? size(d.norm) : step === 3 ? size(outV) : NaN;
    return {
      metrics: [
        { label: { en: 'Vector size', zh: '向量大小' }, value: step >= 1 && step <= 3 ? `${f2(size(d.v))} → ${f2(curSize)}` : step === 4 ? `→ ${f2(size(d.normed[0]))}` : '—' },
        { label: { en: 'Mean', zh: '平均值' }, value: step >= 1 && step <= 3 ? f3(d.mu) : '—' },
        { label: { en: 'Spread', zh: '標準差' }, value: step >= 2 && step <= 3 ? `${f3(d.sd)} → 1` : '—' },
        { label: { en: `${DEPTH} blocks: final size`, zh: `${DEPTH} 區塊：最後大小` }, value: step === 0 ? big(last(deep().plain.stream)) : step === 6 ? big(last(deep().norm.stream)) : '—' },
        { label: { en: 'Shipped model', zh: '課程使用的模型' }, value: t({ en: 'no layer norm', zh: '沒有層正規化' }) },
      ],
      items,
      legend: [
        { tone: 'pos', text: { en: 'positive number', zh: '正數' } },
        { tone: 'neg', text: { en: 'negative number', zh: '負數' } },
        { tone: 'bad', text: { en: 'without layer norm', zh: '沒有層正規化' } },
        { tone: 'a', text: { en: 'with layer norm', zh: '有層正規化' } },
      ],
      controls: step === 3 ? [
        { key: 'gain', kind: 'slider', label: { en: 'Gain of cell 6', zh: '第 6 格的增益' }, min: 0, max: 3, step: 0.1, value: 1, format: (v: number) => v.toFixed(1) },
        { key: 'bias', kind: 'slider', label: { en: 'Bias of cell 10', zh: '第 10 格的偏差' }, min: -2, max: 2, step: 0.1, value: 0, format: (v: number) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1) },
      ] : step === 4 ? [
        { key: 'scale', kind: 'slider', label: { en: 'Multiply the input by', zh: '輸入乘以' }, min: -1, max: 1, step: 0.05, value: 0, format: (v: number) => '×' + fx(10 ** v) },
      ] : undefined,
    };
  },
};
