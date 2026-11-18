import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, TRAIN_NAMES, example, rng } from '../model/data';
import weights from '../model/weights.json';

// Part 1: a tiny chain of three operations, x → a = 2x → b = a + 1 → c = b², worked out live.
// Part 2: the real trained model. We nudge single weights and check the loss change against
// the gradient that one backward pass computed for all 3,115 weights at once.

const H = 0.01; // the small nudge

function makeChain() {
  const f = (x: number) => { const a = 2 * x, b = a + 1, c = b * b; return { x, a, b, c }; };
  const v = f(3);
  const n = f(3 + H);
  const slopes = { xa: 2, ab: 1, bc: 2 * v.b }; // local slope of each operation
  const grad = { c: 1, b: slopes.bc, a: slopes.bc * slopes.ab, x: slopes.bc * slopes.ab * slopes.xa };
  return { v, n, slopes, grad };
}
const C = makeChain(); // a few multiplications, cheap enough to do at start-up
const chain = () => C;
const GAP = C.n.c - C.v.c - C.grad.x * H;
// weight count without building the model
const COUNT = Object.values((weights as Weights).params).reduce((s, v) => s + v.length, 0);
const CN = COUNT.toLocaleString();

interface Probe { name: string; label: string; idx: number; g: number; pred: number; meas: number }
let realCache: ReturnType<typeof makeReal> | undefined;
function makeReal() {
  const model = GPT.fromJSON(weights as Weights);
  const r = rng(7);
  const batch = Array.from({ length: 32 }, () => example(TRAIN_NAMES[Math.floor(r() * TRAIN_NAMES.length)]));
  const L0 = model.loss(batch, true); // forward + backward: fills .g for every weight
  const count = model.paramCount();

  // One weight from three different parts of the model: the one with the largest gradient in each.
  const pick = (name: string): Probe => {
    const m = model.p[name];
    let idx = 0;
    for (let i = 0; i < m.g.length; i++) if (Math.abs(m.g[i]) > Math.abs(m.g[idx])) idx = i;
    const row = Math.floor(idx / m.c), col = idx % m.c;
    const label = name === 'tok' ? `tok “${CHARS[row]}” #${col + 1}` : name === 'u' ? `u #${row + 1} → “${CHARS[col]}”` : `${name} [${row + 1},${col + 1}]`;
    return { name, label, idx, g: m.g[idx], pred: 0, meas: 0 };
  };
  const probes = ['tok', 'q0', 'u'].map(pick);
  const lossWith = (p: Probe, delta: number) => {
    const m = model.p[p.name];
    const keep = m.d[p.idx];
    m.d[p.idx] = keep + delta;
    const l = model.loss(batch);
    m.d[p.idx] = keep;
    return l;
  };
  for (const p of probes) { p.pred = p.g * H; p.meas = lossWith(p, H) - L0; }

  // How well "gradient × nudge" predicts the change as the nudge grows (first probe).
  const main = probes[2];
  const span = 3, N = 25;
  const offsets = Array.from({ length: N }, (_, i) => (span * i) / (N - 1));
  const curve = offsets.map((d) => lossWith(main, d));
  const line = offsets.map((d) => L0 + main.g * d);
  const big = [0.01, 0.3, 1, 3].map((d) => ({ d, pred: main.g * d, meas: lossWith(main, d) - L0 }));

  // Timing, measured right here: forward only vs forward + backward.
  const reps = 6;
  let t0 = performance.now();
  for (let i = 0; i < reps; i++) model.loss(batch);
  const fwd = (performance.now() - t0) / reps;
  t0 = performance.now();
  for (let i = 0; i < reps; i++) model.loss(batch, true);
  const both = (performance.now() - t0) / reps;
  return { L0, count, probes, main, offsets, curve, line, span, big, fwd, both };
}
const real = () => (realCache ??= makeReal());

const sg = (v: number, d = 4) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const ms = (v: number) => (v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString()) + ' ms';

export const gradients: Lesson = {
  id: 'gradients',
  title: { en: 'Gradients and Backpropagation', zh: '梯度與反向傳播' },
  blurb: {
    en: 'How the model finds out, for every one of its weights at once, which way to move.',
    zh: '模型如何一次就知道每一個權重該往哪邊移動。',
  },
  minutes: 12,
  level: 'beginner',
  learn: [
    { text: { en: 'What a slope (gradient) means, with real numbers', zh: '用真實數字看懂斜率（梯度）的意思' }, step: 1 },
    { text: { en: 'The chain rule: multiply the local slopes', zh: '連鎖律：把每一段的斜率相乘' }, step: 3 },
    { text: { en: 'Backpropagation: one pass from the loss backwards', zh: '反向傳播：從損失往回走一趟' }, step: 4 },
    { text: { en: 'Check a real gradient by nudging a real weight', zh: '輕推一個真實權重，驗證真實的梯度' }, step: 5 },
    { text: { en: 'Why a gradient only holds for small steps', zh: '為什麼梯度只對小步伐有效' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: `${CN} weights, one question each`, zh: `${CN} 個權重，各有一個問題` },
      body: [
        { en: 'Training needs to know, for every weight: if I raise this weight a tiny bit, does the loss go up or down, and by how much? That number is the weight’s <b>gradient</b>.', zh: '訓練需要知道每個權重的一件事：如果把這個權重調高一點點，損失會上升還是下降？變化多少？這個數字就是該權重的<b>梯度</b>。' },
        { en: 'The obvious way: nudge one weight, run the model again, see how the loss changed. Then do it again for the next weight, and the next.', zh: '最直接的做法：輕推一個權重，再跑一次模型，看損失變了多少。接著對下一個權重重做一次，再下一個……' },
      ],
      callout: { kind: 'insight', text: { en: `That works, but needs one extra run of the model per weight: ${CN} runs for one training step. Real models have billions of weights. We need a shortcut.`, zh: `這樣做可行，但每個權重都要多跑一次模型：一步訓練就要 ${CN} 次。真實的模型有數十億個權重，所以我們需要捷徑。` } },
    },
    {
      short: { en: 'Slope', zh: '斜率' }, chapter: 'technique',
      title: { en: 'One operation, one slope', zh: '一個運算，一個斜率' },
      body: [
        { en: 'Start tiny. Take <code>x = 3</code> and double it: <code>a = 6</code>. Nudge x by 0.01 and a moves by 0.02, twice as much.', zh: '先從最小的例子開始。取 <code>x = 3</code>，把它乘以 2：<code>a = 6</code>。把 x 輕推 0.01，a 就移動 0.02，剛好是兩倍。' },
        { en: 'That ratio, “output change ÷ input change”, is the <b>slope</b> of the operation. For “×2” it is always 2. A gradient is exactly this kind of slope, measured from a weight all the way to the loss.', zh: '「輸出的變化 ÷ 輸入的變化」這個比值，就是這個運算的<b>斜率</b>。「×2」的斜率永遠是 2。梯度就是這種斜率，只是從某個權重一路量到損失。' },
      ],
      changed: [{ sign: '+', text: { en: 'Slope of “×2” = 2', zh: '「×2」的斜率 = 2' } }],
    },
    {
      short: { en: 'A chain', zh: '一串運算' }, chapter: 'technique',
      title: { en: 'A chain of three steps', zh: '三步串成一條鏈' },
      body: [
        { en: `A model is a long chain of small operations. Ours here has three: double, add 1, then square. From x = ${C.v.x} we get ${C.v.a}, then ${C.v.b}, then ${C.v.c}.`, zh: `模型就是一長串小運算。這裡的例子只有三步：乘以 2、加 1、再平方。從 x = ${C.v.x} 出發，依序得到 ${C.v.a}、${C.v.b}、${C.v.c}。` },
        { en: `Each step has its own <b>local slope</b>: “×2” has 2, “+1” has 1, and squaring near ${C.v.b} has ${C.slopes.bc} (squaring’s slope is 2 × its input). Each one only knows about its own little piece.`, zh: `每一步都有自己的<b>局部斜率</b>：「×2」是 2，「+1」是 1，在 ${C.v.b} 附近平方的斜率是 ${C.slopes.bc}（平方的斜率是輸入的 2 倍）。每一步都只管自己那一小段。` },
      ],
      changed: [{ sign: '+', text: { en: 'Three local slopes: 2, 1, 14', zh: '三個局部斜率：2、1、14' } }],
    },
    {
      short: { en: 'Chain rule', zh: '連鎖律' }, chapter: 'technique',
      title: { en: 'The chain rule: multiply the slopes', zh: '連鎖律：把斜率相乘' },
      body: [
        { en: 'How much does the end, c, move when x moves? A nudge to x is doubled, then passed through unchanged, then multiplied by 14. So the total slope is <b>2 × 1 × 14 = 28</b>. This is the <b>chain rule</b>.', zh: 'x 動一下，最後的 c 會動多少？對 x 的輕推先被放大 2 倍，接著原封不動傳下去，再被放大 14 倍。所以總斜率是 <b>2 × 1 × 14 = 28</b>。這就是<b>連鎖律</b>。' },
        { en: `Check it for real (bottom row): x = ${C.n.x} gives c = ${+C.n.c.toFixed(4)}. The change ${(C.n.c - C.v.c).toFixed(4)} is almost exactly 28 × ${H} = ${(28 * H).toFixed(2)}.`, zh: `實際驗證一下（下面那一列）：x = ${C.n.x} 時 c = ${+C.n.c.toFixed(4)}。變化量 ${(C.n.c - C.v.c).toFixed(4)}，幾乎正好等於 28 × ${H} = ${(28 * H).toFixed(2)}。` },
      ],
      callout: { kind: 'note', text: { en: `The tiny gap (${GAP.toFixed(4)}) is there because squaring curves: its slope changes as you move. The smaller the nudge, the smaller the gap.`, zh: `那一點點差距（${GAP.toFixed(4)}）是因為平方是彎曲的：一移動，斜率就跟著變。輕推得越小，差距就越小。` } },
      changed: [{ sign: '+', text: { en: 'Total slope 28, checked by a nudge', zh: '總斜率 28，用輕推驗證' } }],
    },
    {
      short: { en: 'Backward', zh: '往回算' }, chapter: 'technique',
      title: { en: 'Walk backwards from the end', zh: '從尾端往回走' },
      body: [
        { en: 'Now go backwards. Start at c with gradient 1 (c moves exactly with itself). Each step back, multiply by that step’s local slope: 1 × 14 = 14 at b, 14 × 1 = 14 at a, 14 × 2 = 28 at x.', zh: '現在倒著走。從 c 開始，梯度是 1（c 跟自己同步移動）。每往回一步，就乘上那一步的局部斜率：b 是 1 × 14 = 14，a 是 14 × 1 = 14，x 是 14 × 2 = 28。' },
        { en: 'One walk back gave us the gradient of <i>every</i> value in the chain, not just x. That walk is <b>backpropagation</b>.', zh: '只往回走一趟，就得到鏈上<i>每一個</i>值的梯度，而不只是 x 的。這一趟就叫<b>反向傳播</b>。' },
      ],
      callout: { kind: 'insight', text: { en: 'Forward pass: compute the values, left to right. Backward pass: compute the gradients, right to left, reusing what the forward pass stored.', zh: '前向計算：由左到右算出各個值。反向傳播：由右到左算出各個梯度，並重複利用前向時存下來的值。' } },
      changed: [{ sign: '+', text: { en: 'Gradients for x, a, b and c in one pass', zh: '一趟就得到 x、a、b、c 的梯度' } }],
    },
    {
      short: { en: 'Real check', zh: '真實驗證' }, chapter: 'solution',
      title: { en: 'Check it on the real model', zh: '在真實模型上驗證' },
      body: [
        { en: 'Our trained model reads a batch of 32 names and runs one forward and one backward pass. Every weight now holds a gradient. We pick one weight from three different parts: a letter’s vector (<code>tok</code>), the attention query (<code>q0</code>) and the output layer (<code>u</code>).', zh: '讓訓練好的模型讀一個 32 個名字的批次，做一次前向、一次反向計算。現在每個權重都帶著一個梯度。我們從三個不同的地方各挑一個權重：字母的向量（<code>tok</code>）、注意力的查詢（<code>q0</code>），以及輸出層（<code>u</code>）。' },
        { en: 'For each, the table predicts the loss change for a nudge of 0.01 (gradient × 0.01), then really nudges the weight and measures. They match.', zh: '表格先用「梯度 × 0.01」預測輕推 0.01 後損失的變化，再真的去推那個權重並量出結果。兩者吻合。' },
      ],
      changed: [{ sign: '+', text: { en: 'Prediction vs measurement on 3 real weights', zh: '3 個真實權重：預測 vs 實測' } }],
    },
    {
      short: { en: 'All at once', zh: '一次全拿' }, chapter: 'solution',
      title: { en: 'Every gradient from one backward pass', zh: '一次反向計算，拿到所有梯度' },
      body: [
        { en: `The real model is the same idea with more links: letter vectors, attention, MLP, output, loss. The backward pass walks this chain from the loss back to the start, multiplying local slopes, and fills in all ${CN} gradients on the way.`, zh: `真實模型的道理一樣，只是鏈更長：字母向量、注意力、MLP、輸出、損失。反向傳播從損失沿著這條鏈走回起點，一路乘上局部斜率，順便填好全部 ${CN} 個梯度。` },
        { en: `Timed in your browser just now: one forward pass over the batch, versus forward plus backward. Nudging every weight one by one would instead need ${(COUNT + 1).toLocaleString()} forward passes.`, zh: `這是剛剛在你的瀏覽器裡量到的時間：批次的一次前向計算，以及前向加反向。若改成一個一個去推權重，就得跑 ${(COUNT + 1).toLocaleString()} 次前向計算。` },
      ],
      callout: { kind: 'note', text: { en: 'For large transformers the standard estimate is that a backward pass costs about twice a forward pass. Either way it is a few passes, not billions.', zh: '對大型 Transformer，常用的估計是一次反向計算大約是前向計算的兩倍成本。不管怎樣，都只是幾趟計算，而不是數十億趟。' } },
      changed: [{ sign: '~', text: { en: `${(COUNT + 1).toLocaleString()} runs → 1 forward + 1 backward`, zh: `${(COUNT + 1).toLocaleString()} 次 → 1 次前向 + 1 次反向` } }],
    },
    {
      short: { en: 'Too far', zh: '推太遠' }, chapter: 'problem',
      title: { en: 'Case: a gradient only knows the nearby slope', zh: '案例：梯度只知道附近的坡度' },
      body: [
        { en: 'Take the output-layer weight and move it much further. The teal curve is the real loss; the amber line is “loss + gradient × nudge”.', zh: '把那個輸出層的權重推得遠很多。青色曲線是真實的損失，琥珀色直線是「損失 + 梯度 × 推動量」。' },
        { en: 'Close to the start they agree. Further out the real loss curves up and the straight-line guess is badly wrong. This is why training takes many small steps instead of one big jump.', zh: '一開始兩者一致；推遠了，真實的損失往上彎，直線的預測就錯得離譜。這就是為什麼訓練要走很多小步，而不是一次跳很遠。' },
      ],
      callout: { kind: 'insight', text: { en: 'This is the same reason a too-high learning rate broke training in the last lesson.', zh: '上一課學習率太高會讓訓練失敗，原因正是這個。' } },
      changed: [{ sign: '~', text: { en: `Nudge ${H} → up to ${3}`, zh: `推動量 ${H} → 最多 ${3}` } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Slopes, multiplied, walked backwards', zh: '斜率相乘，倒著走' },
      body: [
        { en: '<b>Gradient</b>: how fast the loss changes when one weight moves.<br><b>Chain rule</b>: the slope through a chain is the product of the local slopes.<br><b>Backpropagation</b>: one walk from the loss backwards gives every gradient.<br><b>Local</b>: a gradient is only accurate for small moves.', zh: '<b>梯度</b>：某個權重移動時，損失變化得多快。<br><b>連鎖律</b>：整條鏈的斜率，就是每一段局部斜率的乘積。<br><b>反向傳播</b>：從損失往回走一趟，就得到所有梯度。<br><b>局部性</b>：梯度只在小幅移動時才準。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Optimizers: SGD and Adam. Now that every weight has a gradient, how big a step should each one take?', zh: '下一課：最佳化器：SGD 與 Adam。每個權重都有了梯度，那每個權重該跨多大一步？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const c = chain();
    const BX = [40, 280, 520, 760], BW = 110, BY = 130;
    const vals = [['x', c.v.x], ['a', c.v.a], ['b', c.v.b], ['c', c.v.c]] as const;
    const ops = ['× 2', '+ 1', t({ en: 'square', zh: '平方' })];
    const localSlope = [c.slopes.xa, c.slopes.ab, c.slopes.bc];
    const nvals = [c.n.x, c.n.a, c.n.b, c.n.c];
    const grads = [c.grad.x, c.grad.a, c.grad.b, c.grad.c];

    if ((step >= 1 && step <= 4) || step === 8) {
      const shown = step === 1 ? 2 : 4;
      items.push({ k: 'chL', t: 'text', x: 40, y: 90, size: 14.5, tone: 'muted', text: t({ en: 'forward: compute the values, left to right', zh: '前向：由左到右算出各個值' }) });
      for (let i = 0; i < shown; i++) {
        items.push({ k: 'cb' + i, t: 'box', x: BX[i], y: BY, w: BW, h: 50, label: `${vals[i][0]} = ${vals[i][1]}`, tone: i === 0 ? 'a' : 'ink' });
        if (i < shown - 1) {
          items.push({ k: 'ca' + i, t: 'arrow', x1: BX[i] + BW + 8, y1: BY + 25, x2: BX[i + 1] - 8, y2: BY + 25 });
          items.push({ k: 'co' + i, t: 'text', x: (BX[i] + BW + BX[i + 1]) / 2, y: BY + 14, anchor: 'middle', size: 15, bold: true, text: ops[i] });
          items.push({ k: 'cs' + i, t: 'text', x: (BX[i] + BW + BX[i + 1]) / 2, y: BY + 46, anchor: 'middle', size: 13.5, tone: 'b', text: t({ en: `slope ${localSlope[i]}`, zh: `斜率 ${localSlope[i]}` }) });
        }
      }
      if (step === 1 || step === 3) {
        const NY = 260;
        items.push({ k: 'nL', t: 'text', x: 40, y: NY - 18, size: 14.5, tone: 'muted', text: t({ en: `nudge x by ${H}:`, zh: `把 x 輕推 ${H}：` }) });
        for (let i = 0; i < shown; i++) {
          items.push({ k: 'nb' + i, t: 'box', x: BX[i], y: NY, w: BW, h: 44, label: `${vals[i][0]} = ${+nvals[i].toFixed(4)}`, tone: i === 0 ? 'b' : 'ink' });
          if (i < shown - 1) items.push({ k: 'na' + i, t: 'arrow', x1: BX[i] + BW + 8, y1: NY + 22, x2: BX[i + 1] - 8, y2: NY + 22 });
        }
        const last = shown - 1;
        const change = nvals[last] - vals[last][1];
        const slope = step === 1 ? c.slopes.xa : c.grad.x;
        items.push({ k: 'nr', t: 'text', x: 450, y: 370, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'a',
          text: step === 1
            ? t({ en: `a changed by ${change.toFixed(4)} = 2 × ${H}`, zh: `a 改變了 ${change.toFixed(4)} = 2 × ${H}` })
            : t({ en: `c changed by ${change.toFixed(4)} ≈ 28 × ${H} = ${(slope * H).toFixed(2)}`, zh: `c 改變了 ${change.toFixed(4)} ≈ 28 × ${H} = ${(slope * H).toFixed(2)}` }) });
        if (step === 3) items.push({ k: 'cr', t: 'text', x: 450, y: 420, anchor: 'middle', size: 16, mono: true, text: t({ en: 'total slope from x to c: 2 × 1 × 14 = 28', zh: '從 x 到 c 的總斜率：2 × 1 × 14 = 28' }) });
      }
      if (step === 2) items.push({ k: 'cq', t: 'text', x: 450, y: 300, anchor: 'middle', size: 16, tone: 'muted', text: t({ en: 'if x moves, how much does c move?', zh: '如果 x 動了，c 會動多少？' }) });
      if (step === 4 || step === 8) {
        const GY = 300;
        items.push({ k: 'gL', t: 'text', x: 40, y: GY - 28, size: 14.5, tone: 'a', text: t({ en: 'backward: gradients, right to left (multiply by each slope)', zh: '反向：由右到左算梯度（乘上每一段的斜率）' }) });
        for (let i = 0; i < 4; i++) {
          items.push({ k: 'gb' + i, t: 'box', x: BX[i], y: GY, w: BW, h: 50, label: t({ en: `gradient ${grads[i]}`, zh: `梯度 ${grads[i]}` }), tone: 'a' });
          if (i > 0) items.push({ k: 'garr' + i, t: 'arrow', x1: BX[i] - 8, y1: GY + 25, x2: BX[i - 1] + BW + 8, y2: GY + 25, tone: 'a' });
          if (i > 0) items.push({ k: 'gm' + i, t: 'text', x: (BX[i - 1] + BW + BX[i]) / 2, y: GY + 50, anchor: 'middle', size: 13.5, tone: 'a', text: `× ${localSlope[i - 1]}` });
        }
        items.push({ k: 'gst', t: 'text', x: BX[3] + BW / 2, y: GY + 82, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'start here: 1', zh: '從這裡開始：1' }) });
        if (step === 8) items.push({ k: 'rcp', t: 'text', x: 450, y: 460, anchor: 'middle', size: 15, tone: 'a', bold: true, text: t({ en: `the real model: the same walk back gives all ${CN} gradients at once`, zh: `真實模型：同樣往回走一趟，${CN} 個梯度一次全拿到` }) });
      }
    }

    if (step === 0 || (step >= 5 && step <= 7)) {
      const R = real();
      if (step === 0) {
        const m = GPT.fromJSON(weights as Weights);
        items.push({ k: 'w0v', t: 'vec', x: 60, y: 110, v: m.p.tok.row(1), cell: 30, label: t({ en: `16 of the ${R.count.toLocaleString()} weights (the vector for “a”); each one needs its own answer`, zh: `${R.count.toLocaleString()} 個權重中的 16 個（「a」的向量），每一個都需要自己的答案` }) });
        items.push({ k: 'w0', t: 'box', x: 60, y: 200, w: 170, h: 56, label: t({ en: 'one weight', zh: '一個權重' }), sub: t({ en: `+ ${H}`, zh: `+ ${H}` }), tone: 'b' });
        items.push({ k: 'w0a', t: 'arrow', x1: 240, y1: 228, x2: 330, y2: 228 });
        items.push({ k: 'm0', t: 'box', x: 340, y: 190, w: 220, h: 76, label: t({ en: 'run the whole model', zh: '整個模型跑一次' }), sub: t({ en: `${R.count.toLocaleString()} weights`, zh: `${R.count.toLocaleString()} 個權重` }) });
        items.push({ k: 'm0a', t: 'arrow', x1: 570, y1: 228, x2: 660, y2: 228 });
        items.push({ k: 'l0', t: 'box', x: 670, y: 200, w: 170, h: 56, label: t({ en: 'loss changed by ?', zh: '損失變了多少？' }), tone: 'a' });
        items.push({ k: 'rep', t: 'text', x: 450, y: 330, anchor: 'middle', size: 16, tone: 'bad', text: t({ en: `…repeat for each of the ${R.count.toLocaleString()} weights, every training step`, zh: `……每一步訓練，都要對 ${R.count.toLocaleString()} 個權重各做一次` }) });
      }
      if (step === 5) {
        const cols = [60, 330, 500, 680];
        const head = [t({ en: 'weight', zh: '權重' }), t({ en: 'gradient', zh: '梯度' }), t({ en: 'predicted change', zh: '預測的變化' }), t({ en: 'measured change', zh: '實測的變化' })];
        head.forEach((h, j) => items.push({ k: 'th' + j, t: 'text', x: cols[j], y: 110, size: 14.5, bold: true, tone: 'muted', text: h }));
        items.push({ k: 'tsub', t: 'text', x: 500, y: 132, size: 13, tone: 'muted', text: t({ en: `gradient × ${H}`, zh: `梯度 × ${H}` }) });
        items.push({ k: 'tsub2', t: 'text', x: 680, y: 132, size: 13, tone: 'muted', text: t({ en: `really nudge by ${H}`, zh: `真的推 ${H}` }) });
        R.probes.forEach((p, i) => {
          const y = 180 + i * 70;
          items.push({ k: 'pr' + i, t: 'box', x: 40, y: y - 28, w: 820, h: 50, tone: i === 2 ? 'a' : 'ink' });
          items.push({ k: 'pn' + i, t: 'text', x: cols[0], y: y + 2, size: 15, mono: true, text: p.label });
          items.push({ k: 'pg' + i, t: 'text', x: cols[1], y: y + 2, size: 15, mono: true, text: sg(p.g) });
          items.push({ k: 'pp' + i, t: 'text', x: cols[2], y: y + 2, size: 15, mono: true, text: sg(p.pred, 6) });
          items.push({ k: 'pm' + i, t: 'text', x: cols[3], y: y + 2, size: 15, mono: true, bold: true, tone: 'a', text: sg(p.meas, 6) });
        });
        items.push({ k: 'tnote', t: 'text', x: 450, y: 420, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: `batch loss before any nudge: ${R.L0.toFixed(4)} · a negative gradient means raising the weight lowers the loss`, zh: `輕推前的批次損失：${R.L0.toFixed(4)} · 梯度為負，代表調高權重會讓損失下降` }) });
      }
      if (step === 6) {
        const parts = [t({ en: 'letter vectors', zh: '字母向量' }), t({ en: 'attention', zh: '注意力' }), 'MLP', t({ en: 'output', zh: '輸出' }), t({ en: 'loss', zh: '損失' })];
        parts.forEach((p, i) => {
          const x = 30 + i * 172;
          items.push({ k: 'mp' + i, t: 'box', x, y: 70, w: 140, h: 50, label: p, tone: i === 4 ? 'a' : 'ink' });
          if (i < 4) {
            items.push({ k: 'mf' + i, t: 'arrow', x1: x + 144, y1: 86, x2: x + 168, y2: 86 });
            items.push({ k: 'mbk' + i, t: 'arrow', x1: x + 168, y1: 106, x2: x + 144, y2: 106, tone: 'a' });
          }
        });
        items.push({ k: 'mfl', t: 'text', x: 30, y: 52, size: 13.5, tone: 'muted', text: t({ en: 'grey: forward (values) · teal: backward (gradients)', zh: '灰色：前向（算值）· 青色：反向（算梯度）' }) });
        const rows = [
          { k: 'tf', label: t({ en: 'one forward pass (loss only)', zh: '一次前向計算（只算損失）' }), v: ms(R.fwd), tone: 'ink' as const },
          { k: 'tb', label: t({ en: `forward + backward: all ${R.count.toLocaleString()} gradients`, zh: `前向 + 反向：全部 ${R.count.toLocaleString()} 個梯度` }), v: ms(R.both), tone: 'a' as const },
          { k: 'tn', label: t({ en: `nudge one by one: ${(R.count + 1).toLocaleString()} forward passes`, zh: `一個一個推：${(R.count + 1).toLocaleString()} 次前向計算` }), v: '≈ ' + ms(R.fwd * (R.count + 1)), tone: 'bad' as const },
        ];
        rows.forEach((r, i) => {
          const y = 180 + i * 76;
          items.push({ k: r.k + 'b', t: 'box', x: 60, y, w: 560, h: 54, label: r.label, tone: r.tone });
          items.push({ k: r.k + 'v', t: 'text', x: 650, y: y + 34, size: 22, bold: true, mono: true, tone: r.tone, text: r.v });
        });
        items.push({ k: 'tm', t: 'text', x: 450, y: 440, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: `measured in this tab on a batch of 32 names (times vary a little from run to run): backward ≈ ${((R.both - R.fwd) / R.fwd).toFixed(1)} × forward`, zh: `在這個分頁、用 32 個名字的批次量測（每次會略有不同）：反向 ≈ 前向的 ${((R.both - R.fwd) / R.fwd).toFixed(1)} 倍` }) });
      }
      if (step === 7) {
        const ymin = Math.min(...R.curve, ...R.line), ymax = Math.max(...R.curve);
        const lo = Math.floor(ymin * 2) / 2, hi = Math.ceil(ymax * 2) / 2;
        items.push({
          k: 'nc', t: 'chart', x: 90, y: 60, w: 480, h: 320, ymin: lo, ymax: hi, xmax: R.span, every: R.span / (R.offsets.length - 1),
          series: [{ pts: R.line, tone: 'b' }, { pts: R.curve, tone: 'a' }],
          xlabel: t({ en: 'how far the weight is pushed', zh: '權重被推了多遠' }), ylabel: t({ en: 'batch loss', zh: '批次損失' }),
          guide: { y: R.L0, label: t({ en: `before the nudge · ${R.L0.toFixed(3)}`, zh: `推之前 · ${R.L0.toFixed(3)}` }) },
        });
        items.push({ k: 'bh0', t: 'text', x: 620, y: 80, size: 14.5, bold: true, tone: 'muted', text: t({ en: 'nudge', zh: '推動量' }) });
        items.push({ k: 'bh1', t: 'text', x: 700, y: 80, size: 14.5, bold: true, tone: 'b', text: t({ en: 'predicted', zh: '預測' }) });
        items.push({ k: 'bh2', t: 'text', x: 790, y: 80, size: 14.5, bold: true, tone: 'a', text: t({ en: 'measured', zh: '實測' }) });
        R.big.forEach((b, i) => {
          const y = 120 + i * 40;
          items.push({ k: 'bd' + i, t: 'text', x: 620, y, size: 15, mono: true, text: '+' + b.d });
          items.push({ k: 'bp' + i, t: 'text', x: 700, y, size: 15, mono: true, tone: 'b', text: sg(b.pred, 4) });
          items.push({ k: 'bm' + i, t: 'text', x: 790, y, size: 15, mono: true, tone: Math.abs(b.meas - b.pred) > 0.05 ? 'bad' : 'a', text: sg(b.meas, 4) });
        });
        items.push({ k: 'bw', t: 'text', x: 620, y: 300, size: 14, tone: 'muted', text: t({ en: `weight: ${R.main.label}`, zh: `權重：${R.main.label}` }) });
      }
    }

    const R = step === 0 || step >= 5 ? real() : null;
    const metrics = step >= 1 && step <= 4
      ? [
        { label: { en: 'x', zh: 'x' }, value: String(c.v.x) },
        { label: { en: 'c (the end)', zh: 'c（終點）' }, value: step === 1 ? '—' : String(c.v.c) },
        { label: { en: 'Slope x → c', zh: '斜率 x → c' }, value: step >= 3 ? String(c.grad.x) : '?' },
        { label: { en: 'Passes needed', zh: '需要的計算趟數' }, value: step === 4 ? t({ en: '1 back', zh: '往回 1 趟' }) : '—' },
      ]
      : [
        { label: { en: 'Weights', zh: '權重數' }, value: R!.count.toLocaleString() },
        { label: { en: 'Batch loss', zh: '批次損失' }, value: R!.L0.toFixed(4) },
        { label: { en: 'Gradients known', zh: '已知梯度' }, value: step === 0 ? '0' : R!.count.toLocaleString() },
        { label: { en: 'Runs needed', zh: '需要跑幾次' }, value: step === 0 ? (R!.count + 1).toLocaleString() : t({ en: '1 fwd + 1 back', zh: '前向 1 + 反向 1' }) },
      ];
    return {
      metrics,
      items,
      legend: step === 7 ? [{ tone: 'a', text: { en: 'real loss', zh: '真實損失' } }, { tone: 'b', text: { en: 'gradient × nudge (straight-line guess)', zh: '梯度 × 推動量（直線預測）' } }] : undefined,
    };
  },
};
