import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, TRAIN_NAMES, example, rng } from '../model/data';
import weights from '../model/weights.json';

// Part 1: a tiny chain with one weight, a = w × x → b = a + 1 → c = b² (c plays the loss), worked out live.
// Part 2: the real trained model (the course model). We nudge single weights and check the loss change
// against the gradient that one backward pass computed for all 3,115 weights at once, then let the
// learner drag one real weight along its real loss curve.

const H = 0.01; // the small nudge
const X = 3; // the input
const W0 = 2; // the weight

function makeChain() {
  const f = (w: number) => { const a = w * X, b = a + 1, c = b * b; return { w, a, b, c }; };
  const v = f(W0);
  const n = f(W0 + H);
  const slopes = { wa: X, ab: 1, bc: 2 * v.b }; // local slope of each operation
  const grad = { c: 1, b: slopes.bc, a: slopes.bc * slopes.ab, w: slopes.bc * slopes.ab * slopes.wa };
  return { v, n, slopes, grad };
}
const C = makeChain(); // a few multiplications, cheap enough to do at start-up
const chain = () => C;
const GAP = C.n.c - C.v.c - C.grad.w * H;
const SQ = (C.v.b + H) ** 2 - C.v.b ** 2; // squaring checked with a nudge
// weight count without building the model
const COUNT = Object.values((weights as Weights).params).reduce((s, v) => s + v.length, 0);
const CN = COUNT.toLocaleString();

interface Probe { name: string; label: L; plain: L; idx: number; g: number; pred: number; meas: number }
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
    const label: L = name === 'tok' ? { en: `tok “${CHARS[row]}” #${col + 1}`, zh: `tok 「${CHARS[row]}」 #${col + 1}` }
      : name === 'u' ? { en: `u #${row + 1} → “${CHARS[col]}”`, zh: `u #${row + 1} → 「${CHARS[col]}」` } : { en: `${name} [${row + 1},${col + 1}]`, zh: `${name} [${row + 1},${col + 1}]` };
    const plain: L = name === 'tok'
      ? { en: `number ${col + 1} of the letter “${CHARS[row]}” vector`, zh: `字母「${CHARS[row]}」向量的第 ${col + 1} 個數字` }
      : name === 'u'
        ? { en: `output weight from cell ${row + 1} to the letter “${CHARS[col]}”`, zh: `輸出層：從第 ${row + 1} 格連到字母「${CHARS[col]}」的權重` }
        : { en: 'one number in the attention query matrix', zh: '注意力查詢矩陣裡的一個數字' };
    return { name, label, plain, idx, g: m.g[idx], pred: 0, meas: 0 };
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
  return { L0, count, probes, main, offsets, curve, line, span, big, fwd, both, model, batch };
}
const real = () => (realCache ??= makeReal());

// The "drag one weight" step: the real batch loss at many values of the output weight (forward only,
// computed once), and the real gradient from backprop at whatever value the learner picks (memoized).
const DSPAN = 1.5, DSTEP = 0.05;
const DN = Math.round((2 * DSPAN) / DSTEP) + 1;
let dragCache: { ds: number[]; curve: number[]; keep: number } | undefined;
const drag = () => {
  if (dragCache) return dragCache;
  const R = real();
  const m = R.model.p[R.main.name], keep = m.d[R.main.idx];
  const ds = Array.from({ length: DN }, (_, i) => -DSPAN + i * DSTEP);
  const curve = ds.map((d) => { m.d[R.main.idx] = keep + d; const l = R.model.loss(R.batch); m.d[R.main.idx] = keep; return l; });
  return (dragCache = { ds, curve, keep });
};
const gradAt = new Map<number, { loss: number; g: number }>();
const pointAt = (d: number) => {
  const key = Math.round(d / DSTEP);
  const hit = gradAt.get(key);
  if (hit) return hit;
  const R = real(), D = drag();
  const m = R.model.p[R.main.name];
  m.d[R.main.idx] = D.keep + key * DSTEP;
  const loss = R.model.loss(R.batch, true); // forward + backward at this value
  const g = m.g[R.main.idx];
  m.d[R.main.idx] = D.keep;
  const out = { loss, g };
  gradAt.set(key, out);
  return out;
};

const sg = (v: number, d = 4) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const ms = (v: number) => (v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString()) + ' ms';

const f4 = (v: number) => String(+v.toFixed(4));

export const gradients: Lesson = {
  id: 'gradients',
  title: { en: 'Gradients and Backpropagation', zh: '梯度與反向傳播' },
  blurb: {
    en: 'How the model finds out, for every one of its weights at once, which way to move.',
    zh: '模型如何一次就知道每一個權重該往哪邊移動。',
  },
  minutes: 14,
  level: 'beginner',
  learn: [
    { text: { en: 'What a slope (gradient) means, with real numbers', zh: '用真實數字看懂斜率（梯度）的意思' }, step: 1 },
    { text: { en: 'The chain rule: multiply the local slopes', zh: '連鎖律：把每一段的斜率相乘' }, step: 3 },
    { text: { en: 'Backpropagation: one pass from the loss back to every weight', zh: '反向傳播：從損失往回走一趟，走到每個權重' }, step: 4 },
    { text: { en: 'Check a real gradient by nudging a real weight', zh: '輕推一個真實權重，驗證真實的梯度' }, step: 5 },
    { text: { en: 'Why a gradient only holds for small steps', zh: '為什麼梯度只對小步伐有效' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: `${CN} weights, one question each`, zh: `${CN} 個權重，各有一個問題` },
      body: [
        { en: 'Training needs to know, for every weight (a number inside the model that training adjusts): if I raise this weight a tiny bit, does the loss go up or down, and by how much? That number is the weight’s <b>gradient</b>.', zh: '訓練需要知道每個權重（模型裡由訓練來調整的數字）的一件事：如果把這個權重調高一點點，損失會上升還是下降？變化多少？這個數字就是該權重的<b>梯度</b>。' },
        { en: 'The obvious way: nudge one weight, run the model again, see how the loss changed. Then do it again for the next weight, and the next.', zh: '最直接的做法：輕推一個權重，再跑一次模型，看損失變了多少。接著對下一個權重重做一次，再下一個……' },
      ],
      callout: { kind: 'insight', text: { en: `That works, but needs one extra run of the model per weight: ${CN} runs for one training step. Real models have billions of weights. We need a shortcut.`, zh: `這樣做可行，但每個權重都要多跑一次模型：一步訓練就要 ${CN} 次。真實的模型有數十億個權重，所以我們需要捷徑。` } },
    },
    {
      short: { en: 'Slope', zh: '斜率' }, chapter: 'technique',
      title: { en: 'One weight, one slope', zh: '一個權重，一個斜率' },
      body: [
        { en: `Start tiny: one weight <code>w = ${W0}</code> multiplies an input <code>x = ${X}</code>, giving <code>a = w × x = ${C.v.a}</code>. This is what every layer of a model does, many times over. Nudge w by ${H} and a moves by ${f4(C.n.a - C.v.a)}, three times as much.`, zh: `先從最小的例子開始：一個權重 <code>w = ${W0}</code> 乘上輸入 <code>x = ${X}</code>，得到 <code>a = w × x = ${C.v.a}</code>。模型的每一層做的都是這件事，只是做很多次。把 w 輕推 ${H}，a 就移動 ${f4(C.n.a - C.v.a)}，是三倍。` },
        { en: 'That ratio, “output change ÷ input change”, is the <b>slope</b>. For <code>w × x</code> the slope with respect to w is simply x, the input it multiplies. A gradient is this kind of slope, measured from a weight all the way to the loss.', zh: '「輸出的變化 ÷ 輸入的變化」這個比值，就是<b>斜率</b>。對 <code>w × x</code> 來說，對 w 的斜率就是 x，也就是它所乘的那個輸入。梯度就是這種斜率，只是從某個權重一路量到損失。' },
      ],
      changed: [{ sign: '+', text: { en: `Slope of w × x for w = x = ${X}`, zh: `w × x 對 w 的斜率 = x = ${X}` } }],
    },
    {
      short: { en: 'A chain', zh: '一串運算' }, chapter: 'technique',
      title: { en: 'A chain of three steps that ends in a loss', zh: '三步串成一條鏈，終點是損失' },
      body: [
        { en: `A model is a long chain of small operations ending in the loss. Ours has three: multiply by the weight, add 1, then square. From w = ${W0} we get ${C.v.a}, then ${C.v.b}, then ${C.v.c}. Pretend c is the loss: training wants to make it smaller.`, zh: `模型就是一長串小運算，最後算出損失。這裡的例子只有三步：乘上權重、加 1、再平方。從 w = ${W0} 出發，依序得到 ${C.v.a}、${C.v.b}、${C.v.c}。就把 c 當成損失：訓練想讓它變小。` },
        { en: `Each step has its own <b>local slope</b>: “× x” has ${X}, “+1” has 1, and squaring near ${C.v.b} has ${C.slopes.bc}. Check the last one: ${C.v.b + H}² − ${C.v.b}² = ${f4(SQ)}, about ${C.slopes.bc} × ${H}. Each step only knows about its own little piece.`, zh: `每一步都有自己的<b>局部斜率</b>：「× x」是 ${X}，「+1」是 1，在 ${C.v.b} 附近平方的斜率是 ${C.slopes.bc}。驗證最後一個：${C.v.b + H}² − ${C.v.b}² = ${f4(SQ)}，大約是 ${C.slopes.bc} × ${H}。每一步都只管自己那一小段。` },
      ],
      changed: [{ sign: '+', text: { en: `Three local slopes: ${X}, 1, ${C.slopes.bc}`, zh: `三個局部斜率：${X}、1、${C.slopes.bc}` } }],
    },
    {
      short: { en: 'Chain rule', zh: '連鎖律' }, chapter: 'technique',
      title: { en: 'The chain rule: multiply the slopes', zh: '連鎖律：把斜率相乘' },
      body: [
        { en: `How much does the loss c move when w moves? A nudge to w is multiplied by ${X}, passed through unchanged, then multiplied by ${C.slopes.bc}. So the total slope is <b>${X} × 1 × ${C.slopes.bc} = ${C.grad.w}</b>. This is the <b>chain rule</b>.`, zh: `w 動一下，損失 c 會動多少？對 w 的輕推先被放大 ${X} 倍，接著原封不動傳下去，再被放大 ${C.slopes.bc} 倍。所以總斜率是 <b>${X} × 1 × ${C.slopes.bc} = ${C.grad.w}</b>。這就是<b>連鎖律</b>。` },
        { en: `Check it for real (bottom row): w = ${C.n.w} gives c = ${f4(C.n.c)}. The change ${(C.n.c - C.v.c).toFixed(4)} is almost exactly ${C.grad.w} × ${H} = ${(C.grad.w * H).toFixed(2)}.`, zh: `實際驗證一下（下面那一列）：w = ${C.n.w} 時 c = ${f4(C.n.c)}。變化量 ${(C.n.c - C.v.c).toFixed(4)}，幾乎正好等於 ${C.grad.w} × ${H} = ${(C.grad.w * H).toFixed(2)}。` },
      ],
      callout: { kind: 'note', text: { en: `The tiny gap (${GAP.toFixed(4)}) is there because squaring curves: its slope changes as you move. The smaller the nudge, the smaller the gap.`, zh: `那一點點差距（${GAP.toFixed(4)}）是因為平方是彎曲的：一移動，斜率就跟著變。輕推得越小，差距就越小。` } },
      changed: [{ sign: '+', text: { en: `Total slope ${C.grad.w}, checked by a nudge`, zh: `總斜率 ${C.grad.w}，用輕推驗證` } }],
    },
    {
      short: { en: 'Backward', zh: '往回算' }, chapter: 'technique',
      title: { en: 'Walk backwards from the loss', zh: '從損失往回走' },
      body: [
        { en: `Now go backwards. Start at the loss c with gradient 1 (c moves exactly with itself). Each step back, multiply by that step’s local slope: 1 × ${C.slopes.bc} = ${C.grad.b} at b, ${C.grad.b} × 1 = ${C.grad.a} at a, and ${C.grad.a} × ${X} = <b>${C.grad.w} at w</b> (the slope of w × x with respect to w is the input x).`, zh: `現在倒著走。從損失 c 開始，梯度是 1（c 跟自己同步移動）。每往回一步，就乘上那一步的局部斜率：b 是 1 × ${C.slopes.bc} = ${C.grad.b}，a 是 ${C.grad.b} × 1 = ${C.grad.a}，w 是 ${C.grad.a} × ${X} = <b>${C.grad.w}</b>（w × x 對 w 的斜率就是輸入 x）。` },
        { en: 'Training only needs the gradients of the weights (here w), but the walk passes through every value to get there, and one walk gives them all. That walk is <b>backpropagation</b>.', zh: '訓練只需要權重的梯度（這裡是 w），但往回走的路上會經過每一個值，而且只走一趟就全部拿到。這一趟就叫<b>反向傳播</b>。' },
      ],
      callout: { kind: 'insight', text: { en: 'Forward pass: compute the values, left to right. Backward pass: compute the gradients, right to left, reusing what the forward pass stored (like x, needed for w’s slope).', zh: '前向計算：由左到右算出各個值。反向傳播：由右到左算出各個梯度，並重複利用前向時存下來的值（例如算 w 的斜率要用到的 x）。' } },
      changed: [{ sign: '+', text: { en: `Gradient of the weight w = ${C.grad.w}, in one pass`, zh: `一趟就得到權重 w 的梯度 = ${C.grad.w}` } }],
    },
    {
      short: { en: 'Real check', zh: '真實驗證' }, chapter: 'solution',
      title: { en: 'Check it on the real model', zh: '在真實模型上驗證' },
      body: [
        { en: 'The course model (the trained model used throughout this course) reads a batch of 32 names and runs one forward and one backward pass. Every weight now holds a gradient. We pick one weight from three different parts: a letter’s vector, the attention query, and the output layer.', zh: '讓課程模型（整門課一直在用的那個訓練好的模型）讀一個 32 個名字的批次，做一次前向、一次反向計算。現在每個權重都帶著一個梯度。我們從三個不同的地方各挑一個權重：字母的向量、注意力的查詢，以及輸出層。' },
        { en: `For each, the table predicts the loss change for a nudge of ${H} (gradient × ${H}), then really nudges the weight and measures. They match.`, zh: `表格先用「梯度 × ${H}」預測輕推 ${H} 後損失的變化，再真的去推那個權重並量出結果。兩者吻合。` },
      ],
      changed: [{ sign: '+', text: { en: 'Prediction vs measurement on 3 real weights', zh: '3 個真實權重：預測 vs 實測' } }],
    },
    {
      short: { en: 'All at once', zh: '一次全拿' }, chapter: 'solution',
      title: { en: 'Every gradient from one backward pass', zh: '一次反向計算，拿到所有梯度' },
      body: [
        { en: `The real model is the same idea with more links: letter vectors, attention, MLP, output, loss. The backward pass walks from the loss back to the start, multiplying local slopes, and fills in all ${CN} gradients. When a value feeds several later steps, the gradients coming back along each path are added up.`, zh: `真實模型的道理一樣，只是鏈更長：字母向量、注意力、MLP、輸出、損失。反向傳播從損失走回起點，一路乘上局部斜率，填好全部 ${CN} 個梯度。如果一個值同時流向後面好幾個地方，從每條路傳回來的梯度就加起來。` },
        { en: `Timed in your browser just now: one forward pass over the batch, versus forward plus backward. Nudging every weight one by one would instead need ${(COUNT + 1).toLocaleString()} forward passes.`, zh: `這是剛剛在你的瀏覽器裡量到的時間：批次的一次前向計算，以及前向加反向。若改成一個一個去推權重，就得跑 ${(COUNT + 1).toLocaleString()} 次前向計算。` },
      ],
      callout: { kind: 'note', text: { en: 'The ratio measured here, on our tiny model, is shown under the boxes. For large transformers the usual estimate is that a backward pass costs about twice a forward pass. Either way it is a few passes, not billions.', zh: '在我們的小模型上量到的倍數，寫在方塊下方。對大型 Transformer，常用的估計是一次反向計算大約是前向計算的兩倍成本。不管怎樣，都只是幾趟計算，而不是數十億趟。' } },
      changed: [{ sign: '~', text: { en: `${(COUNT + 1).toLocaleString()} runs → 1 forward + 1 backward`, zh: `${(COUNT + 1).toLocaleString()} 次 → 1 次前向 + 1 次反向` } }],
    },
    {
      short: { en: 'Too far', zh: '推太遠' }, chapter: 'problem',
      title: { en: 'Case: a gradient only knows the nearby slope', zh: '案例：梯度只知道附近的坡度' },
      body: [
        { en: 'Take the output-layer weight from the table. Its gradient is negative, so raising it should lower the loss: that is the downhill direction, the one training would take. We push it that way, further and further. The teal curve is the real loss; the amber line is “loss + gradient × push”.', zh: '拿表格裡那個輸出層的權重來試。它的梯度是負的，所以調高它應該會讓損失下降：這就是下坡的方向，也是訓練會走的方向。我們往這個方向越推越遠。青色曲線是真實的損失，琥珀色直線是「損失 + 梯度 × 推動量」。' },
        { en: 'A small push lowers the loss, as predicted. Push further and the same direction makes things worse: the real loss curves up while the straight-line guess keeps falling. This is why training takes many small steps instead of one big jump.', zh: '推一點點，損失確實如預測般下降。推遠了，同一個方向反而讓情況變糟：真實的損失往上彎，直線的預測卻一路往下。這就是為什麼訓練要走很多小步，而不是一次跳很遠。' },
      ],
      callout: { kind: 'insight', text: { en: 'This is the same reason a too-high learning rate broke training in the last lesson.', zh: '上一課學習率太高會讓訓練失敗，原因正是這個。' } },
      changed: [{ sign: '~', text: { en: `Nudge ${H} → up to ${3}`, zh: `推動量 ${H} → 最多 ${3}` } }],
    },
    {
      short: { en: 'Drag it', zh: '拖拖看' }, chapter: 'solution',
      title: { en: 'Try it: slide one real weight along its loss', zh: '動手試試：讓一個真實權重沿著損失曲線滑動' },
      body: [
        { en: 'Try it: drag the slider to move that same output weight, and watch the dot and the amber line. The teal curve is the real batch loss at each value of the weight. The amber line touches the curve at the dot; its slope is the gradient that backpropagation computes at that spot.', zh: '動手試試：拖動滑桿，移動同一個輸出層權重，看看圓點和琥珀色直線。青色曲線是這個權重取不同值時的真實批次損失。琥珀色直線在圓點處貼著曲線，它的斜率就是反向傳播在那一點算出的梯度。' },
        { en: 'Notice that the gradient changes as you move. Left of the bottom it is negative (raise the weight); right of it, positive (lower it). From both sides it points to the bottom, but at any spot it only describes the slope right there.', zh: '注意：一移動，梯度就跟著變。在谷底左邊它是負的（該調高權重）；在右邊是正的（該調低）。不論從哪一邊，它都指向谷底，但在每個位置它只描述那裡的坡度。' },
      ],
      callout: { kind: 'note', text: { en: 'The curve is the course model’s real loss on one fixed batch of 32 training names; every other weight stays as trained.', zh: '這條曲線是課程模型在一個固定的 32 名字批次上的真實損失；其他權重都維持訓練好的值。' } },
      changed: [{ sign: '+', text: { en: 'A slider that moves one real weight', zh: '一個能移動真實權重的滑桿' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Slopes, multiplied, walked backwards', zh: '斜率相乘，倒著走' },
      body: [
        { en: '<b>Gradient</b>: how fast the loss changes when one weight moves.<br><b>Chain rule</b>: the slope through a chain is the product of the local slopes; for a weight times an input, the local slope is the input.<br><b>Backpropagation</b>: one walk from the loss backwards gives every weight’s gradient.<br><b>Local</b>: a gradient is only accurate for small moves.', zh: '<b>梯度</b>：某個權重移動時，損失變化得多快。<br><b>連鎖律</b>：整條鏈的斜率，就是每一段局部斜率的乘積；權重乘上輸入時，局部斜率就是那個輸入。<br><b>反向傳播</b>：從損失往回走一趟，就得到每個權重的梯度。<br><b>局部性</b>：梯度只在小幅移動時才準。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Optimizers: SGD and Adam. Now that every weight has a gradient, how big a step should each one take?', zh: '下一課：最佳化器：SGD 與 Adam。每個權重都有了梯度，那每個權重該跨多大一步？' } },
    },
  ],

  scene(step, input = {}) {
    const items: Item[] = [];
    const c = chain();
    const BX = [40, 280, 520, 760], BW = 110, BY = 150;
    const vals = [['w', c.v.w], ['a', c.v.a], ['b', c.v.b], ['c', c.v.c]] as const;
    const ops = ['× x', '+ 1', t({ en: 'square', zh: '平方' })];
    const localSlope = [c.slopes.wa, c.slopes.ab, c.slopes.bc];
    const nvals = [c.n.w, c.n.a, c.n.b, c.n.c];
    const grads = [c.grad.w, c.grad.a, c.grad.b, c.grad.c];
    const RECAP = 9, DRAG = 8;

    if ((step >= 1 && step <= 4) || step === RECAP) {
      const shown = step === 1 ? 2 : 4;
      items.push({ k: 'chL', t: 'text', x: 40, y: 40, size: 14.5, tone: 'muted', text: t({ en: 'forward: compute the values, left to right', zh: '前向：由左到右算出各個值' }) });
      items.push({ k: 'xin', t: 'box', x: 175, y: 62, w: 80, h: 44, label: `x = ${X}`, sub: t({ en: 'input', zh: '輸入' }), tone: 'muted' });
      items.push({ k: 'xina', t: 'arrow', x1: 215, y1: 110, x2: 215, y2: BY + 2 });
      for (let i = 0; i < shown; i++) {
        const sub = i === 0 ? t({ en: 'the weight', zh: '權重' }) : i === 3 ? t({ en: 'the loss', zh: '損失' }) : undefined;
        items.push({ k: 'cb' + i, t: 'box', x: BX[i], y: BY, w: BW, h: 54, label: `${vals[i][0]} = ${vals[i][1]}`, sub, tone: i === 0 ? 'a' : 'ink' });
        if (i < shown - 1) {
          items.push({ k: 'ca' + i, t: 'arrow', x1: BX[i] + BW + 8, y1: BY + 27, x2: BX[i + 1] - 8, y2: BY + 27 });
          items.push({ k: 'co' + i, t: 'text', x: (BX[i] + BW + BX[i + 1]) / 2 + (i === 0 ? 22 : 0), y: BY + 16, anchor: 'middle', size: 15, bold: true, text: ops[i] });
          items.push({ k: 'cs' + i, t: 'text', x: (BX[i] + BW + BX[i + 1]) / 2, y: BY + 48, anchor: 'middle', size: 13.5, tone: 'b', text: t({ en: `slope ${localSlope[i]}`, zh: `斜率 ${localSlope[i]}` }) });
        }
      }
      if (step === 1 || step === 3) {
        const NY = 280;
        items.push({ k: 'nL', t: 'text', x: 40, y: NY - 18, size: 14.5, tone: 'muted', text: t({ en: `nudge w by ${H}:`, zh: `把 w 輕推 ${H}：` }) });
        for (let i = 0; i < shown; i++) {
          items.push({ k: 'nb' + i, t: 'box', x: BX[i], y: NY, w: BW, h: 44, label: `${vals[i][0]} = ${f4(nvals[i])}`, tone: i === 0 ? 'b' : 'ink' });
          if (i < shown - 1) items.push({ k: 'na' + i, t: 'arrow', x1: BX[i] + BW + 8, y1: NY + 22, x2: BX[i + 1] - 8, y2: NY + 22 });
        }
        const last = shown - 1;
        const change = nvals[last] - vals[last][1];
        items.push({ k: 'nr', t: 'text', x: 450, y: 390, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'a',
          text: step === 1
            ? t({ en: `a changed by ${change.toFixed(4)} = ${X} × ${H}  (slope = x)`, zh: `a 改變了 ${change.toFixed(4)} = ${X} × ${H}（斜率 = x）` })
            : t({ en: `c changed by ${change.toFixed(4)} ≈ ${c.grad.w} × ${H} = ${(c.grad.w * H).toFixed(2)}`, zh: `c 改變了 ${change.toFixed(4)} ≈ ${c.grad.w} × ${H} = ${(c.grad.w * H).toFixed(2)}` }) });
        if (step === 3) items.push({ k: 'cr', t: 'text', x: 450, y: 440, anchor: 'middle', size: 16, mono: true, text: t({ en: `total slope from w to the loss c: ${X} × 1 × ${c.slopes.bc} = ${c.grad.w}`, zh: `從 w 到損失 c 的總斜率：${X} × 1 × ${c.slopes.bc} = ${c.grad.w}` }) });
      }
      if (step === 2) {
        items.push({ k: 'cq', t: 'text', x: 450, y: 300, anchor: 'middle', size: 16, tone: 'muted', text: t({ en: 'if the weight w moves, how much does the loss c move?', zh: '如果權重 w 動了，損失 c 會動多少？' }) });
        items.push({ k: 'sqc', t: 'text', x: 450, y: 350, anchor: 'middle', size: 15, mono: true, tone: 'b', text: t({ en: `squaring near ${c.v.b}: ${c.v.b + H}² − ${c.v.b}² = ${f4(SQ)} ≈ ${c.slopes.bc} × ${H}`, zh: `在 ${c.v.b} 附近平方：${c.v.b + H}² − ${c.v.b}² = ${f4(SQ)} ≈ ${c.slopes.bc} × ${H}` }) });
      }
      if (step === 4 || step === RECAP) {
        const GY = 320;
        items.push({ k: 'gL', t: 'text', x: 40, y: GY - 28, size: 14.5, tone: 'a', text: t({ en: 'backward: gradients, right to left (multiply by each slope)', zh: '反向：由右到左算梯度（乘上每一段的斜率）' }) });
        for (let i = 0; i < 4; i++) {
          items.push({ k: 'gb' + i, t: 'box', x: BX[i], y: GY, w: BW, h: 50, label: t({ en: `gradient ${grads[i]}`, zh: `梯度 ${grads[i]}` }), tone: 'a', active: i === 0 });
          if (i > 0) items.push({ k: 'garr' + i, t: 'arrow', x1: BX[i] - 8, y1: GY + 25, x2: BX[i - 1] + BW + 8, y2: GY + 25, tone: 'a' });
          if (i > 0) items.push({ k: 'gm' + i, t: 'text', x: (BX[i - 1] + BW + BX[i]) / 2, y: GY + 50, anchor: 'middle', size: 13.5, tone: 'a', text: i === 1 ? t({ en: `× ${X} (x)`, zh: `× ${X}（x）` }) : `× ${localSlope[i - 1]}` });
        }
        items.push({ k: 'gst', t: 'text', x: BX[3] + BW / 2, y: GY + 82, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'start here: 1', zh: '從這裡開始：1' }) });
        items.push({ k: 'gw', t: 'text', x: BX[0] + BW / 2, y: GY + 82, anchor: 'middle', size: 13.5, tone: 'a', bold: true, text: t({ en: 'what training uses', zh: '訓練要用的就是這個' }) });
        if (step === RECAP) items.push({ k: 'rcp', t: 'text', x: 450, y: 470, anchor: 'middle', size: 15, tone: 'a', bold: true, text: t({ en: `the real model: the same walk back gives all ${CN} gradients at once`, zh: `真實模型：同樣往回走一趟，${CN} 個梯度一次全拿到` }) });
      }
    }

    if (step === 0 || (step >= 5 && step <= DRAG)) {
      const R = real();
      if (step === 0) {
        const m = GPT.fromJSON(weights as Weights);
        items.push({ k: 'w0v', t: 'vec', x: 60, y: 110, v: m.p.tok.row(1), cell: 30, label: t({ en: `16 of the ${R.count.toLocaleString()} weights (the vector for “a”); each one needs its own answer`, zh: `${R.count.toLocaleString()} 個權重中的 16 個（「a」的向量），每一個都需要自己的答案` }) });
        items.push({ k: 'w0', t: 'box', x: 60, y: 200, w: 170, h: 56, label: t({ en: 'one weight', zh: '一個權重' }), sub: `+ ${H}`, tone: 'b' });
        items.push({ k: 'w0a', t: 'arrow', x1: 240, y1: 228, x2: 330, y2: 228 });
        items.push({ k: 'm0', t: 'box', x: 340, y: 190, w: 220, h: 76, label: t({ en: 'run the whole model', zh: '整個模型跑一次' }), sub: t({ en: `${R.count.toLocaleString()} weights`, zh: `${R.count.toLocaleString()} 個權重` }) });
        items.push({ k: 'm0a', t: 'arrow', x1: 570, y1: 228, x2: 660, y2: 228 });
        items.push({ k: 'l0', t: 'box', x: 670, y: 200, w: 170, h: 56, label: t({ en: 'loss changed by ?', zh: '損失變了多少？' }), tone: 'a' });
        items.push({ k: 'rep', t: 'text', x: 450, y: 330, anchor: 'middle', size: 16, tone: 'bad', text: t({ en: `…repeat for each of the ${R.count.toLocaleString()} weights, every training step`, zh: `……每一步訓練，都要對 ${R.count.toLocaleString()} 個權重各做一次` }) });
      }
      if (step === 5) {
        const cols = [60, 400, 545, 705];
        const head = [t({ en: 'weight', zh: '權重' }), t({ en: 'gradient', zh: '梯度' }), t({ en: 'predicted change', zh: '預測的變化' }), t({ en: 'measured change', zh: '實測的變化' })];
        head.forEach((h, j) => items.push({ k: 'th' + j, t: 'text', x: cols[j], y: 100, size: 14.5, bold: true, tone: 'muted', text: h }));
        items.push({ k: 'tsub', t: 'text', x: cols[2], y: 122, size: 13, tone: 'muted', text: t({ en: `gradient × ${H}`, zh: `梯度 × ${H}` }) });
        items.push({ k: 'tsub2', t: 'text', x: cols[3], y: 122, size: 13, tone: 'muted', text: t({ en: `really nudge by ${H}`, zh: `真的推 ${H}` }) });
        R.probes.forEach((p, i) => {
          const y = 170 + i * 78;
          items.push({ k: 'pr' + i, t: 'box', x: 40, y: y - 30, w: 820, h: 62, tone: i === 2 ? 'a' : 'ink' });
          items.push({ k: 'pn' + i, t: 'text', x: cols[0], y: y - 2, size: 15, mono: true, text: t(p.label) });
          items.push({ k: 'pl' + i, t: 'text', x: cols[0], y: y + 20, size: 13, tone: 'muted', text: t(p.plain) });
          items.push({ k: 'pg' + i, t: 'text', x: cols[1], y: y + 6, size: 15, mono: true, text: sg(p.g) });
          items.push({ k: 'pp' + i, t: 'text', x: cols[2], y: y + 6, size: 15, mono: true, text: sg(p.pred, 6) });
          items.push({ k: 'pm' + i, t: 'text', x: cols[3], y: y + 6, size: 15, mono: true, bold: true, tone: 'a', text: sg(p.meas, 6) });
        });
        items.push({ k: 'tnote', t: 'text', x: 450, y: 430, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: `batch loss before any nudge: ${R.L0.toFixed(4)} · a negative gradient means raising the weight lowers the loss`, zh: `輕推前的批次損失：${R.L0.toFixed(4)} · 梯度為負，代表調高權重會讓損失下降` }) });
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
          xlabel: t({ en: 'how far the weight is pushed (downhill direction)', zh: '權重往下坡方向被推了多遠' }), ylabel: t({ en: 'batch loss', zh: '批次損失' }),
          guide: { y: R.L0, label: t({ en: `before the push · ${R.L0.toFixed(3)}`, zh: `推之前 · ${R.L0.toFixed(3)}` }) },
        });
        items.push({ k: 'bh0', t: 'text', x: 620, y: 80, size: 14.5, bold: true, tone: 'muted', text: t({ en: 'push', zh: '推動量' }) });
        items.push({ k: 'bh1', t: 'text', x: 700, y: 80, size: 14.5, bold: true, tone: 'b', text: t({ en: 'predicted', zh: '預測' }) });
        items.push({ k: 'bh2', t: 'text', x: 790, y: 80, size: 14.5, bold: true, tone: 'a', text: t({ en: 'measured', zh: '實測' }) });
        R.big.forEach((b, i) => {
          const y = 120 + i * 40;
          items.push({ k: 'bd' + i, t: 'text', x: 620, y, size: 15, mono: true, text: '+' + b.d });
          items.push({ k: 'bp' + i, t: 'text', x: 700, y, size: 15, mono: true, tone: 'b', text: sg(b.pred, 4) });
          items.push({ k: 'bm' + i, t: 'text', x: 790, y, size: 15, mono: true, tone: Math.abs(b.meas - b.pred) > 0.05 ? 'bad' : 'a', text: sg(b.meas, 4) });
        });
        items.push({ k: 'bw', t: 'text', x: 620, y: 300, size: 14, tone: 'muted', text: t({ en: `weight: ${t(R.main.label)}`, zh: `權重：${t(R.main.label)}` }) });
        items.push({ k: 'bw2', t: 'text', x: 620, y: 322, size: 13, tone: 'muted', text: t(R.main.plain) });
      }
      if (step === DRAG) {
        const D = drag();
        const d = Math.max(-DSPAN, Math.min(DSPAN, Number(input.d ?? 0)));
        const P = pointAt(d);
        const dd = Math.round(d / DSTEP) * DSTEP;
        const X0 = 80, Y0 = 70, PW = 500, PH = 330;
        const lo = Math.floor(Math.min(...D.curve) * 10) / 10 - 0.1, hi = Math.ceil(Math.max(...D.curve) * 10) / 10;
        const px = (v: number) => X0 + ((v + DSPAN) / (2 * DSPAN)) * PW;
        const py = (v: number) => Y0 + PH - ((v - lo) / (hi - lo)) * PH;
        items.push({ k: 'dfr', t: 'box', x: X0, y: Y0, w: PW, h: PH, tone: 'muted' });
        for (let g = 0; lo + g * 0.1 <= hi + 1e-9; g++) {
          const v = lo + g * 0.1;
          items.push({ k: 'dyl' + g, t: 'text', x: X0 - 8, y: py(v) + 4, anchor: 'end', size: 13, tone: 'muted', text: v.toFixed(1) });
        }
        for (const v of [-DSPAN, -0.75, 0, 0.75, DSPAN]) items.push({ k: 'dxl' + v, t: 'text', x: px(v), y: Y0 + PH + 20, anchor: 'middle', size: 13, tone: 'muted', text: (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v) });
        items.push({ k: 'dxt', t: 'text', x: X0 + PW / 2, y: Y0 + PH + 44, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'change to the weight (0 = its trained value)', zh: '權重的改變量（0 = 訓練好的值）' }) });
        items.push({ k: 'dyt', t: 'text', x: X0, y: Y0 - 12, size: 13.5, tone: 'muted', text: t({ en: 'batch loss', zh: '批次損失' }) });
        items.push({ k: 'd0', t: 'path', x: 0, y: 0, pts: [[px(0), Y0], [px(0), Y0 + PH]], tone: 'muted', width: 1, dashed: true });
        items.push({ k: 'dcv', t: 'path', x: 0, y: 0, pts: D.ds.map((v, i) => [px(v), py(D.curve[i])] as [number, number]), tone: 'a', width: 3 });
        // tangent line through the point, slope = gradient, kept inside the frame
        const clip = (e: number) => {
          let x = Math.max(-DSPAN, Math.min(DSPAN, e));
          const y = P.loss + P.g * (x - dd);
          if (y > hi) x = dd + (hi - P.loss) / P.g; else if (y < lo) x = dd + (lo - P.loss) / P.g;
          return [px(x), py(P.loss + P.g * (x - dd))] as [number, number];
        };
        items.push({ k: 'dtg', t: 'path', x: 0, y: 0, pts: [clip(dd - 0.6), clip(dd + 0.6)], tone: 'b', width: 2.5 });
        items.push({ k: 'ddot', t: 'dot', x: px(dd), y: py(P.loss), r: 7, tone: 'b' });
        const RX = 620;
        const flat = Math.abs(P.g) < 0.02;
        const rows: [L, string, 'ink' | 'a' | 'b'][] = [
          [{ en: 'weight value', zh: '權重的值' }, (D.keep + dd).toFixed(2), 'ink'],
          [{ en: 'batch loss here', zh: '這裡的批次損失' }, P.loss.toFixed(4), 'a'],
          [{ en: 'gradient here (backprop)', zh: '這裡的梯度（反向傳播）' }, sg(P.g, 3), 'b'],
        ];
        rows.forEach(([l, v, tone], i) => {
          items.push({ k: 'drl' + i, t: 'text', x: RX, y: 100 + i * 64, size: 13.5, tone: 'muted', text: t(l) });
          items.push({ k: 'drv' + i, t: 'text', x: RX, y: 128 + i * 64, size: 22, bold: true, mono: true, tone, text: v });
        });
        items.push({ k: 'drd', t: 'text', x: RX, y: 312, size: 13.5, tone: 'muted', text: t({ en: 'downhill is', zh: '下坡方向' }) });
        items.push({ k: 'drw', t: 'text', x: RX, y: 340, size: 17, bold: true, tone: 'a', text: flat
          ? t({ en: 'nearly flat: the bottom', zh: '幾乎是平的：谷底' })
          : P.g < 0 ? t({ en: '→ raise the weight', zh: '→ 調高權重' }) : t({ en: '← lower the weight', zh: '← 調低權重' }) });
        items.push({ k: 'drn', t: 'text', x: RX, y: 380, size: 13, tone: 'muted', text: t({ en: `weight: ${t(R.main.label)}`, zh: `權重：${t(R.main.label)}` }) });
      }
    }

    const R = step === 0 || step >= 5 ? real() : null;
    let metrics: { label: L; value: string }[];
    if (step >= 1 && step <= 4) {
      metrics = [
        { label: { en: 'Weight w', zh: '權重 w' }, value: String(c.v.w) },
        { label: { en: 'Loss c', zh: '損失 c' }, value: step === 1 ? '—' : String(c.v.c) },
        { label: { en: 'Gradient of w', zh: 'w 的梯度' }, value: step >= 3 ? String(c.grad.w) : '?' },
        { label: { en: 'Passes needed', zh: '需要的計算趟數' }, value: step === 4 ? t({ en: '1 back', zh: '往回 1 趟' }) : '—' },
      ];
    } else if (step === DRAG) {
      const d = Math.max(-DSPAN, Math.min(DSPAN, Number(input.d ?? 0)));
      const P = pointAt(d);
      metrics = [
        { label: { en: 'Change to weight', zh: '權重改變量' }, value: sg(Math.round(d / DSTEP) * DSTEP, 2) },
        { label: { en: 'Batch loss', zh: '批次損失' }, value: P.loss.toFixed(4) },
        { label: { en: 'Gradient', zh: '梯度' }, value: sg(P.g, 3) },
        { label: { en: 'Trained value', zh: '訓練好的值' }, value: drag().keep.toFixed(2) },
      ];
    } else {
      metrics = [
        { label: { en: 'Weights', zh: '權重數' }, value: R!.count.toLocaleString() },
        { label: { en: 'Batch loss', zh: '批次損失' }, value: R!.L0.toFixed(4) },
        { label: { en: 'Gradients known', zh: '已知梯度' }, value: step === 0 ? '0' : R!.count.toLocaleString() },
        { label: { en: 'Runs needed', zh: '需要跑幾次' }, value: step === 0 ? (R!.count + 1).toLocaleString() : t({ en: '1 fwd + 1 back', zh: '前向 1 + 反向 1' }) },
      ];
    }
    return {
      metrics,
      items,
      legend: step === 7 ? [{ tone: 'a', text: { en: 'real loss', zh: '真實損失' } }, { tone: 'b', text: { en: 'gradient × push (straight-line guess)', zh: '梯度 × 推動量（直線預測）' } }]
        : step === DRAG ? [{ tone: 'a', text: { en: 'real batch loss', zh: '真實批次損失' } }, { tone: 'b', text: { en: 'tangent: slope = gradient', zh: '切線：斜率 = 梯度' } }] : undefined,
      controls: step === DRAG ? [{
        key: 'd', kind: 'slider', label: { en: 'Change to the weight', zh: '權重的改變量' }, min: -DSPAN, max: DSPAN, step: DSTEP, value: 0,
        format: (v: number) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2),
      }] : undefined,
    };
  },
};
