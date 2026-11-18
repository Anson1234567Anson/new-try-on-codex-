import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, VAL_NAMES, encode, example, gaussian, rng } from '../model/data';
import { Mat, add, addRow, backward, matmul, relu, softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';

// We follow the second "m" of ".jimm" (the running example of the attention
// lesson) and watch attention and the MLP add their results onto its vector.

const WORD = '.jimm';
const LABELS = [...WORD];
const F = LABELS.length - 1;
const DEPTH = 12;

const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const row = (v: number[]) => { const m = new Mat(1, v.length); m.d.set(v); return m; };
const size = (v: ArrayLike<number>) => Math.sqrt(Array.from(v).reduce((s, x) => s + x * x, 0));
type Mode = 'normal' | 'noAtt' | 'noMlp';

interface Data {
  x: number[]; att: number[]; x1: number[]; mlp: number[]; x2: number[];
  probs: Record<Mode, number[]>; loss: Record<Mode, number>;
}
let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const P = model.p;
  const mlp = (v: number[]) => addRow(matmul(relu(addRow(matmul(row(v), P.w1), P.b1)), P.w2), P.b2).row(0);
  const probsOf = (v: number[]) => softmaxRow(addRow(matmul(row(v), P.u), P.bu).row(0));
  // the vector that reaches the output when one part is switched off
  const final = (tr: ReturnType<GPT['forward']>, i: number, mode: Mode): number[] => {
    if (mode === 'normal') return tr.x2.row(i);
    if (mode === 'noMlp') return tr.x1.row(i); // x2 = x1: nothing added by the MLP
    const x = tr.x.row(i), o = mlp(x); // x1 = x: nothing added by attention
    return x.map((v, j) => v + o[j]);
  };
  const modes: Mode[] = ['normal', 'noAtt', 'noMlp'];
  const tr = model.forward(encode(WORD));
  const probs = Object.fromEntries(modes.map((m) => [m, probsOf(final(tr, F, m))])) as Record<Mode, number[]>;
  const loss = { normal: 0, noAtt: 0, noMlp: 0 };
  let n = 0;
  for (const name of VAL_NAMES) {
    const ex = example(name);
    const t2 = model.forward(ex.x);
    ex.x.forEach((_, i) => { for (const m of modes) loss[m] -= Math.log(probsOf(final(t2, i, m))[ex.y[i]]); n++; });
  }
  for (const m of modes) loss[m] /= n;
  return { x: tr.x.row(F), att: tr.attOut.row(F), x1: tr.x1.row(F), mlp: tr.mlpOut.row(F), x2: tr.x2.row(F), probs, loss };
})();

// A stack of 12 random MLP layers (set up the way our model starts), run once
// with "add the result" and once with "replace with the result".
interface Deep { sizes: number[]; grad: number }
let deepCache: { add: Deep; replace: Deep } | undefined;
const deep = () => deepCache ??= (() => {
  const runOne = (residual: boolean): Deep => {
    const r = rng(3);
    const layers = Array.from({ length: DEPTH }, () => {
      const a = new Mat(16, 32), b = new Mat(32, 16);
      for (let i = 0; i < a.d.length; i++) a.d[i] = gaussian(r) / 4;
      for (let i = 0; i < b.d.length; i++) b.d[i] = gaussian(r) / Math.sqrt(32);
      return [a, b];
    });
    let x = new Mat(1, 16);
    for (let i = 0; i < 16; i++) x.d[i] = gaussian(r);
    const sizes: number[] = [];
    for (const [a, b] of layers) {
      const h = matmul(relu(matmul(x, a)), b);
      x = residual ? add(x, h) : h;
      sizes.push(size(x.d));
    }
    const ones = new Mat(16, 1); ones.d.fill(1);
    backward(matmul(x, ones)); // a stand-in loss: the sum of the final numbers
    return { sizes, grad: size([...layers[0][0].g, ...layers[0][1].g]) };
  };
  return { add: runOne(true), replace: runOne(false) };
})();

const pct = (v: number) => Math.round(v * 100) + '%';
const sz = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 0.1 ? v.toPrecision(2) : String(Number(v.toPrecision(1))));
const top = (p: number[], n: number) =>
  p.map((v, i) => ({ label: CHARS[i], v })).sort((a, b) => b.v - a.v).slice(0, n).map((b) => ({ ...b, hi: b.label === 'y' }));
const best = (p: number[]) => top(p, 1)[0];

// stream geometry
const SY = 160; // y of the stream line
const ATT = { x: 250, w: 170 }, MLP = { x: 540, w: 170 };

export const residual: Lesson = {
  id: 'residual',
  title: { en: 'The Residual Stream', zh: '殘差流' },
  blurb: {
    en: 'Attention and the MLP never overwrite a token’s vector. They add to it, and that keeps deep models trainable.',
    zh: '注意力和 MLP 從不覆蓋詞元的向量，而是往上加。正是這一點讓很深的模型也訓練得動。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'Why sub-layers add instead of replace', zh: '為什麼子層是「加上去」而不是「取代」' }, step: 1 },
    { text: { en: 'What the residual stream is', zh: '什麼是殘差流' }, step: 3 },
    { text: { en: 'How to test a part by switching it off', zh: '如何把某個部分關掉來測試它' }, step: 5 },
    { text: { en: 'Why adding helps deep models train', zh: '為什麼「相加」能幫助深層模型訓練' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Two parts, one vector', zh: '兩個部分，一個向量' },
      body: [
        { en: 'Our model reads <code>.jimm</code>. The vector of the second “m” knows which letter it is and where it sits. Then attention produces 16 new numbers about the context, and the MLP produces 16 more.', zh: '模型讀進 <code>.jimm</code>。第二個「m」的向量知道自己是哪個字母、在第幾個位置。接著注意力會產生 16 個關於上下文的新數字，MLP 也會再產生 16 個。' },
        { en: 'If each part simply <i>replaced</i> the vector with its own result, the token would forget what it was. How should the results be combined?', zh: '如果每個部分都直接用自己的結果<i>取代</i>原本的向量，詞元就會忘記自己是誰。那這些結果該怎麼合起來？' },
      ],
      callout: { kind: 'insight', text: { en: 'Each part should contribute something new without wiping out what is already there.', zh: '每個部分都應該貢獻新的東西，同時不抹掉原本已有的資訊。' } },
    },
    {
      short: { en: 'Add', zh: '相加' }, chapter: 'technique',
      title: { en: 'Add, don’t replace', zh: '用加的，不用取代' },
      body: [
        { en: 'The answer is plain addition. The attention output is added onto the vector, cell by cell: <code>x1 = x + attention output</code>.', zh: '答案就是單純的加法。注意力的輸出逐格加到向量上：<code>x1 = x + 注意力輸出</code>。' },
        { en: 'Check any column below: the bottom number is the sum of the two above (up to rounding). Whatever <code>x</code> held is still in there, now with context mixed in.', zh: '隨便看下面任何一欄：最下面的數字就是上面兩個數字的和（只差四捨五入）。<code>x</code> 原本的內容都還在，只是多混入了上下文。' },
      ],
      changed: [{ sign: '+', text: { en: 'x1 = x + attention output', zh: 'x1 = x + 注意力輸出' } }],
    },
    {
      short: { en: 'Add again', zh: '再加一次' }, chapter: 'technique',
      title: { en: 'The MLP adds its part too', zh: 'MLP 也把它那份加上去' },
      body: [
        { en: 'The MLP reads <code>x1</code> and its 16 output numbers are added the same way: <code>x2 = x1 + MLP output</code>.', zh: 'MLP 讀取 <code>x1</code>，它輸出的 16 個數字也用同樣的方式加上去：<code>x2 = x1 + MLP 輸出</code>。' },
        { en: '<code>x2</code> is what the output layer turns into a prediction. It holds the original token, plus what attention found, plus what the MLP worked out.', zh: '<code>x2</code> 就是輸出層用來產生預測的向量。它包含原本的詞元資訊，加上注意力找到的東西，再加上 MLP 算出來的東西。' },
      ],
      changed: [{ sign: '+', text: { en: 'x2 = x1 + MLP output', zh: 'x2 = x1 + MLP 輸出' } }],
    },
    {
      short: { en: 'The stream', zh: '殘差流' }, chapter: 'technique',
      title: { en: 'A stream that parts write into', zh: '一條讓各部分寫入的「流」' },
      body: [
        { en: 'Picture the token’s vector as a stream flowing from the input to the output. Attention and the MLP branch off, read it, and add their result back in. This is the <b>residual stream</b>; the added results are called <b>residuals</b>.', zh: '把詞元的向量想像成一條從輸入流到輸出的河。注意力和 MLP 從旁邊分出來，讀取它，再把結果加回去。這就是<b>殘差流（residual stream）</b>，加回去的結果叫做<b>殘差（residual）</b>。' },
        lz(() => ({ en: `Each addition makes the vector longer. Its size (the length of the 16 numbers seen as an arrow) grows from ${size(data().x).toFixed(2)} to ${size(data().x1).toFixed(2)} to ${size(data().x2).toFixed(2)}.`, zh: `每加一次，向量就變長一點。它的大小（把 16 個數字看成一支箭頭時的長度）從 ${size(data().x).toFixed(2)} 變成 ${size(data().x1).toFixed(2)}，再變成 ${size(data().x2).toFixed(2)}。` })),
      ],
      callout: { kind: 'note', text: { en: 'That growing size matters later: the Layer Normalization lesson deals with it.', zh: '這個越來越大的尺寸之後會造成問題，「層正規化」那一課會處理它。' } },
      changed: [{ sign: '~', text: { en: 'x → x1 → x2 on one stream', zh: 'x → x1 → x2 在同一條流上' } }],
    },
    {
      short: { en: 'Prediction', zh: '預測' }, chapter: 'solution',
      title: { en: 'Everything adds up to a prediction', zh: '全部加起來，變成預測' },
      body: [
        lz(() => ({ en: `With both parts adding their share, the model predicts “${best(data().probs.normal).label}” after <code>.jimm</code> with ${pct(best(data().probs.normal).v)}, as in “jimmy”.`, zh: `兩個部分都加上自己的那份之後，模型在 <code>.jimm</code> 後面以 ${pct(best(data().probs.normal).v)} 預測「${best(data().probs.normal).label}」，就像「jimmy」。` })),
        { en: 'Because the stream is a sum, we can test each part: take its contribution out and see what changes.', zh: '因為這條流是一個總和，我們可以逐一測試每個部分：把它的貢獻拿掉，看看會有什麼改變。' },
      ],
      changed: [{ sign: '+', text: { en: 'Prediction from x2', zh: '由 x2 產生的預測' } }],
    },
    {
      short: { en: 'No attention', zh: '關掉注意力' }, chapter: 'problem',
      title: { en: 'Case: switch off attention', zh: '案例：關掉注意力' },
      body: [
        lz(() => ({ en: `Set the attention output to zero, so <code>x1 = x</code>. Now the second “m” cannot see the first one. The top guess becomes “${best(data().probs.noAtt).label}” (${pct(best(data().probs.noAtt).v)}) and “y” drops to ${pct(data().probs.noAtt[CHARS.indexOf('y')])}.`, zh: `把注意力的輸出設成零，於是 <code>x1 = x</code>。現在第二個「m」看不到第一個「m」了。最可能的答案變成「${best(data().probs.noAtt).label}」（${pct(best(data().probs.noAtt).v)}），「y」則降到 ${pct(data().probs.noAtt[CHARS.indexOf('y')])}。` })),
        lz(() => ({ en: `Over 615 names the model never trained on, the loss rises from ${data().loss.normal.toFixed(2)} to ${data().loss.noAtt.toFixed(2)}.`, zh: `在模型沒練過的 615 個名字上，損失從 ${data().loss.normal.toFixed(2)} 升到 ${data().loss.noAtt.toFixed(2)}。` })),
      ],
      callout: { kind: 'note', text: { en: 'Switching a part off and measuring the damage is called an ablation. It is a common way to find out what a part does.', zh: '把某個部分關掉、再量測造成的損害，這叫做消融實驗（ablation），是了解某個部分功用的常見方法。' } },
      changed: [{ sign: '~', text: { en: 'Attention output → 0', zh: '注意力輸出 → 0' } }],
    },
    {
      short: { en: 'No MLP', zh: '關掉 MLP' }, chapter: 'problem',
      title: { en: 'Case: switch off the MLP', zh: '案例：關掉 MLP' },
      body: [
        lz(() => ({ en: `Now keep attention but zero the MLP output, so <code>x2 = x1</code>. The prediction smears out: the top guess is “${best(data().probs.noMlp).label}” with only ${pct(best(data().probs.noMlp).v)}.`, zh: `這次保留注意力，但把 MLP 的輸出設成零，於是 <code>x2 = x1</code>。預測變得很分散：最可能的「${best(data().probs.noMlp).label}」只有 ${pct(best(data().probs.noMlp).v)}。` })),
        lz(() => ({ en: `The held-out loss jumps to ${data().loss.noMlp.toFixed(2)}, worse than removing attention. In this model, the MLP does much of the work of turning context into an answer.`, zh: `保留資料上的損失跳到 ${data().loss.noMlp.toFixed(2)}，比拿掉注意力還糟。在這個模型裡，把上下文轉成答案的工作，很大一部分是 MLP 做的。` })),
      ],
      changed: [{ sign: '~', text: { en: 'MLP output → 0 (attention back on)', zh: 'MLP 輸出 → 0（注意力恢復）' } }],
    },
    {
      short: { en: 'Why it trains', zh: '為何好訓練' }, chapter: 'technique',
      title: { en: 'Why adding helps deep models learn', zh: '為什麼相加能讓深層模型學得動' },
      body: [
        lz(() => ({ en: `Real models stack dozens of layers. Here are ${DEPTH} random MLP layers, set up the way our model starts. If each layer <i>replaces</i> the vector, the signal shrinks layer after layer, to ${sz(deep().replace.sizes[DEPTH - 1])}. If each layer <i>adds</i>, it survives.`, zh: `真實的模型會疊上幾十層。這裡有 ${DEPTH} 層隨機的 MLP，初始設定和我們的模型一樣。如果每一層都<i>取代</i>向量，訊號會一層一層縮小，最後只剩 ${sz(deep().replace.sizes[DEPTH - 1])}；如果每一層都<i>相加</i>，訊號就能保留下來。` })),
        lz(() => ({ en: `Learning runs the same path backwards. The <b>gradient</b> (the “which way to change” signal) reaching layer 1 is ${sz(deep().add.grad)} with adding, but only ${sz(deep().replace.grad)} with replacing. Adding gives the signal a straight path through every layer.`, zh: `學習時訊號沿著同一條路往回走。傳到第 1 層的<b>梯度</b>（「該往哪邊改」的訊號），用相加時是 ${sz(deep().add.grad)}，用取代時只有 ${sz(deep().replace.grad)}。相加讓訊號有一條穿過每一層的直達路線。` })),
      ],
      callout: { kind: 'insight', text: { en: 'With addition, a layer that has not learned anything yet simply passes the stream through unchanged. Every layer starts as “do no harm”.', zh: '有了加法，一個還沒學到東西的層，頂多就是讓殘差流原封不動地通過。每一層一開始都是「至少不搞砸」。' } },
      changed: [{ sign: '+', text: { en: `${DEPTH}-layer test: add vs replace`, zh: `${DEPTH} 層測試：相加 vs 取代` } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'The residual stream in one line', zh: '一句話說完殘差流' },
      body: [
        { en: 'Each token’s vector flows along a stream. Attention and the MLP read it and <b>add</b> their results: <code>x1 = x + attention</code>, <code>x2 = x1 + MLP</code>.', zh: '每個詞元的向量沿著一條流前進。注意力和 MLP 讀取它，再把結果<b>加</b>回去：<code>x1 = x + 注意力</code>、<code>x2 = x1 + MLP</code>。' },
        { en: 'Adding keeps what the token knew, lets us switch parts off to study them, and gives learning signals a straight path through deep stacks.', zh: '相加保留了詞元原本知道的東西，讓我們可以關掉某個部分來研究它，也讓學習訊號能直達很深的堆疊。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Layer Normalization. All that adding makes vectors grow; layer normalization keeps their size under control.', zh: '下一課：層正規化。一直相加會讓向量越來越大，層正規化負責把它們的大小控制住。' } },
    },
  ],

  scene(step) {
    const d = data();
    const items: Item[] = [];

    // tokens, with the second "m" highlighted
    if (step <= 6) LABELS.forEach((ch, i) => items.push({ k: 'tok' + i, t: 'token', text: ch, x: 20 + i * 50, y: 16, tone: i === F ? 'a' : ch === '.' ? 'b' : 'ink', dim: i !== F }));

    // the stream and its two branches
    if (step <= 6) {
      const offA = step === 5, offM = step === 6;
      items.push({ k: 'stream', t: 'arrow', x1: 40, y1: SY, x2: 870, y2: SY, tone: 'a' });
      items.push({ k: 'sl', t: 'text', x: 870, y: SY - 10, text: t({ en: 'to the output', zh: '往輸出' }), size: 13, tone: 'muted', anchor: 'end' });
      items.push({ k: 'attb', t: 'box', x: ATT.x, y: 70, w: ATT.w, h: 46, label: t({ en: 'attention', zh: '注意力' }), sub: offA ? t({ en: 'switched off', zh: '已關閉' }) : undefined, tone: offA ? 'bad' : step === 1 ? 'a' : 'ink', active: step === 1 });
      items.push({ k: 'mlpb', t: 'box', x: MLP.x, y: 70, w: MLP.w, h: 46, label: 'MLP', sub: offM ? t({ en: 'switched off', zh: '已關閉' }) : undefined, tone: offM ? 'bad' : step === 2 ? 'a' : 'ink', active: step === 2 });
      for (const [k, b, off] of [['a', ATT, offA], ['m', MLP, offM]] as const) {
        items.push({ k: 'in' + k, t: 'arrow', x1: b.x + 30, y1: SY - 2, x2: b.x + 30, y2: 120, tone: 'muted' });
        items.push({ k: 'out' + k, t: 'arrow', x1: b.x + b.w - 30, y1: 120, x2: b.x + b.w - 30, y2: SY - 4, tone: off ? 'bad' : 'a', dashed: off });
        items.push({ k: 'plus' + k, t: 'text', x: b.x + b.w - 12, y: SY + 22, text: step === 0 ? '?' : '+', size: 20, bold: true, tone: off ? 'bad' : 'a' });
      }
      const labs = [{ x: 60, v: 'x' }, { x: ATT.x + ATT.w + 20, v: 'x1' }, { x: MLP.x + MLP.w + 20, v: 'x2' }];
      if (step >= 1) labs.slice(0, step >= 2 ? 3 : 2).forEach((l, i) => items.push({ k: 'sv' + i, t: 'text', x: l.x, y: SY + 26, text: l.v, size: 15, bold: true, mono: true, tone: 'a' }));
      else items.push({ k: 'sv0', t: 'text', x: 60, y: SY + 26, text: 'x', size: 15, bold: true, mono: true, tone: 'a' });
    }

    if (step === 0) {
      items.push({ k: 'qx', t: 'vec', x: 258, y: 270, v: d.x, cell: 24, label: t({ en: 'x: the second “m” (letter + position)', zh: 'x：第二個「m」（字母 + 位置）' }) });
      items.push({ k: 'qa', t: 'vec', x: 258, y: 360, v: d.att, cell: 24, label: t({ en: 'attention output: what it found in the context', zh: '注意力輸出：它在上下文中找到的東西' }) });
      items.push({ k: 'qq', t: 'text', x: 450, y: 450, text: t({ en: 'replace x with it, or combine them?', zh: '用它取代 x，還是把兩者合起來？' }), size: 16, bold: true, tone: 'b', anchor: 'middle' });
    }

    const sumRows = step === 1 ? [d.x, d.att, d.x1] : [d.x1, d.mlp, d.x2];
    if (step === 1 || step === 2) {
      const names = step === 1
        ? ['x', t({ en: '+ attention', zh: '+ 注意力' }), '= x1']
        : ['x1', '+ MLP', '= x2'];
      items.push({ k: 'sum', t: 'heat', x: 130, y: 270, cell: 46, fmt: 'num', max: Math.max(...sumRows.flat().map(Math.abs)), m: sumRows, rows: names, cols: sumRows[0].map((_, j) => String(j + 1)), hi: 2 });
      items.push({ k: 'sumn', t: 'text', x: 130, y: 440, text: t({ en: 'each column: top + middle = bottom (all 16 numbers of the second “m”)', zh: '每一欄：上 + 中 = 下（第二個「m」的全部 16 個數字）' }), size: 13, tone: 'muted' });
    }

    if (step === 3) {
      const vs = [d.x, d.x1, d.x2];
      const xs = [60, ATT.x + ATT.w + 20, MLP.x + MLP.w + 20];
      vs.forEach((v, i) => {
        items.push({ k: 'sv' + i + 'v', t: 'vec', x: xs[i], y: SY + 50, v, cell: 9, max: 2.5 });
        items.push({ k: 'sz' + i, t: 'text', x: xs[i], y: SY + 110, text: t({ en: `size ${size(v).toFixed(2)}`, zh: `大小 ${size(v).toFixed(2)}` }), size: 15, mono: true, bold: true });
      });
      items.push({ k: 'szn', t: 'text', x: 450, y: 380, text: t({ en: 'same colour scale for all three: the stream gets stronger as parts add to it', zh: '三條用同樣的顏色比例：各部分一直往上加，流就越來越強' }), size: 13, tone: 'muted', anchor: 'middle' });
    }

    if (step >= 4 && step <= 6) {
      const mode: Mode = step === 4 ? 'normal' : step === 5 ? 'noAtt' : 'noMlp';
      const title = step === 4 ? t({ en: 'prediction after “.jimm” (both parts on)', zh: '「.jimm」之後的預測（兩個部分都開）' })
        : step === 5 ? t({ en: 'prediction with attention switched off', zh: '關掉注意力時的預測' })
          : t({ en: 'prediction with the MLP switched off', zh: '關掉 MLP 時的預測' });
      items.push({ k: 'pred', t: 'bars', x: 170, y: 250, w: 560, h: 230, title, items: top(d.probs[mode], 6) });
    }

    if (step === 7) {
      const dp = deep();
      const bw = 60, gap = 6, x0 = 87;
      items.push({ k: 'dh', t: 'text', x: 450, y: 30, text: t({ en: `${DEPTH} random layers: size of the vector after each layer`, zh: `${DEPTH} 層隨機的層：每一層之後的向量大小` }), size: 15, bold: true, anchor: 'middle' });
      ([['add', 110, 'good'], ['replace', 290, 'bad']] as const).forEach(([kind, y, tone]) => {
        items.push({ k: 'dl' + kind, t: 'text', x: x0, y: y - 14, text: kind === 'add' ? t({ en: 'add: x = x + layer(x)', zh: '相加：x = x + 層(x)' }) : t({ en: 'replace: x = layer(x)', zh: '取代：x = 層(x)' }), size: 14, bold: true, tone, mono: true });
        dp[kind].sizes.forEach((v, i) => {
          items.push({ k: `d${kind}${i}`, t: 'box', x: x0 + i * (bw + gap), y, w: bw, h: 50, label: sz(v), sub: t({ en: `L${i + 1}`, zh: `第${i + 1}層` }), tone: kind === 'replace' && v < 0.1 ? 'bad' : 'ink' });
        });
        items.push({ k: 'dg' + kind, t: 'text', x: x0, y: y + 84, text: t({ en: `gradient reaching layer 1: ${sz(dp[kind].grad)}`, zh: `傳回第 1 層的梯度：${sz(dp[kind].grad)}` }), size: 15, mono: true, bold: true, tone });
      });
      items.push({ k: 'dfw', t: 'arrow', x1: 60, y1: 440, x2: 380, y2: 440, tone: 'muted' });
      items.push({ k: 'dfwl', t: 'text', x: 60, y: 466, text: t({ en: 'forward: the signal', zh: '前向：訊號' }), size: 13, tone: 'muted' });
      items.push({ k: 'dbw', t: 'arrow', x1: 840, y1: 440, x2: 520, y2: 440, tone: 'b', dashed: true });
      items.push({ k: 'dbwl', t: 'text', x: 840, y: 466, text: t({ en: 'backward: the learning signal', zh: '反向：學習訊號' }), size: 13, tone: 'b', anchor: 'end' });
    }

    if (step === 8) {
      const boxes: [L, boolean][] = [
        [{ en: 'x (token + position)', zh: 'x（詞元 + 位置）' }, false], [{ en: '+ attention', zh: '+ 注意力' }, true], [{ en: 'x1', zh: 'x1' }, false],
        [{ en: '+ MLP', zh: '+ MLP' }, true], [{ en: 'x2 → prediction', zh: 'x2 → 預測' }, false],
      ];
      boxes.forEach(([b, part], i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 210, w: 150, h: 60, label: t(b), tone: part ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 240, x2: bx + 170, y2: 240 });
      });
      items.push({ k: 'rt', t: 'text', x: 450, y: 320, text: t({ en: `held-out loss: ${d.loss.normal.toFixed(2)} · no attention ${d.loss.noAtt.toFixed(2)} · no MLP ${d.loss.noMlp.toFixed(2)}`, zh: `保留資料損失：${d.loss.normal.toFixed(2)} · 無注意力 ${d.loss.noAtt.toFixed(2)} · 無 MLP ${d.loss.noMlp.toFixed(2)}` }), size: 14, tone: 'muted', anchor: 'middle', mono: true });
    }

    const vec = step === 0 ? d.x : step === 1 ? d.x1 : d.x2;
    const mode: Mode | null = step === 4 ? 'normal' : step === 5 ? 'noAtt' : step === 6 ? 'noMlp' : null;
    return {
      metrics: [
        { label: { en: 'Following', zh: '追蹤對象' }, value: t({ en: 'second “m”', zh: '第二個「m」' }) },
        { label: { en: 'Vector size', zh: '向量大小' }, value: step <= 3 ? size(vec).toFixed(2) : '—' },
        { label: { en: 'Top prediction', zh: '最可能的下一個' }, value: mode ? `“${best(d.probs[mode]).label}” ${pct(best(d.probs[mode]).v)}` : step >= 7 ? `“${best(d.probs.normal).label}” ${pct(best(d.probs.normal).v)}` : '—' },
        { label: { en: 'Held-out loss', zh: '保留資料損失' }, value: mode ? d.loss[mode].toFixed(2) : step >= 7 ? d.loss.normal.toFixed(2) : '—' },
      ],
      items,
      legend: [
        { tone: 'pos', text: { en: 'positive number', zh: '正數' } },
        { tone: 'neg', text: { en: 'negative number', zh: '負數' } },
        { tone: 'a', text: { en: 'the stream', zh: '殘差流' } },
      ],
    };
  },
};
