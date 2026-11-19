import type { Control, Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, encode } from '../model/data';
import weights from '../model/weights.json';

// The trained model reads ".mary". We follow the last position of ".ma":
// its final vector × u (+ bu) gives 27 logits, and softmax turns them into probabilities.

const NAME = '.mary';
const P = 2; // the position we follow: the "a" of ".ma"
const CASE = '.sam';

interface Data {
  final: number[]; logits: number[]; probs: number[]; total: number;
  rows: number[][]; // probabilities at every position of NAME
  dot: number; bias: number; // how the logit of the top token is built
  top: number[]; caseProbs: number[]; caseLogits: number[];
  ucol: number[]; // the column of u for the top token
}

const softmax = (l: number[]) => {
  const e = l.map((x) => Math.exp(x));
  const s = e.reduce((a, b) => a + b, 0);
  return { p: e.map((x) => x / s), s };
};

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const tr = model.forward(encode(NAME));
  const final = tr.final.row(P);
  const logits = tr.logits.row(P);
  const { p: probs, s: total } = softmax(logits);
  const rows = encode(NAME).map((_, i) => softmax(tr.logits.row(i)).p);
  const top = probs.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).map((x) => x.i);
  const best = top[0];
  let dot = 0;
  const ucol = final.map((_, k) => model.p.u.d[k * model.p.u.c + best]);
  for (let k = 0; k < final.length; k++) dot += final[k] * ucol[k];
  const c = model.forward(encode(CASE));
  const caseLogits = c.logits.row(CASE.length - 1);
  return { final, logits, probs, total, rows, dot, bias: model.p.bu.d[best], top, caseProbs: softmax(caseLogits).p, caseLogits, ucol };
})();

const lz = (f: (d: Data) => L): L => ({ get en() { return f(data()).en; }, get zh() { return f(data()).zh; } });
const f2 = (v: number) => (v >= 0 ? '' : '−') + Math.abs(v).toFixed(2);
const pct = (v: number) => Math.round(v * 100) + '%';
const pct1 = (v: number) => (v * 100).toFixed(1) + '%';
const C = (i: number) => CHARS[i];
const q = (s: string) => t({ en: `“${s}”`, zh: `「${s}」` });
/** A tiny share written so it never reads as 0: 0.02%, 0.0004%. */
const tiny = (v: number) => (v * 100 >= 0.1 ? (v * 100).toFixed(1) : (v * 100).toPrecision(1)) + '%';
const mult = (by: number) => { const m = Math.exp(by); return m >= 10 ? m.toFixed(0) : m >= 0.1 ? m.toFixed(2) : m.toPrecision(2); };
const sgn = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1);

// the interactive step: drag one letter's logit and watch all 27 shares
const TRY = 5;
const tryControls = (d: Data): Control[] => [
  { key: 'letter', kind: 'choice', label: { en: 'Letter', zh: '字母' }, value: String(d.top[0]), options: [d.top[0], d.top[1], d.top[26]].map((i) => ({ value: String(i), label: { en: `“${C(i)}”`, zh: `「${C(i)}」` } })) },
  { key: 'nudge', kind: 'slider', label: { en: 'Add to its logit', zh: '它的 logit 加上' }, min: -6, max: 6, step: 0.1, value: 0, format: sgn },
];
/** Exact softmax of the .ma logits with one logit changed. */
const nudged = (d: Data, id: number, by: number) => {
  const l = d.logits.map((v, i) => (i === id ? v + by : v));
  const m = Math.max(...l), e = l.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
};
const caseTop = (d: Data) => d.caseProbs.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);

/** Letters shown as columns in the "every position" grid: each row's top 2 plus the true next letter. */
const gridCols = (d: Data) => {
  const ids = encode(NAME + '.');
  const set = new Set<number>();
  d.rows.forEach((p, i) => {
    p.map((v, j) => ({ v, j })).sort((a, b) => b.v - a.v).slice(0, 2).forEach((x) => set.add(x.j));
    set.add(ids[i + 1]);
  });
  return [...set].sort((a, b) => a - b);
};

export const nextToken: Lesson = {
  id: 'next-token',
  title: { en: 'Predicting the Next Token', zh: '預測下一個詞元' },
  blurb: {
    en: 'How 16 numbers at the end of the model become a probability for every possible next token.',
    zh: '模型最後的 16 個數字，如何變成每個可能的下一個詞元的機率。',
  },
  minutes: 9,
  level: 'beginner',
  learn: [
    { text: { en: 'Where the prediction comes from', zh: '預測從哪裡來' }, step: 0 },
    { text: { en: 'How one score (logit) is made per token', zh: '每個詞元的分數（logit）怎麼算出來' }, step: 1 },
    { text: { en: 'How softmax turns scores into probabilities', zh: 'softmax 如何把分數變成機率' }, step: 3 },
    { text: { en: 'Why a bigger logit means a much bigger share', zh: '為什麼 logit 大一點，比例就大很多' }, step: 5 },
    { text: { en: 'That the model predicts at every position at once', zh: '模型在每個位置同時做預測' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'From 16 numbers to a guess', zh: '從 16 個數字到一個猜測' },
      body: [
        { en: 'The model has read <code>.ma</code> and must guess the next letter. Inside, each token is a <b>vector</b> of 16 numbers (a vector is just a list of numbers), and the layers you will meet later keep updating it.', zh: '模型讀了 <code>.ma</code>，必須猜出下一個字母。在模型內部，每個詞元都是 16 個數字組成的<b>向量</b>（向量就是一串數字），後面會介紹的各層會不斷更新它。' },
        { en: 'At the very end, the last position holds its <b>final vector</b>. But we need something else: a probability for each of the 27 tokens. How do 16 numbers become 27 probabilities?', zh: '到了最後，最後一個位置留下它的<b>最終向量</b>。但我們要的是另一種東西：27 個詞元各一個機率。16 個數字要怎麼變成 27 個機率？' },
      ],
      callout: { kind: 'insight', text: { en: 'Everything inside a transformer exists to make this one vector as useful as possible for guessing the next token.', zh: 'Transformer 內部的一切，都是為了讓這個向量盡可能有助於猜出下一個詞元。' } },
    },
    {
      short: { en: 'Scores', zh: '分數' }, chapter: 'technique',
      title: { en: 'One score for every token', zh: '每個詞元一個分數' },
      body: [
        { en: 'The model has an <b>output matrix</b> called <code>u</code>, 16 rows × 27 columns: one column of 16 numbers for each token. It was learned in training, like every other weight (the adjustable numbers inside the model).', zh: '模型有一個叫做 <code>u</code> 的<b>輸出矩陣</b>，16 列 × 27 行：每個詞元各有一行 16 個數字。它和其他權重（模型裡可以調整的數字）一樣，是在訓練中學出來的。' },
        lz((d) => ({ en: `To score “${C(d.top[0])}”, multiply the final vector by its column cell by cell and add up: ${f2(d.dot)}. Add a small learned extra number (from <code>bu</code>), ${f2(d.bias)}, and the score is ${f2(d.logits[d.top[0]])}. Do that for all 27 columns.`, zh: `要替「${C(d.top[0])}」打分數，就把最終向量和它那一行逐格相乘再加總：${f2(d.dot)}。再加上一個學習得來的小數字（來自 <code>bu</code>）${f2(d.bias)}，分數就是 ${f2(d.logits[d.top[0]])}。27 行全部照做一遍。` })),
      ],
      changed: [{ sign: '+', text: { en: 'Final vector × u + bu = 27 scores', zh: '最終向量 × u + bu = 27 個分數' } }],
    },
    {
      short: { en: 'Logits', zh: 'Logits' }, chapter: 'technique',
      title: { en: 'Raw scores, called logits', zh: '原始分數，叫做 logits' },
      body: [
        { en: 'These 27 scores are called <b>logits</b>. A higher logit means “more likely next”. But they are not probabilities yet: they can be negative, and they do not add up to anything in particular.', zh: '這 27 個分數叫做 <b>logits</b>。logit 越高，代表「越可能是下一個」。但它們還不是機率：可能是負數，加起來也不等於任何特定的值。' },
        lz((d) => ({ en: `After <code>.ma</code>, “${C(d.top[0])}” leads with ${f2(d.logits[d.top[0]])}. The lowest, “${C(d.top[26])}”, is ${f2(d.logits[d.top[26]])}.`, zh: `在 <code>.ma</code> 之後，「${C(d.top[0])}」以 ${f2(d.logits[d.top[0]])} 領先；最低的「${C(d.top[26])}」是 ${f2(d.logits[d.top[26]])}。` })),
      ],
      changed: [{ sign: '~', text: { en: 'Sorted: highest to lowest', zh: '排序：由高到低' } }],
    },
    {
      short: { en: 'Softmax', zh: 'Softmax' }, chapter: 'technique',
      title: { en: 'Softmax: make them positive, then share', zh: 'Softmax：先變正數，再分比例' },
      body: [
        { en: '<b>Softmax</b> does two things. First it raises e (about 2.718) to the power of each logit: e^logit. The result is always positive, and every +1 in the logit multiplies it by about 2.7.', zh: '<b>Softmax</b> 做兩件事。首先算出 e（約 2.718）的 logit 次方，也就是 e^logit。結果一定是正數，而且 logit 每多 1，結果就乘上約 2.7 倍。' },
        lz((d) => ({ en: `Then it divides each result by the total of all 27, here ${d.total.toFixed(1)}. Each token gets its share of that total.`, zh: `接著把每個結果除以 27 個結果的總和，這裡是 ${d.total.toFixed(1)}。每個詞元就分到總和中屬於自己的那一份。` })),
      ],
      changed: [{ sign: '+', text: { en: 'e^logit, then ÷ total', zh: 'e^logit，再 ÷ 總和' } }],
    },
    {
      short: { en: 'Probabilities', zh: '機率' }, chapter: 'solution',
      title: { en: 'Probabilities that add up to 100%', zh: '加起來等於 100% 的機率' },
      body: [
        lz((d) => ({ en: `Here are all 27 shares. “${C(d.top[0])}” gets ${pct(d.probs[d.top[0]])}, as in “mary” or “mark”. The shares add up to exactly 100%.`, zh: `這是 27 個比例。「${C(d.top[0])}」分到 ${pct(d.probs[d.top[0]])}，就像「mary」或「mark」。全部加起來剛好 100%。` })),
        lz((d) => ({ en: `Its logit is only ${(d.logits[d.top[0]] - d.logits[d.top[1]]).toFixed(2)} above “${C(d.top[1])}”, yet its share is ${(d.probs[d.top[0]] / d.probs[d.top[1]]).toFixed(1)} times bigger. Softmax rewards the leader a lot.`, zh: `它的 logit 只比「${C(d.top[1])}」高 ${(d.logits[d.top[0]] - d.logits[d.top[1]]).toFixed(2)}，比例卻大了 ${(d.probs[d.top[0]] / d.probs[d.top[1]]).toFixed(1)} 倍。Softmax 會大大獎勵領先者。` })),
      ],
      callout: { kind: 'insight', text: { en: 'A difference in logits becomes a ratio in probabilities: +1 in logit means about 2.7× the share.', zh: 'logit 的差距會變成機率的倍數：logit 多 1，比例大約變成 2.7 倍。' } },
      changed: [{ sign: '~', text: { en: 'Logits → probabilities', zh: 'Logits → 機率' } }],
    },
    {
      short: { en: 'Try it', zh: '動手試試' }, chapter: 'solution',
      title: { en: 'Drag one logit, watch all 27 shares', zh: '拖動一個 logit，看 27 個比例怎麼變' },
      body: [
        { en: 'Try it: pick a letter and drag the slider to add to its logit. All 27 bars are recomputed with the exact softmax. Watch the readout: every +1 multiplies that letter’s e^logit by about 2.7.', zh: '動手試試：選一個字母，拖動滑桿替它的 logit 加減。27 根長條都會用精確的 softmax 重新計算。看看上方的數字：logit 每加 1，那個字母的 e^logit 就乘上約 2.7 倍。' },
        { en: 'Notice where its gain comes from: the other letters shrink, so the bars still add up to 100%. Then raise the least likely letter by 6 and see how far behind the leader it still is.', zh: '注意它多出來的比例是從哪裡來的：其他字母都變小了，所以全部加起來仍是 100%。再把最不可能的字母加 6 看看，它離領先者還有多遠。' },
      ],
      changed: [{ sign: '+', text: { en: 'A slider on one logit', zh: '一個 logit 的滑桿' } }],
    },
    {
      short: { en: 'Every position', zh: '每個位置' }, chapter: 'solution',
      title: { en: 'A prediction at every position, at once', zh: '每個位置同時都在預測' },
      body: [
        { en: 'We followed one position, but the model does this for every position in a single pass: each row of the output is a full set of 27 probabilities for “what comes after here”.', zh: '我們只跟著一個位置，但模型一次計算就替每個位置都做了這件事：輸出的每一列，都是「這裡之後接什麼」的一整組 27 個機率。' },
        lz((d) => ({ en: `Reading <code>.mary</code>, the right column shows what really came next and the probability the model gave it. After <code>.mar</code> it gave “y” only ${pct1(d.rows[3][CHARS.indexOf('y')])}: many names continue differently.`, zh: `讀入 <code>.mary</code> 時，右邊那一欄是實際接下來的字母，以及模型給它的機率。在 <code>.mar</code> 之後，它只給「y」${pct1(d.rows[3][CHARS.indexOf('y')])}：很多名字會接別的字母。` })),
      ],
      callout: { kind: 'note', text: { en: 'This is why training is efficient: one name gives one prediction to check at every position. The Training Loop lesson turns these checks into a score called loss.', zh: '這就是訓練有效率的原因：一個名字在每個位置都提供一個可以檢查的預測。「訓練迴圈」那一課會把這些檢查變成一個叫做損失的分數。' } },
      changed: [{ sign: '+', text: { en: 'One row of probabilities per position', zh: '每個位置一列機率' } }],
    },
    {
      short: { en: 'No winner', zh: '沒有贏家' }, chapter: 'problem',
      title: { en: 'Case: no clear winner', zh: '案例：沒有明顯的贏家' },
      body: [
        lz((d) => {
          const o = caseTop(d).filter((x) => x.v >= 0.15);
          const list = o.map((x) => C(x.i)).join(', ');
          return { en: `After <code>.sam</code>, ${o.length} letters (${list}) have nearly equal logits, so each gets a similar share, between ${pct(o[o.length - 1].v)} and ${pct(o[0].v)}. No letter stands out.`, zh: `在 <code>.sam</code> 之後，有 ${o.length} 個字母（${o.map((x) => C(x.i)).join('、')}）的 logit 幾乎一樣，所以分到的比例也差不多，介於 ${pct(o[o.length - 1].v)} 到 ${pct(o[0].v)} 之間，沒有哪個特別突出。` };
        }),
        lz((d) => {
          const w = caseTop(d)[26];
          return { en: `That is not a failure: our list has “samir”, “samara”, “sammy” and “samuel”, so many letters really can follow. Softmax also never gives exactly 0%: even the least likely letter, “${C(w.i)}”, keeps ${tiny(w.v)}. (The bars round to whole percent, so tiny shares show as 0%.)`, zh: `這不是失敗：我們的清單裡有「samir」、「samara」、「sammy」和「samuel」，「sam」後面真的可以接很多種字母。Softmax 也從不給出剛好 0%：就算最不可能的「${C(w.i)}」，也保留了 ${tiny(w.v)}。（長條上的數字四捨五入到整數百分比，所以很小的比例會顯示成 0%。）` };
        }),
      ],
      changed: [{ sign: '~', text: { en: 'Prefix .ma → .sam', zh: '前綴 .ma → .sam' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Vector in, probabilities out', zh: '向量進去，機率出來' },
      body: [
        { en: 'The last position’s <b>final vector</b> is multiplied by the output matrix <code>u</code> (plus <code>bu</code>).<br>That gives one score per token: the <b>logits</b>.<br><b>Softmax</b> makes them positive and divides by the total, so they add up to 100%.<br>A bigger logit wins a much bigger share.<br>Every position gets its own prediction in the same pass.', zh: '最後一個位置的<b>最終向量</b>乘上輸出矩陣 <code>u</code>（再加上 <code>bu</code>）。<br>得到每個詞元一個分數：<b>logits</b>。<br><b>Softmax</b> 把它們變成正數再除以總和，讓它們加起來等於 100%。<br>logit 大一點，就能分到大很多的比例。<br>每個位置在同一次計算中都有自己的預測。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Self-Attention, Step by Step. How the final vector of “a” comes to know about the letters before it.', zh: '下一課：自注意力，一步一步來。「a」的最終向量，是怎麼得知它前面有哪些字母的。' } },
    },
  ],

  scene(step, input) {
    const d = data();
    const items: Item[] = [];
    const prefix = step === 7 ? CASE : NAME.slice(0, P + 1);
    const strip = 45; // x where the 27-cell logits strip starts
    const lmax = Math.max(...d.logits.map(Math.abs));

    if (step <= TRY || step === 7) {
      const chips = [...prefix];
      const x0 = step === 0 ? 300 : 40;
      chips.forEach((ch, i) => items.push({ k: 'tok' + i, t: 'token', x: x0 + i * 56, y: 24, text: ch, tone: i === chips.length - 1 ? 'a' : ch === '.' ? 'b' : 'ink' }));
      if (step !== 7) items.push({ k: 'q', t: 'token', x: x0 + chips.length * 56, y: 24, text: '?', tone: 'a' });
    }

    if (step <= 3) {
      const fx = step === 0 ? 242 : 300;
      items.push({ k: 'fa', t: 'arrow', x1: (step === 0 ? 300 : 40) + P * 56 + 22, y1: 74, x2: fx + 30, y2: 112, tone: 'a' });
      items.push({ k: 'final', t: 'vec', x: fx, y: 130, v: d.final, cell: step === 0 ? 26 : 19, dim: step >= 2, label: t({ en: 'final vector of the last position (16 numbers)', zh: '最後一個位置的最終向量（16 個數字）' }) });
    }
    if (step === 0) {
      items.push({ k: 'need', t: 'text', x: 450, y: 260, anchor: 'middle', size: 30, tone: 'muted', text: '↓ ?' });
      items.push({ k: 'needt', t: 'text', x: 450, y: 320, anchor: 'middle', size: 17, text: t({ en: 'wanted: 27 probabilities, one per token, adding up to 100%', zh: '想要的：27 個機率，每個詞元一個，加起來 100%' }) });
      items.push({ k: 'needs', t: 'text', x: 450, y: 360, anchor: 'middle', size: 15, tone: 'muted', text: '. a b c d e f g h i j k l m n o p q r s t u v w x y z' });
    }

    if (step >= 1 && step <= 3) {
      items.push({ k: 'logits', t: 'vec', x: strip, y: 250, v: d.logits, cell: 30, max: lmax, label: t({ en: 'logits: one score per token', zh: 'logits：每個詞元一個分數' }) });
      [...CHARS].forEach((ch, i) => items.push({ k: 'll' + i, t: 'text', x: strip + i * 30 + 14, y: 296, text: ch, size: 14, anchor: 'middle', mono: true, tone: i === d.top[0] ? 'a' : 'muted', bold: i === d.top[0] }));
    }
    if (step >= 1 && step <= 3) items.push({ k: 'mul', t: 'arrow', x1: 450, y1: 166, x2: 450, y2: 226, tone: step === 1 ? 'a' : 'muted' });
    if (step === 1) {
      items.push({ k: 'mult', t: 'text', x: 466, y: 202, size: 15, tone: 'a', text: t({ en: '× u (16 × 27)  + bu', zh: '× u（16 × 27）＋ bu' }) });
      items.push({ k: 'fs', t: 'vec', x: 330, y: 330, v: d.final, cell: 16 });
      items.push({ k: 'fsl', t: 'text', x: 318, y: 346, anchor: 'end', size: 14, tone: 'muted', text: t({ en: 'final vector', zh: '最終向量' }) });
      items.push({ k: 'uc', t: 'vec', x: 330, y: 366, v: d.ucol, cell: 16 });
      items.push({ k: 'ucl', t: 'text', x: 318, y: 382, anchor: 'end', size: 14, tone: 'a', text: t({ en: `column of u for ${q(C(d.top[0]))}`, zh: `u 裡${q(C(d.top[0]))}的那一行` }) });
      items.push({ k: 'ucx', t: 'text', x: 600, y: 366, size: 14, tone: 'muted', text: t({ en: '× cell by cell, then add up', zh: '逐格相乘，再加總' }) });
      items.push({ k: 'how', t: 'text', x: 450, y: 432, anchor: 'middle', size: 16, mono: true, text: `${q(C(d.top[0]))}: ${f2(d.dot)} + ${f2(d.bias)} = ${f2(d.logits[d.top[0]])}` });
      items.push({ k: 'hown', t: 'text', x: 450, y: 460, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: '(final vector · column of u) + bu', zh: '（最終向量 · u 的那一行）＋ bu' }) });
    }
    if (step === 2) {
      const show = [...d.top.slice(0, 5), d.top[25], d.top[26]];
      show.forEach((id, r) => {
        const y = 350 + (r < 5 ? r : r + 0.6) * 22;
        items.push({ k: 'rk' + r, t: 'text', x: 400, y, text: `${C(id)}`, size: 15, mono: true, bold: r === 0, tone: r === 0 ? 'a' : 'ink', anchor: 'end' });
        items.push({ k: 'rv' + r, t: 'text', x: 480, y, text: f2(d.logits[id]), size: 15, mono: true, bold: r === 0, tone: r === 0 ? 'a' : r >= 5 ? 'muted' : 'ink', anchor: 'end' });
      });
      items.push({ k: 'rdots', t: 'text', x: 440, y: 350 + 5 * 22, anchor: 'middle', size: 13, tone: 'muted', text: '⋮' });
    }
    if (step === 3) {
      const heads = [{ en: 'token', zh: '詞元' }, { en: 'logit', zh: 'logit' }, { en: 'e^logit', zh: 'e^logit' }, { en: '÷ total', zh: '÷ 總和' }];
      const cols = [260, 380, 510, 650];
      heads.forEach((h, i) => items.push({ k: 'sh' + i, t: 'text', x: cols[i], y: 342, anchor: 'end', size: 13, tone: 'muted', text: t(h) }));
      d.top.slice(0, 4).forEach((id, r) => {
        const y = 370 + r * 24, tone = r === 0 ? 'a' as const : 'ink' as const;
        items.push({ k: 'sa' + r, t: 'text', x: cols[0], y, anchor: 'end', size: 15, mono: true, tone, text: C(id) });
        items.push({ k: 'sb' + r, t: 'text', x: cols[1], y, anchor: 'end', size: 15, mono: true, tone, text: f2(d.logits[id]) });
        items.push({ k: 'sc' + r, t: 'text', x: cols[2], y, anchor: 'end', size: 15, mono: true, tone, text: Math.exp(d.logits[id]).toFixed(1) });
        items.push({ k: 'sd' + r, t: 'text', x: cols[3], y, anchor: 'end', size: 15, mono: true, tone, bold: r === 0, text: pct1(d.probs[id]) });
      });
      items.push({ k: 'stot', t: 'text', x: cols[2], y: 370 + 4 * 24 + 6, anchor: 'end', size: 14, mono: true, tone: 'muted', text: t({ en: `total of all 27: ${d.total.toFixed(1)}`, zh: `27 個的總和：${d.total.toFixed(1)}` }) });
    }

    let controls: Control[] | undefined;
    let tryVal = '', tryTop = '';
    if (step === TRY) {
      controls = tryControls(d);
      const id = Math.max(0, Math.min(26, Math.round(Number(input?.letter ?? d.top[0])) || 0));
      const by = Number(input?.nudge ?? 0) || 0;
      const pr = nudged(d, id, by);
      const best = pr.indexOf(Math.max(...pr));
      items.push({
        k: 'bars', t: 'bars', x: 40, y: 150, w: 820, h: 260,
        items: pr.map((v, i) => ({ label: CHARS[i], v, hi: i === id })),
        title: t({ en: `after “${prefix}”, with the logit of “${C(id)}” changed by ${sgn(by)}`, zh: `「${prefix}」之後，「${C(id)}」的 logit 改變 ${sgn(by)}` }),
      });
      const was = d.probs[id], now = pr[id];
      items.push({ k: 'r1', t: 'text', x: 450, y: 88, anchor: 'middle', size: 16, mono: true, bold: true, tone: 'a', text: t({ en: `“${C(id)}”: logit ${f2(d.logits[id])} → ${f2(d.logits[id] + by)} · share ${tiny(was)} → ${tiny(now)}`, zh: `「${C(id)}」：logit ${f2(d.logits[id])} → ${f2(d.logits[id] + by)} · 比例 ${tiny(was)} → ${tiny(now)}` }) });
      items.push({ k: 'r2', t: 'text', x: 450, y: 112, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: `all other letters together: ${pct1(1 - was)} → ${pct1(1 - now)} · e^logit × ${mult(by)}`, zh: `其他字母合計：${pct1(1 - was)} → ${pct1(1 - now)} · e^logit × ${mult(by)}` }) });
      items.push({ k: 'sum', t: 'text', x: 450, y: 460, anchor: 'middle', size: 16, tone: 'muted', text: t({ en: `all 27 bars still add up to 100% · top now: “${C(best)}” ${pct(pr[best])}`, zh: `27 根長條加起來仍是 100% · 現在第一名：「${C(best)}」${pct(pr[best])}` }) });
      tryVal = `${q(C(id))} ${tiny(now)}`;
      tryTop = `${q(C(best))} ${pct(pr[best])}`;
    }

    if (step === 4 || step === 7) {
      const pr = step === 4 ? d.probs : d.caseProbs;
      const best = pr.indexOf(Math.max(...pr));
      items.push({
        k: 'bars', t: 'bars', x: 40, y: 150, w: 820, h: 260,
        items: pr.map((v, i) => ({ label: CHARS[i], v, hi: step === 4 ? i === best : v >= 0.15 })),
        title: t({ en: `probability of each next token after “${prefix}” (softmax of the logits)`, zh: `「${prefix}」之後每個下一個詞元的機率（logits 經過 softmax）` }),
      });
      items.push({ k: 'sum', t: 'text', x: 450, y: 460, anchor: 'middle', size: 16, tone: 'muted', text: step === 7
        ? t({ en: 'bars are rounded to whole percent: “0%” means under 0.5%, never exactly 0', zh: '長條上的數字四捨五入到整數：「0%」代表不到 0.5%，不是剛好 0' })
        : t({ en: 'all 27 bars add up to 100%', zh: '27 根長條加起來是 100%' }) });
    }

    if (step === 6) {
      const ids = encode(NAME + '.');
      const cols = gridCols(d);
      const m = d.rows.map((p) => cols.map((j) => p[j]));
      const cell = 46, gx = 150, gy = 70;
      items.push({ k: 'grid', t: 'heat', x: gx, y: gy, cell, m, rows: d.rows.map((_, i) => NAME.slice(0, i + 1)), cols: cols.map(C), fmt: 'pct', hi: P, max: 0.6 });
      items.push({ k: 'gh', t: 'text', x: gx + cols.length * cell + 30, y: gy - 10, size: 13, tone: 'muted', text: t({ en: 'really next', zh: '實際下一個' }) });
      d.rows.forEach((p, i) => items.push({ k: 'gt' + i, t: 'text', x: gx + cols.length * cell + 30, y: gy + i * cell + cell / 2 + 5, size: 15, mono: true, tone: 'a', bold: true, text: `${q(C(ids[i + 1]))} ${pct1(p[ids[i + 1]])}` }));
      items.push({ k: 'gl', t: 'text', x: gx - 12, y: gy + d.rows.length * cell + 30, anchor: 'end', size: 13, tone: 'muted', text: t({ en: '↑ text read so far', zh: '↑ 目前讀到的文字' }) });
      items.push({ k: 'gn', t: 'text', x: gx + cols.length * cell, y: gy + d.rows.length * cell + 30, anchor: 'end', size: 13, tone: 'muted', text: t({ en: `showing ${cols.length} of the 27 columns`, zh: `只列出 27 行中的 ${cols.length} 行` }) });
    }

    if (step === 8) {
      const boxes = [
        { en: 'final vector', zh: '最終向量' }, { en: '× u + bu', zh: '× u ＋ bu' }, { en: '27 logits', zh: '27 個 logits' },
        { en: 'softmax', zh: 'softmax' }, { en: '27 probabilities', zh: '27 個機率' },
      ];
      boxes.forEach((b, i) => {
        items.push({ k: 'rb' + i, t: 'box', x: 30 + i * 172, y: 200, w: 150, h: 60, label: t(b), tone: i === 4 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'ra' + i, t: 'arrow', x1: 184 + i * 172, y1: 230, x2: 198 + i * 172, y2: 230 });
      });
      items.push({ k: 'rs', t: 'text', x: 450, y: 330, anchor: 'middle', size: 16, tone: 'a', bold: true, text: t({ en: `after “.ma”: “${C(d.top[0])}” ${pct(d.probs[d.top[0]])}`, zh: `「.ma」之後：「${C(d.top[0])}」${pct(d.probs[d.top[0]])}` }) });
    }

    const showTop = step >= 4 && step !== 7;
    const ct = caseTop(d)[0];
    return {
      metrics: [
        { label: { en: 'Text read', zh: '讀入的文字' }, value: step === 6 ? NAME : prefix },
        { label: { en: 'Scores', zh: '分數' }, value: step >= 1 ? '27' : '—' },
        { label: { en: 'Top token', zh: '最可能的' }, value: step === TRY ? tryTop : showTop ? `${q(C(d.top[0]))} ${pct(d.probs[d.top[0]])}` : step >= 2 && step <= 3 ? `${q(C(d.top[0]))} ${f2(d.logits[d.top[0]])}` : step === 7 ? `${q(C(ct.i))} ${pct(ct.v)}` : '—' },
        ...(step === TRY ? [{ label: { en: 'Your letter now', zh: '你的字母現在' }, value: tryVal }] : []),
        { label: { en: 'Sum', zh: '總和' }, value: step >= 4 ? '100%' : '—' },
      ],
      items,
      controls,
      legend: step >= 1 && step <= 3
        ? [{ tone: 'pos', text: { en: 'positive score', zh: '正的分數' } }, { tone: 'neg', text: { en: 'negative score', zh: '負的分數' } }]
        : step === 6 ? [{ tone: 'heat', text: { en: 'probability', zh: '機率' } }] : undefined,
    };
  },
};
