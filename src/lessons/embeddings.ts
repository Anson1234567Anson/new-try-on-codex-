import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, DEFAULT_CONFIG, type Weights } from '../model/gpt';
import { CHARS, rng } from '../model/data';
import weights from '../model/weights.json';

// The embedding table `tok` (27 rows × 16 numbers), read from the trained model
// and from a fresh, untrained one. Similarity and the 2-D map are computed here.

const VOWELS = 'aeiou';
const isVowel = (i: number) => VOWELS.includes(CHARS[i]);
const E = CHARS.indexOf('e');

const cos = (a: number[], b: number[]) => {
  let d = 0, x = 0, y = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; }
  return d / Math.sqrt(x * y);
};

/** Top-2 principal components of the rows (each scaled to length 1 first), by power iteration. */
function pca2(rows: number[][]): { pts: [number, number][]; share: number } {
  const n = rows.length, dim = rows[0].length;
  const unit = rows.map((r) => { const l = Math.hypot(...r); return r.map((v) => v / l); });
  const mean = new Array(dim).fill(0);
  unit.forEach((r) => r.forEach((v, k) => { mean[k] += v / n; }));
  const X = unit.map((r) => r.map((v, k) => v - mean[k]));
  const C = Array.from({ length: dim }, (_, a) => Array.from({ length: dim }, (_, b) => X.reduce((s, r) => s + r[a] * r[b], 0)));
  const total = C.reduce((s, r, i) => s + r[i], 0);
  const pcs: number[][] = [];
  let share = 0;
  for (let p = 0; p < 2; p++) {
    let v = Array.from({ length: dim }, (_, i) => Math.sin(i + 1 + p));
    for (let it = 0; it < 300; it++) {
      let nv = C.map((row) => row.reduce((s, c, j) => s + c * v[j], 0));
      for (const q of pcs) { const d = nv.reduce((s, x, j) => s + x * q[j], 0); nv = nv.map((x, j) => x - d * q[j]); }
      const l = Math.hypot(...nv);
      v = nv.map((x) => x / l);
    }
    share += v.reduce((s, x, i) => s + x * C[i].reduce((u, c, j) => u + c * v[j], 0), 0) / total;
    pcs.push(v);
  }
  // point the first direction so that the vowels sit on the right, for easy comparison
  const pts = X.map((r) => pcs.map((q) => r.reduce((s, x, j) => s + x * q[j], 0)) as [number, number]);
  const vowelSide = pts.filter((_, i) => isVowel(i)).reduce((s, p) => s + p[0], 0);
  if (vowelSide < 0) pts.forEach((p) => { p[0] = -p[0]; });
  return { pts, share };
}

/** Place chips on the map, then push apart any that overlap. */
function layout(pts: [number, number][], x0: number, y0: number, w: number, h: number): { x: number; y: number }[] {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [ax, bx, ay, by] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pos = pts.map((p) => ({ x: x0 + ((p[0] - ax) / (bx - ax)) * (w - 44), y: y0 + (1 - (p[1] - ay) / (by - ay)) * (h - 44) }));
  for (let it = 0; it < 80; it++)
    for (let i = 0; i < pos.length; i++)
      for (let j = i + 1; j < pos.length; j++) {
        const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
        if (Math.abs(dx) < 48 && Math.abs(dy) < 48) {
          const push = (48 - Math.max(Math.abs(dx), Math.abs(dy))) / 2 + 0.5;
          if (Math.abs(dx) > Math.abs(dy)) { const s = dx >= 0 ? 1 : -1; pos[i].x -= s * push; pos[j].x += s * push; }
          else { const s = dy >= 0 ? 1 : -1; pos[i].y -= s * push; pos[j].y += s * push; }
        }
      }
  return pos.map((p) => ({ x: Math.min(x0 + w - 44, Math.max(x0, p.x)), y: Math.min(y0 + h - 44, Math.max(y0, p.y)) }));
}

interface Side { rows: number[][]; vv: number; vc: number; cc: number; map: { x: number; y: number }[]; share: number; rightSix: string }
interface Data { trained: Side; fresh: Side; change: number; pairs: { a: string; b: string; v: number }[] }

function side(model: GPT): Side {
  const rows = [...CHARS].map((_, i) => model.p.tok.row(i));
  const letters = [...CHARS].map((_, i) => i).filter((i) => i > 0);
  const V = letters.filter(isVowel), C = letters.filter((i) => !isVowel(i));
  const avg = (A: number[], B: number[]) => { let s = 0, n = 0; for (const i of A) for (const j of B) if (i !== j) { s += cos(rows[i], rows[j]); n++; } return s / n; };
  const { pts, share } = pca2(rows);
  const rightSix = pts.map((p, i) => ({ p: p[0], i })).sort((a, b) => b.p - a.p).slice(0, 6).map((x) => CHARS[x.i]).join(' ');
  return { rows, vv: avg(V, V), vc: avg(V, C), cc: avg(C, C), map: layout(pts, 40, 40, 820, 410), share, rightSix };
}

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const trained = side(GPT.fromJSON(weights as Weights));
  const fresh = side(new GPT(DEFAULT_CONFIG, rng(1)));
  const a = trained.rows[E], b = fresh.rows[E];
  const change = a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;
  const pair = (x: string, y: string) => ({ a: x, b: y, v: cos(trained.rows[CHARS.indexOf(x)], trained.rows[CHARS.indexOf(y)]) });
  return { trained, fresh, change, pairs: [pair('a', 'o'), pair('a', 'v')] };
})();

/** Text whose numbers are only computed when it is first shown. */
const lz = (f: (d: Data) => L): L => ({ get en() { return f(data()).en; }, get zh() { return f(data()).zh; } });
const f2 = (v: number) => (v >= 0 ? '' : '−') + Math.abs(v).toFixed(2);

const TABLE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 26]; // rows of the table we draw (the rest are skipped with ⋮)
const tableY = (r: number) => 70 + (r < 9 ? r : 10) * 40;
const GRID = 'aeioubdmnrst';

export const embeddings: Lesson = {
  id: 'embeddings',
  title: { en: 'Embeddings', zh: '嵌入' },
  blurb: {
    en: 'How an id becomes a list of numbers the model can work with, and what training does to those numbers.',
    zh: '一個編號如何變成模型能運算的一串數字，以及訓練對這些數字做了什麼。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'Why an id alone is not enough', zh: '為什麼光有編號是不夠的' }, step: 0 },
    { text: { en: 'What an embedding table is and how lookup works', zh: '嵌入表是什麼，查表怎麼運作' }, step: 2 },
    { text: { en: 'That embeddings start random and are learned', zh: '嵌入從隨機開始，靠學習得來' }, step: 3 },
    { text: { en: 'How to measure whether two tokens are similar', zh: '怎麼衡量兩個詞元像不像' }, step: 4 },
    { text: { en: 'What the trained table actually learned', zh: '訓練好的表實際學到了什麼' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'An id is only a label', zh: '編號只是一個標籤' },
      body: [
        { en: 'The Tokens lesson turned each letter into an id: “a” is 1, “e” is 5, “m” is 13. But these numbers are just positions in an alphabetical list.', zh: '在「詞元」那一課，每個字母都變成了一個編號：「a」是 1，「e」是 5，「m」是 13。但這些數字只是在字母順序清單裡的位置。' },
        { en: 'By id, “m” (13) sits right next to “n” (14) and far from “a” (1). That says nothing about how letters behave in names. Doing arithmetic on ids would be meaningless.', zh: '照編號來看，「m」（13）緊挨著「n」（14），離「a」（1）很遠。但這跟字母在名字裡的用法毫無關係，拿編號直接做算術是沒有意義的。' },
      ],
      callout: { kind: 'insight', text: { en: 'The model needs a description of each token that it can compute with, and that it can improve as it learns.', zh: '模型需要一種能拿來運算、而且能在學習中慢慢改進的方式來描述每個詞元。' } },
    },
    {
      short: { en: 'The table', zh: '嵌入表' }, chapter: 'technique',
      title: { en: 'One row of numbers per token', zh: '每個詞元一列數字' },
      body: [
        { en: 'The model keeps a table with one row for each of the 27 tokens. Each row holds 16 numbers. In the code it is called <code>tok</code>: 27 × 16 = 432 numbers.', zh: '模型保存一張表，27 個詞元各佔一列，每列有 16 個數字。在程式裡它叫做 <code>tok</code>：27 × 16 = 432 個數字。' },
        { en: 'These are the trained model’s real rows. Orange cells are positive, blue negative; a stronger colour means a bigger number.', zh: '這些是訓練好的模型裡真實的數值。橘色格子是正數，藍色是負數，顏色越深代表數值越大。' },
      ],
      changed: [{ sign: '+', text: { en: 'Table tok: 27 rows × 16 numbers', zh: '表 tok：27 列 × 16 個數字' } }],
    },
    {
      short: { en: 'Lookup', zh: '查表' }, chapter: 'technique',
      title: { en: 'The id picks a row', zh: '編號選出一列' },
      body: [
        { en: 'To read “e”, the model takes its id, 5, and copies row 5 out of the table. Those 16 numbers are the token’s <b>embedding</b>: its description inside the model.', zh: '要讀入「e」，模型拿它的編號 5，把表中第 5 列複製出來。這 16 個數字就是這個詞元的<b>嵌入（embedding）</b>，也就是它在模型內部的描述。' },
        { en: 'That is all an embedding is: a lookup, no arithmetic. The id finally does a useful job, as a row number.', zh: '嵌入就只是這樣：查表，沒有任何計算。編號終於派上用場了：當作列號。' },
      ],
      callout: { kind: 'note', text: { en: 'Every “e” in every name gets the same 16 numbers. Where the letter sits is added later, in the Positional Information lesson.', zh: '每個名字裡的每個「e」都拿到同樣的 16 個數字。字母在第幾個位置，會在「位置資訊」那一課再加進來。' } },
      changed: [{ sign: '+', text: { en: 'Embedding of “e” = row 5', zh: '「e」的嵌入 = 第 5 列' } }],
    },
    {
      short: { en: 'Learned', zh: '學出來的' }, chapter: 'technique',
      title: { en: 'Nobody fills in the table', zh: '沒有人去填這張表' },
      body: [
        { en: 'Before training, every number in the table is random. The upper strip is row “e” of a fresh, untrained model; the lower strip is the same row after training.', zh: '訓練之前，表裡的每個數字都是隨機的。上面那條是一個全新、未訓練模型的「e」列，下面那條是訓練之後的同一列。' },
        lz((d) => ({ en: `Training nudged every number, by ${d.change.toFixed(2)} on average. The table is part of the model’s weights, and it is learned the same way as all the others.`, zh: `訓練把每個數字都推動了一些，平均移動 ${d.change.toFixed(2)}。這張表是模型權重的一部分，學習的方式和其他權重完全一樣。` })),
      ],
      changed: [{ sign: '~', text: { en: 'Random row → trained row', zh: '隨機的列 → 訓練後的列' } }],
    },
    {
      short: { en: 'Similarity', zh: '相似度' }, chapter: 'technique',
      title: { en: 'Do two tokens point the same way?', zh: '兩個詞元指向同一個方向嗎？' },
      body: [
        { en: 'Think of each row as an arrow in 16 directions. <b>Cosine similarity</b> asks how much two arrows point the same way: +1 means the same direction, 0 unrelated, −1 opposite.', zh: '把每一列想成指向 16 個方向的箭頭。<b>餘弦相似度（cosine similarity）</b>問的是兩個箭頭有多同向：+1 代表方向相同，0 代表無關，−1 代表方向相反。' },
        lz((d) => ({ en: `In the trained table, “a” and “o” score ${f2(d.pairs[0].v)}. “a” and “v” score ${f2(d.pairs[1].v)}: they point in nearly opposite directions.`, zh: `在訓練好的表裡，「a」和「o」的分數是 ${f2(d.pairs[0].v)}；「a」和「v」是 ${f2(d.pairs[1].v)}，幾乎指向相反的方向。` })),
      ],
      changed: [{ sign: '+', text: { en: 'A way to compare two embeddings', zh: '比較兩個嵌入的方法' } }],
    },
    {
      short: { en: 'Vowels', zh: '母音' }, chapter: 'solution',
      title: { en: 'Vowels ended up close together', zh: '母音最後聚在一起' },
      body: [
        lz((d) => ({ en: `The grid compares 12 letters. The top-left block, vowel with vowel, is mostly orange: on average ${f2(d.trained.vv)}. Vowel with consonant averages ${f2(d.trained.vc)}.`, zh: `這張表比較了 12 個字母。左上角「母音對母音」的區塊大多是橘色，平均 ${f2(d.trained.vv)}；「母音對子音」平均 ${f2(d.trained.vc)}。` })),
        { en: 'Nobody told the model what a vowel is. Vowels play similar roles in names (they sit between consonants), so being pushed to predict well gave them similar rows.', zh: '沒有人告訴模型什麼是母音。母音在名字裡扮演相似的角色（夾在子音之間），所以模型被推著把預測做好時，就讓它們的列變得相似。' },
      ],
      callout: { kind: 'insight', text: { en: 'Tokens that behave alike end up with embeddings that point alike. In big models the same happens with words and meanings.', zh: '用法相似的詞元，最後會有方向相似的嵌入。在大型模型裡，單字和意義也會發生同樣的事。' } },
      changed: [{ sign: '+', text: { en: '12 × 12 grid of similarities', zh: '12 × 12 的相似度表' } }],
    },
    {
      short: { en: 'The map', zh: '地圖' }, chapter: 'solution',
      title: { en: 'A map of all 27 tokens', zh: '27 個詞元的地圖' },
      body: [
        lz((d) => ({ en: `We cannot draw 16 directions, so we squash them to the 2 that spread the tokens out the most (a method called PCA). Those 2 keep ${Math.round(d.trained.share * 100)}% of the spread, so this is a rough view.`, zh: `16 個方向沒辦法畫出來，所以我們把它壓成最能把詞元分開的 2 個方向（這種方法叫 PCA）。這 2 個方向只保留了 ${Math.round(d.trained.share * 100)}% 的差異，所以只是個粗略的樣貌。` })),
        lz((d) => ({ en: `Along the left–right direction, the six tokens furthest right are ${d.trained.rightSix}: the five vowels plus “y”, which often acts as a vowel in names. The consonants spread into two loose bunches; we found no simple rule behind them, so we claim nothing more.`, zh: `沿著左右方向，最右邊的六個詞元是 ${d.trained.rightSix}：五個母音，加上在名字裡常當母音用的「y」。子音則鬆散地分成兩堆，我們找不到簡單的規則來解釋，所以不多做推論。` })),
      ],
      callout: { kind: 'note', text: { en: 'Chips are nudged a little so they do not overlap; their rough place is what matters.', zh: '為了不重疊，字塊的位置稍微挪動過，重點是大致的位置。' } },
      changed: [{ sign: '~', text: { en: '16 numbers → 2 map directions', zh: '16 個數字 → 地圖上的 2 個方向' } }],
    },
    {
      short: { en: 'Untrained', zh: '未訓練' }, chapter: 'problem',
      title: { en: 'Case: the same map before training', zh: '案例：訓練之前的同一張地圖' },
      body: [
        lz((d) => ({ en: `Here is the untrained table, mapped the same way. The vowels no longer form one group: some sit among the consonants. Vowel-with-vowel similarity is only ${f2(d.fresh.vv)}, against ${f2(d.trained.vv)} after training.`, zh: `這是未訓練的表，用同樣的方法畫出來。母音不再聚成一群，有些混在子音之間。母音對母音的相似度只有 ${f2(d.fresh.vv)}，訓練後則是 ${f2(d.trained.vv)}。` })),
        { en: 'Random rows can be close by chance (here “e” and “o” happen to start close), but nothing about how letters behave is in them yet. The structure on the previous map was learned.', zh: '隨機的列偶爾也會碰巧很接近（這裡「e」和「o」一開始就碰巧靠得很近），但裡面還沒有任何關於字母用法的資訊。上一張地圖的結構，是學出來的。' },
      ],
      changed: [{ sign: '~', text: { en: 'Trained table → untrained table', zh: '訓練後的表 → 未訓練的表' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'An id becomes a learned description', zh: '編號變成學出來的描述' },
      body: [
        { en: 'The <b>embedding table</b> has one row of 16 numbers per token.<br>The id picks the row: that row is the token’s <b>embedding</b>.<br>The rows start random and training moves them.<br>Tokens that behave alike, like vowels, end up pointing alike.', zh: '<b>嵌入表</b>為每個詞元存一列 16 個數字。<br>編號選出那一列，那一列就是詞元的<b>嵌入</b>。<br>這些列一開始是隨機的，由訓練推動。<br>用法相似的詞元（例如母音），最後會指向相似的方向。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Predicting the Next Token. How the model turns vectors back into a probability for every letter.', zh: '下一課：預測下一個詞元。模型如何把向量變回每個字母的機率。' } },
    },
  ],

  scene(step) {
    const d = data();
    const items: Item[] = [];
    const trained = d.trained.rows;
    const maxAbs = 0.8; // a few rare letters have much larger numbers; colours saturate above this

    if (step === 0) {
      items.push({ k: 'line', t: 'arrow', x1: 40, y1: 300, x2: 870, y2: 300, tone: 'muted' });
      [...CHARS].forEach((ch, i) => items.push({ k: 'n' + i, t: 'text', x: 60 + i * 30, y: 330, text: ch, size: 15, anchor: 'middle', mono: true, tone: 'muted' }));
      [...CHARS].forEach((_, i) => { if (i % 5 === 0 || i === 26) items.push({ k: 'ni' + i, t: 'text', x: 60 + i * 30, y: 356, text: String(i), size: 13, anchor: 'middle', tone: 'muted' }); });
      for (const ch of 'aemn') {
        const i = CHARS.indexOf(ch);
        items.push({ k: 'chip' + ch, t: 'token', x: 60 + i * 30 - 22, y: ch === 'n' ? 150 : 210, text: ch, sub: String(i), tone: ch === 'a' || ch === 'e' ? 'a' : 'ink' });
      }
      items.push({ k: 'q0', t: 'text', x: 450, y: 120, anchor: 'middle', size: 17, text: t({ en: 'by id, “m” is next to “n” and far from “a”. So what?', zh: '照編號，「m」緊挨著「n」、離「a」很遠。那又怎樣？' }) });
      items.push({ k: 'q1', t: 'text', x: 450, y: 420, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: 'the order of the alphabet says nothing about how letters are used', zh: '字母表的順序，跟字母怎麼被使用毫無關係' }) });
    }

    if (step >= 1 && step <= 3) {
      items.push({ k: 'th', t: 'text', x: 130, y: 44, text: t({ en: 'table tok (trained)', zh: '表 tok（訓練後）' }), size: 14, tone: 'muted' });
      TABLE.forEach((id, r) => {
        const y = tableY(r);
        const on = step === 1 || id === E;
        items.push({ k: 'rl' + id, t: 'text', x: 118, y: y + 16, text: `${CHARS[id]}  ${id}`, anchor: 'end', size: 14, mono: true, tone: id === E && step >= 2 ? 'a' : 'ink' });
        items.push({ k: 'r' + id, t: 'vec', x: 130, y, v: trained[id], cell: 16, max: maxAbs, dim: !on });
      });
      items.push({ k: 'dots', t: 'text', x: 254, y: tableY(9) - 16, anchor: 'middle', size: 16, tone: 'muted', text: t({ en: '⋮  17 more rows', zh: '⋮  還有 17 列' }) });
    }

    if (step === 2) {
      items.push({ k: 'chipE', t: 'token', x: 600, y: 60, text: 'e', sub: t({ en: 'id 5', zh: '編號 5' }), tone: 'a' });
      items.push({ k: 'look', t: 'arrow', x1: 594, y1: 82, x2: 396, y2: tableY(5) + 11, tone: 'a', dashed: true });
      items.push({ k: 'copy', t: 'arrow', x1: 396, y1: tableY(5) + 30, x2: 520, y2: 318, tone: 'a' });
      items.push({ k: 'big', t: 'vec', x: 440, y: 345, v: trained[E], cell: 26, max: maxAbs, label: t({ en: 'embedding of “e”: 16 numbers', zh: '「e」的嵌入：16 個數字' }) });
      items.push({ k: 'vals', t: 'text', x: 440, y: 400, size: 13, mono: true, tone: 'muted', text: trained[E].slice(0, 6).map(f2).join('  ') + '  …' });
    }

    if (step === 3) {
      items.push({ k: 'old', t: 'vec', x: 440, y: 180, v: d.fresh.rows[E], cell: 26, max: maxAbs, label: t({ en: '“e” before training (random)', zh: '訓練前的「e」（隨機）' }) });
      items.push({ k: 'train', t: 'arrow', x1: 648, y1: 214, x2: 648, y2: 318, tone: 'a' });
      items.push({ k: 'trl', t: 'text', x: 660, y: 262, size: 14, tone: 'a', text: t({ en: 'training', zh: '訓練' }) });
      items.push({ k: 'big', t: 'vec', x: 440, y: 345, v: trained[E], cell: 26, max: maxAbs, label: t({ en: '“e” after training', zh: '訓練後的「e」' }) });
      items.push({ k: 'vals', t: 'text', x: 440, y: 400, size: 13, mono: true, tone: 'muted', text: t({ en: `average change per number: ${d.change.toFixed(2)}`, zh: `每個數字平均改變：${d.change.toFixed(2)}` }) });
    }

    if (step === 4) {
      const show = ['a', 'o', 'v'];
      show.forEach((ch, i) => {
        const id = CHARS.indexOf(ch);
        items.push({ k: 'cl' + ch, t: 'token', x: 120, y: 90 + i * 110, text: ch, tone: ch === 'v' ? 'ink' : 'a' });
        items.push({ k: 'cv' + ch, t: 'vec', x: 190, y: 101 + i * 110, v: trained[id], cell: 24, max: maxAbs });
      });
      items.push({ k: 'br1', t: 'arrow', x1: 600, y1: 112, x2: 600, y2: 200, tone: 'a' });
      items.push({ k: 'cs1', t: 'text', x: 650, y: 162, size: 20, bold: true, mono: true, tone: 'a', text: `a · o  ${f2(d.pairs[0].v)}` });
      items.push({ k: 'br2', t: 'arrow', x1: 630, y1: 112, x2: 630, y2: 310, tone: 'bad', dashed: true });
      items.push({ k: 'cs2', t: 'text', x: 650, y: 300, size: 20, bold: true, mono: true, tone: 'bad', text: `a · v  ${f2(d.pairs[1].v)}` });
      items.push({ k: 'cn', t: 'text', x: 450, y: 450, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: 'cosine similarity: +1 same direction · 0 unrelated · −1 opposite', zh: '餘弦相似度：+1 同方向 · 0 無關 · −1 相反' }) });
    }

    if (step === 5) {
      const ids = [...GRID].map((c) => CHARS.indexOf(c));
      const m = ids.map((i) => ids.map((j) => cos(trained[i], trained[j])));
      items.push({ k: 'grid', t: 'heat', x: 250, y: 64, cell: 36, m, rows: [...GRID], cols: [...GRID], fmt: 'num', max: 1 });
      items.push({ k: 'gl1', t: 'text', x: 214, y: 160, anchor: 'end', size: 14, tone: 'a', text: t({ en: 'vowels', zh: '母音' }) });
      items.push({ k: 'gl2', t: 'text', x: 214, y: 360, anchor: 'end', size: 14, tone: 'muted', text: t({ en: 'consonants', zh: '子音' }) });
      items.push({ k: 'gs1', t: 'text', x: 795, y: 150, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'vowel · vowel', zh: '母音 · 母音' }) });
      items.push({ k: 'gs1v', t: 'text', x: 795, y: 180, anchor: 'middle', size: 22, bold: true, mono: true, tone: 'a', text: f2(d.trained.vv) });
      items.push({ k: 'gs2', t: 'text', x: 795, y: 250, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'vowel · consonant', zh: '母音 · 子音' }) });
      items.push({ k: 'gs2v', t: 'text', x: 795, y: 280, anchor: 'middle', size: 22, bold: true, mono: true, text: f2(d.trained.vc) });
      items.push({ k: 'gs3', t: 'text', x: 795, y: 350, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'consonant · consonant', zh: '子音 · 子音' }) });
      items.push({ k: 'gs3v', t: 'text', x: 795, y: 380, anchor: 'middle', size: 22, bold: true, mono: true, text: f2(d.trained.cc) });
    }

    if (step === 6 || step === 7) {
      const s = step === 6 ? d.trained : d.fresh;
      s.map.forEach((p, i) => items.push({ k: 'm' + i, t: 'token', x: p.x, y: p.y, text: CHARS[i], tone: isVowel(i) ? 'a' : CHARS[i] === 'y' ? 'b' : i === 0 ? 'muted' : 'ink' }));
      items.push({ k: 'ax', t: 'text', x: 860, y: 505, anchor: 'end', size: 13, tone: 'muted', text: t({ en: 'map direction 1 →', zh: '地圖方向 1 →' }) });
      items.push({ k: 'mt', t: 'text', x: 40, y: 505, size: 14, tone: step === 6 ? 'a' : 'b', bold: true, text: step === 6 ? t({ en: 'trained table', zh: '訓練後的表' }) : t({ en: 'untrained table (random)', zh: '未訓練的表（隨機）' }) });
    }

    if (step === 8) {
      const y = 190;
      items.push({ k: 'chipE', t: 'token', x: 70, y: y + 8, text: 'e', tone: 'a' });
      items.push({ k: 'r1', t: 'box', x: 160, y, w: 120, h: 60, label: '5', sub: t({ en: 'id', zh: '編號' }) });
      items.push({ k: 'r2', t: 'box', x: 330, y, w: 170, h: 60, label: t({ en: 'row 5 of tok', zh: 'tok 的第 5 列' }), sub: t({ en: 'lookup', zh: '查表' }) });
      items.push({ k: 'big', t: 'vec', x: 550, y: y + 19, v: trained[E], cell: 18, max: maxAbs, label: t({ en: 'embedding (learned)', zh: '嵌入（學出來的）' }) });
      items.push({ k: 'ra0', t: 'arrow', x1: 118, y1: y + 30, x2: 154, y2: y + 30 });
      items.push({ k: 'ra1', t: 'arrow', x1: 284, y1: y + 30, x2: 324, y2: y + 30 });
      items.push({ k: 'ra2', t: 'arrow', x1: 504, y1: y + 30, x2: 544, y2: y + 30 });
      items.push({ k: 'rs', t: 'text', x: 450, y: 360, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: `vowel similarity: ${f2(d.fresh.vv)} before training → ${f2(d.trained.vv)} after`, zh: `母音相似度：訓練前 ${f2(d.fresh.vv)} → 訓練後 ${f2(d.trained.vv)}` }) });
    }

    const isFresh = step === 7;
    return {
      metrics: [
        { label: { en: 'Table size', zh: '表的大小' }, value: step >= 1 ? '27 × 16' : '—' },
        { label: { en: 'Numbers per token', zh: '每個詞元的數字' }, value: step >= 1 ? '16' : '1' },
        { label: { en: 'Table', zh: '表' }, value: isFresh ? t({ en: 'untrained', zh: '未訓練' }) : step >= 1 ? t({ en: 'trained', zh: '訓練後' }) : '—' },
        { label: { en: 'Vowel similarity', zh: '母音相似度' }, value: step >= 5 ? f2(isFresh ? d.fresh.vv : d.trained.vv) : '—' },
      ],
      items,
      legend: step >= 6 && step <= 7
        ? [{ tone: 'a', text: { en: 'vowel', zh: '母音' } }, { tone: 'b', text: { en: '“y”', zh: '「y」' } }]
        : step >= 1
          ? [{ tone: 'pos', text: { en: 'positive number', zh: '正數' } }, { tone: 'neg', text: { en: 'negative number', zh: '負數' } }]
          : undefined,
    };
  },
};
