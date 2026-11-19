import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, DEFAULT_CONFIG, type Weights } from '../model/gpt';
import { CHARS, encode, rng } from '../model/data';
import { Mat, addRow, matmul, softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';
import experiments from '../model/experiments.json';

// A block = attention + MLP, each adding onto the residual stream.
// Real models stack many identical blocks; we count weights and compare
// our real 1-block and 2-block training runs.

const WORD = '.jimm';
const LABELS = [...WORD];
const F = LABELS.length - 1;
const V = experiments.variants;

// GPT-2 small (verified outside facts): 12 blocks, vectors of 768, MLP width 3,072.
const G = { blocks: 12, d: 768, hidden: 3072 };
const gAtt = 4 * G.d * G.d; // query, key, value and output matrices
const gMlp = 2 * G.d * G.hidden; // w1 and w2
const gBlock = gAtt + gMlp;

const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const fmt = (v: number) => v.toLocaleString('en-US');
const mil = (v: number) => (v / 1e6).toFixed(1);
const pct = (v: number) => Math.round(v * 100) + '%';
const last = (a: number[]) => a[a.length - 1];
const best = (a: number[]) => Math.min(...a);
const f3 = (v: number) => v.toFixed(3);

interface Data {
  att: number; mlp: number; block: number; base: number; two: number; emb: number; out: number;
  early: number[]; late: number[];
}
let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const P = model.p;
  const c = (m: GPT, names: string[]) => names.reduce((s, k) => s + m.p[k].d.length, 0);
  const two = new GPT({ ...DEFAULT_CONFIG, layers: 2 }, rng(1)); // only used to count its weights
  const tr = model.forward(encode(WORD));
  // read the stream with the output layer before and after the block
  const read = (v: number[]) => { const m = new Mat(1, 16); m.d.set(v); return softmaxRow(addRow(matmul(m, P.u), P.bu).row(0)); };
  return {
    att: c(model, ['q0', 'k0', 'v0', 'o']), mlp: c(model, ['w1', 'b1', 'w2', 'b2']),
    block: c(two, ['q0_1', 'k0_1', 'v0_1', 'o_1', 'w1_1', 'b1_1', 'w2_1', 'b2_1']),
    base: model.paramCount(), two: two.paramCount(), emb: c(model, ['tok', 'pos']), out: c(model, ['u', 'bu']),
    early: read(tr.x.row(F)), late: read(tr.x2.row(F)),
  };
})();

const top = (p: number[], n: number) =>
  p.map((v, i) => ({ label: CHARS[i], v })).sort((a, b) => b.v - a.v).slice(0, n).map((b) => ({ ...b, hi: b.label === 'y' }));
const Y = CHARS.indexOf('y');

export const stacking: Lesson = {
  id: 'stacking',
  title: { en: 'Stacking Blocks', zh: '堆疊區塊' },
  blurb: {
    en: 'A transformer is one block design repeated many times. See what a block holds and what depth buys.',
    zh: 'Transformer 就是同一種區塊重複很多次。來看一個區塊裡有什麼，以及深度能換到什麼。',
  },
  minutes: 9,
  level: 'beginner',
  learn: [
    { text: { en: 'What one transformer block contains', zh: '一個 Transformer 區塊包含什麼' }, step: 1 },
    { text: { en: 'Why identical blocks can be stacked', zh: '為什麼相同的區塊可以疊起來' }, step: 2 },
    { text: { en: 'How each block refines the vectors', zh: '每個區塊如何改進向量' }, step: 3 },
    { text: { en: 'How big real stacks are', zh: '真實模型的堆疊有多大' }, step: 4 },
    { text: { en: 'When more depth does not help', zh: '什麼時候加深沒有幫助' }, step: 5 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'One round of thinking', zh: '只想一輪' },
      body: [
        { en: 'Our model reads <code>.jimm</code>, turns each letter into a vector, passes the vectors once through attention and the MLP, and predicts. That middle part is one <b>block</b>.', zh: '我們的模型讀進 <code>.jimm</code>，把每個字母變成向量，讓這些向量通過一次注意力和 MLP，然後做預測。中間這一段就叫做一個<b>區塊（block）</b>。' },
        { en: 'One block means one round: each token gathers context once and processes it once. Harder patterns may need a token to use what <i>another</i> token already gathered. For example (an illustration): in block 1 the second “m” could learn “the letter before me is also m”; only a second block lets a later token read that finding and use it, say “after a double letter, a vowel is likely”. How do we get more rounds?', zh: '一個區塊代表只有一輪：每個詞元收集一次上下文、加工一次。比較難的模式可能需要一個詞元去利用<i>另一個</i>詞元已經收集好的資訊。舉例來說（示意）：在第 1 個區塊裡，第二個「m」可以學到「我前面那個字母也是 m」；要到第 2 個區塊，後面的詞元才能讀到這個發現並加以利用，例如「雙字母之後很可能接母音」。要怎麼多想幾輪？' },
      ],
      callout: { kind: 'insight', text: { en: 'Depth is about rounds: each block lets every token look around and think one more time.', zh: '深度講的就是「輪數」：每多一個區塊，每個詞元就能多看一次、多想一次。' } },
    },
    {
      short: { en: 'A block', zh: '一個區塊' }, chapter: 'technique',
      title: { en: 'What is inside a block', zh: '區塊裡有什麼' },
      body: [
        lz(() => ({ en: `A block is the two parts from the last lessons, each adding onto the residual stream: attention (${fmt(data().att)} weights in our model) and the MLP (${fmt(data().mlp)}).`, zh: `一個區塊就是前幾課的兩個部分，各自把結果加到殘差流上：注意力（我們的模型裡有 ${fmt(data().att)} 個權重）和 MLP（${fmt(data().mlp)} 個）。` })),
        lz(() => ({ en: `That is ${fmt(data().att + data().mlp)} weights per block. The rest of our ${fmt(data().base)} weights are the embeddings at the start and the output layer at the end.`, zh: `所以每個區塊有 ${fmt(data().att + data().mlp)} 個權重。我們全部 ${fmt(data().base)} 個權重裡，其餘的是開頭的嵌入和最後的輸出層。` })),
      ],
      changed: [{ sign: '+', text: { en: 'Block = attention + MLP, each added', zh: '區塊 = 注意力 + MLP，各自相加' } }],
    },
    {
      short: { en: 'Stack', zh: '堆疊' }, chapter: 'technique',
      title: { en: 'Stack identical blocks', zh: '把相同的區塊疊起來' },
      body: [
        { en: 'A block takes 16 numbers per token and gives back 16 numbers per token. Because the shape going out matches the shape coming in, the output of block 1 can feed straight into block 2.', zh: '一個區塊輸入每個詞元 16 個數字，輸出也是每個詞元 16 個數字。因為出來的形狀和進去的一樣，第 1 個區塊的輸出可以直接接進第 2 個區塊。' },
        lz(() => ({ en: `Every block has the same design but its <i>own</i> weights. Our model with 2 blocks has ${fmt(data().base)} + ${fmt(data().block)} = ${fmt(data().two)} weights.`, zh: `每個區塊的設計相同，但權重是<i>各自</i>的。我們的模型改成 2 個區塊後，有 ${fmt(data().base)} + ${fmt(data().block)} = ${fmt(data().two)} 個權重。` })),
      ],
      changed: [{ sign: '+', text: { en: 'Block 2, with its own weights', zh: '第 2 個區塊，有自己的權重' } }],
    },
    {
      short: { en: 'Refine', zh: '改進' }, chapter: 'technique',
      title: { en: 'Each block refines the guess', zh: '每個區塊都讓猜測更好' },
      body: [
        lz(() => ({ en: `We can peek at the stream before the block by feeding it straight to the output layer. With only the embeddings, the guess after <code>.jimm</code> is rough: “y” gets ${pct(data().early[Y])}. The vector still mostly says “I am an ${top(data().early, 1)[0].label}”, so the early reading guesses “${top(data().early, 1)[0].label}” (${pct(top(data().early, 1)[0].v)}).`, zh: `我們可以偷看區塊之前的殘差流：直接把它送進輸出層。只靠嵌入的話，<code>.jimm</code> 之後的猜測很粗略：「y」只有 ${pct(data().early[Y])}。這時向量大多還在說「我是 ${top(data().early, 1)[0].label}」，所以提早讀出的猜測是「${top(data().early, 1)[0].label}」（${pct(top(data().early, 1)[0].v)}）。` })),
        lz(() => ({ en: `After one block, “y” gets ${pct(data().late[Y])}. In a deep model each block adds its own correction to the stream, and the guess sharpens bit by bit.`, zh: `通過一個區塊之後，「y」變成 ${pct(data().late[Y])}。在很深的模型裡，每個區塊都往殘差流加上自己的修正，猜測就一點一點變準。` })),
      ],
      callout: { kind: 'note', text: { en: 'The output layer was trained to read the stream only at the very end, so the early reading is rough. It still shows the trend.', zh: '輸出層只學過讀取最後的殘差流，所以提早讀出的結果比較粗略，但仍看得出趨勢。' } },
      changed: [{ sign: '+', text: { en: 'Prediction before vs after the block', zh: '區塊前後的預測對比' } }],
    },
    {
      short: { en: 'Real stacks', zh: '真實堆疊' }, chapter: 'solution',
      title: { en: 'Real models stack many blocks', zh: '真實的模型疊了很多區塊' },
      body: [
        { en: `GPT-2 small, a well-known model from 2019, stacks ${G.blocks} blocks with vectors of ${G.d} numbers. Each block holds about ${mil(gAtt)} million attention weights and ${mil(gMlp)} million MLP weights.`, zh: `GPT-2 small 是 2019 年一個知名的模型，疊了 ${G.blocks} 個區塊，每個向量有 ${G.d} 個數字。每個區塊大約有 ${mil(gAtt)} 百萬個注意力權重和 ${mil(gMlp)} 百萬個 MLP 權重。` },
        { en: 'GPT-3 (2020) stacks 96 blocks for 175 billion weights. The design inside each block is the same idea you now know: attention, MLP, add, repeat.', zh: 'GPT-3（2020 年）疊了 96 個區塊，總共 1,750 億個權重。每個區塊裡的設計，就是你現在已經懂的想法：注意力、MLP、相加，然後重複。' },
      ],
      callout: { kind: 'note', text: { en: 'Block weights here are rounded and ignore small extras like biases; the totals also include embeddings.', zh: '這裡的區塊權重是四捨五入，並忽略了偏差等小項目；模型總數另外還包含嵌入。' } },
      changed: [{ sign: '~', text: { en: `1 block → ${G.blocks} blocks (GPT-2 small)`, zh: `1 個區塊 → ${G.blocks} 個區塊（GPT-2 small）` } }],
    },
    {
      short: { en: 'Two blocks', zh: '兩個區塊' }, chapter: 'problem',
      title: { en: 'Case: two blocks on our task', zh: '案例：在我們的任務上用兩個區塊' },
      body: [
        { en: `We really trained a 1-block and a 2-block model the same way for ${fmt(V.steps)} steps (comparison runs, constant learning rate ${V.base.lr[0]}). Held-out loss (the loss on 615 names the model never trained on) at the end: ${f3(last(V.base.val))} with 1 block, ${f3(last(V.layers2.val))} with 2. Best along the way: ${f3(best(V.base.val))} vs ${f3(best(V.layers2.val))}.`, zh: `我們真的用同樣的方法訓練了 1 區塊和 2 區塊的模型，各 ${fmt(V.steps)} 步（比較實驗，學習率固定 ${V.base.lr[0]}）。最後的驗證損失（模型在訓練時從沒看過的 615 個名字上的損失）：1 個區塊是 ${f3(last(V.base.val))}，2 個區塊是 ${f3(last(V.layers2.val))}；過程中最好的分別是 ${f3(best(V.base.val))} 和 ${f3(best(V.layers2.val))}。` },
        { en: 'No gain. Our names are at most 7 letters and the patterns are simple, so one round already catches most of what the data can teach. The second block mostly fits the training names a little closer.', zh: '沒有進步。我們的名字最多 7 個字母，模式也很簡單，一輪就已經學到資料能教的大部分東西了。第二個區塊多半只是把訓練用的名字背得更緊一點。' },
      ],
      changed: [{ sign: '+', text: { en: 'Real runs: 1 vs 2 blocks', zh: '真實訓練：1 vs 2 個區塊' } }],
    },
    {
      short: { en: 'Cost', zh: '代價' }, chapter: 'problem',
      title: { en: 'Case: depth is not free', zh: '案例：加深不是免費的' },
      body: [
        { en: `The 2-block run took ${V.layers2.seconds} seconds against ${V.base.seconds}: every extra block is more work on every token, both when training and when writing.`, zh: `2 區塊的訓練花了 ${V.layers2.seconds} 秒，1 區塊是 ${V.base.seconds} 秒：每多一個區塊，每個詞元都要多算一輪，訓練和寫字時都一樣。` },
        { en: `Its training loss (measured on the names it trains on) is a bit lower (${f3(last(V.layers2.train))} vs ${f3(last(V.base.train))}) while its held-out loss (on names it never saw) is not. Extra capacity helps only when the task and the data are rich enough to use it.`, zh: `它的訓練損失（在訓練用的名字上量的）低了一點（${f3(last(V.layers2.train))} 對 ${f3(last(V.base.train))}），驗證損失（在沒看過的名字上量的）卻沒有變好。多出來的容量，只有在任務和資料夠豐富時才派得上用場。` },
      ],
      callout: { kind: 'insight', text: { en: 'Big models are deep because their tasks are hard and their data is huge. Depth is a tool to match the problem, not a goal.', zh: '大型模型之所以深，是因為它們的任務難、資料又多。深度是用來配合問題的工具，不是目標。' } },
      changed: [{ sign: '+', text: { en: 'Time and training-vs-held-out gap', zh: '時間，以及訓練損失與驗證損失的落差' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Stacking in one line', zh: '一句話說完堆疊' },
      body: [
        { en: 'A <b>block</b> is attention + MLP, each adding onto the residual stream. Blocks keep the shape of the vectors, so identical blocks, each with its own weights, can be stacked.', zh: '一個<b>區塊</b>就是注意力 + MLP，各自把結果加到殘差流上。區塊不改變向量的形狀，所以相同設計、各有權重的區塊可以一直疊上去。' },
        { en: 'Each block refines the vectors a bit more. Big models stack dozens; our tiny task gains nothing from a second block.', zh: '每個區塊都把向量再改進一點。大型模型疊了幾十個；我們的小任務多加一個區塊卻沒有好處。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: From Logits to Text. The last block gives scores for every letter; how do we turn those into an actual name?', zh: '下一課：從 logits 到文字。最後一個區塊給出每個字母的分數，要怎麼把它們變成真正的名字？' } },
    },
  ],

  scene(step) {
    const d = data();
    const items: Item[] = [];
    const two = step === 2;

    if (step <= 3) {
      LABELS.forEach((ch, i) => items.push({ k: 'tok' + i, t: 'token', text: ch, x: 70 + i * 50, y: 16, tone: i === F ? 'a' : ch === '.' ? 'b' : 'ink' }));
      const bh = two ? 110 : step === 1 ? 190 : 150;
      const b1y = 150, b2y = b1y + bh + 30;
      const outY = two ? b2y + bh + 30 : b1y + bh + 40;
      items.push({ k: 'emb', t: 'box', x: 60, y: 82, w: 260, h: 40, label: t({ en: 'embeddings: letter + position', zh: '嵌入：字母 + 位置' }) });
      items.push({ k: 'b1', t: 'box', x: 60, y: b1y, w: 260, h: bh, label: step === 1 ? '' : t({ en: 'block 1', zh: '區塊 1' }), sub: step === 1 ? undefined : t({ en: 'attention + MLP', zh: '注意力 + MLP' }), tone: 'a', active: step <= 1 });
      if (step === 1) {
        items.push({ k: 'b1t', t: 'text', x: 190, y: b1y + 24, text: t({ en: 'block', zh: '區塊' }), size: 15, bold: true, tone: 'a', anchor: 'middle' });
        items.push({ k: 'ia', t: 'box', x: 90, y: b1y + 40, w: 200, h: 50, label: t({ en: 'attention', zh: '注意力' }), sub: t({ en: `${fmt(d.att)} weights · add`, zh: `${fmt(d.att)} 個權重 · 相加` }) });
        items.push({ k: 'im', t: 'box', x: 90, y: b1y + 115, w: 200, h: 50, label: 'MLP', sub: t({ en: `${fmt(d.mlp)} weights · add`, zh: `${fmt(d.mlp)} 個權重 · 相加` }) });
        items.push({ k: 'iam', t: 'arrow', x1: 190, y1: b1y + 92, x2: 190, y2: b1y + 113, tone: 'muted' });
      }
      if (two) items.push({ k: 'b2', t: 'box', x: 60, y: b2y, w: 260, h: bh, label: t({ en: 'block 2', zh: '區塊 2' }), sub: t({ en: 'same design, own weights', zh: '設計相同，權重各自' }), tone: 'b', active: true });
      items.push({ k: 'out', t: 'box', x: 60, y: outY, w: 260, h: 40, label: t({ en: 'output layer → next letter', zh: '輸出層 → 下一個字母' }) });
      items.push({ k: 'a0', t: 'arrow', x1: 190, y1: 124, x2: 190, y2: b1y - 2, tone: 'a' });
      items.push({ k: 'a1', t: 'arrow', x1: 190, y1: b1y + bh + 2, x2: 190, y2: (two ? b2y : outY) - 2, tone: 'a' });
      if (two) items.push({ k: 'a2', t: 'arrow', x1: 190, y1: b2y + bh + 2, x2: 190, y2: outY - 2, tone: 'a' });
      if (two || step === 0) {
        items.push({ k: 'shape', t: 'text', x: 202, y: b1y + bh + 20, text: t({ en: '16 numbers per token', zh: '每個詞元 16 個數字' }), size: 13, tone: 'muted' });
      }
    }

    if (step === 0) {
      items.push({ k: 'q', t: 'text', x: 600, y: 200, text: t({ en: 'one block = one round', zh: '一個區塊 = 一輪' }), size: 20, bold: true, anchor: 'middle' });
      items.push({ k: 'q2', t: 'text', x: 600, y: 236, text: t({ en: 'gather context once, process it once', zh: '收集一次上下文，加工一次' }), size: 15, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'q3', t: 'text', x: 600, y: 300, text: t({ en: 'more rounds?', zh: '想多幾輪？' }), size: 20, bold: true, tone: 'b', anchor: 'middle' });
    }

    if (step === 1 || step === 2) {
      const rows: [L, number, boolean][] = step === 1
        ? [[{ en: 'attention (q, k, v, o)', zh: '注意力（q、k、v、o）' }, d.att, false], [{ en: 'MLP (w1, b1, w2, b2)', zh: 'MLP（w1、b1、w2、b2）' }, d.mlp, false], [{ en: 'one block', zh: '一個區塊' }, d.att + d.mlp, true]]
        : [[{ en: 'embeddings', zh: '嵌入' }, d.emb, false], [{ en: 'block 1', zh: '區塊 1' }, d.block, false], [{ en: 'block 2', zh: '區塊 2' }, d.block, false], [{ en: 'output layer', zh: '輸出層' }, d.out, false], [{ en: 'total, 2 blocks', zh: '合計（2 個區塊）' }, d.two, true]];
      items.push({ k: 'th', t: 'text', x: 430, y: 130, text: t({ en: 'weights', zh: '權重數量' }), size: 15, bold: true, tone: 'muted' });
      rows.forEach(([label, n, total], i) => {
        const y = 170 + i * 40;
        items.push({ k: 'tl' + i, t: 'text', x: 430, y, text: t(label), size: 15, bold: total, tone: total ? 'a' : 'ink' });
        items.push({ k: 'tn' + i, t: 'text', x: 840, y, text: fmt(n), size: 16, mono: true, bold: true, anchor: 'end', tone: total ? 'a' : 'ink' });
      });
    }

    if (step === 3) {
      items.push({ k: 'pe', t: 'bars', x: 400, y: 70, w: 460, h: 170, items: top(d.early, 6), title: t({ en: 'reading the stream before the block', zh: '在區塊之前讀取殘差流' }) });
      items.push({ k: 'pl', t: 'bars', x: 400, y: 300, w: 460, h: 170, items: top(d.late, 6), title: t({ en: 'reading it after the block (the real output)', zh: '在區塊之後讀取（真正的輸出）' }) });
      items.push({ k: 'ra', t: 'arrow', x1: 330, y1: 120, x2: 392, y2: 140, tone: 'muted', dashed: true });
      items.push({ k: 'rb', t: 'arrow', x1: 330, y1: 330, x2: 392, y2: 380, tone: 'a' });
    }

    if (step === 4) {
      items.push({ k: 'oh', t: 'text', x: 130, y: 60, text: t({ en: 'our model', zh: '我們的模型' }), size: 15, bold: true, anchor: 'middle' });
      items.push({ k: 'ob', t: 'box', x: 60, y: 80, w: 140, h: 26, tone: 'a' });
      items.push({ k: 'on', t: 'text', x: 130, y: 136, text: t({ en: `1 block · ${fmt(d.base)} weights`, zh: `1 個區塊 · ${fmt(d.base)} 個權重` }), size: 13, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'gh', t: 'text', x: 400, y: 60, text: 'GPT-2 small', size: 15, bold: true, anchor: 'middle' });
      for (let i = 0; i < G.blocks; i++) items.push({ k: 'g' + i, t: 'box', x: 330, y: 80 + i * 30, w: 140, h: 26, tone: 'b' });
      items.push({ k: 'gn', t: 'text', x: 400, y: 80 + G.blocks * 30 + 22, text: t({ en: `${G.blocks} blocks · ~${mil(gBlock)} M weights each`, zh: `${G.blocks} 個區塊 · 每個約 ${mil(gBlock)} 百萬個權重` }), size: 13, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'g3h', t: 'text', x: 700, y: 60, text: 'GPT-3', size: 15, bold: true, anchor: 'middle' });
      items.push({ k: 'g3b', t: 'box', x: 630, y: 80, w: 140, h: 356, label: t({ en: '96 blocks', zh: '96 個區塊' }), sub: t({ en: '(too many to draw)', zh: '（多到畫不下）' }), tone: 'ink' });
      items.push({ k: 'g3n', t: 'text', x: 700, y: 80 + G.blocks * 30 + 22, text: t({ en: '175 billion weights', zh: '1,750 億個權重' }), size: 13, tone: 'muted', anchor: 'middle' });
    }

    if (step === 5 || step === 6) {
      const s5 = step === 5;
      items.push({
        k: 'runs', t: 'chart', x: 80, y: 60, w: 480, h: 360, ymin: 1.9, ymax: 2.7, xmax: V.steps, every: V.base.every,
        series: s5 ? [{ pts: V.base.val, tone: 'a' }, { pts: V.layers2.val, tone: 'b' }]
          : [{ pts: V.base.train, tone: 'a' }, { pts: V.layers2.train, tone: 'b' }, { pts: V.base.val, tone: 'muted' }, { pts: V.layers2.val, tone: 'muted' }],
        xlabel: t({ en: 'training step', zh: '訓練步數' }),
        ylabel: s5 ? t({ en: 'held-out loss', zh: '驗證損失' }) : t({ en: 'training loss (grey: held-out)', zh: '訓練損失（灰色：驗證損失）' }),
      });
      const rowsAt = (y: number, label: L, tone: 'a' | 'b', vals: string) => {
        items.push({ k: 'rl' + y, t: 'text', x: 610, y, text: t(label), size: 15, bold: true, tone });
        items.push({ k: 'rv' + y, t: 'text', x: 610, y: y + 30, text: vals, size: 15, mono: true });
      };
      if (s5) {
        rowsAt(120, { en: '1 block', zh: '1 個區塊' }, 'a', t({ en: `end ${f3(last(V.base.val))} · best ${f3(best(V.base.val))}`, zh: `最後 ${f3(last(V.base.val))} · 最佳 ${f3(best(V.base.val))}` }));
        rowsAt(210, { en: '2 blocks', zh: '2 個區塊' }, 'b', t({ en: `end ${f3(last(V.layers2.val))} · best ${f3(best(V.layers2.val))}`, zh: `最後 ${f3(last(V.layers2.val))} · 最佳 ${f3(best(V.layers2.val))}` }));
        items.push({ k: 'verdict', t: 'text', x: 610, y: 320, text: t({ en: 'no gain on our task', zh: '在我們的任務上沒有進步' }), size: 18, bold: true });
      } else {
        rowsAt(120, { en: '1 block', zh: '1 個區塊' }, 'a', t({ en: `${V.base.seconds} s · train ${f3(last(V.base.train))}`, zh: `${V.base.seconds} 秒 · 訓練 ${f3(last(V.base.train))}` }));
        rowsAt(210, { en: '2 blocks', zh: '2 個區塊' }, 'b', t({ en: `${V.layers2.seconds} s · train ${f3(last(V.layers2.train))}`, zh: `${V.layers2.seconds} 秒 · 訓練 ${f3(last(V.layers2.train))}` }));
        items.push({ k: 'verdict', t: 'text', x: 610, y: 320, text: t({ en: `${(V.layers2.seconds / V.base.seconds).toFixed(1)}× the time`, zh: `花了 ${(V.layers2.seconds / V.base.seconds).toFixed(1)} 倍時間` }), size: 18, bold: true });
      }
    }

    if (step === 7) {
      const boxes: L[] = [
        { en: 'embeddings', zh: '嵌入' }, { en: 'block 1', zh: '區塊 1' }, { en: 'block 2', zh: '區塊 2' }, { en: '… block N', zh: '… 區塊 N' }, { en: 'output', zh: '輸出' },
      ];
      boxes.forEach((b, i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 210, w: 150, h: 60, label: t(b), tone: i >= 1 && i <= 3 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 240, x2: bx + 170, y2: 240 });
      });
      items.push({ k: 'rt', t: 'text', x: 450, y: 320, text: t({ en: 'each block: attention + MLP, both adding onto the stream', zh: '每個區塊：注意力 + MLP，都加到殘差流上' }), size: 14, tone: 'muted', anchor: 'middle' });
    }

    const blocks = step === 2 ? '2' : step === 4 ? `1 vs ${G.blocks}` : step === 5 || step === 6 ? '1 vs 2' : '1';
    return {
      metrics: [
        { label: { en: 'Blocks', zh: '區塊數' }, value: blocks },
        { label: { en: 'Weights per block', zh: '每個區塊的權重' }, value: step >= 1 ? fmt(d.att + d.mlp) : '—' },
        { label: { en: 'Total weights', zh: '總權重' }, value: step === 2 ? fmt(d.two) : step === 5 || step === 6 ? `${fmt(d.base)}/${fmt(d.two)}` : fmt(d.base) },
        { label: { en: 'P(“y”) after .jimm', zh: '.jimm 之後 P(「y」)' }, value: step === 3 ? `${pct(d.early[Y])} → ${pct(d.late[Y])}` : step === 0 ? pct(d.late[Y]) : '—' },
        { label: { en: 'Held-out loss', zh: '驗證損失' }, value: step >= 5 ? `${last(V.base.val).toFixed(2)}/${last(V.layers2.val).toFixed(2)}` : '—' },
      ],
      items,
      legend: [
        { tone: 'a', text: { en: 'our model (1 block)', zh: '我們的模型（1 個區塊）' } },
        { tone: 'b', text: { en: 'extra blocks', zh: '多加的區塊' } },
      ],
    };
  },
};
