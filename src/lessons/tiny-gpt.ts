import type { Lesson, Step } from './types';
import type { Item, Tone } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, NAMES, encode } from '../model/data';
import { softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';

// The whole shipped model in one picture, then one real name through it.
// We feed ".sof" (the start of "sofia") and watch it predict the next letter.

const PREFIX = 'sof';

interface Part { name: string; label: string; r: number; c: number; n: number }
interface Data {
  parts: Record<string, Part[]>; // weights grouped by stage
  total: number;
  ids: number[];
  tokEmb: number[][]; posEmb: number[][]; x: number[][];
  att: (number | null)[][];
  x3: number[]; attOut: number[]; x1: number[];
  hidden: number[]; active: number; mlpOut: number[]; x2: number[];
  logits: number[]; probs: number[];
  look: { at: number; v: number };
  pos: number[][];
  longest: string; maxLen: number;
}

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const P = model.p;
  const part = (name: string, label = name): Part => ({ name, label, r: P[name].r, c: P[name].c, n: P[name].d.length });
  const parts = {
    ids: [] as Part[],
    emb: [part('tok'), part('pos')],
    att: [part('q0', 'q'), part('k0', 'k'), part('v0', 'v'), part('o')],
    mlp: [part('w1'), part('b1'), part('w2'), part('b2')],
    out: [part('u'), part('bu')],
  };
  const ids = encode('.' + PREFIX);
  const tr = model.forward(ids);
  const last = ids.length - 1;
  const rows = (m: { row(i: number): number[] }) => ids.map((_, i) => m.row(i));
  const att = rows(tr.att[0]).map((r, i) => r.map((v, j) => (j > i ? null : v)));
  const attLast = tr.att[0].row(last);
  const at = attLast.indexOf(Math.max(...attLast));
  const hidden = tr.hidden.row(last);
  const logits = tr.logits.row(last);
  const maxLen = Math.max(...NAMES.map((n) => n.length));
  return {
    parts, total: model.paramCount(), ids,
    tokEmb: rows(tr.tokEmb), posEmb: rows(tr.posEmb!), x: rows(tr.x),
    att, x3: tr.x.row(last), attOut: tr.attOut.row(last), x1: tr.x1.row(last),
    hidden, active: hidden.filter((v) => v > 0).length, mlpOut: tr.mlpOut.row(last), x2: tr.x2.row(last),
    logits, probs: softmaxRow(logits),
    look: { at, v: attLast[at] },
    pos: Array.from({ length: P.pos.r }, (_, i) => P.pos.row(i)),
    longest: NAMES.find((n) => n.length === maxLen)!, maxLen,
  };
})();

const sum = (ps: Part[]) => ps.reduce((s, p) => s + p.n, 0);
const n0 = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => Math.round(v * 100) + '%';
const top = () => { const p = data().probs; const i = p.indexOf(Math.max(...p)); return { ch: CHARS[i], p: p[i] }; };

type StageId = 'ids' | 'emb' | 'att' | 'mlp' | 'out';
const STAGES: { id: StageId; label: L; from: L }[] = [
  { id: 'ids', label: { en: 'Token ids', zh: '詞元編號' }, from: { en: 'Tokens', zh: '詞元' } },
  { id: 'emb', label: { en: 'Embeddings', zh: '嵌入' }, from: { en: 'Embeddings', zh: '嵌入 · 位置資訊' } },
  { id: 'att', label: { en: 'Attention', zh: '注意力' }, from: { en: 'Self-Attention', zh: '自注意力' } },
  { id: 'mlp', label: { en: 'MLP', zh: 'MLP' }, from: { en: 'The MLP Layer', zh: 'MLP 層' } },
  { id: 'out', label: { en: 'Output', zh: '輸出' }, from: { en: 'From Logits to Text', zh: '從 logits 到文字' } },
];
// which stages have been assembled at each step of the build-up
const BUILT: StageId[][] = [[], ['ids', 'emb'], ['ids', 'emb', 'att', 'mlp'], ['ids', 'emb', 'att', 'mlp', 'out']];
// which stage is lit during the forward pass (steps 4–7)
const LIT: Record<number, StageId[]> = { 4: ['ids', 'emb'], 5: ['att'], 6: ['mlp'], 7: ['out'], 8: ['emb'] };

const counted = (step: number) => {
  const b = step >= 3 ? BUILT[3] : BUILT[step];
  return b.reduce((s, id) => s + sum(data().parts[id]), 0);
};

let stepsCache: Step[] | undefined;
function buildSteps(): Step[] {
  const D = data();
  const tp = top();
  const letters = D.ids.map((i) => CHARS[i]);
  const emb = sum(D.parts.emb), att = sum(D.parts.att), mlp = sum(D.parts.mlp), q = D.parts.att[0].n;
  return [
    {
      short: { en: 'The pieces', zh: '零件' }, chapter: 'problem',
      title: { en: 'Five lessons, five loose pieces', zh: '五堂課，五個散落的零件' },
      body: [
        { en: 'Across the course you met each part of a transformer on its own: tokens, embeddings, attention, the MLP, and turning logits into text.', zh: '在這門課裡，你已經一個一個認識了 Transformer 的零件：詞元、嵌入、注意力、MLP，以及把 logits 變回文字。' },
        { en: 'But a model is not five ideas. It is one machine that a name flows through. In this lesson we bolt the pieces together and count every weight.', zh: '但模型不是五個獨立的概念，而是一台讓名字流過去的機器。這一課我們把零件鎖在一起，並且數清楚每一個權重。' },
      ],
      callout: { kind: 'insight', text: { en: 'Each box below carries the name of the lesson that explained it. Nothing new is added here: the whole GPT is just these parts in order.', zh: '下面每個方塊都標著介紹它的那一課。這裡沒有新東西：整個 GPT 就只是這些零件依序排好。' } },
    },
    {
      short: { en: 'Input', zh: '輸入' }, chapter: 'technique',
      title: { en: 'Input: ids and two lookup tables', zh: '輸入：編號與兩張查詢表' },
      body: [
        { en: 'A name enters as token <b>ids</b> (Tokens lesson). The first table, <code>tok</code>, has one row of 16 numbers for each of the 27 tokens. The second, <code>pos</code>, has one row for each of the 8 positions.', zh: '名字以詞元<b>編號</b>的形式進來（「詞元」那一課）。第一張表 <code>tok</code>，替 27 個詞元各準備一列 16 個數字；第二張表 <code>pos</code>，替 8 個位置各準備一列。' },
        { en: `A table of 27 rows × 16 columns holds 27 × 16 = ${n0(D.parts.emb[0].n)} weights. Together the two tables hold ${n0(emb)}.`, zh: `一張 27 列 × 16 欄的表，裝了 27 × 16 = ${n0(D.parts.emb[0].n)} 個權重。兩張表合起來是 ${n0(emb)} 個。` },
      ],
      changed: [
        { sign: '+', text: { en: 'tok 27×16, pos 8×16', zh: 'tok 27×16、pos 8×16' } },
      ],
    },
    {
      short: { en: 'Block', zh: '區塊' }, chapter: 'technique',
      title: { en: 'The block: attention, then the MLP', zh: '區塊：先注意力，再 MLP' },
      body: [
        { en: 'Attention needs three 16×16 matrices to make queries, keys and values, plus one more, <code>o</code>, to mix the result back. The MLP widens each vector to 32 numbers (<code>w1</code>, <code>b1</code>) and narrows it back to 16 (<code>w2</code>, <code>b2</code>).', zh: '注意力需要三個 16×16 矩陣來產生查詢、鍵、值，再加一個 <code>o</code> 把結果整理回來。MLP 先把每個向量撐寬到 32 個數字（<code>w1</code>、<code>b1</code>），再縮回 16 個（<code>w2</code>、<code>b2</code>）。' },
        { en: 'Neither replaces the vector. Each one <i>adds</i> its result onto it: that running sum is the residual stream (The Residual Stream lesson).', zh: '兩者都不會取代原本的向量，而是把結果<i>加</i>上去：這條一路累加的線，就是殘差流（「殘差流」那一課）。' },
      ],
      callout: { kind: 'note', text: { en: 'Ours is the smallest version: one block with one attention head and no layer normalization. Bigger models stack many blocks (Stacking Blocks lesson).', zh: '我們用的是最小的版本：一個區塊、一個注意力頭、沒有層正規化。更大的模型會堆疊很多個區塊（「堆疊區塊」那一課）。' } },
      changed: [
        { sign: '+', text: { en: `Attention: 4 × ${q} = ${n0(att)} weights`, zh: `注意力：4 × ${q} = ${n0(att)} 個權重` } },
        { sign: '+', text: { en: `MLP: ${n0(mlp)} weights`, zh: `MLP：${n0(mlp)} 個權重` } },
      ],
    },
    {
      short: { en: 'Output', zh: '輸出' }, chapter: 'technique',
      title: { en: 'The output layer, and the full count', zh: '輸出層，以及總數' },
      body: [
        { en: 'Last, <code>u</code> (16×27) and <code>bu</code> turn the final 16 numbers into 27 scores, one per possible next token. Those scores are the logits.', zh: '最後，<code>u</code>（16×27）和 <code>bu</code> 把最後的 16 個數字變成 27 個分數，每個可能的下一個詞元一個。這些分數就是 logits。' },
        { en: 'Add every box and you get the whole model. The bars show where the weights sit: the MLP and attention hold most of them.', zh: '把每個方塊加起來，就是整個模型。長條圖顯示權重都在哪裡：大部分在 MLP 和注意力裡。' },
      ],
      changed: [{ sign: '+', text: { en: 'u 16×27, bu 27', zh: 'u 16×27、bu 27' } }],
    },
    {
      short: { en: 'Embed', zh: '嵌入' }, chapter: 'technique',
      title: { en: 'Follow a name: ids become vectors', zh: '跟著名字走：編號變成向量' },
      body: [
        { en: `Now we run the real model on <code>.${PREFIX}</code>, the start of “sofia”. Each id picks its row from <code>tok</code> (Embeddings lesson).`, zh: `現在讓真正的模型讀 <code>.${PREFIX}</code>，也就是「sofia」的開頭。每個編號從 <code>tok</code> 取出自己那一列（「嵌入」那一課）。` },
        { en: 'Then we add the row of <code>pos</code> for its place (Positional Information lesson). The sum on the right is what the block actually reads.', zh: '接著加上 <code>pos</code> 裡對應它位置的那一列（「位置資訊」那一課）。右邊的總和，才是區塊真正讀進去的東西。' },
      ],
      changed: [{ sign: '+', text: { en: '4 tokens → 4 vectors of 16', zh: '4 個詞元 → 4 個 16 維向量' } }],
    },
    {
      short: { en: 'Attend', zh: '注意' }, chapter: 'technique',
      title: { en: 'Attention: look back at earlier letters', zh: '注意力：回頭看前面的字母' },
      body: [
        { en: 'Each letter compares its query with the keys of the letters up to and including itself; future letters are masked (×). Each row adds up to 100% (Self-Attention lesson).', zh: '每個字母用自己的查詢，和自己以及前面字母的鍵做比對；未來的字母被遮住（×）。每一列加起來是 100%（「自注意力」那一課）。' },
        { en: `The last letter, “${letters[3]}”, puts ${pct(D.look.v)} of its attention on “${letters[D.look.at]}”. What it gathers is added to its own vector (right).`, zh: `最後一個字母「${letters[3]}」把 ${pct(D.look.v)} 的注意力放在「${letters[D.look.at]}」上。它收集到的東西，會加到自己的向量上（右邊）。` },
      ],
      changed: [{ sign: '~', text: { en: 'Last vector: x + attention output', zh: '最後一個向量：x + 注意力輸出' } }],
    },
    {
      short: { en: 'MLP', zh: 'MLP' }, chapter: 'technique',
      title: { en: 'The MLP: think about what was gathered', zh: 'MLP：消化剛剛收集到的資訊' },
      body: [
        { en: 'The MLP works on each position alone. It widens the 16 numbers to 32 neurons, sets negative ones to 0, and squeezes the rest back to 16 (The MLP Layer lesson).', zh: 'MLP 對每個位置各自運算。它把 16 個數字撐寬成 32 個神經元，把負的設成 0，再把剩下的壓回 16 個（「MLP 層」那一課）。' },
        { en: `For “${letters[3]}”, ${D.active} of the 32 neurons switch on. Their output is again added to the stream, so the vector now carries what both attention and the MLP found.`, zh: `對「${letters[3]}」來說，32 個神經元中有 ${D.active} 個被打開。它們的輸出同樣加回殘差流，所以這個向量現在同時帶著注意力與 MLP 找到的資訊。` },
      ],
      changed: [{ sign: '~', text: { en: 'Last vector: + MLP output', zh: '最後一個向量：+ MLP 輸出' } }],
    },
    {
      short: { en: 'Predict', zh: '預測' }, chapter: 'solution',
      title: { en: 'Logits, then probabilities', zh: '先是 logits，再變成機率' },
      body: [
        { en: 'The last vector meets <code>u</code> and <code>bu</code> and becomes 27 logits. Softmax turns them into probabilities that add up to 100% (From Logits to Text lesson).', zh: '最後一個向量經過 <code>u</code> 和 <code>bu</code>，變成 27 個 logits。softmax 再把它們轉成加總為 100% 的機率（「從 logits 到文字」那一課）。' },
        { en: `Top guess: “${tp.ch}” at ${pct(tp.p)}, which is right for “sofia”. The model is not sure, and that is fine: sampling picks among these, so it can also write other names.`, zh: `最有可能的是「${tp.ch}」，機率 ${pct(tp.p)}，對「sofia」來說正是答案。模型並不是百分之百確定，這沒關係：取樣會在這些選項裡挑，所以它也能寫出別的名字。` },
      ],
      callout: { kind: 'insight', text: { en: 'That is a whole forward pass: a lookup, one attention mix, one MLP, one output layer. Writing a name just repeats it, one letter at a time.', zh: '這就是一次完整的前向計算：查表、一次注意力混合、一次 MLP、一次輸出層。寫出整個名字，只是一個字母一個字母地重複這件事。' } },
      changed: [{ sign: '+', text: { en: '27 probabilities for the next letter', zh: '下一個字母的 27 個機率' } }],
    },
    {
      short: { en: 'Limit', zh: '上限' }, chapter: 'problem',
      title: { en: 'Case: the model can only read 8 tokens', zh: '案例：模型最多只能讀 8 個詞元' },
      body: [
        { en: `The <code>pos</code> table has exactly ${D.pos.length} rows, one per position. The longest name in our list, “${D.longest}”, has ${D.maxLen} letters, so with the start mark it fills all ${D.pos.length}.`, zh: `<code>pos</code> 表剛好有 ${D.pos.length} 列，每個位置一列。清單中最長的名字「${D.longest}」有 ${D.maxLen} 個字母，加上起始符號，正好用滿 ${D.pos.length} 個位置。` },
        { en: 'So the model simply cannot see a ninth token: there is no position vector for it. Our names fit, but this is a hard limit built into the shapes, called the <b>context length</b>.', zh: '所以模型根本看不到第 9 個詞元：沒有對應的位置向量。我們的名字剛好放得下，但這是寫死在形狀裡的硬性上限，叫做<b>上下文長度</b>。' },
      ],
      callout: { kind: 'note', text: { en: 'Real models have the same kind of limit, only far longer. Making it longer means a bigger position table and much more attention work.', zh: '真實的模型也有同樣的上限，只是長得多。要拉長它，就需要更大的位置表，以及多得多的注意力運算。' } },
      changed: [{ sign: '+', text: { en: 'Context length: 8', zh: '上下文長度：8' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'A GPT is a short pipeline of simple parts', zh: 'GPT 是一條由簡單零件組成的短管線' },
      body: [
        { en: `<b>Embed</b>: ids pick rows from <code>tok</code> and <code>pos</code>.<br><b>Attend</b>: each letter gathers from earlier letters.<br><b>MLP</b>: each position is processed on its own.<br><b>Output</b>: 27 logits, then softmax.<br>All of it is ${n0(D.total)} numbers that training must find.`, zh: `<b>嵌入</b>：編號從 <code>tok</code> 和 <code>pos</code> 取出一列。<br><b>注意力</b>：每個字母從前面的字母收集資訊。<br><b>MLP</b>：每個位置各自處理。<br><b>輸出</b>：27 個 logits，再經過 softmax。<br>全部加起來就是 ${n0(D.total)} 個數字，等著訓練去找出來。` },
      ],
      callout: { kind: 'next', text: { en: 'Next: Train It Live. Start this exact model from random numbers and train it in your browser, watching every curve move.', zh: '下一課：現場訓練它。讓這個模型從隨機數字開始，在你的瀏覽器裡訓練，看著每條曲線變化。' } },
    },
  ];
}

export const tinyGpt: Lesson = {
  id: 'tiny-gpt',
  title: { en: 'Build a Tiny GPT', zh: '打造一個迷你 GPT' },
  blurb: {
    en: 'Put every piece from the course together into one working model, and follow a real name through it.',
    zh: '把整門課學過的零件組裝成一個能運作的模型，並跟著一個真實的名字走完它的每一步。',
  },
  minutes: 12,
  level: 'intermediate',
  learn: [
    { text: { en: 'How the parts connect into one model', zh: '各個零件如何接成一個模型' }, step: 2 },
    { text: { en: 'Where every weight lives', zh: '每個權重分別在哪裡' }, step: 3 },
    { text: { en: 'What happens at each stage of a forward pass', zh: '前向計算每一站發生了什麼' }, step: 6 },
    { text: { en: 'How the last vector becomes a guess', zh: '最後一個向量如何變成一個猜測' }, step: 7 },
    { text: { en: 'Why the model has a length limit', zh: '為什麼模型有長度上限' }, step: 8 },
  ],
  get steps() { return stepsCache ??= buildSteps(); },

  scene(step) {
    const D = data();
    const items: Item[] = [];

    // ---------- the pipeline of five stages ----------
    const assembled = step <= 3 || step === 9;
    const built = step === 9 ? BUILT[3] : step <= 3 ? BUILT[step] : BUILT[3];
    const W = 150, GAP = 25, X0 = 20;
    const bx = (i: number) => X0 + i * (W + GAP);
    const scatter = [[60, 120], [380, 70], [640, 160], [140, 330], [520, 330]];
    STAGES.forEach((s, i) => {
      const on = built.includes(s.id);
      const lit = LIT[step]?.includes(s.id) ?? false;
      const n = sum(D.parts[s.id]);
      let x = bx(i), y = 120, h = 56, tone: Tone = on ? 'a' : 'muted';
      let sub = on ? (n ? t({ en: `${n0(n)} weights`, zh: `${n0(n)} 個權重` }) : t({ en: 'no weights', zh: '沒有權重' })) : t(s.from);
      if (step === 0) { [x, y] = scatter[i]; sub = t(s.from); tone = 'muted'; }
      if (!assembled) { y = 20; h = 50; tone = lit ? 'a' : 'muted'; sub = t(s.from); }
      items.push({ k: 'st' + i, t: 'box', x, y, w: W, h, label: t(s.label), sub, tone, active: !assembled && lit });
      if (step > 0 && i < 4) {
        const ay = assembled ? 148 : 45;
        items.push({ k: 'sa' + i, t: 'arrow', x1: bx(i) + W + 2, y1: ay, x2: bx(i + 1) - 2, y2: ay });
      }
      // weight shapes under each assembled box
      if (assembled && on) {
        const lines = s.id === 'ids'
          ? [t({ en: `${CHARS.length} possible tokens`, zh: `${CHARS.length} 種可能的詞元` })]
          : D.parts[s.id].map((p) => p.r === 1 ? `${p.label} ${p.c} = ${n0(p.n)}` : `${p.label} ${p.r}×${p.c} = ${n0(p.n)}`);
        lines.forEach((ln, j) => items.push({ k: `w${i}_${j}`, t: 'text', x: bx(i) + W / 2, y: 205 + j * 22, anchor: 'middle', mono: s.id !== 'ids', size: 13.5, text: ln, tone: s.id === 'ids' ? 'muted' : 'ink' }));
      }
    });
    if (step === 0) {
      items.push({ k: 'q0', t: 'text', x: 450, y: 470, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: 'labels show the lesson that explained each piece', zh: '方塊上的小字是介紹該零件的課程' }) });
    }

    // residual stream over the block
    if ((step >= 2 && step <= 3) || step === 9) {
      items.push({ k: 'res', t: 'arrow', x1: bx(1) + W / 2, y1: 98, x2: bx(4) + W / 2, y2: 98, tone: 'a', dashed: true });
      items.push({ k: 'resl', t: 'text', x: bx(2) + W + GAP / 2, y: 88, anchor: 'middle', size: 13.5, tone: 'a', text: t({ en: 'residual stream: attention and MLP add onto it', zh: '殘差流：注意力與 MLP 都把結果加上去' }) });
    }

    // where the weights sit
    if (step === 3 || step === 9) {
      const groups: StageId[] = ['emb', 'att', 'mlp', 'out'];
      items.push({
        k: 'share', t: 'bars', x: 150, y: 345, w: 600, h: 150,
        title: t({ en: `share of all ${n0(D.total)} weights`, zh: `占全部 ${n0(D.total)} 個權重的比例` }),
        items: groups.map((g) => ({ label: t(STAGES.find((s) => s.id === g)!.label), v: sum(D.parts[g]) / D.total, hi: g === 'mlp' })),
      });
      items.push({ k: 'tot', t: 'text', x: 870, y: 420, anchor: 'end', size: 24, bold: true, tone: 'a', text: n0(D.total) });
      items.push({ k: 'totl', t: 'text', x: 870, y: 444, anchor: 'end', size: 13.5, tone: 'muted', text: t({ en: 'weights in total', zh: '個權重' }) });
    }

    // ---------- forward pass ----------
    const letters = D.ids.map((i) => CHARS[i]);
    if (step === 4) {
      const heads = [
        { x: 110, text: t({ en: 'token row (tok)', zh: '詞元那一列（tok）' }) },
        { x: 330, text: t({ en: 'position row (pos)', zh: '位置那一列（pos）' }) },
        { x: 550, text: t({ en: 'sum: what the block reads', zh: '總和：區塊讀進去的' }) },
      ];
      heads.forEach((h, j) => items.push({ k: 'fh' + j, t: 'text', x: h.x, y: 122, size: 13.5, tone: 'muted', text: h.text }));
      letters.forEach((ch, i) => {
        const y = 140 + i * 82;
        items.push({ k: 'ft' + i, t: 'token', x: 30, y, text: ch, sub: t({ en: `id ${D.ids[i]}`, zh: `編號 ${D.ids[i]}` }), tone: ch === '.' ? 'b' : 'ink' });
        items.push({ k: 'fe' + i, t: 'vec', x: 110, y: y + 11, v: D.tokEmb[i] });
        items.push({ k: 'fp' + i, t: 'text', x: 310, y: y + 28, anchor: 'middle', size: 18, bold: true, tone: 'muted', text: '+' });
        items.push({ k: 'fq' + i, t: 'vec', x: 330, y: y + 11, v: D.posEmb[i] });
        items.push({ k: 'fr' + i, t: 'text', x: 528, y: y + 28, anchor: 'middle', size: 18, bold: true, tone: 'muted', text: '=' });
        items.push({ k: 'fx' + i, t: 'vec', x: 550, y: y + 11, v: D.x[i] });
        items.push({ k: 'fz' + i, t: 'text', x: 745, y: y + 28, size: 13.5, tone: 'muted', text: t({ en: `position ${i}`, zh: `位置 ${i}` }) });
      });
    }

    if (step === 5) {
      items.push({ k: 'ah', t: 'text', x: 90, y: 112, size: 13.5, tone: 'muted', text: t({ en: 'row = the letter looking · column = the letter looked at', zh: '列 = 正在看的字母 · 欄 = 被看的字母' }) });
      items.push({ k: 'heat', t: 'heat', x: 110, y: 150, m: D.att, rows: letters, cols: letters, fmt: 'pct', hi: letters.length - 1 });
      const vx = 470;
      items.push({ k: 'v_x', t: 'vec', x: vx, y: 160, v: D.x3, cell: 14, label: t({ en: `“${letters[3]}” going in`, zh: `進來時的「${letters[3]}」` }) });
      items.push({ k: 'v_a', t: 'vec', x: vx, y: 240, v: D.attOut, cell: 14, label: t({ en: '+ attention output (after o)', zh: '+ 注意力輸出（經過 o）' }) });
      items.push({ k: 'v_1', t: 'vec', x: vx, y: 320, v: D.x1, cell: 14, label: t({ en: '= new vector in the residual stream', zh: '= 殘差流裡的新向量' }) });
      items.push({ k: 'va1', t: 'arrow', x1: vx + 112, y1: 190, x2: vx + 112, y2: 220 });
      items.push({ k: 'va2', t: 'arrow', x1: vx + 112, y1: 270, x2: vx + 112, y2: 300 });
      items.push({ k: 'alook', t: 'text', x: 110, y: 400, size: 15, bold: true, tone: 'a', text: t({ en: `“${letters[3]}” puts ${pct(D.look.v)} of its attention on “${letters[D.look.at]}”`, zh: `「${letters[3]}」把 ${pct(D.look.v)} 的注意力放在「${letters[D.look.at]}」上` }) });
    }

    if (step === 6) {
      const vx = 120;
      items.push({ k: 'v_1', t: 'vec', x: vx, y: 130, v: D.x1, cell: 14, label: t({ en: `“${letters[3]}” after attention · 16`, zh: `注意力之後的「${letters[3]}」· 16` }) });
      items.push({ k: 'm_h', t: 'vec', x: vx, y: 220, v: D.hidden, cell: 14, label: t({ en: '32 neurons after w1, b1 and ReLU (negatives → 0)', zh: '經過 w1、b1 與 ReLU 的 32 個神經元（負數 → 0）' }) });
      items.push({ k: 'm_o', t: 'vec', x: vx, y: 310, v: D.mlpOut, cell: 14, label: t({ en: 'MLP output after w2, b2 · 16', zh: '經過 w2、b2 的 MLP 輸出 · 16' }) });
      items.push({ k: 'm_x', t: 'vec', x: vx, y: 400, v: D.x2, cell: 14, label: t({ en: '= vector after the block (added to the stream)', zh: '= 區塊之後的向量（加回殘差流）' }) });
      [[160, 190], [250, 280], [340, 370]].forEach(([a, b], j) => items.push({ k: 'ma' + j, t: 'arrow', x1: vx + 112, y1: a, x2: vx + 112, y2: b }));
      items.push({ k: 'm_n', t: 'text', x: 610, y: 245, size: 24, bold: true, tone: 'a', text: `${D.active} / 32` });
      items.push({ k: 'm_nl', t: 'text', x: 610, y: 270, size: 13.5, tone: 'muted', text: t({ en: 'neurons switched on for this letter', zh: '這個字母打開的神經元' }) });
    }

    if (step === 7) {
      const tp = top();
      items.push({ k: 'm_x', t: 'vec', x: 40, y: 130, v: D.x2, label: t({ en: 'last vector · 16', zh: '最後的向量 · 16' }) });
      items.push({ k: 'oa', t: 'arrow', x1: 230, y1: 141, x2: 290, y2: 141 });
      items.push({ k: 'ou', t: 'text', x: 260, y: 128, anchor: 'middle', size: 13, mono: true, tone: 'muted', text: '× u + bu' });
      items.push({ k: 'lg', t: 'vec', x: 300, y: 130, v: D.logits, label: t({ en: '27 logits, one per token ( . a b … z )', zh: '27 個 logits，每個詞元一個（ . a b … z ）' }) });
      const order = D.probs.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, 10);
      items.push({
        k: 'pb', t: 'bars', x: 60, y: 225, w: 560, h: 210,
        title: t({ en: 'after softmax: the 10 likeliest next letters', zh: 'softmax 之後：最可能的 10 個下一個字母' }),
        items: order.map((o) => ({ label: CHARS[o.i], v: o.v, hi: o.i === CHARS.indexOf(tp.ch) })),
      });
      letters.forEach((ch, i) => items.push({ k: 'pt' + i, t: 'token', x: 660 + i * 50, y: 240, text: ch, tone: ch === '.' ? 'b' : 'ink' }));
      items.push({ k: 'pn', t: 'token', x: 680, y: 320, text: tp.ch, tone: 'a', sub: t({ en: `next · ${pct(tp.p)}`, zh: `下一個 · ${pct(tp.p)}` }) });
      items.push({ k: 'pna', t: 'arrow', x1: 755, y1: 290, x2: 726, y2: 318, tone: 'a' });
    }

    if (step === 8) {
      const name = '.' + D.longest;
      [...name].forEach((ch, i) => {
        const y = 86 + i * 52;
        items.push({ k: 'ct' + i, t: 'token', x: 40, y, text: ch, tone: ch === '.' ? 'b' : 'ink' });
        items.push({ k: 'cp' + i, t: 'vec', x: 110, y: y + 11, v: D.pos[i] });
        items.push({ k: 'cl' + i, t: 'text', x: 300, y: y + 28, size: 13.5, tone: 'muted', text: t({ en: `pos row ${i}`, zh: `pos 第 ${i} 列` }) });
      });
      items.push({ k: 'cx', t: 'box', x: 470, y: 300, w: 380, h: 64, tone: 'bad', label: t({ en: 'a 9th token: no row 8 in pos', zh: '第 9 個詞元：pos 沒有第 8 列' }), sub: t({ en: 'the model has nowhere to put it', zh: '模型沒有地方放它' }) });
      items.push({ k: 'cxa', t: 'arrow', x1: 390, y1: 470, x2: 470, y2: 345, tone: 'bad', dashed: true });
      items.push({ k: 'ch', t: 'text', x: 470, y: 130, size: 15, bold: true, tone: 'a', text: t({ en: `longest name in the list: ${D.maxLen} letters`, zh: `清單中最長的名字：${D.maxLen} 個字母` }) });
      items.push({ k: 'ch2', t: 'text', x: 470, y: 156, size: 13.5, tone: 'muted', text: t({ en: `“${D.longest}” plus the start mark = 8 tokens, exactly full`, zh: `「${D.longest}」加上起始符號 = 8 個詞元，剛好塞滿` }) });
    }

    const tp = top();
    return {
      metrics: [
        { label: { en: 'Weights counted', zh: '已計算權重' }, value: n0(step === 9 || step > 3 ? D.total : counted(step)) },
        { label: { en: 'Vector width', zh: '向量寬度' }, value: '16' },
        { label: { en: 'Context', zh: '上下文' }, value: t({ en: '8 tokens', zh: '8 個詞元' }) },
        { label: { en: 'Next letter', zh: '下一個字母' }, value: step >= 7 && step !== 8 ? `${tp.ch} · ${pct(tp.p)}` : '—' },
      ],
      items,
      legend: step >= 4 && step <= 8
        ? [{ tone: 'pos', text: { en: 'positive', zh: '正數' } }, { tone: 'neg', text: { en: 'negative', zh: '負數' } }]
        : undefined,
    };
  },
};

