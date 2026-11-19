import type { Control, Input, Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, NAMES, VAL_NAMES, encode } from '../model/data';
import { Mat, add, addRow, matmul, relu, softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';

// The course model reads ".jimm" and must predict what follows the second m.
// We compare it with ".jiam": same last letter, same position, only the letter before differs.
const model = GPT.fromJSON(weights as Weights);
const WORD = '.jimm';
const OTHER = '.jiam';
const IDS = encode(WORD);
const LABELS = [...WORD];
const F = IDS.length - 1; // the token we follow: the second "m"
const Y = CHARS.indexOf('y');
const tr = model.forward(IDS);
const x = IDS.map((_, i) => tr.x.row(i));
const keys = IDS.map((_, i) => tr.k[0].row(i));
const vals = IDS.map((_, i) => tr.v[0].row(i));
const query = tr.q[0].row(F);
const scores = IDS.map((_, i) => tr.scores[0].row(i));
const att = IDS.map((_, i) => tr.att[0].row(i));
const mixed = tr.mixed[0].row(F);

type Bar = { label: string; v: number; hi?: boolean };
const top = (p: number[], n: number, hi?: string): Bar[] =>
  p.map((v, i) => ({ label: CHARS[i], v })).sort((a, b) => b.v - a.v).slice(0, n).map((b) => ({ ...b, hi: b.label === hi }));

/**
 * Attention for the last token with some earlier tokens hidden: hidden tokens get
 * zero attention and the others are renormalised (exactly like the causal mask does
 * for the future). Then the rest of the course model runs as usual on that token.
 */
interface Masked { att: number[]; probs: number[] }
const maskCache = new Map<string, Masked>();
function attendMasked(ids: number[], visible: boolean[]): Masked {
  const key = ids.join(',') + '|' + visible.map((v) => (v ? 1 : 0)).join('');
  const hit = maskCache.get(key);
  if (hit) return hit;
  const P = model.p;
  const r = model.forward(ids);
  const L = ids.length - 1;
  const sc = r.scores[0].row(L);
  const open = sc.map((_, j) => j === L || visible[j] !== false);
  const mx = Math.max(...sc.filter((_, j) => open[j]));
  const e = sc.map((s, j) => (open[j] ? Math.exp(s - mx) : 0));
  const z = e.reduce((a, b) => a + b, 0);
  const a = e.map((v) => v / z);
  const mix = new Mat(1, P.o.r);
  for (let j = 0; j <= L; j++) { const v = r.v[0].row(j); for (let c = 0; c < v.length; c++) mix.d[c] += a[j] * v[c]; }
  const xL = new Mat(1, P.o.c); xL.d.set(r.x.row(L));
  const x1 = add(xL, matmul(mix, P.o));
  const h = relu(addRow(matmul(x1, P.w1), P.b1));
  const x2 = add(x1, addRow(matmul(h, P.w2), P.b2));
  const out = { att: a, probs: softmaxRow(addRow(matmul(x2, P.u), P.bu).row(0)) };
  if (maskCache.size > 400) maskCache.clear();
  maskCache.set(key, out);
  return out;
}

const pJimm = model.next(IDS);
const pJiam = model.next(encode(OTHER));
const attJiam = model.forward(encode(OTHER)).att[0].row(F);

// What follows "m", and what follows "mm", across the whole name list, ignoring anything earlier.
const follow = (pre: string) => {
  const c = new Array(27).fill(0);
  let n = 0;
  for (const name of NAMES) {
    const s = '.' + name + '.';
    for (let i = pre.length; i < s.length; i++) if (s.slice(i - pre.length, i) === pre) { c[CHARS.indexOf(s[i])]++; n++; }
  }
  return { p: c.map((v) => v / n), n };
};
const afterM = follow('m');
const afterMM = follow('mm');

// How much each token gives to the token just before it, over names the model never trained on.
let habitCache: { prev: number; topPrev: number } | undefined;
const habit = () => habitCache ??= (() => {
  let s = 0, best = 0, n = 0;
  for (const name of VAL_NAMES) {
    const ids = encode('.' + name).slice(0, model.cfg.ctx);
    const r = model.forward(ids);
    for (let i = 2; i < ids.length; i++) {
      const row = r.att[0].row(i).slice(0, i + 1);
      s += row[i - 1]; n++;
      if (row.indexOf(Math.max(...row)) === i - 1) best++;
    }
  }
  return { prev: s / n, topPrev: best / n };
})();

const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const best = att[F].indexOf(Math.max(...att[F]));
const pct = (v: number) => Math.round(v * 100) + '%';
const argmax = (p: number[]) => p.indexOf(Math.max(...p));
const ROW = 64;
const rowY = (i: number) => 120 + i * ROW;
const MAXN = 7; // longest name the course model can read after "."
const q = (s: string) => t({ en: `“${s}”`, zh: `「${s}」` });

// ---------- the "hide a letter" step ----------
const TRY = 9;
// one toggle per earlier position, labelled with its letter; toggles past the typed name are hidden
const tryControls = (raw: string): Control[] => {
  const chars = ['.', ...raw];
  return [
    { key: 'name', kind: 'text', label: { en: 'Name', zh: '名字' }, value: WORD.slice(1), maxLength: MAXN, pattern: 'a-z', placeholder: { en: 'a–z', zh: 'a–z' } },
    ...Array.from({ length: MAXN }, (_, i): Control => {
      const ch = i < chars.length - 1 ? chars[i] : '';
      // a letter that appears twice gets its position too, to tell the boxes apart
      const twice = chars.slice(0, -1).filter((c) => c === ch).length > 1;
      const name = twice ? `${ch} (#${i + 1})` : ch;
      return { key: 'see' + i, kind: 'toggle', label: ch ? { en: `see ${name}`, zh: `看得到 ${name}` } : { en: `see #${i + 1}`, zh: `看得到 #${i + 1}` }, value: i !== F - 1, hidden: !ch };
    }),
  ];
};
const readTry = (input?: Input) => {
  const raw = String(input?.name ?? WORD.slice(1)).toLowerCase().replace(/[^a-z]/g, '').slice(0, MAXN);
  const ids = encode('.' + raw);
  const L = ids.length - 1;
  const visible = Array.from({ length: L + 1 }, (_, j) => j === L || (input?.['see' + j] ?? j !== F - 1) !== false);
  return { raw, ids, L, visible };
};

export const attention: Lesson = {
  id: 'self-attention',
  title: { en: 'Self-Attention, Step by Step', zh: '自注意力，一步一步來' },
  blurb: {
    en: 'How a token looks back at earlier tokens and decides which ones matter.',
    zh: '一個詞元如何回頭看前面的詞元，並決定哪些比較重要。',
  },
  minutes: 14,
  level: 'beginner',
  learn: [
    { text: { en: 'Why a token needs context', zh: '為什麼詞元需要上下文' }, step: 0 },
    { text: { en: 'What queries and keys are for', zh: '查詢與鍵的用途' }, step: 3 },
    { text: { en: 'Why the future is masked', zh: '為什麼要遮住未來' }, step: 5 },
    { text: { en: 'How scores become attention weights', zh: '分數如何變成注意力權重' }, step: 6 },
    { text: { en: 'How values are mixed into a new vector', zh: '值如何混合成新的向量' }, step: 7 },
    { text: { en: 'What hiding a letter from attention does', zh: '把一個字母藏起來，注意力會怎樣' }, step: TRY },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'One letter is not enough', zh: '只看一個字母是不夠的' },
      body: [
        { en: 'Our model is writing a name. So far it has <code>.jimm</code>, and it must choose the next letter.', zh: '我們的模型正在寫一個名字。目前寫到 <code>.jimm</code>，它必須選出下一個字母。' },
        { en: `If it looked only at the last letter, the best it could use is the left chart: across all ${NAMES.length.toLocaleString('en-US')} names, “m” is mostly followed by a, i or e, and “y” only ${pct(afterM.p[Y])} of the time. After a double “mm” (right chart, only ${afterMM.n} cases) “y” comes first, as in jimmy or tommy. Only the letter <i>before</i> the m reveals that.`, zh: `如果只看最後一個字母，它頂多只能參考左邊那張圖：在全部 ${NAMES.length.toLocaleString('en-US')} 個名字裡，「m」後面多半接 a、i 或 e，接「y」只佔 ${pct(afterM.p[Y])}。但在連續兩個「mm」之後（右圖，只有 ${afterMM.n} 次），排第一的是「y」，就像 jimmy 或 tommy。只有 m <i>前面</i>的那個字母能透露這件事。` },
      ],
      callout: { kind: 'insight', text: { en: 'The meaning of a token depends on its context. Attention is the mechanism that lets each token gather information from the others.', zh: '一個詞元的意義取決於它的上下文。注意力（attention）就是讓每個詞元從其他詞元收集資訊的機制。' } },
    },
    {
      short: { en: 'Vectors', zh: '向量' }, chapter: 'technique',
      title: { en: 'Every token is a list of numbers', zh: '每個詞元都是一串數字' },
      body: [
        { en: 'Inside the model each token is a <b>vector</b>: here, 16 numbers. Orange cells are positive, blue cells negative, and stronger colour means a larger number.', zh: '在模型內部，每個詞元都是一個<b>向量</b>：在這裡是 16 個數字。橘色格子是正數，藍色格子是負數，顏色越深代表數值越大。' },
        { en: 'The vector encodes which letter it is <i>and</i> where it sits. That is why the two m’s have different vectors. (The Embeddings lesson showed the letter part; the Positional Information lesson, two lessons from now, shows how the position part is added.)', zh: '這個向量同時記錄了「是哪個字母」<i>以及</i>「在第幾個位置」，所以兩個 m 的向量並不相同。（「嵌入」那一課講過字母的部分；位置的部分怎麼加進來，兩課之後的「位置資訊」會說明。）' },
      ],
      changed: [{ sign: '+', text: { en: `${IDS.length} vectors of 16 numbers`, zh: `${IDS.length} 個向量，各 16 個數字` } }],
    },
    {
      short: { en: 'Query', zh: '查詢' }, chapter: 'technique',
      title: { en: 'The token asks a question', zh: '詞元提出一個問題' },
      body: [
        { en: 'We follow the last token, the second “m”. It needs to find out what came before it.', zh: '我們跟著最後一個詞元，也就是第二個「m」。它需要知道自己前面有什麼。' },
        { en: 'Its vector is multiplied by a learned 16 × 16 matrix to produce a <b>query</b>: “this is what I am looking for”. (The same move as the output matrix <code>u</code> in the last lesson: multiply cell by cell and add up, once per column, so 16 columns give 16 new numbers.)', zh: '它的向量乘上一個學習得來的 16 × 16 矩陣，產生一個<b>查詢（query）</b>：「我在找這樣的東西」。（和上一課的輸出矩陣 <code>u</code> 是同一個動作：每一行都逐格相乘再加總，16 行就得到 16 個新數字。）' },
      ],
      callout: { kind: 'note', text: { en: 'Nobody writes down what the query means. The matrix starts random and training shapes it.', zh: '沒有人規定查詢代表什麼。這個矩陣一開始是隨機的，由訓練慢慢塑造出來。' } },
      changed: [{ sign: '+', text: { en: 'Query vector for the second “m”', zh: '第二個「m」的查詢向量' } }],
    },
    {
      short: { en: 'Keys', zh: '鍵' }, chapter: 'technique',
      title: { en: 'Every token offers a key', zh: '每個詞元都提供一把鍵' },
      body: [
        { en: 'Each token, including the “m” itself, is multiplied by a second learned matrix to produce a <b>key</b>: “this is what I contain”.', zh: '每個詞元（包括「m」自己）都會乘上第二個學習得來的矩陣，產生一把<b>鍵（key）</b>：「我這裡有這樣的東西」。' },
        { en: 'A query is a question; keys are the labels it gets compared against. What a token is looking for is not the same as what it offers, so each gets its own matrix.', zh: '查詢是一個問題，而鍵就是用來和它比對的標籤。一個詞元「想找的」和它「能提供的」並不一樣，所以兩者各有自己的矩陣。' },
      ],
      changed: [{ sign: '+', text: { en: `${IDS.length} key vectors`, zh: `${IDS.length} 個鍵向量` } }],
    },
    {
      short: { en: 'Scores', zh: '分數' }, chapter: 'technique',
      title: { en: 'Compare the query with each key', zh: '把查詢和每把鍵比對' },
      body: [
        { en: 'To compare two vectors we take their <b>dot product</b>: multiply them cell by cell and add everything up. It is the cosine similarity from the Embeddings lesson without dividing by the lengths: vectors that point the same way score high, and longer vectors score higher too.', zh: '要比較兩個向量，我們計算它們的<b>內積（dot product）</b>：把對應的格子兩兩相乘，再全部加起來。它就是「嵌入」那一課的餘弦相似度，只是沒有除以長度：方向越一致分數越高，向量越長分數也越高。' },
        { en: 'That gives one <b>score</b> per token: how well its key answers the query. (Each score is also divided by √16 = 4 to keep the numbers small.)', zh: '這樣每個詞元都得到一個<b>分數</b>：它的鍵有多符合這個查詢。（每個分數還會除以 √16 = 4，讓數值不要太大。）' },
      ],
      changed: [{ sign: '+', text: { en: 'One score per token', zh: '每個詞元一個分數' } }],
    },
    {
      short: { en: 'Mask', zh: '遮罩' }, chapter: 'technique',
      title: { en: 'No peeking at the future', zh: '不可以偷看未來' },
      body: [
        { en: 'Every token does this at once, so we get a whole grid of scores: one row per token asking, one column per token being looked at.', zh: '其實每個詞元都同時在做這件事，所以我們得到一整張分數表：每一列是發問的詞元，每一行是被看的詞元。' },
        { en: 'The upper-right half is crossed out. A token may only look at itself and at earlier tokens, because when the model writes, the later tokens do not exist yet.', zh: '右上半邊被劃掉了。一個詞元只能看自己和前面的詞元，因為模型在寫字時，後面的詞元根本還不存在。' },
      ],
      callout: { kind: 'insight', text: { en: 'This is the causal mask. Without it the model could copy the answer during training and would learn nothing.', zh: '這就是因果遮罩（causal mask）。沒有它，模型在訓練時就能直接抄答案，什麼也學不到。' } },
      changed: [
        { sign: '+', text: { en: `${IDS.length} × ${IDS.length} grid of scores`, zh: `${IDS.length} × ${IDS.length} 的分數表` } },
        { sign: '~', text: { en: 'Future positions masked', zh: '未來的位置被遮住' } },
      ],
    },
    {
      short: { en: 'Weights', zh: '權重' }, chapter: 'technique',
      title: { en: 'Turn scores into shares of attention', zh: '把分數變成注意力的比例' },
      body: [
        { en: '<b>Softmax</b> turns each row of scores into positive numbers that add up to 100%. Higher scores get a bigger share. (The first row is 100% because “.” can only see itself.)', zh: '<b>Softmax</b> 把每一列的分數轉成加起來等於 100% 的正數，分數越高，分到的比例越大。（第一列是 100%，因為「.」只看得到自己。）' },
        { en: `Read the highlighted row: the second “m” gives ${pct(att[F][best])} of its attention to the letter just before it, the first “m”, and spreads the rest over the others.`, zh: `看被框起來的那一列：第二個「m」把 ${pct(att[F][best])} 的注意力給了緊鄰在它前面的字母，也就是第一個「m」，其餘的分給其他詞元。` },
      ],
      callout: {
        kind: 'insight', text: lz(() => {
          const h = habit();
          return { en: `This head has a habit: it mostly looks at the letter just before. On the ${VAL_NAMES.length} names the model never trained on, a letter gives the previous token ${pct(h.prev)} of its attention on average, and that is its biggest share ${pct(h.topPrev)} of the time.`, zh: `這個頭有個習慣：它主要看緊鄰在前的字母。在模型訓練時從沒看過的 ${VAL_NAMES.length} 個名字上，一個字母平均把 ${pct(h.prev)} 的注意力給前一個詞元，而且有 ${pct(h.topPrev)} 的時候那是它給得最多的。` };
        }),
      },
      changed: [{ sign: '~', text: { en: 'Scores → attention weights', zh: '分數 → 注意力權重' } }],
    },
    {
      short: { en: 'Values', zh: '值' }, chapter: 'solution',
      title: { en: 'Mix the values', zh: '把值混合起來' },
      body: [
        { en: 'Each token also has a <b>value</b> vector, made with a third learned matrix: the information it actually hands over.', zh: '每個詞元還有一個<b>值（value）</b>向量，由第三個學習得來的矩陣產生：這才是它真正交出去的資訊。' },
        { en: `The second “m” scales every value by its attention weight and adds them up. The biggest part, ${pct(att[F][best])}, comes from the first “m”, so what flows in is roughly “the letter before me was m”. A double m is a strong hint for “y”.`, zh: `第二個「m」把每個值乘上對應的注意力權重，再全部加起來。最大的一份（${pct(att[F][best])}）來自第一個「m」，所以流進來的資訊大致是「我前面的字母是 m」。連續兩個 m，正是「y」的強烈暗示。` },
      ],
      callout: { kind: 'note', text: { en: 'This mix passes through a fourth learned matrix and is then added back onto the token’s own vector, so the token keeps what it knew and gains context. (Why <i>added</i>: see The Residual Stream.)', zh: '這個混合結果還會通過第四個學習得來的矩陣，然後加回詞元原本的向量上，讓詞元既保留原本知道的，又多了上下文。（為什麼是「加上去」：見「殘差流」那一課。）' } },
      changed: [{ sign: '+', text: { en: 'Context vector for the second “m”', zh: '第二個「m」的上下文向量' } }],
    },
    {
      short: { en: 'Result', zh: '結果' }, chapter: 'solution',
      title: { en: 'Same letter, same place, different answer', zh: '同一個字母、同一個位置，不同的答案' },
      body: [
        { en: '<code>.jimm</code> and <code>.jiam</code> end in the same letter at the same position. The only difference is the letter just before it, and only attention can see that letter: everything else in our model works on each token alone.', zh: '<code>.jimm</code> 和 <code>.jiam</code> 的最後一個字母相同，位置也相同，唯一的差別是它前面那個字母。而能看到那個字母的只有注意力：模型的其他部分都是一個詞元一個詞元各自處理。' },
        { en: `With an “m” before it, the model says “y” ${pct(pJimm[Y])}, as in jimmy. With an “a” before it, “y” gets only ${pct(pJiam[Y])} and the top answer is “${CHARS[argmax(pJiam)]}” ${pct(Math.max(...pJiam))}. The attention row (under each chip) shows where the context came from.`, zh: `前面是「m」時，模型以 ${pct(pJimm[Y])} 選「y」，就像 jimmy；前面是「a」時，「y」只剩 ${pct(pJiam[Y])}，第一名變成「${CHARS[argmax(pJiam)]}」${pct(Math.max(...pJiam))}。每個字塊下方的注意力比例，顯示上下文是從哪裡來的。` },
      ],
      callout: { kind: 'note', text: { en: 'Position matters too: after <code>.jim</code>, where the m sits one place earlier, the model leans to other letters. The Positional Information lesson shows why.', zh: '位置也有影響：在 <code>.jim</code> 之後（m 早一個位置），模型會偏向別的字母。原因會在「位置資訊」那一課說明。' } },
      changed: [{ sign: '~', text: { en: 'Same last letter, different letter before it', zh: '最後一個字母相同，前一個字母不同' } }],
    },
    {
      short: { en: 'Try it', zh: '動手試試' }, chapter: 'solution',
      title: { en: 'Hide a letter from attention', zh: '把一個字母藏起來，不讓注意力看' },
      body: [
        { en: 'Try it: untick a letter to hide it from the last letter’s attention, or type another name. A hidden token gets 0%, the others share its attention, and the rest of the real model runs as usual. (The last letter always sees itself, so it has no box.)', zh: '動手試試：取消勾選某個字母，就能把它藏起來，不讓最後一個字母的注意力看到；也可以輸入別的名字。被藏起來的詞元分到 0%，它的注意力由其他詞元重新分配，模型其餘部分照常計算。（最後一個字母永遠看得到自己，所以它沒有勾選框。）' },
        { en: 'It starts with the first m of <code>jimm</code> hidden: compare the two charts. Then tick it again, try hiding “.” instead, or type <code>jiam</code> or <code>tomm</code> and see whether the letter just before the last one matters most.', zh: '一開始，<code>jimm</code> 的第一個 m 被藏起來了：比較兩張圖。接著把它勾回來，改藏「.」看看，或輸入 <code>jiam</code>、<code>tomm</code>，看看是不是最後一個字母的前一個字母最重要。' },
      ],
      changed: [{ sign: '+', text: { en: 'Your own name and hidden letters', zh: '自己的名字和藏起來的字母' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Attention in one line', zh: '一句話說完注意力' },
      body: [
        { en: 'Each token asks a question (<b>query</b>), compares it with every earlier token’s label (<b>key</b>), turns the scores into shares (<b>softmax</b>), and collects a weighted mix of what they offer (<b>values</b>).', zh: '每個詞元提出一個問題（<b>查詢</b>），和前面每個詞元的標籤（<b>鍵</b>）比對，把分數變成比例（<b>softmax</b>），再依比例收集它們提供的資訊（<b>值</b>）。' },
        { en: 'Three learned matrices (query, key, value) do the looking, and a fourth mixes the result back in. Nothing is hand-written. Our head learned to read mostly the letter just before, which is how <code>.jimm</code> knows it has a double m.', zh: '三個學習得來的矩陣（查詢、鍵、值）負責「看」，第四個把結果混回去，沒有任何一條規則是人寫的。我們這個頭學會主要看前一個字母，<code>.jimm</code> 就是這樣知道自己有兩個 m。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Multi-Head Attention. One question per token is limiting, so real models ask several at once.', zh: '下一課：多頭注意力。每個詞元只問一個問題太受限了，所以真實的模型會同時問好幾個。' } },
    },
  ],

  scene(step, input) {
    const items: Item[] = [];
    const column = step >= 1 && step <= 7;
    const grid = step === 5 || step === 6;
    const tokX = grid ? 296 : 50;

    // the input tokens are on screen until the comparison
    if (step <= 7)
      LABELS.forEach((ch, i) => {
        items.push({
          k: 'tok' + i, t: 'token', text: ch,
          x: column ? tokX : 300 + i * 62, y: column ? rowY(i) : 40,
          tone: i === F && step >= 2 ? 'a' : ch === '.' ? 'b' : 'ink',
        });
      });

    if (step === 0) {
      items.push({ k: 'next', t: 'token', x: 300 + IDS.length * 62, y: 40, text: '?', tone: 'a' });
      const yb = top(afterM.p, 7, 'y');
      items.push({ k: 'data', t: 'bars', x: 40, y: 180, w: 480, h: 250, items: yb, title: t({ en: `what follows “m” (${afterM.n} times in the list)`, zh: `「m」後面接什麼（清單裡共 ${afterM.n} 次）` }) });
      items.push({ k: 'data2', t: 'bars', x: 580, y: 180, w: 280, h: 250, items: top(afterMM.p, 4, 'y'), title: t({ en: `what follows “mm” (${afterMM.n} times)`, zh: `「mm」後面接什麼（${afterMM.n} 次）` }) });
      items.push({ k: 'd0n', t: 'text', x: 450, y: 480, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'counted over every name in the list, ignoring everything before', zh: '統計清單裡的每個名字，不管更前面是什麼' }) });
    }

    const strips = step >= 1 && step <= 4;
    if (strips) {
      items.push({ k: 'hx', t: 'text', x: 130, y: rowY(0) - 24, text: t({ en: 'token vector', zh: '詞元向量' }), size: 14, tone: 'muted' });
      x.forEach((v, i) => items.push({ k: 'x' + i, t: 'vec', x: 130, y: rowY(i) + 11, v, dim: step === 2 && i !== F }));
    }
    if (step >= 2 && step <= 4) {
      items.push({ k: 'hq', t: 'text', x: 660, y: rowY(F) - 8, text: t({ en: 'query', zh: '查詢' }), size: 14, tone: 'a', bold: true });
      items.push({ k: 'q', t: 'vec', x: 660, y: rowY(F) + 11, v: query });
      if (step === 2) {
        items.push({ k: 'qa', t: 'arrow', x1: 314, y1: rowY(F) + 22, x2: 652, y2: rowY(F) + 22, tone: 'a' });
        items.push({ k: 'qm', t: 'text', x: 483, y: rowY(F) + 12, text: t({ en: '× query matrix (16 × 16)', zh: '× 查詢矩陣（16 × 16）' }), size: 14, tone: 'a', anchor: 'middle' });
      }
    }
    if (step === 3 || step === 4) {
      items.push({ k: 'hk', t: 'text', x: 370, y: rowY(0) - 24, text: t({ en: 'key', zh: '鍵' }), size: 14, tone: 'muted' });
      keys.forEach((v, i) => {
        items.push({ k: 'k' + i, t: 'vec', x: 370, y: rowY(i) + 11, v });
        if (step === 3) items.push({ k: 'ka' + i, t: 'arrow', x1: 314, y1: rowY(i) + 22, x2: 362, y2: rowY(i) + 22 });
      });
      if (step === 3) items.push({ k: 'km', t: 'text', x: 370, y: rowY(F) + 64, text: t({ en: '× key matrix (a second 16 × 16)', zh: '× 鍵矩陣（第二個 16 × 16）' }), size: 14, tone: 'muted' });
    }
    if (step === 4) {
      items.push({ k: 'hs', t: 'text', x: 600, y: rowY(0) - 24, text: t({ en: 'score', zh: '分數' }), size: 14, tone: 'muted', anchor: 'middle' });
      scores[F].forEach((s, i) => items.push({ k: 's' + i, t: 'text', x: 600, y: rowY(i) + 28, text: s.toFixed(2), size: 17, anchor: 'middle', mono: true, bold: true, tone: i === best ? 'a' : 'ink' }));
      items.push({ k: 'sn', t: 'text', x: 600, y: rowY(F) + 80, text: t({ en: 'score = query · key ÷ 4', zh: '分數 = 查詢 · 鍵 ÷ 4' }), size: 14, tone: 'muted', anchor: 'middle' });
    }

    if (grid) {
      const m = IDS.map((_, i) => IDS.map((_, j) => (j > i ? null : step === 5 ? scores[i][j] : att[i][j])));
      items.push({
        k: 'grid', t: 'heat', x: 360, y: rowY(0) - 10, cell: ROW, m, rows: LABELS.map(() => ''), cols: LABELS,
        fmt: step === 5 ? 'num' : 'pct', hi: step === 6 ? F : undefined,
        max: step === 5 ? Math.max(...scores.flat().map(Math.abs)) : 1,
      });
      items.push({ k: 'gl1', t: 'text', x: 360 + ROW * 2, y: rowY(0) - 44, text: t({ en: 'token being looked at →', zh: '被看的詞元 →' }), size: 14, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'gl2', t: 'text', x: 280, y: rowY(4) + 16, text: t({ en: '↑ token asking', zh: '↑ 發問的詞元' }), size: 14, tone: 'muted', anchor: 'end' });
    }

    if (step === 7) {
      items.push({ k: 'hw', t: 'text', x: 150, y: rowY(0) - 24, text: t({ en: 'weight', zh: '權重' }), size: 14, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'hv', t: 'text', x: 210, y: rowY(0) - 24, text: t({ en: 'value', zh: '值' }), size: 14, tone: 'muted' });
      vals.forEach((v, i) => {
        items.push({ k: 'w' + i, t: 'text', x: 150, y: rowY(i) + 28, text: pct(att[F][i]), size: 16, anchor: 'middle', mono: true, bold: true, tone: i === best ? 'a' : 'ink' });
        items.push({ k: 'v' + i, t: 'vec', x: 210, y: rowY(i) + 11, v });
        items.push({ k: 'va' + i, t: 'arrow', x1: 394, y1: rowY(i) + 22, x2: 592, y2: rowY(F / 2) + 22, tone: i === best ? 'a' : 'muted' });
      });
      items.push({ k: 'hm', t: 'text', x: 600, y: rowY(F / 2) - 8, text: t({ en: 'weighted mix = context', zh: '加權混合 = 上下文' }), size: 14, tone: 'a', bold: true });
      items.push({ k: 'mix', t: 'vec', x: 600, y: rowY(F / 2) + 11, v: mixed });
      items.push({ k: 'mixn', t: 'text', x: 600, y: rowY(F / 2) + 64, text: t({ en: `mostly the first “m” (${pct(att[F][best])})`, zh: `大部分來自第一個「m」（${pct(att[F][best])}）` }), size: 14, tone: 'muted' });
    }

    if (step === 8) {
      const rows = [
        { w: WORD, a: att[F], p: pJimm, y: 40, key: 'tok' },
        { w: OTHER, a: attJiam, p: pJiam, y: 290, key: 'tj' },
      ];
      rows.forEach((r, ri) => {
        items.push({ k: 'cw' + ri, t: 'text', x: 40, y: r.y + 4, size: 15, bold: true, mono: true, tone: ri === 0 ? 'a' : 'b', text: r.w });
        items.push({ k: 'cwl' + ri, t: 'text', x: 40, y: r.y + 128, size: 13, tone: 'muted', text: t({ en: 'attention of the last “m”', zh: '最後一個「m」的注意力' }) });
        [...r.w].forEach((ch, i) => items.push({
          k: r.key + i, t: 'token', text: ch, x: 40 + i * 56, y: r.y + 40,
          sub: pct(r.a[i]), tone: i === F ? 'a' : ch === '.' ? 'b' : 'ink',
        }));
        items.push({ k: 'cp' + ri, t: 'bars', x: 360, y: r.y + 20, w: 500, h: 180, items: top(r.p, 6, 'y'), title: t({ en: `model’s prediction after “${r.w}”`, zh: `模型在「${r.w}」之後的預測` }) });
      });
    }

    let tryMetrics: { label: L; value: string }[] | undefined;
    let controls: Control[] | undefined;
    if (step === TRY) {
      // the first pass (no input yet) lists every control so their values are kept
      const { raw, ids, L, visible } = readTry(input);
      controls = tryControls(raw);
      const chipX = (i: number) => 450 - ((L + 1) * 62 - 18) / 2 + i * 62;
      items.push({ k: 'trl1', t: 'text', x: 20, y: 110, size: 13, tone: 'muted', text: t({ en: 'attention now', zh: '現在的注意力' }) });
      items.push({ k: 'trl2', t: 'text', x: 20, y: 130, size: 13, tone: 'muted', text: t({ en: '(all visible)', zh: '（全部看得到）' }) });
      if (!raw) {
        items.push({ k: 'tok0', t: 'token', text: '.', x: chipX(0), y: 50, tone: 'b' });
        items.push({ k: 'thint', t: 'text', x: 450, y: 260, anchor: 'middle', size: 17, tone: 'muted', text: t({ en: 'type a name (letters a–z) in the box above', zh: '請在上方的方框輸入一個名字（字母 a–z）' }) });
        tryMetrics = [
          { label: { en: 'Name', zh: '名字' }, value: '.' },
          { label: { en: 'Hidden', zh: '藏起來的' }, value: '—' },
          { label: { en: 'Top, all visible', zh: '全部看得到時的第一名' }, value: '—' },
          { label: { en: 'Top, now', zh: '現在的第一名' }, value: '—' },
        ];
      } else {
        const full = attendMasked(ids, ids.map(() => true));
        const now = attendMasked(ids, visible);
        const chars = ['.', ...raw];
        chars.forEach((ch, i) => {
          const hidden = !visible[i];
          items.push({ k: 'tok' + i, t: 'token', text: ch, x: chipX(i), y: 50, sub: pct(now.att[i]), tone: i === L ? 'a' : hidden ? 'bad' : ch === '.' ? 'b' : 'ink', dim: hidden });
          items.push({ k: 'tn' + i, t: 'text', x: chipX(i) + 22, y: 38, anchor: 'middle', size: 13, tone: hidden ? 'bad' : 'muted', mono: true, text: i === L ? t({ en: 'last', zh: '最後' }) : hidden ? `#${i + 1} ✕` : `#${i + 1}` });
          items.push({ k: 'tw' + i, t: 'text', x: chipX(i) + 22, y: 128, anchor: 'middle', size: 13, tone: 'muted', mono: true, text: `(${pct(full.att[i])})` });
        });
        const ref = argmax(full.probs);
        const hiL = CHARS[ref];
        items.push({ k: 'tb1', t: 'bars', x: 40, y: 200, w: 390, h: 230, items: top(full.probs, 6, hiL), title: t({ en: 'prediction, all letters visible', zh: '預測：全部字母都看得到' }) });
        items.push({ k: 'tb2', t: 'bars', x: 470, y: 200, w: 390, h: 230, items: top(now.probs, 6, hiL), title: t({ en: 'prediction, with your hidden letters', zh: '預測：藏起你選的字母' }) });
        const nHidden = visible.filter((v) => !v).length;
        items.push({ k: 'tsum', t: 'text', x: 450, y: 480, anchor: 'middle', size: 15, tone: nHidden ? 'a' : 'muted', bold: !!nHidden, text: nHidden
          ? t({ en: `“${hiL}”: ${pct(full.probs[ref])} with everything visible → ${pct(now.probs[ref])} now`, zh: `「${hiL}」：全部看得到時 ${pct(full.probs[ref])} → 現在 ${pct(now.probs[ref])}` })
          : t({ en: 'nothing hidden: both charts are the same', zh: '沒有藏任何字母：兩張圖一樣' }) });
        const tn = argmax(now.probs);
        tryMetrics = [
          { label: { en: 'Name', zh: '名字' }, value: '.' + raw },
          { label: { en: 'Hidden', zh: '藏起來的' }, value: String(nHidden) },
          { label: { en: 'Top, all visible', zh: '全部看得到時的第一名' }, value: `${q(hiL)} ${pct(full.probs[ref])}` },
          { label: { en: 'Top, now', zh: '現在的第一名' }, value: `${q(CHARS[tn])} ${pct(now.probs[tn])}` },
        ];
      }
    }

    if (step === 10) {
      const boxes = [
        { en: 'token vectors', zh: '詞元向量' }, { en: 'query · key', zh: '查詢 · 鍵' }, { en: 'mask future', zh: '遮住未來' },
        { en: 'softmax', zh: 'softmax' }, { en: 'mix values', zh: '混合值' }, { en: '× 4th matrix, add back', zh: '× 第四個矩陣，加回去' },
      ];
      boxes.forEach((b, i) => {
        const bx = 120 + (i % 3) * 230, by = 150 + Math.floor(i / 3) * 130;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: by, w: 190, h: 56, label: t(b), tone: i === 5 ? 'a' : 'ink' });
        if (i % 3 < 2) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 194, y1: by + 28, x2: bx + 226, y2: by + 28 });
      });
      items.push({ k: 'rbd', t: 'arrow', x1: 675, y1: 210, x2: 215, y2: 276, dashed: true });
    }

    const legend: { tone: 'pos' | 'neg' | 'heat' | 'a' | 'b' | 'bad'; text: L }[] | undefined =
      step >= 1 && step <= 5 || step === 7 ? [{ tone: 'pos', text: { en: 'positive number', zh: '正數' } }, { tone: 'neg', text: { en: 'negative number', zh: '負數' } }]
        : step === 6 ? [{ tone: 'heat', text: { en: 'attention weight', zh: '注意力權重' } }]
          : step === 8 ? [{ tone: 'a', text: { en: 'the token we follow', zh: '追蹤的詞元' } }, { tone: 'b', text: { en: 'start mark', zh: '開頭記號' } }]
            : step === TRY ? [{ tone: 'a', text: { en: 'last letter', zh: '最後一個字母' } }, { tone: 'bad', text: { en: 'hidden from attention', zh: '藏起來，注意力看不到' } }]
              : undefined;

    return {
      metrics: tryMetrics ?? [
        { label: { en: 'Tokens in view', zh: '可見詞元' }, value: String(IDS.length) },
        { label: { en: 'Following', zh: '追蹤對象' }, value: step >= 2 ? t({ en: 'second “m”', zh: '第二個「m」' }) : '—' },
        { label: { en: 'Strongest attention', zh: '最強的注意力' }, value: step >= 6 ? `${q(LABELS[best])} ${pct(att[F][best])}` : '—' },
        { label: { en: '“y” after .jimm', zh: '「.jimm」後接「y」' }, value: step >= 8 ? pct(pJimm[Y]) : '—' },
        { label: { en: '“y” after .jiam', zh: '「.jiam」後接「y」' }, value: step >= 8 ? pct(pJiam[Y]) : '—' },
      ],
      items,
      legend,
      controls,
    };
  },
};
