import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, NAMES, encode } from '../model/data';
import weights from '../model/weights.json';

// The trained model reads ".jimm" and must predict what follows the second m.
const model = GPT.fromJSON(weights as Weights);
const IDS = encode('.jimm');
const LABELS = IDS.map((i) => CHARS[i]);
const F = IDS.length - 1; // the token we follow: the second "m"
const tr = model.forward(IDS);
const x = IDS.map((_, i) => tr.x.row(i));
const keys = IDS.map((_, i) => tr.k[0].row(i));
const vals = IDS.map((_, i) => tr.v[0].row(i));
const query = tr.q[0].row(F);
const scores = IDS.map((_, i) => tr.scores[0].row(i));
const att = IDS.map((_, i) => tr.att[0].row(i));
const mixed = tr.mixed[0].row(F);

const top = (p: number[], n: number, hi?: string) =>
  p.map((v, i) => ({ label: CHARS[i], v })).sort((a, b) => b.v - a.v).slice(0, n).map((b) => ({ ...b, hi: b.label === hi }));

const afterFirstM = top(model.next(IDS.slice(0, -1)), 6);
const afterSecondM = top(model.next(IDS), 6);

// What follows "m" across the whole training list, ignoring context.
const counts = new Array(27).fill(0);
let total = 0;
for (const n of NAMES) {
  const s = n + '.';
  for (let i = 0; i < n.length; i++) if (s[i] === 'm') { counts[CHARS.indexOf(s[i + 1])]++; total++; }
}
const afterAnyM = top(counts.map((c) => c / total), 6);

const best = att[F].indexOf(Math.max(...att[F]));
const pct = (v: number) => Math.round(v * 100) + '%';
const ROW = 64;
const rowY = (i: number) => 120 + i * ROW;

export const attention: Lesson = {
  id: 'self-attention',
  title: { en: 'Self-Attention, Step by Step', zh: '自注意力，一步一步來' },
  blurb: {
    en: 'How a token looks back at earlier tokens and decides which ones matter.',
    zh: '一個詞元如何回頭看前面的詞元，並決定哪些比較重要。',
  },
  minutes: 12,
  level: 'beginner',
  learn: [
    { text: { en: 'Why a token needs context', zh: '為什麼詞元需要上下文' }, step: 0 },
    { text: { en: 'What queries and keys are for', zh: '查詢與鍵的用途' }, step: 3 },
    { text: { en: 'How scores become attention weights', zh: '分數如何變成注意力權重' }, step: 6 },
    { text: { en: 'Why the future is masked', zh: '為什麼要遮住未來' }, step: 5 },
    { text: { en: 'How values are mixed into a new vector', zh: '值如何混合成新的向量' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'One letter is not enough', zh: '只看一個字母是不夠的' },
      body: [
        { en: 'Our model is writing a name. So far it has <code>.jimm</code>, and it must choose the next letter.', zh: '我們的模型正在寫一個名字。目前寫到 <code>.jimm</code>，它必須選出下一個字母。' },
        { en: `If it looked only at the last letter, “m”, the best it could do is this chart: what follows “m” across all ${NAMES.length} names. No answer stands out. To do better it must use the letters that came before.`, zh: `如果只看最後一個字母「m」，它頂多只能參考這張圖：在全部 ${NAMES.length} 個名字裡，「m」後面接什麼。沒有哪個答案特別突出。想猜得更準，就必須用到前面的字母。` },
      ],
      callout: { kind: 'insight', text: { en: 'The meaning of a token depends on its context. Attention is the mechanism that lets each token gather information from the others.', zh: '一個詞元的意義取決於它的上下文。注意力（attention）就是讓每個詞元從其他詞元收集資訊的機制。' } },
    },
    {
      short: { en: 'Vectors', zh: '向量' }, chapter: 'technique',
      title: { en: 'Every token is a list of numbers', zh: '每個詞元都是一串數字' },
      body: [
        { en: 'Inside the model each token is a <b>vector</b>: here, 16 numbers. Orange cells are positive, blue cells negative, and stronger colour means a larger number.', zh: '在模型內部，每個詞元都是一個<b>向量</b>：在這裡是 16 個數字。橘色格子是正數，藍色格子是負數，顏色越深代表數值越大。' },
        { en: 'The vector encodes which letter it is <i>and</i> where it sits. That is why the two m’s have different vectors. (The Embeddings lesson shows where these numbers come from.)', zh: '這個向量同時記錄了「是哪個字母」<i>以及</i>「在第幾個位置」，所以兩個 m 的向量並不相同。（這些數字的來源會在「嵌入」那一課說明。）' },
      ],
      changed: [{ sign: '+', text: { en: `${IDS.length} vectors of 16 numbers`, zh: `${IDS.length} 個向量，各 16 個數字` } }],
    },
    {
      short: { en: 'Query', zh: '查詢' }, chapter: 'technique',
      title: { en: 'The token asks a question', zh: '詞元提出一個問題' },
      body: [
        { en: 'We follow the last token, the second “m”. It needs to find out what came before it.', zh: '我們跟著最後一個詞元，也就是第二個「m」。它需要知道自己前面有什麼。' },
        { en: 'Its vector is multiplied by a learned matrix to produce a <b>query</b>: a new vector that means, roughly, “this is what I am looking for”.', zh: '它的向量會乘上一個學習得來的矩陣，產生一個<b>查詢（query）</b>：一個新的向量，大致的意思是「我在找這樣的東西」。' },
      ],
      callout: { kind: 'note', text: { en: 'Nobody writes down what the query means. The matrix starts random and training shapes it.', zh: '沒有人規定查詢代表什麼。這個矩陣一開始是隨機的，由訓練慢慢塑造出來。' } },
      changed: [{ sign: '+', text: { en: 'Query vector for the second “m”', zh: '第二個「m」的查詢向量' } }],
    },
    {
      short: { en: 'Keys', zh: '鍵' }, chapter: 'technique',
      title: { en: 'Every token offers a key', zh: '每個詞元都提供一把鍵' },
      body: [
        { en: 'Each token, including the “m” itself, is multiplied by a second learned matrix to produce a <b>key</b>: “this is what I contain”.', zh: '每個詞元（包括「m」自己）都會乘上第二個學習得來的矩陣，產生一把<b>鍵（key）</b>：「我這裡有這樣的東西」。' },
        { en: 'A query is a question; keys are the labels it gets compared against.', zh: '查詢是一個問題，而鍵就是用來和它比對的標籤。' },
      ],
      changed: [{ sign: '+', text: { en: `${IDS.length} key vectors`, zh: `${IDS.length} 個鍵向量` } }],
    },
    {
      short: { en: 'Scores', zh: '分數' }, chapter: 'technique',
      title: { en: 'Compare the query with each key', zh: '把查詢和每把鍵比對' },
      body: [
        { en: 'To compare two vectors we take their <b>dot product</b>: multiply them cell by cell and add everything up. When the two vectors point the same way the result is large.', zh: '要比較兩個向量，我們計算它們的<b>內積（dot product）</b>：把對應的格子兩兩相乘，再全部加起來。兩個向量方向越一致，結果就越大。' },
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
        { en: '<b>Softmax</b> turns each row of scores into positive numbers that add up to 100%. Higher scores get a bigger share.', zh: '<b>Softmax</b> 把每一列的分數轉成加起來等於 100% 的正數，分數越高，分到的比例越大。' },
        { en: `Read the highlighted row: the second “m” gives ${pct(att[F][best])} of its attention to “${LABELS[best]}” (position ${best + 1}), and spreads the rest over the others.`, zh: `看被框起來的那一列：第二個「m」把 ${pct(att[F][best])} 的注意力放在「${LABELS[best]}」（第 ${best + 1} 個位置），其餘的分給其他詞元。` },
      ],
      changed: [{ sign: '~', text: { en: 'Scores → attention weights', zh: '分數 → 注意力權重' } }],
    },
    {
      short: { en: 'Values', zh: '值' }, chapter: 'solution',
      title: { en: 'Mix the values', zh: '把值混合起來' },
      body: [
        { en: 'Each token also has a <b>value</b> vector, made with a third learned matrix: the information it actually hands over.', zh: '每個詞元還有一個<b>值（value）</b>向量，由第三個學習得來的矩陣產生：這才是它真正交出去的資訊。' },
        { en: 'The second “m” takes every value, scales it by its attention weight, and adds them up. The result is one new vector that blends what it found in its context.', zh: '第二個「m」拿到每個值，乘上對應的注意力權重，再全部加起來。結果是一個新的向量，融合了它在上下文裡找到的資訊。' },
      ],
      callout: { kind: 'note', text: { en: 'This mix passes through one more learned matrix and is added back onto the token’s own vector, so the token keeps what it knew and gains context.', zh: '這個混合結果還會再通過一個學習得來的矩陣，然後加回詞元原本的向量上。這樣詞元既保留原本知道的，又多了上下文。' } },
      changed: [{ sign: '+', text: { en: 'Context vector for the second “m”', zh: '第二個「m」的上下文向量' } }],
    },
    {
      short: { en: 'Result', zh: '結果' }, chapter: 'solution',
      title: { en: 'Same letter, different answer', zh: '同一個字母，不同的答案' },
      body: [
        { en: 'Here are the model’s real predictions. After the <i>first</i> m (<code>.jim</code>) it is torn between several letters.', zh: '這是模型真正的預測。在<i>第一個</i> m 之後（<code>.jim</code>），它在好幾個字母之間猶豫。' },
        { en: `After the <i>second</i> m (<code>.jimm</code>) it says “${afterSecondM[0].label}” with ${pct(afterSecondM[0].v)}, as in “jimmy”. The token is the same letter both times. Only the context differs, and attention is what carried it in.`, zh: `在<i>第二個</i> m 之後（<code>.jimm</code>），它以 ${pct(afterSecondM[0].v)} 的機率選了「${afterSecondM[0].label}」，就像「jimmy」。兩次都是同一個字母，不同的只有上下文，而把上下文帶進來的正是注意力。` },
      ],
      changed: [{ sign: '~', text: { en: 'Prediction now depends on context', zh: '預測現在取決於上下文' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Attention in one line', zh: '一句話說完注意力' },
      body: [
        { en: 'Each token asks a question (<b>query</b>), compares it with every earlier token’s label (<b>key</b>), turns the scores into shares (<b>softmax</b>), and collects a weighted mix of what they offer (<b>values</b>).', zh: '每個詞元提出一個問題（<b>查詢</b>），和前面每個詞元的標籤（<b>鍵</b>）比對，把分數變成比例（<b>softmax</b>），再依比例收集它們提供的資訊（<b>值</b>）。' },
        { en: 'Three learned matrices do all of it. Nothing is hand-written.', zh: '這一切只靠三個學習得來的矩陣完成，沒有任何一條規則是人寫的。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Multi-Head Attention. One question per token is limiting, so real models ask several at once.', zh: '下一課：多頭注意力。每個詞元只問一個問題太受限了，所以真實的模型會同時問好幾個。' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const column = step >= 1 && step <= 8;
    const grid = step === 5 || step === 6;
    const tokX = grid ? 296 : 50;

    // the four input tokens are on screen for the whole lesson
    if (step <= 8)
      LABELS.forEach((ch, i) => {
        items.push({
          k: 'tok' + i, t: 'token', text: ch,
          x: column ? tokX : 300 + i * 62, y: column ? rowY(i) : 50,
          tone: i === F && step >= 2 ? 'a' : ch === '.' ? 'b' : 'ink',
        });
      });

    if (step === 0) {
      items.push({ k: 'next', t: 'token', x: 300 + IDS.length * 62, y: 50, text: '?', tone: 'a' });
      items.push({ k: 'data', t: 'bars', x: 200, y: 190, w: 500, h: 230, items: afterAnyM, title: t({ en: 'what follows “m” across all names in the list', zh: '在清單的所有名字裡，「m」後面接的字母' }) });
    }

    const strips = (step >= 1 && step <= 4);
    if (strips) {
      items.push({ k: 'hx', t: 'text', x: 130, y: rowY(0) - 24, text: t({ en: 'token vector', zh: '詞元向量' }), size: 14, tone: 'muted' });
      x.forEach((v, i) => items.push({ k: 'x' + i, t: 'vec', x: 130, y: rowY(i) + 11, v, dim: step >= 2 && step < 3 && i !== F }));
    }
    if (step >= 2 && step <= 4) {
      items.push({ k: 'hq', t: 'text', x: 660, y: rowY(F) - 8, text: t({ en: 'query', zh: '查詢' }), size: 14, tone: 'a', bold: true });
      items.push({ k: 'q', t: 'vec', x: 660, y: rowY(F) + 11, v: query });
      if (step === 2) {
        items.push({ k: 'qa', t: 'arrow', x1: 314, y1: rowY(F) + 22, x2: 652, y2: rowY(F) + 22, tone: 'a' });
        items.push({ k: 'qm', t: 'text', x: 483, y: rowY(F) + 12, text: t({ en: '× query matrix', zh: '× 查詢矩陣' }), size: 14, tone: 'a', anchor: 'middle' });
      }
    }
    if (step === 3 || step === 4) {
      items.push({ k: 'hk', t: 'text', x: 370, y: rowY(0) - 24, text: t({ en: 'key', zh: '鍵' }), size: 14, tone: 'muted' });
      keys.forEach((v, i) => {
        items.push({ k: 'k' + i, t: 'vec', x: 370, y: rowY(i) + 11, v });
        if (step === 3) items.push({ k: 'ka' + i, t: 'arrow', x1: 314, y1: rowY(i) + 22, x2: 362, y2: rowY(i) + 22 });
      });
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
    }

    if (step === 8) {
      items.push({ k: 'p1', t: 'bars', x: 190, y: 62, w: 520, h: 170, items: afterFirstM, title: t({ en: 'model’s prediction after “.jim” (first m)', zh: '模型在「.jim」之後的預測（第一個 m）' }) });
      items.push({ k: 'p2', t: 'bars', x: 190, y: 300, w: 520, h: 170, items: afterSecondM, title: t({ en: 'model’s prediction after “.jimm” (second m)', zh: '模型在「.jimm」之後的預測（第二個 m）' }) });
    }

    if (step === 9) {
      const boxes = [
        { en: 'token vectors', zh: '詞元向量' }, { en: 'query · key', zh: '查詢 · 鍵' }, { en: 'mask future', zh: '遮住未來' },
        { en: 'softmax', zh: 'softmax' }, { en: 'mix values', zh: '混合值' }, { en: 'new vector', zh: '新向量' },
      ];
      boxes.forEach((b, i) => {
        const bx = 120 + (i % 3) * 230, by = 150 + Math.floor(i / 3) * 130;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: by, w: 190, h: 56, label: t(b), tone: i === 5 ? 'a' : 'ink' });
        if (i % 3 < 2) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 194, y1: by + 28, x2: bx + 226, y2: by + 28 });
      });
      items.push({ k: 'rbd', t: 'arrow', x1: 675, y1: 210, x2: 215, y2: 276, dashed: true });
    }

    return {
      metrics: [
        { label: { en: 'Tokens in view', zh: '可見詞元' }, value: String(IDS.length) },
        { label: { en: 'Following', zh: '追蹤對象' }, value: step >= 2 ? t({ en: 'second “m”', zh: '第二個「m」' }) : '—' },
        { label: { en: 'Strongest attention', zh: '最強的注意力' }, value: step >= 6 ? `“${LABELS[best]}” ${pct(att[F][best])}` : '—' },
        { label: { en: 'Top prediction', zh: '最可能的下一個' }, value: step >= 8 ? `“${afterSecondM[0].label}” ${pct(afterSecondM[0].v)}` : '—' },
      ],
      items,
      legend: [
        { tone: 'pos', text: { en: 'positive number', zh: '正數' } },
        { tone: 'neg', text: { en: 'negative number', zh: '負數' } },
        { tone: 'heat', text: { en: 'attention weight', zh: '注意力權重' } },
      ],
    };
  },
};
