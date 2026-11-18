import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, NAMES, encode } from '../model/data';
import weights from '../model/weights.json';
import weights2 from '../model/weights-2h.json';
import experiments from '../model/experiments.json';

// A second trained model, the same size as the course model but with 2 attention heads
// of 8 numbers each. We follow the last letter of ".samir" (the "r").

const WORD = '.samir';
const IDS = encode(WORD);
const LABELS = [...WORD];
const F = IDS.length - 1;
const KINDS = 4; // itself, 1 back, 2+ back (a letter), the start "."

interface Data {
  one: number[]; // the 1-head model's attention for "r"
  att: number[][][]; // [head][row][col]
  x: number[]; q: number[][]; mixed: number[][]; concat: number[]; out: number[];
  habits: number[][]; // [head][kind] average share over all names
  count1: number; count2: number;
  base: number[]; heads2: number[];
}

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const m1 = GPT.fromJSON(weights as Weights);
  const m2 = GPT.fromJSON(weights2 as unknown as Weights);
  const one = m1.forward(IDS).att[0].row(F).slice(0, F + 1);
  const tr = m2.forward(IDS);
  const att = [0, 1].map((h) => IDS.map((_, i) => tr.att[h].row(i).slice(0, IDS.length)));
  const mixed = [0, 1].map((h) => tr.mixed[h].row(F));
  // what each head looks at, on average, for every letter that has at least 3 letters before it
  const habits = [new Array(KINDS).fill(0), new Array(KINDS).fill(0)];
  let n = 0;
  for (const name of NAMES) {
    const ids = encode('.' + name).slice(0, m2.cfg.ctx);
    if (ids.length < 5) continue;
    const r = m2.forward(ids);
    for (let i = 4; i < ids.length; i++) {
      n++;
      for (let h = 0; h < 2; h++) {
        const a = r.att[h].row(i);
        for (let j = 0; j <= i; j++) habits[h][j === 0 ? 3 : i - j === 0 ? 0 : i - j === 1 ? 1 : 2] += a[j];
      }
    }
  }
  habits.forEach((hb) => hb.forEach((_, k) => { hb[k] /= n; }));
  const v = experiments.variants;
  return {
    one, att, x: tr.x.row(F), q: [0, 1].map((h) => tr.q[h].row(F)), mixed, concat: [...mixed[0], ...mixed[1]], out: tr.attOut.row(F),
    habits, count1: m1.paramCount(), count2: m2.paramCount(), base: v.base.val, heads2: v.heads2.val,
  };
})();

const lz = (f: (d: Data) => L): L => ({ get en() { return f(data()).en; }, get zh() { return f(data()).zh; } });
const pct = (v: number) => Math.round(v * 100) + '%';
const argmax = (a: number[]) => a.indexOf(Math.max(...a));
const focus = (d: Data, h: number) => { const r = d.att[h][F]; const j = argmax(r); return { j, ch: LABELS[j], v: r[j] }; };
const last = (a: number[]) => a[a.length - 1];
const best = (a: number[]) => Math.min(...a);
const vowelPos = (j: number) => 'aeiouy'.includes(LABELS[j]);

const GX = [100, 520]; // x of each head's grid
const GY = 110;
const CELL = 44;

export const multiHead: Lesson = {
  id: 'multi-head',
  title: { en: 'Multi-Head Attention', zh: '多頭注意力' },
  blurb: {
    en: 'Instead of one set of attention shares, run several smaller attentions side by side and join what they find.',
    zh: '不只算一組注意力比例，而是並排跑好幾個較小的注意力，再把它們找到的東西合起來。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'Why one attention head can be limiting', zh: '為什麼只有一個注意力頭會受限' }, step: 0 },
    { text: { en: 'How the work is split into heads', zh: '工作怎麼分給各個頭' }, step: 1 },
    { text: { en: 'That different heads learn different habits', zh: '不同的頭會學到不同的習慣' }, step: 4 },
    { text: { en: 'How heads are joined and mixed by the output matrix', zh: '各頭的結果怎麼接起來，再由輸出矩陣混合' }, step: 6 },
    { text: { en: 'When more heads help, and when they do not', zh: '多頭什麼時候有幫助，什麼時候沒有' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'One head, one set of shares', zh: '一個頭，只有一組比例' },
      body: [
        { en: 'In the Self-Attention lesson each token handed out its attention once: one set of shares that adds up to 100%. Here is our 1-head model reading <code>.samir</code>, for the last letter “r”.', zh: '在「自注意力」那一課，每個詞元只分配一次注意力：一組加起來 100% 的比例。這是我們的單頭模型讀入 <code>.samir</code> 時，最後一個字母「r」的分配。' },
        lz((d) => ({ en: `But a letter may want two different things at once: the letter right before it, and something further back, such as the earlier vowel. With one set of shares, attention given to one is taken from the other. Here it chose the letter just before (“i”, ${pct(d.one[F - 1])}), so almost nothing is left for the earlier vowel “a” (${pct(d.one[2])}).`, zh: `但一個字母可能同時想要兩種不同的東西：緊鄰在前的字母，以及更前面的資訊，例如前面那個母音。只有一組比例的話，給了這邊，那邊就少了。這裡它選了緊鄰在前的字母（「i」，${pct(d.one[F - 1])}），前面的母音「a」幾乎什麼都沒分到（${pct(d.one[2])}）。` })),
      ],
      callout: { kind: 'insight', text: { en: 'One attention head is one question per token. Multi-head attention lets each token ask several questions at once.', zh: '一個注意力頭，就是每個詞元問一個問題。多頭注意力讓每個詞元同時問好幾個問題。' } },
    },
    {
      short: { en: 'Split', zh: '分頭' }, chapter: 'technique',
      title: { en: 'Two smaller heads instead of one', zh: '用兩個較小的頭取代一個' },
      body: [
        { en: 'We trained a second model (6,000 steps, like the course model) that is identical except for attention: it has <b>2 heads</b>. Each <b>head</b> is a complete attention of its own, with its own query, key and value matrices.', zh: '我們另外訓練了一個模型（和課程模型一樣訓練 6,000 步），除了注意力以外完全相同：它有 <b>2 個頭</b>。每個<b>頭（head）</b>都是一套完整的注意力，有自己的查詢、鍵、值矩陣。' },
        lz((d) => ({ en: `Each head reads all 16 numbers of the token, but its matrices are 16 × 8, so its query, key and value have 8 numbers instead of 16. Both models have exactly ${d.count2.toLocaleString()} weights: the heads split the work, they do not add to it.`, zh: `每個頭都讀詞元全部的 16 個數字，但它的矩陣是 16 × 8，所以它的查詢、鍵、值只有 8 個數字，而不是 16 個。兩個模型的權重數完全一樣，都是 ${d.count2.toLocaleString()} 個：多頭是把工作拆開，不是多加工作。` })),
      ],
      changed: [
        { sign: '~', text: { en: '1 head × 16 → 2 heads × 8', zh: '1 個頭 × 16 → 2 個頭 × 8' } },
        { sign: '+', text: { en: 'Separate query, key, value per head', zh: '每個頭有自己的查詢、鍵、值' } },
      ],
    },
    {
      short: { en: 'Head 1', zh: '第 1 頭' }, chapter: 'technique',
      title: { en: 'Head 1 usually looks just before', zh: '第 1 頭通常看緊鄰在前的字母' },
      body: [
        { en: 'This is head 1’s full attention grid for <code>.samir</code>: each row is a letter asking, each column a letter being looked at, and future positions are masked.', zh: '這是第 1 頭在 <code>.samir</code> 上完整的注意力表：每一列是發問的字母，每一行是被看的字母，未來的位置被遮住。' },
        lz((d) => { const f = focus(d, 0); return { en: `Read the bottom row: “r” gives ${pct(f.v)} of head 1’s attention to “${f.ch}”${f.j === F - 1 ? ', the letter right before it' : ''}.`, zh: `看最下面那一列：「r」把第 1 頭 ${pct(f.v)} 的注意力給了「${f.ch}」${f.j === F - 1 ? '，也就是緊鄰在它前面的字母' : ''}。` }; }),
      ],
      changed: [{ sign: '+', text: { en: 'Head 1 attention grid', zh: '第 1 頭的注意力表' } }],
    },
    {
      short: { en: 'Head 2', zh: '第 2 頭' }, chapter: 'technique',
      title: { en: 'Head 2 often looks somewhere else', zh: '第 2 頭常常看別的地方' },
      body: [
        lz((d) => { const f = focus(d, 1); return { en: `Head 2 reads the same letters at the same time, with its own matrices. Its bottom row gives ${pct(f.v)} to “${f.ch}”${vowelPos(f.j) ? `, the vowel ${F - f.j} letters back` : ''}.`, zh: `第 2 頭在同一時間讀同樣的字母，但用自己的矩陣。它的最下面一列把 ${pct(f.v)} 給了「${f.ch}」${vowelPos(f.j) ? `，也就是往前數 ${F - f.j} 個的母音` : ''}。` }; }),
        { en: 'So for this letter the two heads collect two different pieces of context, without competing for the same 100%. Each head has its own 100% to hand out.', zh: '所以對這個字母來說，兩個頭收集了兩份不同的上下文，而且不必搶同一個 100%：每個頭都有自己的 100% 可以分配。' },
      ],
      changed: [{ sign: '+', text: { en: 'Head 2 attention grid, side by side', zh: '第 2 頭的注意力表，並排' } }],
    },
    {
      short: { en: 'Habits', zh: '習慣' }, chapter: 'solution',
      title: { en: 'Each head has its own habit', zh: '每個頭都有自己的習慣' },
      body: [
        { en: 'One word could be a fluke, so we measured every name in the list: where does each head put its attention, on average, for letters with at least 3 letters before them?', zh: '一個字可能只是巧合，所以我們把清單裡每個名字都量了一遍：對前面至少有 3 個字母的字母來說，每個頭平均把注意力放在哪裡？' },
        lz((d) => ({ en: `Head 1 gives ${pct(d.habits[0][1])} to the letter right before; head 2 gives ${pct(d.habits[1][1])}. Head 2 reaches further back more: ${pct(d.habits[1][2])} versus ${pct(d.habits[0][2])}. Nobody assigned these roles; training produced them.`, zh: `第 1 頭把 ${pct(d.habits[0][1])} 給緊鄰在前的字母，第 2 頭給 ${pct(d.habits[1][1])}。第 2 頭比較常往更前面看：${pct(d.habits[1][2])} 對 ${pct(d.habits[0][2])}。沒有人指派這些角色，是訓練自己形成的。` })),
      ],
      callout: { kind: 'note', text: { en: 'The habits overlap a lot. Heads are tendencies, not clean rules.', zh: '兩者的習慣其實重疊很多。頭的分工是傾向，不是乾淨俐落的規則。' } },
      changed: [{ sign: '+', text: { en: 'Averages over all names', zh: '所有名字的平均' } }],
    },
    {
      short: { en: 'Join', zh: '接起來' }, chapter: 'technique',
      title: { en: 'Join the two results end to end', zh: '把兩個結果頭尾接起來' },
      body: [
        { en: 'Each head mixes its own values with its own shares, and gives back 8 numbers for “r”.', zh: '每個頭用自己的比例混合自己的值，替「r」交回 8 個數字。' },
        { en: 'The two results are simply placed side by side (<b>concatenated</b>): 8 + 8 = 16 numbers, the same width as one big head would give.', zh: '兩個結果直接並排放在一起（<b>串接</b>）：8 + 8 = 16 個數字，和一個大頭給出的寬度相同。' },
      ],
      changed: [{ sign: '+', text: { en: 'Concatenated: 8 + 8 = 16 numbers', zh: '串接：8 + 8 = 16 個數字' } }],
    },
    {
      short: { en: 'Mix', zh: '混合' }, chapter: 'technique',
      title: { en: 'The output matrix mixes the heads', zh: '輸出矩陣把各頭混合起來' },
      body: [
        { en: 'Side by side, the two halves cannot yet talk to each other. A learned 16 × 16 <b>output matrix</b>, <code>o</code>, multiplies the joined vector, so every output number can draw on both heads.', zh: '並排之後，兩半還沒辦法互相溝通。一個學習得來的 16 × 16 <b>輸出矩陣</b> <code>o</code> 乘上接好的向量，讓每個輸出數字都能同時用到兩個頭的資訊。' },
        { en: 'The result is added onto the token’s own vector, exactly as with one head. Everything after attention is unchanged.', zh: '結果會加回詞元原本的向量上，和單頭時一模一樣。注意力之後的流程完全不變。' },
      ],
      changed: [{ sign: '+', text: { en: '× o (16 × 16) → attention output', zh: '× o（16 × 16）→ 注意力輸出' } }],
    },
    {
      short: { en: 'Did it help?', zh: '有幫助嗎？' }, chapter: 'problem',
      title: { en: 'Case: do two heads help on our task?', zh: '案例：在我們的任務上，兩個頭有幫助嗎？' },
      body: [
        { en: 'To compare, we use the <b>loss</b>: how surprised the model is by the real next letter, on average. Pure guessing among 27 letters scores 3.30, and lower is better. We measure the <b>held-out loss</b>: the loss on 615 names the model never trained on.', zh: '要比較，我們看<b>損失（loss）</b>：模型對真正的下一個字母平均有多「意外」。在 27 個字母裡瞎猜是 3.30，越低越好。我們量的是<b>驗證損失</b>：模型在訓練時從沒看過的 615 個名字上的損失。' },
        lz((d) => ({ en: `In two comparison runs, a 1-head and a 2-head design were trained the same way for 1,500 steps. Final held-out loss: 1 head ${last(d.base).toFixed(3)}, 2 heads ${last(d.heads2).toFixed(3)}; best along the way ${best(d.base).toFixed(3)} and ${best(d.heads2).toFixed(3)}. About the same: names are at most 8 tokens, so one 16-wide head already sees enough.`, zh: `我們做了兩次比較實驗：單頭和雙頭的設計用同樣的方式各訓練 1,500 步。最後的驗證損失：單頭 ${last(d.base).toFixed(3)}，雙頭 ${last(d.heads2).toFixed(3)}；過程中最好的是 ${best(d.base).toFixed(3)} 和 ${best(d.heads2).toFixed(3)}。差不多一樣：名字最多 8 個詞元，一個 16 數字寬的頭已經看得夠了。` })),
      ],
      callout: { kind: 'note', text: { en: 'These are short comparison runs (constant learning rate 0.01), not the 2-head model in the grids above. Large models read long texts with far wider vectors, and there many heads per layer is standard: the original Transformer paper (2017) used 8.', zh: '這些是較短的比較實驗（學習率固定 0.01），不是上面注意力表用的雙頭模型。大型模型讀的是長篇文字、向量也寬得多，在那裡每層用很多個頭是標準做法：最早的 Transformer 論文（2017）用了 8 個。' } },
      changed: [{ sign: '~', text: { en: '1 head vs 2 heads, real training runs', zh: '單頭與雙頭，真實的訓練紀錄' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Several questions at once', zh: '同時問好幾個問題' },
      body: [
        { en: 'Each <b>head</b> is a full attention with its own query, key and value matrices, each producing 8 numbers from the full 16.<br>Heads hand out their own 100% and can learn different habits.<br>Their results are <b>concatenated</b> and mixed by the output matrix <code>o</code>.<br>Same number of weights; on our tiny task, about the same loss.', zh: '每個<b>頭</b>都是一套完整的注意力，有自己的查詢、鍵、值矩陣，各自從完整的 16 個數字算出 8 個數字。<br>每個頭分配自己的 100%，可以學到不同的習慣。<br>各頭的結果<b>串接</b>起來，再由輸出矩陣 <code>o</code> 混合。<br>權重數相同；在我們的小任務上，損失也差不多。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Positional Information. Attention compares contents, so how does the model know which letter came first?', zh: '下一課：位置資訊。注意力比較的是內容，那模型怎麼知道哪個字母在前面？' } },
    },
  ],

  scene(step) {
    const d = data();
    const items: Item[] = [];
    const grid = (h: number, x: number, hi: boolean) => {
      const m = d.att[h].map((row, i) => row.map((v, j) => (j > i ? null : v)));
      items.push({ k: 'g' + h, t: 'heat', x, y: GY, cell: CELL, m, rows: LABELS, cols: LABELS, fmt: 'pct', hi: hi ? F : undefined, max: 1 });
      const f = focus(d, h);
      items.push({ k: 'gt' + h, t: 'text', x: x + (CELL * IDS.length) / 2, y: GY - 40, anchor: 'middle', size: 16, bold: true, tone: h === 0 ? 'a' : 'b', text: t({ en: `head ${h + 1}`, zh: `第 ${h + 1} 頭` }) });
      if (hi) items.push({ k: 'gf' + h, t: 'text', x: x + (CELL * IDS.length) / 2, y: GY + CELL * IDS.length + 34, anchor: 'middle', size: 15, tone: h === 0 ? 'a' : 'b', text: t({ en: `“r” → “${f.ch}” ${pct(f.v)}`, zh: `「r」→「${f.ch}」${pct(f.v)}` }) });
    };

    if (step === 0) {
      items.push({ k: 'one', t: 'heat', x: 258, y: 190, cell: 64, m: [d.one], rows: ['r'], cols: LABELS, fmt: 'pct', max: 1 });
      items.push({ k: 'onet', t: 'text', x: 450, y: 130, anchor: 'middle', size: 16, text: t({ en: 'the 1-head model: where the last “r” of “.samir” looks', zh: '單頭模型：「.samir」最後的「r」看向哪裡' }) });
      items.push({ k: 'oneb', t: 'text', x: 450, y: 310, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: 'one set of shares, adding up to 100%', zh: '一組比例，加起來 100%' }) });
      const want = [{ en: 'the letter just before?', zh: '緊鄰在前的字母？' }, { en: 'the earlier vowel?', zh: '前面的母音？' }];
      want.forEach((w, i) => items.push({ k: 'want' + i, t: 'box', x: 210 + i * 260, y: 370, w: 220, h: 50, label: t(w), tone: i === 0 ? 'a' : 'b' }));
    }

    if (step === 1) {
      items.push({ k: 'x', t: 'vec', x: 290, y: 60, v: d.x, cell: 20, label: t({ en: 'vector of the last “r” (16 numbers)', zh: '最後一個「r」的向量（16 個數字）' }) });
      [0, 1].forEach((h) => {
        const bx = h === 0 ? 120 : 520;
        items.push({ k: 'hb' + h, t: 'box', x: bx, y: 170, w: 260, h: 60, label: t({ en: `head ${h + 1}`, zh: `第 ${h + 1} 頭` }), sub: t({ en: 'own query, key, value matrices', zh: '自己的查詢、鍵、值矩陣' }), tone: h === 0 ? 'a' : 'b' });
        items.push({ k: 'ha' + h, t: 'arrow', x1: 450, y1: 92, x2: bx + 130, y2: 164, tone: h === 0 ? 'a' : 'b' });
        items.push({ k: 'hq' + h, t: 'vec', x: bx + 50, y: 290, v: d.q[h], cell: 20, label: t({ en: 'query: 8 numbers', zh: '查詢：8 個數字' }) });
        items.push({ k: 'hqa' + h, t: 'arrow', x1: bx + 130, y1: 234, x2: bx + 130, y2: 270 });
      });
      items.push({ k: 'wc', t: 'text', x: 450, y: 400, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: '3 matrices of 16 × 16  =  2 heads × 3 matrices of 16 × 8  =  768 weights', zh: '3 個 16 × 16 矩陣 = 2 個頭 × 3 個 16 × 8 矩陣 = 768 個權重' }) });
    }

    if (step === 2) grid(0, 318, true);
    if (step === 3) { grid(0, GX[0], true); grid(1, GX[1], true); }

    if (step === 4) {
      const kinds = [{ en: 'itself', zh: '自己' }, { en: '1 back', zh: '前 1 個' }, { en: '2+ back', zh: '更前面' }, { en: 'start “.”', zh: '開頭「.」' }];
      [0, 1].forEach((h) => items.push({
        k: 'hab' + h, t: 'bars', x: h === 0 ? 70 : 490, y: 150, w: 340, h: 270,
        items: kinds.map((kd, i) => ({ label: t(kd), v: d.habits[h][i], hi: h === 0 ? i === 1 : i === 2 })),
        title: t({ en: `head ${h + 1}: average share, all names`, zh: `第 ${h + 1} 頭：所有名字的平均比例` }),
      }));
    }

    if (step === 5 || step === 6) {
      [0, 1].forEach((h) => {
        const x = h === 0 ? 140 : 552;
        items.push({ k: 'mx' + h, t: 'vec', x, y: 70, v: d.mixed[h], cell: 26, label: t({ en: `head ${h + 1} result (8)`, zh: `第 ${h + 1} 頭的結果（8）` }) });
        items.push({ k: 'mxa' + h, t: 'arrow', x1: x + 104, y1: 102, x2: h === 0 ? 346 : 554, y2: 186, tone: h === 0 ? 'a' : 'b' });
      });
      items.push({ k: 'cat', t: 'vec', x: 242, y: 200, v: d.concat, cell: 26, label: t({ en: 'joined: 16 numbers (head 1 | head 2)', zh: '接起來：16 個數字（第 1 頭 | 第 2 頭）' }) });
    }
    if (step === 6) {
      items.push({ k: 'oa', t: 'arrow', x1: 450, y1: 236, x2: 450, y2: 320, tone: 'a' });
      items.push({ k: 'ot', t: 'text', x: 466, y: 284, size: 15, tone: 'a', text: t({ en: '× o (16 × 16)', zh: '× o（16 × 16）' }) });
      items.push({ k: 'out', t: 'vec', x: 242, y: 340, v: d.out, cell: 26, label: t({ en: 'attention output for “r”: every number uses both heads', zh: '「r」的注意力輸出：每個數字都用到兩個頭' }) });
      items.push({ k: 'res', t: 'text', x: 450, y: 420, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: '→ added onto the token’s own vector, as before', zh: '→ 和以前一樣，加回詞元原本的向量' }) });
    }

    if (step === 7) {
      items.push({
        k: 'chart', t: 'chart', x: 90, y: 50, w: 740, h: 320, ymin: 2.1, ymax: 2.6, xmax: 1500, every: experiments.variants.base.every,
        series: [{ pts: d.base, tone: 'b' }, { pts: d.heads2, tone: 'a' }],
        xlabel: t({ en: 'training steps', zh: '訓練步數' }), ylabel: t({ en: 'held-out loss (lower is better)', zh: '驗證損失（越低越好）' }),
      });
      items.push({ k: 'fin1', t: 'text', x: 300, y: 450, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'b', text: t({ en: `1 head: ${last(d.base).toFixed(3)}`, zh: `單頭：${last(d.base).toFixed(3)}` }) });
      items.push({ k: 'fin2', t: 'text', x: 620, y: 450, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'a', text: t({ en: `2 heads: ${last(d.heads2).toFixed(3)}`, zh: `雙頭：${last(d.heads2).toFixed(3)}` }) });
    }

    if (step === 8) {
      items.push({ k: 'r0', t: 'box', x: 30, y: 200, w: 150, h: 60, label: t({ en: 'token vectors', zh: '詞元向量' }) });
      items.push({ k: 'hb0', t: 'box', x: 230, y: 140, w: 150, h: 56, label: t({ en: 'head 1', zh: '第 1 頭' }), tone: 'a' });
      items.push({ k: 'hb1', t: 'box', x: 230, y: 264, w: 150, h: 56, label: t({ en: 'head 2', zh: '第 2 頭' }), tone: 'b' });
      items.push({ k: 'r2', t: 'box', x: 430, y: 200, w: 130, h: 60, label: t({ en: 'join', zh: '串接' }) });
      items.push({ k: 'r3', t: 'box', x: 600, y: 200, w: 110, h: 60, label: '× o' });
      items.push({ k: 'r4', t: 'box', x: 750, y: 200, w: 120, h: 60, label: t({ en: 'output', zh: '輸出' }), tone: 'a' });
      items.push({ k: 'ra0', t: 'arrow', x1: 182, y1: 222, x2: 226, y2: 170 });
      items.push({ k: 'ra1', t: 'arrow', x1: 182, y1: 238, x2: 226, y2: 290 });
      items.push({ k: 'ra2', t: 'arrow', x1: 384, y1: 170, x2: 426, y2: 222 });
      items.push({ k: 'ra3', t: 'arrow', x1: 384, y1: 290, x2: 426, y2: 238 });
      items.push({ k: 'ra4', t: 'arrow', x1: 564, y1: 230, x2: 596, y2: 230 });
      items.push({ k: 'ra5', t: 'arrow', x1: 714, y1: 230, x2: 746, y2: 230 });
    }

    const f0 = step >= 2 && step <= 6 ? focus(d, 0) : null;
    const f1 = step >= 3 && step <= 6 ? focus(d, 1) : null;
    return {
      metrics: [
        { label: { en: 'Heads', zh: '頭數' }, value: step === 0 ? '1' : '2' },
        { label: { en: 'Numbers per head', zh: '每頭的數字' }, value: step === 0 ? '16' : '8' },
        { label: { en: 'Head 1 → ', zh: '第 1 頭 →' }, value: f0 ? t({ en: `“${f0.ch}” ${pct(f0.v)}`, zh: `「${f0.ch}」 ${pct(f0.v)}` }) : '—' },
        { label: { en: 'Head 2 → ', zh: '第 2 頭 →' }, value: f1 ? t({ en: `“${f1.ch}” ${pct(f1.v)}`, zh: `「${f1.ch}」 ${pct(f1.v)}` }) : '—' },
        { label: { en: 'Weights', zh: '權重數' }, value: (step === 0 ? d.count1 : d.count2).toLocaleString() },
      ],
      items,
      legend: step === 7
        ? [{ tone: 'b', text: { en: '1 head', zh: '單頭' } }, { tone: 'a', text: { en: '2 heads', zh: '雙頭' } }]
        : step === 1 || step === 5 || step === 6
          ? [{ tone: 'pos', text: { en: 'positive number', zh: '正數' } }, { tone: 'neg', text: { en: 'negative number', zh: '負數' } }]
          : [{ tone: 'heat', text: { en: 'attention share', zh: '注意力比例' } }],
    };
  },
};
