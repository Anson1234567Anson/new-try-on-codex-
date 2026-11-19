import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, encode } from '../model/data';
import weights from '../model/weights.json';
import experiments from '../model/experiments.json';

// Two words with the same letters in a different order. Attention on token contents
// alone cannot tell them apart; the learned position table `pos` fixes that.

const A = '.nora', B = '.rona';
const SAME = '.emma'; // two m's at positions 2 and 3
const D = 16;

interface Side { shares: number[]; mix: number[] }
interface Data {
  plain: [Side, Side]; // attention for the last "a" using token contents only (positions left out)
  real: [Side, Side]; // the real model, with positions
  plainDiff: number; realDiff: number;
  pos: number[][]; tokM: number[]; pos2: number[]; pos3: number[]; x2: number[]; x3: number[]; cosX: number;
  base: number[]; noPos: number[]; heads2: number[];
}

const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const cos = (a: number[], b: number[]) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
const dist = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0));

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const m = GPT.fromJSON(weights as Weights);
  const mul = (v: number[], name: string) => { const W = m.p[name]; return Array.from({ length: W.c }, (_, c) => v.reduce((s, x, r) => s + x * W.d[r * W.c + c], 0)); };
  // the same attention head, fed only the token embeddings (no position added)
  const plain = (word: string): Side => {
    const rows = encode(word).map((id) => m.p.tok.row(id));
    const q = mul(rows[rows.length - 1], 'q0');
    const ks = rows.map((r) => mul(r, 'k0')), vs = rows.map((r) => mul(r, 'v0'));
    const sc = ks.map((k) => dot(q, k) / Math.sqrt(D));
    const mx = Math.max(...sc), e = sc.map((s) => Math.exp(s - mx)), z = e.reduce((a, b) => a + b, 0);
    const shares = e.map((x) => x / z);
    const mix = vs[0].map((_, c) => vs.reduce((s, v, j) => s + shares[j] * v[c], 0));
    return { shares, mix };
  };
  const real = (word: string): Side => {
    const tr = m.forward(encode(word));
    const L = word.length - 1;
    return { shares: tr.att[0].row(L).slice(0, word.length), mix: tr.mixed[0].row(L) };
  };
  const pa = plain(A), pb = plain(B), ra = real(A), rb = real(B);
  const tr = m.forward(encode(SAME));
  const x2 = tr.x.row(2), x3 = tr.x.row(3);
  const v = experiments.variants;
  return {
    plain: [pa, pb], real: [ra, rb], plainDiff: dist(pa.mix, pb.mix), realDiff: dist(ra.mix, rb.mix),
    pos: Array.from({ length: m.cfg.ctx }, (_, i) => m.p.pos.row(i)),
    tokM: m.p.tok.row(CHARS.indexOf('m')), pos2: m.p.pos.row(2), pos3: m.p.pos.row(3), x2, x3, cosX: cos(x2, x3),
    base: v.base.val, noPos: v.noPos.val, heads2: v.heads2.val,
  };
})();

const lz = (f: (d: Data) => L): L => ({ get en() { return f(data()).en; }, get zh() { return f(data()).zh; } });
const pct = (v: number) => Math.round(v * 100) + '%';
const last = (a: number[]) => a[a.length - 1];
const share = (s: Side, word: string, ch: string) => s.shares[word.indexOf(ch)];

// one word drawn as chips with an attention share under each letter
const ROWY = [70, 250];
const chipX = (i: number) => 120 + i * 62;

export const positions: Lesson = {
  id: 'positions',
  title: { en: 'Positional Information', zh: '位置資訊' },
  blurb: {
    en: 'Attention compares contents, not order. See how a learned table of positions tells the model where each token sits.',
    zh: '注意力比較的是內容，不是順序。來看一張學習得來的位置表，如何告訴模型每個詞元在哪裡。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'Why attention alone ignores word order', zh: '為什麼光靠注意力會忽略順序' }, step: 1 },
    { text: { en: 'What the position table is', zh: '位置表是什麼' }, step: 2 },
    { text: { en: 'How the same letter gets different vectors at different places', zh: '同一個字母在不同位置，如何得到不同的向量' }, step: 3 },
    { text: { en: 'Why a model without positions still gets some order', zh: '為什麼沒有位置資訊的模型，仍然知道一點順序' }, step: 6 },
    { text: { en: 'Other ways real models encode position', zh: '真實模型表示位置的其他方法' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Same letters, different order', zh: '同樣的字母，不同的順序' },
      body: [
        { en: '<code>.nora</code> and <code>.rona</code> use the same letters in a different order. Let the last “a” attend using only the letters’ contents: their embeddings, with nothing about where they sit (the course model’s trained attention, with the position rows left out).', zh: '<code>.nora</code> 和 <code>.rona</code> 用的是同樣的字母，只是順序不同。讓最後的「a」只根據字母的內容（也就是嵌入，不含任何位置資訊）來分配注意力（用的是課程模型訓練好的注意力，只是不加位置那幾列）。' },
        lz((d) => ({ en: `The shares under each letter are the same in both words: “n” gets ${pct(share(d.plain[0], A, 'n'))} either way. The mixed vectors for “a” are identical; their difference is ${d.plainDiff.toFixed(3)}.`, zh: `兩個字裡，每個字母分到的比例都一樣：「n」不論在哪個字都分到 ${pct(share(d.plain[0], A, 'n'))}。「a」得到的混合向量完全相同，差距是 ${d.plainDiff.toFixed(3)}。` })),
      ],
      callout: { kind: 'insight', text: { en: 'To attention, a set of tokens with no positions is a bag: it can see what is in the bag, not the order things went in.', zh: '對注意力來說，沒有位置的詞元就像一袋東西：它看得到袋子裡有什麼，卻看不出放進去的順序。' } },
    },
    {
      short: { en: 'Why', zh: '為什麼' }, chapter: 'technique',
      title: { en: 'Adding up does not care about order', zh: '加法不在乎順序' },
      body: [
        { en: 'Each share comes from comparing the “a” with one letter’s contents, so “n” gets the same share wherever it stands. The mix is then a sum: share × value, added up.', zh: '每個比例都來自「a」和某個字母內容的比較，所以「n」不管站在哪裡，分到的比例都一樣。接著混合結果就是一個總和：比例 × 值，全部加起來。' },
        { en: 'Shuffle the letters and you only shuffle the terms of the sum. 2 + 5 is 5 + 2. Nothing in this calculation knows which letter came first.', zh: '把字母洗牌，只是把總和裡的項目換個順序而已。2 + 5 就是 5 + 2。這個計算裡，沒有任何東西知道哪個字母在前面。' },
      ],
      changed: [{ sign: '+', text: { en: 'The mix written as a sum', zh: '把混合寫成總和' } }],
    },
    {
      short: { en: 'Position table', zh: '位置表' }, chapter: 'technique',
      title: { en: 'A second table, one row per position', zh: '第二張表，每個位置一列' },
      body: [
        { en: 'The fix: the model keeps another learned table, <code>pos</code>, with one row of 16 numbers for each of the 8 positions it can read (8 × 16 = 128 numbers).', zh: '解法是：模型再保存一張學習得來的表 <code>pos</code>，它能讀的 8 個位置，各有一列 16 個數字（8 × 16 = 128 個數字）。' },
        { en: 'Like the token table, it starts random and training shapes it. Row 0 means “first place”, row 1 “second place”, and so on.', zh: '它和詞元表一樣，一開始是隨機的，由訓練塑造。第 0 列代表「第一個位置」，第 1 列代表「第二個位置」，依此類推。' },
      ],
      changed: [{ sign: '+', text: { en: 'Table pos: 8 rows × 16 numbers', zh: '表 pos：8 列 × 16 個數字' } }],
    },
    {
      short: { en: 'Add', zh: '相加' }, chapter: 'technique',
      title: { en: 'Add the position to the token', zh: '把位置加到詞元上' },
      body: [
        { en: 'Before anything else, each token’s embedding is added, number by number, to the row for its position. Take <code>.emma</code>: both “m”s start from the very same token row.', zh: '在做任何事之前，每個詞元的嵌入都會和它所在位置的那一列逐個數字相加。以 <code>.emma</code> 為例：兩個「m」一開始拿到的是完全相同的詞元列。' },
        lz((d) => ({ en: `The first “m” gets position 2 added, the second gets position 3. Now they differ: their cosine similarity is ${d.cosX.toFixed(2)} instead of 1.00. Every later layer can tell them apart.`, zh: `第一個「m」加上位置 2，第二個加上位置 3。這樣它們就不同了：餘弦相似度變成 ${d.cosX.toFixed(2)}，而不是 1.00。之後的每一層都分得出它們。` })),
      ],
      callout: { kind: 'note', text: { en: 'Adding mixes both kinds of information into the same 16 numbers. Training learns to keep them usable.', zh: '相加會把兩種資訊混進同樣的 16 個數字裡，而訓練會學著讓兩者都派得上用場。' } },
      changed: [{ sign: '+', text: { en: 'token + position = input vector', zh: '詞元 + 位置 = 輸入向量' } }],
    },
    {
      short: { en: 'Order counts', zh: '順序有用了' }, chapter: 'solution',
      title: { en: 'Now order changes attention', zh: '現在順序會改變注意力' },
      body: [
        lz((d) => ({ en: `Here is the real trained model, positions included. In <code>.nora</code> the last “a” gives “n” ${pct(share(d.real[0], A, 'n'))}; in <code>.rona</code> it gives “n” ${pct(share(d.real[1], B, 'n'))}.`, zh: `這是真正訓練好的模型，包含位置資訊。在 <code>.nora</code> 裡，最後的「a」給「n」${pct(share(d.real[0], A, 'n'))}；在 <code>.rona</code> 裡給「n」${pct(share(d.real[1], B, 'n'))}。` })),
        lz((d) => ({ en: `The two mixes now differ by ${d.realDiff.toFixed(2)} instead of 0. The same letters in a different order finally look different to the model.`, zh: `兩個混合結果現在相差 ${d.realDiff.toFixed(2)}，不再是 0。同樣的字母換了順序，模型終於看得出差別。` })),
      ],
      changed: [{ sign: '~', text: { en: 'Contents only → contents + position', zh: '只有內容 → 內容 + 位置' } }],
    },
    {
      short: { en: 'No positions', zh: '沒有位置' }, chapter: 'problem',
      title: { en: 'Case: train without the position table', zh: '案例：不用位置表來訓練' },
      body: [
        { en: 'So how much does it matter? In two comparison runs we trained the same design for 1,500 steps (constant learning rate 0.01), once without <code>pos</code>. We compare their <b>held-out loss</b>: how surprised the model is by the real next letter, on average, on 615 names it never trained on. Lower is better; pure guessing among 27 letters scores 3.30.', zh: '那到底有多重要？我們做了兩次比較實驗：同樣的設計各訓練 1,500 步（學習率固定 0.01），其中一次拿掉 <code>pos</code>。比較的是<b>驗證損失</b>：在模型訓練時從沒看過的 615 個名字上，模型對真正的下一個字母平均有多「意外」。越低越好；在 27 個字母裡瞎猜是 3.30。' },
        lz((d) => ({ en: `With positions: ${last(d.base).toFixed(3)}. Without: ${last(d.noPos).toFixed(3)}, worse by ${(last(d.noPos) - last(d.base)).toFixed(3)}. That looks tiny, but every change on this small task is small (2 heads instead of 1 moved it by ${Math.abs(last(d.heads2) - last(d.base)).toFixed(3)}), and it is still far from pure guessing. Surprising: how can it work at all without order?`, zh: `有位置：${last(d.base).toFixed(3)}。沒有位置：${last(d.noPos).toFixed(3)}，差了 ${(last(d.noPos) - last(d.base)).toFixed(3)}。看起來很小，但在這個小任務上，任何改變的影響都很小（把 1 個頭換成 2 個頭只差 ${Math.abs(last(d.heads2) - last(d.base)).toFixed(3)}），而且它離瞎猜還很遠。奇怪的是：沒有順序，它怎麼還能運作？` })),
      ],
      changed: [{ sign: '~', text: { en: 'Real runs: with vs without pos', zh: '真實訓練：有 pos 與沒有 pos' } }],
    },
    {
      short: { en: 'Counting', zh: '數數' }, chapter: 'technique',
      title: { en: 'A hidden clue: how many tokens can I see?', zh: '隱藏的線索：我看得到幾個詞元？' },
      body: [
        { en: 'The causal mask lets each token see only itself and the tokens before it. So the 1st token sees 1 token, the 5th sees 5. That count is a clue to position, even with no position table.', zh: '因果遮罩讓每個詞元只看得到自己和前面的詞元。所以第 1 個詞元看到 1 個，第 5 個看到 5 個。即使沒有位置表，這個數量本身就是位置的線索。' },
        { en: 'Illustration: if a token spread its attention evenly, the start mark “.” would get 100%, then 50%, 33%, 25%… The amount of “.” in the mix reveals roughly how far along it is. In our one-layer model exact order is still lost, which is why the loss is worse.', zh: '示意：如果一個詞元把注意力平均分配，開頭記號「.」分到的比例會是 100%、50%、33%、25%…… 混合結果裡「.」佔多少，就透露了它大概走到第幾個。但在我們這個只有一層的模型裡，確切的順序還是不知道，所以損失比較差。' },
      ],
      callout: { kind: 'note', text: { en: 'This is why the comparison is not “works” versus “fails”. The causal mask leaks some order; the position table gives the full picture. With many layers, models without positions recover much more order (Haviv et al., 2022).', zh: '所以這不是「能用」和「不能用」的差別。因果遮罩會洩漏一部分順序，而位置表提供完整的資訊。層數多的模型，即使沒有位置資訊，也能找回多得多的順序（Haviv 等人，2022）。' } },
      changed: [{ sign: '+', text: { en: 'Visible tokens: 1, 2, 3, …', zh: '看得到的詞元：1、2、3、……' } }],
    },
    {
      short: { en: 'Other ways', zh: '其他做法' }, chapter: 'technique',
      title: { en: 'Real models use other position schemes too', zh: '真實模型也用其他位置方法' },
      body: [
        { en: 'A learned table, like ours, is one choice. The original Transformer (2017) instead added <b>fixed waves</b>: each of the 16 numbers follows a sine or cosine wave of a different speed, so every position gets a unique pattern with nothing to learn.', zh: '像我們這樣學一張表，是其中一種選擇。最早的 Transformer（2017）則是加上<b>固定的波形</b>：16 個數字各自跟著一條速度不同的正弦或餘弦波，所以每個位置都有獨一無二的花樣，而且不用學。' },
        { en: 'Many recent models, such as LLaMA, use <b>rotary position embedding (RoPE)</b>: instead of adding anything, they rotate each query and key by an angle that depends on its position, so attention scores depend on how far apart two tokens are.', zh: '許多較新的模型（例如 LLaMA）使用<b>旋轉位置嵌入（RoPE）</b>：不額外加任何東西，而是依照位置把每個查詢和鍵旋轉一個角度，讓注意力分數取決於兩個詞元相隔多遠。' },
      ],
      changed: [{ sign: '+', text: { en: 'Fixed waves (computed from the 2017 formula)', zh: '固定波形（依 2017 年的公式算出）' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Where each token sits', zh: '每個詞元在哪裡' },
      body: [
        { en: 'Attention on contents alone treats tokens as a bag, with no order.<br>A learned <b>position table</b> gives each position its own row of numbers.<br>It is added to the token embedding, so the same letter differs by place.<br>Without it our model is worse but not blind: the causal mask leaks a count.<br>Real models also use fixed waves or rotations (RoPE).', zh: '只看內容的注意力，把詞元當成一袋東西，沒有順序。<br>學習得來的<b>位置表</b>讓每個位置都有自己的一列數字。<br>它會加到詞元嵌入上，讓同一個字母在不同位置有所不同。<br>拿掉它，我們的模型會變差但不至於全盲：因果遮罩會洩漏數量。<br>真實的模型也會用固定波形或旋轉（RoPE）。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: The MLP Layer. After attention gathers context, each token thinks it over on its own.', zh: '下一課：MLP 層。注意力收集完上下文後，每個詞元會各自消化這些資訊。' } },
    },
  ],

  scene(step) {
    const d = data();
    const items: Item[] = [];
    const pmax = 1.2;

    const word = (w: string, s: Side, r: number, tone: 'a' | 'b') => {
      items.push({ k: 'wl' + r, t: 'text', x: 100, y: ROWY[r] + 28, anchor: 'end', size: 15, mono: true, bold: true, tone, text: w });
      [...w].forEach((ch, i) => items.push({
        k: `w${r}-${ch}${w.indexOf(ch) === i ? '' : i}`, t: 'token', x: chipX(i), y: ROWY[r], text: ch,
        tone: i === w.length - 1 ? 'a' : ch === '.' ? 'b' : 'ink', sub: i === w.length - 1 ? undefined : pct(s.shares[i]),
      }));
      items.push({ k: 'mix' + r, t: 'vec', x: 500, y: ROWY[r] + 11, v: s.mix, cell: 22, max: pmax, label: t({ en: 'mixed vector for the last “a”', zh: '最後「a」的混合向量' }) });
    };

    if (step <= 1 || step === 4) {
      const sides = step === 4 ? d.real : d.plain;
      word(A, sides[0], 0, 'a');
      word(B, sides[1], 1, 'b');
      const diff = step === 4 ? d.realDiff : d.plainDiff;
      items.push({ k: 'diff', t: 'text', x: 676, y: 190, anchor: 'middle', size: 17, bold: true, mono: true, tone: step === 4 ? 'a' : 'bad', text: t({ en: `difference: ${diff.toFixed(3)}`, zh: `差距：${diff.toFixed(3)}` }) });
      if (step !== 1) items.push({ k: 'mode', t: 'text', x: 450, y: 400, anchor: 'middle', size: 15, tone: 'muted', text: step === 4 ? t({ en: 'real model: token + position', zh: '真實模型：詞元 + 位置' }) : t({ en: 'contents only: no position information', zh: '只有內容：沒有位置資訊' }) });
    }
    if (step === 1) {
      const term = (w: string, s: Side) => [...w].slice(0, -1).map((ch, i) => `${pct(s.shares[i])}·${ch}`).join(' + ') + ` + ${pct(s.shares[w.length - 1])}·a`;
      items.push({ k: 'sumA', t: 'text', x: 450, y: 440, anchor: 'middle', size: 15, mono: true, tone: 'a', text: term(A, d.plain[0]) });
      items.push({ k: 'sumB', t: 'text', x: 450, y: 470, anchor: 'middle', size: 15, mono: true, tone: 'b', text: term(B, d.plain[1]) });
      items.push({ k: 'mode', t: 'text', x: 450, y: 400, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: 'mix = share × value of each letter, added up (same terms, different order)', zh: '混合 = 每個字母的比例 × 值，全部加總（同樣的項，順序不同）' }) });
    }

    if (step === 2) {
      items.push({ k: 'ph', t: 'text', x: 250, y: 40, size: 14, tone: 'muted', text: t({ en: 'table pos (trained): one row per position', zh: '表 pos（訓練後）：每個位置一列' }) });
      d.pos.forEach((v, i) => {
        items.push({ k: 'pl' + i, t: 'text', x: 236, y: 75 + i * 52, anchor: 'end', size: 14, mono: true, text: t({ en: `position ${i}`, zh: `位置 ${i}` }) });
        items.push({ k: 'p' + i, t: 'vec', x: 250, y: 58 + i * 52, v, cell: 24, max: pmax });
      });
    }

    if (step === 3) {
      const rows = [
        { k: 'a', label: { en: 'first “m” (position 2)', zh: '第一個「m」（位置 2）' }, pos: d.pos2, x: d.x2, pk: 'p2' },
        { k: 'b', label: { en: 'second “m” (position 3)', zh: '第二個「m」（位置 3）' }, pos: d.pos3, x: d.x3, pk: 'p3' },
      ];
      [...SAME].forEach((ch, i) => items.push({ k: 'e' + i, t: 'token', x: 300 + i * 62, y: 20, text: ch, sub: String(i), tone: i === 2 || i === 3 ? 'a' : ch === '.' ? 'b' : 'ink' }));
      rows.forEach((r, j) => {
        const y = 130 + j * 190;
        items.push({ k: 'rl' + r.k, t: 'text', x: 40, y: y - 8, size: 15, bold: true, tone: 'a', text: t(r.label) });
        items.push({ k: 'tm' + r.k, t: 'vec', x: 40, y: y + 20, v: d.tokM, cell: 14, max: pmax, label: t({ en: 'token “m”', zh: '詞元「m」' }) });
        items.push({ k: 'plus' + r.k, t: 'text', x: 280, y: y + 38, anchor: 'middle', size: 22, bold: true, text: '+' });
        items.push({ k: r.pk, t: 'vec', x: 305, y: y + 20, v: r.pos, cell: 14, max: pmax, label: t({ en: `position ${j + 2}`, zh: `位置 ${j + 2}` }) });
        items.push({ k: 'eq' + r.k, t: 'text', x: 545, y: y + 38, anchor: 'middle', size: 22, bold: true, text: '=' });
        items.push({ k: 'x' + r.k, t: 'vec', x: 570, y: y + 20, v: r.x, cell: 18, max: pmax, label: t({ en: 'input vector', zh: '輸入向量' }) });
      });
      items.push({ k: 'cs', t: 'text', x: 450, y: 495, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: `similarity of the two token parts: 1.00 · of the two input vectors: ${d.cosX.toFixed(2)}`, zh: `兩個詞元部分的相似度：1.00 · 兩個輸入向量的相似度：${d.cosX.toFixed(2)}` }) });
    }

    if (step === 5) {
      items.push({
        k: 'chart', t: 'chart', x: 90, y: 50, w: 740, h: 320, ymin: 2.1, ymax: 2.6, xmax: 1500, every: experiments.variants.base.every,
        series: [{ pts: d.noPos, tone: 'bad' }, { pts: d.base, tone: 'a' }],
        xlabel: t({ en: 'training steps', zh: '訓練步數' }), ylabel: t({ en: 'held-out loss (lower is better)', zh: '驗證損失（越低越好）' }),
      });
      items.push({ k: 'fin1', t: 'text', x: 300, y: 450, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'a', text: t({ en: `with pos: ${last(d.base).toFixed(3)}`, zh: `有位置：${last(d.base).toFixed(3)}` }) });
      items.push({ k: 'fin2', t: 'text', x: 620, y: 450, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'bad', text: t({ en: `without: ${last(d.noPos).toFixed(3)}`, zh: `沒有位置：${last(d.noPos).toFixed(3)}` }) });
    }

    if (step === 6) {
      const n = 8;
      items.push({
        k: 'even', t: 'bars', x: 120, y: 120, w: 660, h: 280,
        items: Array.from({ length: n }, (_, i) => ({ label: t({ en: `sees ${i + 1}`, zh: `看到 ${i + 1}` }), v: 1 / (i + 1), hi: i === 0 })),
        title: t({ en: 'illustration: share of the start “.” if attention were spread evenly', zh: '示意：如果注意力平均分配，開頭「.」分到的比例' }),
      });
      items.push({ k: 'evn', t: 'text', x: 450, y: 450, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: 'later tokens see more tokens, so each one gets a smaller share', zh: '越後面的詞元看到越多詞元，所以每個分到的比例越小' }) });
    }

    if (step === 7) {
      const every = 0.25, xmax = 7;
      const wave = (k: number) => Array.from({ length: xmax / every + 1 }, (_, i) => Math.sin((i * every) / Math.pow(10000, (2 * k) / D)));
      items.push({
        k: 'waves', t: 'chart', x: 90, y: 60, w: 520, h: 280, ymin: -1, ymax: 1, xmax, every,
        series: [{ pts: wave(0), tone: 'a' }, { pts: wave(1), tone: 'b' }, { pts: wave(2), tone: 'muted' }],
        xlabel: t({ en: 'position', zh: '位置' }), ylabel: t({ en: 'fixed waves: 3 of the 16 numbers', zh: '固定波形：16 個數字中的 3 個' }),
      });
      const boxes = [
        { l: { en: 'learned table', zh: '學習得來的表' }, s: { en: 'our model', zh: '我們的模型' }, tone: 'a' as const },
        { l: { en: 'fixed waves', zh: '固定波形' }, s: { en: 'original Transformer', zh: '最早的 Transformer' }, tone: 'ink' as const },
        { l: { en: 'rotations (RoPE)', zh: '旋轉（RoPE）' }, s: { en: 'e.g. LLaMA', zh: '例如 LLaMA' }, tone: 'ink' as const },
      ];
      boxes.forEach((b, i) => items.push({ k: 'ob' + i, t: 'box', x: 660, y: 70 + i * 100, w: 210, h: 66, label: t(b.l), sub: t(b.s), tone: b.tone }));
      items.push({ k: 'wn', t: 'text', x: 350, y: 410, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'each position gets a different mix of fast and slow waves', zh: '每個位置都得到快慢不同的波形組合' }) });
    }

    if (step === 8) {
      const y = 190;
      items.push({ k: 'r1', t: 'box', x: 60, y, w: 170, h: 60, label: t({ en: 'token row', zh: '詞元列' }), sub: 'tok' });
      items.push({ k: 'plus', t: 'text', x: 265, y: y + 38, anchor: 'middle', size: 26, bold: true, text: '+' });
      items.push({ k: 'r2', t: 'box', x: 300, y, w: 170, h: 60, label: t({ en: 'position row', zh: '位置列' }), sub: 'pos', tone: 'a' });
      items.push({ k: 'eq', t: 'text', x: 505, y: y + 38, anchor: 'middle', size: 26, bold: true, text: '=' });
      items.push({ k: 'r3', t: 'box', x: 540, y, w: 300, h: 60, label: t({ en: 'what attention reads', zh: '注意力讀到的東西' }), sub: t({ en: 'what + where', zh: '是什麼 + 在哪裡' }) });
      items.push({ k: 'rs', t: 'text', x: 450, y: 340, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: `held-out loss (comparison runs): ${last(d.base).toFixed(3)} with positions · ${last(d.noPos).toFixed(3)} without`, zh: `驗證損失（比較實驗）：有位置 ${last(d.base).toFixed(3)} · 沒有位置 ${last(d.noPos).toFixed(3)}` }) });
    }

    const usesPos = step >= 2 && step !== 5 && step !== 6;
    return {
      metrics: [
        { label: { en: 'Position info', zh: '位置資訊' }, value: step <= 1 || step === 6 ? t({ en: 'none', zh: '無' }) : step === 5 ? t({ en: 'both runs', zh: '兩種都試' }) : t({ en: 'added', zh: '有加入' }) },
        { label: { en: 'Table pos', zh: '表 pos' }, value: usesPos || step === 5 ? '8 × 16' : '—' },
        { label: { en: 'Mix difference', zh: '混合差距' }, value: step <= 1 ? d.plainDiff.toFixed(3) : step === 4 ? d.realDiff.toFixed(3) : '—' },
        { label: { en: 'Loss: with / without', zh: '損失：有／無位置' }, value: step >= 5 ? `${last(d.base).toFixed(2)} / ${last(d.noPos).toFixed(2)}` : '—' },
      ],
      items,
      legend: step === 5
        ? [{ tone: 'a', text: { en: 'with positions', zh: '有位置' } }, { tone: 'bad', text: { en: 'without positions', zh: '沒有位置' } }]
        : step === 7
          ? [{ tone: 'a', text: { en: 'fastest wave', zh: '最快的波' } }, { tone: 'b', text: { en: 'slower wave', zh: '較慢的波' } }]
          : step <= 4 ? [{ tone: 'pos', text: { en: 'positive number', zh: '正數' } }, { tone: 'neg', text: { en: 'negative number', zh: '負數' } }] : undefined,
    };
  },
};
