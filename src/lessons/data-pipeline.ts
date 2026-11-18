import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { NAMES, TRAIN_NAMES, VAL_NAMES } from '../model/data';

// How the course's name list was built, then byte-pair encoding (BPE) learned live on it.

// Counts from the source package ("human-names" 1.0.13, English lists), worked out once
// from its files: they are not shipped with the course, only the cleaned 3,071 names are.
const SRC = { female: 2015, male: 1573, total: 3588, unique: 3475, longer: 404 };
// Letters per name among the 3,475 unique names (lengths 8+ come from the package files).
const LONG_LENGTHS: Record<number, number> = { 8: 266, 9: 110, 10: 22, 11: 5, 13: 1 };
const RAW = ['Aaliyah', 'Abbey', 'Ashley', 'Aaron', 'Abdullah', 'Ashton'];
const BOTH = ['Alex', 'Angel', 'Ashley'];

const EXAMPLES = ['emma', 'cecilia', 'anthony'];
const MAX_MERGES = 200;

interface Bpe {
  merges: { a: string; b: string; count: number }[];
  trainAvg: number[]; // average tokens per training name after 0, 1, 2, … merges
  valAvg: number[];
  topPairs: { pair: string; count: number }[][]; // the 8 most frequent pairs before merge 1 and before merge 2
  split: Record<number, string[][]>; // the example names after 0, 1, 10, 200 merges
}

function countPairs(seqs: string[][]): Map<string, number> {
  const c = new Map<string, number>();
  for (const s of seqs) for (let i = 0; i < s.length - 1; i++) {
    const k = s[i] + '\u0001' + s[i + 1];
    c.set(k, (c.get(k) ?? 0) + 1);
  }
  return c;
}
function merge(s: string[], a: string, b: string): string[] {
  const o: string[] = [];
  for (let i = 0; i < s.length; i++) {
    if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]);
  }
  return o;
}
const avg = (seqs: string[][]) => seqs.reduce((n, s) => n + s.length, 0) / seqs.length;

// Learn the merges on the training names only, then apply the same merges to held-out names.
function learn(): Bpe {
  let train = TRAIN_NAMES.map((n) => [...n]);
  let val = VAL_NAMES.map((n) => [...n]);
  let ex = EXAMPLES.map((n) => [...n]);
  const out: Bpe = { merges: [], trainAvg: [avg(train)], valAvg: [avg(val)], topPairs: [], split: { 0: ex } };
  for (let m = 0; m < MAX_MERGES; m++) {
    const c = countPairs(train);
    const ranked = [...c.entries()].sort((x, y) => y[1] - x[1]); // stable: ties keep first-seen order
    if (m < 2) out.topPairs.push(ranked.slice(0, 8).map(([k, v]) => ({ pair: k.replace('\u0001', ' + '), count: v })));
    const [best, count] = ranked[0];
    const [a, b] = best.split('\u0001');
    train = train.map((s) => merge(s, a, b));
    val = val.map((s) => merge(s, a, b));
    ex = ex.map((s) => merge(s, a, b));
    out.merges.push({ a, b, count });
    out.trainAvg.push(avg(train));
    out.valAvg.push(avg(val));
    if (m + 1 === 1 || m + 1 === 10 || m + 1 === MAX_MERGES) out.split[m + 1] = ex;
  }
  return out;
}
let cache: Bpe | undefined;
const bpe = () => (cache ??= learn());

const fmt = (n: number) => n.toLocaleString('en-US');
const f2 = (v: number) => v.toFixed(2);
const PER_TOKEN = 16 + 16 + 1; // one row of the embedding table + one column of the output matrix + one output bias

const lengthsOfUnique = () => {
  const c: Record<number, number> = { ...LONG_LENGTHS };
  for (const n of NAMES) c[n.length] = (c[n.length] ?? 0) + 1;
  return c;
};

export const dataPipeline: Lesson = {
  id: 'data-pipeline',
  title: { en: 'Data and Tokenizer Pipeline', zh: '資料與分詞流程' },
  blurb: {
    en: 'From a raw list of names to clean training data, then a tokenizer that learns its own pieces.',
    zh: '從一份原始名字清單，到乾淨的訓練資料，再到一個自己學出詞元的分詞器。',
  },
  minutes: 12,
  level: 'intermediate',
  learn: [
    { text: { en: 'Why raw data must be cleaned first', zh: '為什麼原始資料要先清理' }, step: 1 },
    { text: { en: 'Why some data is held back for testing', zh: '為什麼要留一部分資料做測試' }, step: 3 },
    { text: { en: 'How byte-pair encoding learns tokens', zh: '位元組對編碼（BPE）如何學出詞元' }, step: 5 },
    { text: { en: 'The trade-off: shorter sequences vs a bigger vocabulary', zh: '取捨：較短的序列 vs 較大的詞彙表' }, step: 7 },
    { text: { en: 'How real tokenizers are built', zh: '真實的分詞器是怎麼做出來的' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'Raw list', zh: '原始清單' }, chapter: 'problem',
      title: { en: 'A raw list is not training data yet', zh: '原始清單還不是訓練資料' },
      body: [
        { en: `Our names come from the MIT-licensed <code>human-names</code> package: ${fmt(SRC.female)} English female names plus ${fmt(SRC.male)} male names, <b>${fmt(SRC.total)}</b> in all.`, zh: `我們的名字來自採用 MIT 授權的 <code>human-names</code> 套件：${fmt(SRC.female)} 個英文女性名字，加上 ${fmt(SRC.male)} 個男性名字，一共 <b>${fmt(SRC.total)}</b> 個。` },
        { en: 'They start with capital letters, some appear twice, some are too long for our model, and nothing is set aside to test the model on names it has never seen.', zh: '這些名字開頭是大寫，有些重複出現，有些對我們的模型來說太長，而且也沒有預留任何名字，用來測試模型面對沒看過的名字表現如何。' },
      ],
      callout: { kind: 'insight', text: { en: 'A model can only be as good as its data. Real projects spend a lot of their effort here, before any training starts.', zh: '模型的好壞，取決於它的資料。真實的專案在開始訓練之前，會在這一步花上大量心力。' } },
    },
    {
      short: { en: 'Clean', zh: '清理' }, chapter: 'technique',
      title: { en: 'Make every name look the same', zh: '讓每個名字的格式一致' },
      body: [
        { en: 'First we lowercase everything and keep only the letters a–z, because our vocabulary has nothing else. In this list every name already passed that check, so none were lost.', zh: '首先把所有字母改成小寫，並且只保留 a–z，因為我們的詞彙表裡沒有其他符號。這份清單裡每個名字都已經符合這個條件，所以沒有任何名字被刪掉。' },
        { en: `Then we remove repeats. ${fmt(SRC.total - SRC.unique)} names, such as Alex and Ashley, are on both the female and the male list. Keeping them twice would make them count double. ${fmt(SRC.unique)} names remain.`, zh: `接著移除重複。有 ${fmt(SRC.total - SRC.unique)} 個名字（例如 Alex、Ashley）同時出現在女性和男性清單上，如果保留兩份，它們就會被算兩次。最後剩下 ${fmt(SRC.unique)} 個名字。` },
      ],
      changed: [
        { sign: '~', text: { en: 'Lowercase, letters a–z only', zh: '全部小寫，只留 a–z' } },
        { sign: '~', text: { en: `${fmt(SRC.total - SRC.unique)} duplicates removed`, zh: `移除 ${fmt(SRC.total - SRC.unique)} 個重複` } },
      ],
    },
    {
      short: { en: 'Length', zh: '長度' }, chapter: 'technique',
      title: { en: 'Fit the model’s window', zh: '配合模型的視窗大小' },
      body: [
        { en: 'Our model reads at most <b>8 tokens</b>, and the first one is always the start mark <code>.</code>. That leaves room for 7 letters. A longer name would be cut off, and the model would never learn how it ends.', zh: '我們的模型一次最多讀 <b>8 個詞元</b>，而第一個永遠是開頭符號 <code>.</code>，所以最多只能放 7 個字母。更長的名字會被截斷，模型就永遠學不到它怎麼結尾。' },
        { en: `So we keep names of 2 to 7 letters (the teal bars). ${fmt(SRC.longer)} longer names go, and <b>${fmt(NAMES.length)}</b> remain.`, zh: `所以我們只保留 2 到 7 個字母的名字（青綠色的長條）。${fmt(SRC.longer)} 個較長的名字被移除，剩下 <b>${fmt(NAMES.length)}</b> 個。` },
      ],
      callout: { kind: 'note', text: { en: 'Real models face the same limit with a much longer window: long documents are split into pieces that fit.', zh: '真實的模型也有同樣的限制，只是視窗長得多：很長的文件會被切成放得進去的片段。' } },
      changed: [{ sign: '~', text: { en: `${fmt(SRC.longer)} names longer than 7 letters removed`, zh: `移除 ${fmt(SRC.longer)} 個超過 7 個字母的名字` } }],
    },
    {
      short: { en: 'Split', zh: '切分' }, chapter: 'technique',
      title: { en: 'Shuffle, then hold some back', zh: '打亂，再保留一部分' },
      body: [
        { en: 'The list was alphabetical, so we shuffle it first. Then every fifth name goes into a <b>held-out</b> set (also called a validation set) that training never touches.', zh: '原本的清單是按字母排序的，所以要先打亂。接著每五個名字取一個，放進<b>保留集</b>（也就是驗證集），訓練時完全不碰它。' },
        { en: `That gives <b>${fmt(TRAIN_NAMES.length)}</b> training names and <b>${fmt(VAL_NAMES.length)}</b> held-out names. Only the held-out loss tells us whether the model learned how names work, rather than memorizing a list.`, zh: `這樣得到 <b>${fmt(TRAIN_NAMES.length)}</b> 個訓練用的名字和 <b>${fmt(VAL_NAMES.length)}</b> 個保留的名字。只有保留集上的損失，才能告訴我們模型是真的學會名字的規律，而不是背下一份清單。` },
      ],
      callout: { kind: 'insight', text: { en: 'Shuffling matters: without it, the held-out set would be skewed toward some letters and the test would be unfair.', zh: '打亂很重要：如果不打亂，保留集就可能偏向某些字母，測試就不公平了。' } },
      changed: [{ sign: '+', text: { en: `${fmt(TRAIN_NAMES.length)} training / ${fmt(VAL_NAMES.length)} held-out`, zh: `訓練 ${fmt(TRAIN_NAMES.length)} / 保留 ${fmt(VAL_NAMES.length)}` } }],
    },
    {
      short: { en: 'Count pairs', zh: '數字母對' }, chapter: 'technique',
      title: { en: 'Which two tokens sit side by side most?', zh: '哪兩個詞元最常相鄰？' },
      body: [
        { en: 'Our tokenizer is one letter per token. Real models use bigger pieces, learned from data by <b>byte-pair encoding</b> (BPE). We can run it on our own names right now.', zh: '我們的分詞器是一個字母一個詞元。真實的模型使用更大的片段，這些片段是用<b>位元組對編碼</b>（byte-pair encoding，BPE）從資料中學出來的。我們現在就可以拿自己的名字來跑一次。' },
        { en: 'Step one: in all 2,456 training names, count every pair of neighbouring tokens. The winner is “a + n”, side by side 333 times (as in anthony, ana, brian).', zh: '第一步：在全部 2,456 個訓練名字裡，數出每一對相鄰的詞元。冠軍是「a + n」，相鄰出現 333 次（像 anthony、ana、brian）。' },
      ],
      changed: [{ sign: '+', text: { en: 'Pair counts over the training names', zh: '訓練名字中每一對的次數' } }],
    },
    {
      short: { en: 'Merge', zh: '合併' }, chapter: 'technique',
      title: { en: 'Merge the winner, then repeat', zh: '合併冠軍，然後重複' },
      body: [
        { en: 'Step two: make “an” a new token and replace every a-next-to-n with it. Then count again and merge the new winner. That is the whole algorithm.', zh: '第二步：把「an」變成一個新的詞元，並把所有相鄰的 a、n 換成它。然後重新計數，再合併新的冠軍。整個演算法就是這樣。' },
        { en: 'Here are the first 10 merges. Look how the example names now need fewer tokens: <code>emma</code> becomes em + ma.', zh: '這裡是前 10 次合併。看看範例名字現在需要的詞元變少了：<code>emma</code> 變成 em + ma。' },
      ],
      callout: { kind: 'note', text: { en: 'Merges can build on earlier merges: later on, “t + on” joins the token “on” made by merge 3.', zh: '合併可以建立在先前的合併上：之後的「t + on」，就是接上第 3 次合併做出來的「on」。' } },
      changed: [{ sign: '+', text: { en: '10 merges: an, ar, on, el, en, …', zh: '10 次合併：an、ar、on、el、en……' } }],
    },
    {
      short: { en: 'Grow', zh: '成長' }, chapter: 'solution',
      title: { en: 'Bigger vocabulary, shorter names', zh: '詞彙表變大，名字變短' },
      body: [
        { en: 'Every merge adds one token to the vocabulary and makes the average name shorter. After 200 merges our vocabulary has 227 tokens and a training name needs about 2.9 tokens instead of 5.4.', zh: '每合併一次，詞彙表就多一個詞元，而平均每個名字需要的詞元就變少。合併 200 次之後，詞彙表有 227 個詞元，一個訓練名字平均只要大約 2.9 個詞元，而不是 5.4 個。' },
        { en: 'The amber line is the held-out names, which BPE never saw. It shrinks almost the same way, so the pieces are real patterns of names, not a memorized list.', zh: '琥珀色的線是 BPE 從沒看過的保留名字，它也幾乎以同樣的方式變短。這表示學到的片段是名字真正的規律，而不是背下來的清單。' },
      ],
      changed: [{ sign: '~', text: { en: 'Vocabulary 27 → 227, tokens per name 5.4 → 2.9', zh: '詞彙表 27 → 227，每個名字的詞元 5.4 → 2.9' } }],
    },
    {
      short: { en: 'Case: limits', zh: '案例：極限' }, chapter: 'problem',
      title: { en: 'Case: each merge helps less', zh: '案例：每次合併的幫助越來越小' },
      body: [
        { en: 'The chart shows how many tokens each merge saves. The first saves 333; by merge 200 a merge saves only 7. Meanwhile every new token costs weights.', zh: '圖中是每次合併能省下多少詞元。第一次省下 333 個；到第 200 次，一次只省 7 個。與此同時，每個新詞元都要付出權重的代價。' },
        { en: `In our model each token needs ${PER_TOKEN} weights (a row in the embedding table, a column in the output matrix, and one bias). 200 merges would add ${fmt(200 * PER_TOKEN)} weights, more than the whole model has now.`, zh: `在我們的模型裡，每個詞元需要 ${PER_TOKEN} 個權重（嵌入表的一列、輸出矩陣的一欄，以及一個偏差值）。合併 200 次會多出 ${fmt(200 * PER_TOKEN)} 個權重，比整個模型現在的權重還多。` },
      ],
      callout: { kind: 'insight', text: { en: 'Choosing the vocabulary size is a trade-off: shorter sequences against a bigger model and rarer tokens. For our tiny list we keep plain letters.', zh: '決定詞彙表大小是一種取捨：序列變短，但模型變大、詞元也變得更少見。對我們這份小清單來說，我們維持單純的字母。' } },
    },
    {
      short: { en: 'Real tokenizers', zh: '真實分詞器' }, chapter: 'technique',
      title: { en: 'Real tokenizers use the same recipe', zh: '真實的分詞器用的是同一套方法' },
      body: [
        { en: 'BPE for language models was introduced by Sennrich and colleagues (2016). Real tokenizers run it over a huge amount of text, usually starting from the 256 possible <b>bytes</b> instead of letters, so any text in any language can be written.', zh: '把 BPE 用在語言模型上，是由 Sennrich 等人（2016）提出的。真實的分詞器會在海量文字上執行它，而且通常從 256 種可能的<b>位元組</b>開始，而不是從字母開始，因此任何語言的任何文字都寫得出來。' },
        { en: 'GPT-2, for example, learned 50,000 merges: 256 bytes + 50,000 merges + 1 end-of-text token = 50,257 tokens.', zh: '舉例來說，GPT-2 學了 50,000 次合併：256 個位元組 + 50,000 次合併 + 1 個「文字結束」詞元 = 50,257 個詞元。' },
      ],
      callout: { kind: 'note', text: { en: 'The tokenizer is trained once, before the model, and then frozen. The model only ever sees the ids it produces.', zh: '分詞器會在模型之前先訓練一次，之後就固定不變。模型看到的永遠只有它產生的編號。' } },
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'From a raw list to tokens', zh: '從原始清單到詞元' },
      body: [
        { en: `<b>Clean</b>: lowercase, a–z, no repeats (${fmt(SRC.total)} → ${fmt(SRC.unique)}).<br><b>Fit</b>: 2–7 letters for an 8-token window (→ ${fmt(NAMES.length)}).<br><b>Split</b>: shuffle, ${fmt(TRAIN_NAMES.length)} training / ${fmt(VAL_NAMES.length)} held-out.<br><b>BPE</b>: merge the most frequent pair, repeat; vocabulary grows, sequences shrink.`, zh: `<b>清理</b>：小寫、只留 a–z、移除重複（${fmt(SRC.total)} → ${fmt(SRC.unique)}）。<br><b>配合視窗</b>：8 個詞元的視窗，只留 2–7 個字母（→ ${fmt(NAMES.length)}）。<br><b>切分</b>：打亂後分成訓練 ${fmt(TRAIN_NAMES.length)} / 保留 ${fmt(VAL_NAMES.length)}。<br><b>BPE</b>：合併最常見的一對，重複；詞彙表變大，序列變短。` },
      ],
      callout: { kind: 'next', text: { en: 'Next: Compute and Mixed Precision. How much arithmetic does training take, and why do big models do it with 16-bit numbers?', zh: '下一課：運算資源與混合精度。訓練到底要做多少運算？為什麼大模型要用 16 位元的數字來算？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    let names = '—', perName = '—', vocab = '27', merges = '0';

    // ── steps 0–3: the cleaning funnel on the left, details on the right ──
    if (step <= 3) {
      const stages = [
        { label: t({ en: `${fmt(SRC.total)} names`, zh: `${fmt(SRC.total)} 個名字` }), sub: t({ en: 'source list', zh: '原始清單' }) },
        { label: t({ en: `${fmt(SRC.unique)} names`, zh: `${fmt(SRC.unique)} 個名字` }), sub: t({ en: 'lowercase · a–z · no repeats', zh: '小寫 · a–z · 不重複' }) },
        { label: t({ en: `${fmt(NAMES.length)} names`, zh: `${fmt(NAMES.length)} 個名字` }), sub: t({ en: '2–7 letters', zh: '2–7 個字母' }) },
      ];
      stages.forEach((s, i) => {
        if (i > step) return;
        items.push({ k: 'st' + i, t: 'box', x: 30, y: 40 + i * 100, w: 250, h: 60, label: s.label, sub: s.sub, tone: i === step ? 'a' : 'ink', active: i === step });
        if (i > 0) items.push({ k: 'sa' + i, t: 'arrow', x1: 155, y1: 104 + (i - 1) * 100, x2: 155, y2: 136 + (i - 1) * 100 });
      });
      if (step === 3) {
        items.push({ k: 'sa3', t: 'arrow', x1: 120, y1: 304, x2: 90, y2: 356 });
        items.push({ k: 'sa4', t: 'arrow', x1: 190, y1: 304, x2: 220, y2: 356 });
        items.push({ k: 'trainbox', t: 'box', x: 20, y: 360, w: 130, h: 60, label: fmt(TRAIN_NAMES.length), sub: t({ en: 'training', zh: '訓練集' }), tone: 'a' });
        items.push({ k: 'valbox', t: 'box', x: 160, y: 360, w: 130, h: 60, label: fmt(VAL_NAMES.length), sub: t({ en: 'held-out', zh: '保留集' }), tone: 'b' });
      }
      names = step === 0 ? fmt(SRC.total) : step === 1 ? fmt(SRC.unique) : step === 2 ? fmt(NAMES.length) : fmt(TRAIN_NAMES.length);
      if (step === 2) perName = f2(NAMES.reduce((n, s) => n + s.length, 0) / NAMES.length);
      if (step === 3) perName = f2(bpe().trainAvg[0]);
    }

    if (step === 0) {
      items.push({ k: 'rawl', t: 'text', x: 360, y: 70, text: t({ en: 'as they appear in the package', zh: '套件裡的原始樣子' }), tone: 'muted', size: 14 });
      RAW.forEach((n, i) => items.push({ k: 'raw' + i, t: 'box', x: 360 + (i % 3) * 170, y: 90 + Math.floor(i / 3) * 62, w: 150, h: 46, label: n }));
      items.push({ k: 'rawf', t: 'text', x: 360, y: 250, text: t({ en: `${fmt(SRC.female)} female + ${fmt(SRC.male)} male = ${fmt(SRC.total)}`, zh: `女性 ${fmt(SRC.female)} + 男性 ${fmt(SRC.male)} = ${fmt(SRC.total)}` }), size: 16, bold: true });
      const issues = [
        t({ en: '· capital letters', zh: '· 有大寫字母' }),
        t({ en: '· the same name on both lists', zh: '· 同一個名字出現在兩份清單' }),
        t({ en: '· names too long for the model', zh: '· 名字對模型來說太長' }),
        t({ en: '· nothing held back for testing', zh: '· 沒有保留資料做測試' }),
      ];
      issues.forEach((s, i) => items.push({ k: 'iss' + i, t: 'text', x: 360, y: 300 + i * 30, text: s, tone: 'bad', size: 15 }));
    }

    if (step === 1) {
      items.push({ k: 'rawl', t: 'text', x: 360, y: 70, text: t({ en: 'on both lists → kept once, lowercase', zh: '兩份清單都有 → 只留一份，改小寫' }), tone: 'muted', size: 14 });
      BOTH.forEach((n, i) => {
        const y = 100 + i * 80;
        items.push({ k: 'bf' + i, t: 'box', x: 360, y, w: 120, h: 44, label: n, sub: undefined });
        items.push({ k: 'bm' + i, t: 'box', x: 490, y, w: 120, h: 44, label: n });
        items.push({ k: 'bl' + i, t: 'text', x: 420, y: y + 62, text: t({ en: 'female', zh: '女性' }), anchor: 'middle', tone: 'muted', size: 13 });
        items.push({ k: 'bl2' + i, t: 'text', x: 550, y: y + 62, text: t({ en: 'male', zh: '男性' }), anchor: 'middle', tone: 'muted', size: 13 });
        items.push({ k: 'ba' + i, t: 'arrow', x1: 616, y1: y + 22, x2: 690, y2: y + 22, tone: 'a' });
        items.push({ k: 'bo' + i, t: 'box', x: 696, y, w: 120, h: 44, label: n.toLowerCase(), tone: 'a' });
      });
      items.push({ k: 'dupn', t: 'text', x: 360, y: 370, text: t({ en: `${fmt(SRC.total - SRC.unique)} names like these → ${fmt(SRC.total)} − ${fmt(SRC.total - SRC.unique)} = ${fmt(SRC.unique)}`, zh: `像這樣的名字有 ${fmt(SRC.total - SRC.unique)} 個 → ${fmt(SRC.total)} − ${fmt(SRC.total - SRC.unique)} = ${fmt(SRC.unique)}` }), size: 15, bold: true });
    }

    if (step === 2) {
      const c = lengthsOfUnique();
      const lens = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13];
      items.push({
        k: 'lenbars', t: 'bars', x: 330, y: 90, w: 540, h: 260,
        items: lens.map((n) => ({ label: String(n), v: (c[n] ?? 0) / SRC.unique, hi: n <= 7 })),
        title: t({ en: `letters per name · share of the ${fmt(SRC.unique)} names`, zh: `每個名字的字母數 · 佔 ${fmt(SRC.unique)} 個名字的比例` }),
      });
      items.push({ k: 'win', t: 'text', x: 330, y: 400, text: t({ en: 'window: . + up to 7 letters = 8 tokens', zh: '視窗：. + 最多 7 個字母 = 8 個詞元' }), size: 15, bold: true, tone: 'a' });
      items.push({ k: 'winn', t: 'text', x: 330, y: 428, text: t({ en: `kept ${fmt(NAMES.length)} · removed ${fmt(SRC.longer)} with 8 or more letters`, zh: `保留 ${fmt(NAMES.length)} 個 · 移除 ${fmt(SRC.longer)} 個 8 個字母以上的名字` }), size: 14, tone: 'muted' });
    }

    if (step === 3) {
      items.push({ k: 'rawl', t: 'text', x: 360, y: 70, text: t({ en: 'the first 15 names after shuffling', zh: '打亂後的前 15 個名字' }), tone: 'muted', size: 14 });
      NAMES.slice(0, 15).forEach((n, i) => {
        const held = i % 5 === 0;
        items.push({ k: 'sh' + i, t: 'box', x: 360 + (i % 5) * 100, y: 90 + Math.floor(i / 5) * 60, w: 90, h: 44, label: n, tone: held ? 'b' : 'a' });
      });
      items.push({ k: 'shn', t: 'text', x: 360, y: 300, text: t({ en: 'every fifth name (amber) is held out', zh: '每五個取一個（琥珀色）放進保留集' }), size: 15, bold: true, tone: 'b' });
      items.push({ k: 'shn2', t: 'text', x: 360, y: 330, text: t({ en: 'the model trains only on the teal ones', zh: '模型只用青綠色的名字訓練' }), size: 14, tone: 'muted' });
    }

    // ── steps 4–8: byte-pair encoding on the training names ──
    if (step >= 4 && step <= 7) {
      const B = bpe();
      const at = step === 4 ? 0 : step === 5 ? 10 : 200;
      items.push({ k: 'trainbox', t: 'box', x: 20, y: 20, w: 190, h: 56, label: t({ en: `${fmt(TRAIN_NAMES.length)} names`, zh: `${fmt(TRAIN_NAMES.length)} 個名字` }), sub: t({ en: 'training set', zh: '訓練集' }), tone: 'a' });
      // the example names, split into the current tokens
      B.split[at].forEach((parts, r) => {
        const y = 20 + r * 52;
        let x = 250;
        parts.forEach((p, j) => {
          const w = 30 + p.length * 11;
          items.push({ k: `ex${r}-${j}`, t: 'box', x, y, w, h: 40, label: p, tone: p.length > 1 ? 'a' : 'ink' });
          x += w + 6;
        });
        items.push({ k: 'exn' + r, t: 'text', x: 880, y: y + 26, text: t({ en: `${parts.length} tokens`, zh: `${parts.length} 個詞元` }), anchor: 'end', tone: 'muted', size: 14 });
      });
      perName = f2(B.trainAvg[at]);
      vocab = String(27 + at);
      merges = String(at);
      names = fmt(TRAIN_NAMES.length);

      if (step === 4) {
        items.push({ k: 'pl', t: 'text', x: 60, y: 210, text: t({ en: 'most frequent neighbouring pairs in the training names', zh: '訓練名字中最常相鄰的詞元對' }), tone: 'muted', size: 14 });
        const top = B.topPairs[0];
        top.forEach((p, i) => {
          const y = 226 + i * 34;
          items.push({ k: 'pp' + i, t: 'text', x: 60, y: y + 20, text: p.pair, mono: true, bold: i === 0, tone: i === 0 ? 'a' : 'ink', size: 15 });
          items.push({ k: 'pb' + i, t: 'box', x: 140, y, w: Math.max(6, (p.count / top[0].count) * 620), h: 26, tone: i === 0 ? 'a' : 'ink', active: i === 0 });
          items.push({ k: 'pc' + i, t: 'text', x: 150 + (p.count / top[0].count) * 620, y: y + 18, text: fmt(p.count), tone: 'muted', size: 13, mono: true });
        });
      }
      if (step === 5) {
        items.push({ k: 'pl', t: 'text', x: 60, y: 210, text: t({ en: 'the first 10 merges · how often the pair appeared', zh: '前 10 次合併 · 這一對出現的次數' }), tone: 'muted', size: 14 });
        B.merges.slice(0, 10).forEach((m, i) => {
          const x = 60 + Math.floor(i / 5) * 410, y = 226 + (i % 5) * 50;
          items.push({ k: 'mg' + i, t: 'box', x, y, w: 380, h: 40, label: `${i + 1}.  ${m.a} + ${m.b} → ${m.a + m.b}   ·   ${fmt(m.count)}`, tone: i === 0 ? 'a' : 'ink' });
        });
      }
      if (step === 6 || step === 7) {
        const chart: Item = step === 6
          ? {
            k: 'bchart', t: 'chart', x: 90, y: 210, w: 760, h: 230, ymin: 2, ymax: 6, xmax: MAX_MERGES, every: 1,
            series: [{ pts: B.valAvg, tone: 'b' }, { pts: B.trainAvg, tone: 'a' }],
            xlabel: t({ en: 'merges (each one adds a token to the vocabulary)', zh: '合併次數（每次都讓詞彙表多一個詞元）' }), ylabel: t({ en: 'average tokens per name', zh: '每個名字平均的詞元數' }),
          }
          : {
            k: 'bchart', t: 'chart', x: 90, y: 210, w: 760, h: 230, ymin: 0, ymax: 400, xmax: MAX_MERGES, every: 1,
            series: [{ pts: B.merges.map((m) => m.count), tone: 'bad' }],
            xlabel: t({ en: 'merge number', zh: '第幾次合併' }), ylabel: t({ en: 'tokens saved by this merge', zh: '這次合併省下的詞元數' }),
          };
        items.push(chart);
      }
    }

    if (step === 8) {
      const row = (k: string, y: number, parts: { label: string; sub: string; tone?: 'a' | 'b' | 'ink' }[]) => {
        parts.forEach((p, i) => {
          items.push({ k: `${k}${i}`, t: 'box', x: 60 + i * 280, y, w: 220, h: 64, label: p.label, sub: p.sub, tone: p.tone });
          if (i > 0) items.push({ k: `${k}a${i}`, t: 'arrow', x1: 60 + i * 280 - 56, y1: y + 32, x2: 60 + i * 280 - 4, y2: y + 32 });
        });
      };
      items.push({ k: 'ours', t: 'text', x: 60, y: 90, text: t({ en: 'our names (computed above)', zh: '我們的名字（上面實際算出）' }), tone: 'muted', size: 14 });
      row('or', 104, [
        { label: t({ en: '27 tokens', zh: '27 個詞元' }), sub: t({ en: '. and a–z', zh: '. 與 a–z' }) },
        { label: t({ en: '+ 200 merges', zh: '+ 200 次合併' }), sub: t({ en: `on ${fmt(TRAIN_NAMES.length)} names`, zh: `在 ${fmt(TRAIN_NAMES.length)} 個名字上` }) },
        { label: t({ en: '227 tokens', zh: '227 個詞元' }), sub: t({ en: 'vocabulary', zh: '詞彙表' }), tone: 'a' },
      ]);
      items.push({ k: 'gpt', t: 'text', x: 60, y: 270, text: t({ en: 'GPT-2’s tokenizer (published numbers)', zh: 'GPT-2 的分詞器（公開的數字）' }), tone: 'muted', size: 14 });
      row('gp', 284, [
        { label: t({ en: '256 bytes', zh: '256 個位元組' }), sub: t({ en: 'any text can be written', zh: '任何文字都寫得出來' }) },
        { label: t({ en: '+ 50,000 merges', zh: '+ 50,000 次合併' }), sub: t({ en: 'learned from lots of text', zh: '從大量文字中學出' }) },
        { label: t({ en: '50,257 tokens', zh: '50,257 個詞元' }), sub: t({ en: '+ 1 end-of-text token', zh: '+ 1 個文字結束詞元' }), tone: 'b' },
      ]);
      items.push({ k: 'same', t: 'text', x: 450, y: 440, text: t({ en: 'same algorithm, much more data', zh: '同一個演算法，只是資料多得多' }), anchor: 'middle', size: 16, bold: true });
      names = fmt(TRAIN_NAMES.length); vocab = '227'; merges = '200'; perName = f2(bpe().trainAvg[200]);
    }

    if (step === 9) {
      const boxes = [
        { label: fmt(SRC.total), sub: t({ en: 'raw', zh: '原始' }) },
        { label: fmt(SRC.unique), sub: t({ en: 'clean', zh: '清理後' }) },
        { label: fmt(NAMES.length), sub: t({ en: '2–7 letters', zh: '2–7 個字母' }) },
        { label: `${fmt(TRAIN_NAMES.length)} / ${fmt(VAL_NAMES.length)}`, sub: t({ en: 'train / held-out', zh: '訓練 / 保留' }), tone: 'a' as const },
      ];
      boxes.forEach((b, i) => {
        items.push({ k: 'rc' + i, t: 'box', x: 40 + i * 215, y: 130, w: 175, h: 64, label: b.label, sub: b.sub, tone: b.tone });
        if (i > 0) items.push({ k: 'rca' + i, t: 'arrow', x1: 40 + i * 215 - 38, y1: 162, x2: 40 + i * 215 - 4, y2: 162 });
      });
      items.push({ k: 'rcb', t: 'arrow', x1: 772, y1: 198, x2: 772, y2: 262, tone: 'a' });
      items.push({ k: 'rbpe', t: 'box', x: 470, y: 268, w: 400, h: 64, label: t({ en: 'tokenizer: letters, or BPE pieces', zh: '分詞器：字母，或 BPE 片段' }), sub: t({ en: '200 merges: 5.4 → 2.9 tokens per name', zh: '200 次合併：每個名字 5.4 → 2.9 個詞元' }), tone: 'b' });
      items.push({ k: 'rmod', t: 'box', x: 470, y: 380, w: 400, h: 56, label: t({ en: 'the model sees only ids', zh: '模型看到的只有編號' }) });
      items.push({ k: 'rcc', t: 'arrow', x1: 670, y1: 336, x2: 670, y2: 376 });
      names = fmt(TRAIN_NAMES.length); perName = f2(bpe().trainAvg[0]);
    }

    return {
      metrics: [
        { label: { en: 'Names', zh: '名字數' }, value: names },
        { label: { en: 'Tokens per name', zh: '每名詞元數' }, value: perName },
        { label: { en: 'Vocabulary', zh: '詞彙表' }, value: vocab },
        { label: { en: 'Merges', zh: '合併次數' }, value: merges },
      ],
      items,
      legend: step === 6
        ? [{ tone: 'a', text: { en: 'training names', zh: '訓練名字' } }, { tone: 'b', text: { en: 'held-out names (BPE never saw them)', zh: '保留名字（BPE 沒看過）' } }]
        : step === 7 ? [{ tone: 'bad', text: { en: 'tokens saved by each merge', zh: '每次合併省下的詞元' } }] : undefined,
    };
  },
};
