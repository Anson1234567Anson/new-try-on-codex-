import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { CHARS, NAMES, encode } from '../model/data';

const NAME = 'emma';
const ids = encode(NAME); // [5, 13, 13, 1]
const full = encode('.' + NAME + '.'); // [0, 5, 13, 13, 1, 0]

// vocabulary grid: 9 columns × 3 rows
const vocabAt = (i: number) => ({ x: 196 + (i % 9) * 58, y: 250 + Math.floor(i / 9) * 82 });

export const tokens: Lesson = {
  id: 'tokens',
  title: { en: 'Tokens', zh: '詞元（Token）' },
  blurb: {
    en: 'A model can only do arithmetic. See how text becomes numbers it can work with.',
    zh: '模型只會做算術。來看文字怎麼變成它能處理的數字。',
  },
  minutes: 8,
  level: 'beginner',
  learn: [
    { text: { en: 'Why text must become numbers', zh: '為什麼文字必須先變成數字' }, step: 0 },
    { text: { en: 'What a token and a vocabulary are', zh: '什麼是詞元與詞彙表' }, step: 2 },
    { text: { en: 'How a name becomes a list of ids', zh: '一個名字如何變成一串編號' }, step: 3 },
    { text: { en: 'How one name gives many training examples', zh: '一個名字如何變成多筆訓練資料' }, step: 5 },
    { text: { en: 'How real models split text', zh: '真實的模型如何切分文字' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'A model cannot read', zh: '模型不會「讀」字' },
      body: [
        { en: 'Through this course we will build a tiny model that invents first names. We teach it with a list of real names, such as <code>emma</code>.', zh: '在這門課裡，我們會打造一個會自己發明英文名字的小模型。我們用一份真實的名字清單來教它，例如 <code>emma</code>。' },
        { en: 'But inside, a model is only multiplication and addition. It has no idea what the letter “e” is. Before anything else, text has to become numbers.', zh: '但是模型的內部只有乘法和加法，它完全不知道字母「e」是什麼。所以第一件事，就是把文字變成數字。' },
      ],
      callout: { kind: 'insight', text: { en: 'Everything a language model does starts with turning text into numbers, and ends with turning numbers back into text.', zh: '語言模型做的每一件事，都從「把文字變成數字」開始，並以「把數字變回文字」結束。' } },
    },
    {
      short: { en: 'Split', zh: '切開' }, chapter: 'technique',
      title: { en: 'Cut the text into pieces', zh: '把文字切成小塊' },
      body: [
        { en: 'First we cut the text into small pieces called <b>tokens</b>. Our model uses the simplest possible rule: one letter is one token.', zh: '首先，把文字切成一小塊一小塊，每一塊叫做<b>詞元（token）</b>。我們的模型用最簡單的規則：一個字母就是一個詞元。' },
        { en: 'So <code>emma</code> becomes four tokens: e, m, m, a.', zh: '所以 <code>emma</code> 變成四個詞元：e、m、m、a。' },
      ],
      changed: [{ sign: '+', text: { en: '4 tokens, one per letter', zh: '4 個詞元，每個字母一個' } }],
    },
    {
      short: { en: 'Vocabulary', zh: '詞彙表' }, chapter: 'technique',
      title: { en: 'List every token that can exist', zh: '列出所有可能的詞元' },
      body: [
        { en: 'Next we write down every token the model will ever see. This list is the <b>vocabulary</b>. Ours has the 26 letters, plus one special mark we will meet in a moment: 27 entries.', zh: '接著，把模型可能看到的所有詞元列成一張表，這張表叫做<b>詞彙表（vocabulary）</b>。我們的表有 26 個字母，再加上一個等一下會介紹的特殊符號，一共 27 項。' },
        { en: 'Each entry gets a number, its <b>id</b>, simply its position in the list.', zh: '每一項都有一個號碼，叫做<b>編號（id）</b>，其實就是它在表中的位置。' },
      ],
      callout: { kind: 'note', text: { en: 'The ids carry no meaning. “m” being 13 does not make it bigger than “a”. Meaning comes later, in the Embeddings lesson.', zh: '編號本身沒有意義。「m」是 13 並不代表它比「a」大。意義要到後面的「嵌入」那一課才會出現。' } },
      changed: [{ sign: '+', text: { en: 'Vocabulary of 27 tokens', zh: '27 個詞元的詞彙表' } }],
    },
    {
      short: { en: 'Ids', zh: '編號' }, chapter: 'technique',
      title: { en: 'Look up each token’s id', zh: '查出每個詞元的編號' },
      body: [
        { en: 'Now replace every token by its id. Follow the lines: e is entry 5, m is entry 13, a is entry 1.', zh: '現在把每個詞元換成它的編號。沿著線看：e 是第 5 項，m 是第 13 項，a 是第 1 項。' },
        { en: 'The name <code>emma</code> is now <code>[5, 13, 13, 1]</code>. This list of numbers is what the model actually receives.', zh: '名字 <code>emma</code> 現在變成 <code>[5, 13, 13, 1]</code>。模型真正收到的，就是這串數字。' },
      ],
      changed: [{ sign: '~', text: { en: 'emma → [5, 13, 13, 1]', zh: 'emma → [5, 13, 13, 1]' } }],
    },
    {
      short: { en: 'Start & end', zh: '開頭與結尾' }, chapter: 'technique',
      title: { en: 'Mark where a name starts and ends', zh: '標出名字的開頭與結尾' },
      body: [
        { en: 'The model also has to learn how names begin, and when to stop writing. So we add the special token <code>.</code> (id 0) at both ends.', zh: '模型還得學會名字怎麼開頭、什麼時候該停筆。所以我們在頭尾各加上一個特殊詞元 <code>.</code>（編號 0）。' },
        { en: 'Later, when the model writes a name by itself, it starts from <code>.</code> and keeps adding letters until it produces <code>.</code> again.', zh: '之後模型自己寫名字時，會從 <code>.</code> 開始，一個字母一個字母往下寫，直到它再寫出 <code>.</code> 為止。' },
      ],
      changed: [
        { sign: '+', text: { en: 'Special token “.” at both ends', zh: '頭尾加上特殊詞元「.」' } },
        { sign: '~', text: { en: 'Sequence is now [0, 5, 13, 13, 1, 0]', zh: '序列變成 [0, 5, 13, 13, 1, 0]' } },
      ],
    },
    {
      short: { en: 'Examples', zh: '訓練資料' }, chapter: 'solution',
      title: { en: 'One name, five little quizzes', zh: '一個名字，五道小考題' },
      body: [
        { en: 'A language model has one job: <b>given the tokens so far, predict the next one</b>. So we line the sequence up against itself, shifted by one.', zh: '語言模型只有一個任務：<b>根據目前為止的詞元，預測下一個</b>。所以我們把序列和「往後移一格的自己」對齊。' },
        { en: 'Each column is one quiz. After <code>.</code> comes e. After <code>.e</code> comes m. After <code>.emma</code> comes <code>.</code>, meaning “stop”. One name gives five examples.', zh: '每一欄都是一道考題。<code>.</code> 後面是 e；<code>.e</code> 後面是 m；<code>.emma</code> 後面是 <code>.</code>，意思是「停」。一個名字就產生了五筆訓練資料。' },
      ],
      callout: { kind: 'insight', text: { en: 'Nobody labels this data by hand. The text itself supplies the right answers. That is why models can learn from enormous amounts of plain text.', zh: '這些資料不需要人工標註，文字本身就提供了正確答案。這正是模型能從海量純文字中學習的原因。' } },
      changed: [{ sign: '+', text: { en: '5 input → target pairs', zh: '5 組「輸入 → 目標」' } }],
    },
    {
      short: { en: 'Real models', zh: '真實模型' }, chapter: 'technique',
      title: { en: 'Real models use bigger pieces', zh: '真實的模型用更大的小塊' },
      body: [
        { en: 'Letters make long sequences. Whole words make a vocabulary that is far too large and breaks on any new word. Real models use something in between: <b>sub-word tokens</b>, pieces learned from the data so that common chunks get their own token.', zh: '用字母切，序列會很長；用整個單字切，詞彙表會大到無法處理，而且遇到新字就沒輒。真實的模型採取折衷做法：<b>子詞（sub-word）詞元</b>，也就是從資料中學出來的片段，常見的片段會有自己的詞元。' },
        { en: 'Their vocabularies hold tens of thousands of tokens or more. The split shown here is an illustration; each real tokenizer cuts a little differently.', zh: '它們的詞彙表有數萬個以上的詞元。這裡畫的切法只是示意，每個真實的分詞器切法都略有不同。' },
      ],
      callout: { kind: 'note', text: { en: 'Chinese works the same way: a character is often one token, sometimes more. Everything after tokenizing is identical for every language.', zh: '中文也是一樣的道理：一個字通常是一個詞元，有時會是好幾個。切完詞元之後，不論哪種語言，後面的流程完全相同。' } },
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Text in, numbers out', zh: '文字進去，數字出來' },
      body: [
        { en: '<b>Tokens</b> are the small pieces text is cut into.<br>The <b>vocabulary</b> lists every token and gives each an id.<br>A special token marks start and end.<br>Shifting the sequence by one gives training examples for free.', zh: '<b>詞元</b>是文字被切成的小塊。<br><b>詞彙表</b>列出所有詞元，並各給一個編號。<br>特殊詞元標示開頭與結尾。<br>把序列錯開一格，就免費得到訓練資料。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Embeddings. An id is just a label. How does 13 become something the model can do useful math with?', zh: '下一課：嵌入。編號只是個標籤，13 要怎麼變成模型可以拿來做有用運算的東西？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const letters = [...NAME];
    const withMarks = step >= 4;
    const shown = withMarks ? ['.', ...letters, '.'] : letters;
    const shownIds = withMarks ? full : ids;
    const top = step === 5 ? 110 : 70;
    const x0 = 450 - (shown.length * 62 - 18) / 2;

    if (step === 0) {
      items.push({ k: 'word', t: 'text', x: 450, y: 210, text: '“emma”', size: 54, anchor: 'middle', bold: true, mono: true });
      items.push({ k: 'arrow0', t: 'arrow', x1: 450, y1: 240, x2: 450, y2: 300, tone: 'muted' });
      items.push({ k: 'q', t: 'text', x: 450, y: 345, text: '? ? ? ?', size: 30, anchor: 'middle', tone: 'muted', mono: true });
    }

    if (step >= 1 && step <= 5) {
      shown.forEach((ch, i) => {
        const key = withMarks ? (i === 0 ? 'dot0' : i === shown.length - 1 ? 'dot1' : 'tok' + (i - 1)) : 'tok' + i;
        items.push({
          k: key, t: 'token', x: x0 + i * 62, y: top, text: ch,
          tone: ch === '.' ? 'b' : 'ink',
          sub: step >= 3 ? String(shownIds[i]) : undefined,
        });
      });
    }

    if (step === 2 || step === 3 || step === 4) {
      items.push({ k: 'vlabel', t: 'text', x: 196, y: 228, text: t({ en: 'vocabulary · 27 tokens', zh: '詞彙表 · 27 個詞元' }), size: 14.5, tone: 'muted' });
      [...CHARS].forEach((ch, i) => {
        const p = vocabAt(i);
        const used = (step >= 3 && ids.includes(i)) || (step === 4 && i === 0);
        items.push({ k: 'v' + i, t: 'token', x: p.x, y: p.y, text: ch, sub: String(i), tone: used ? 'a' : i === 0 ? 'b' : 'ink', dim: step >= 3 && !used });
      });
      if (step === 3)
        letters.forEach((_, i) => {
          const p = vocabAt(ids[i]);
          items.push({ k: 'look' + i, t: 'arrow', x1: x0 + i * 62 + 22, y1: top + 66, x2: p.x + 22, y2: p.y - 4, tone: 'a', dashed: true });
        });
    }

    if (step === 5) {
      const targets = [...shown.slice(1)];
      items.push({ k: 'inl', t: 'text', x: x0 - 20, y: top + 28, text: t({ en: 'model reads', zh: '模型讀到' }), anchor: 'end', tone: 'muted', size: 13 });
      items.push({ k: 'tgl', t: 'text', x: x0 - 20, y: top + 178, text: t({ en: 'should predict', zh: '應該預測' }), anchor: 'end', tone: 'muted', size: 13 });
      targets.forEach((ch, i) => {
        items.push({ k: 'tg' + i, t: 'token', x: x0 + i * 62, y: top + 150, text: ch, tone: 'a', sub: String(full[i + 1]) });
        items.push({ k: 'ta' + i, t: 'arrow', x1: x0 + i * 62 + 22, y1: top + 68, x2: x0 + i * 62 + 22, y2: top + 144, tone: 'a' });
      });
      items.push({ k: 'note5', t: 'text', x: 450, y: top + 260, text: t({ en: 'each column is one training example', zh: '每一欄都是一筆訓練資料' }), anchor: 'middle', tone: 'muted', size: 13 });
    }

    if (step === 6) {
      const rows: { label: string; parts: string[]; y: number }[] = [
        { label: t({ en: 'by letter · 11 tokens', zh: '按字母 · 11 個詞元' }), parts: [...'transformer'], y: 80 },
        { label: t({ en: 'by word · 1 token, huge vocabulary', zh: '按單字 · 1 個詞元，詞彙表極大' }), parts: ['transformer'], y: 220 },
        { label: t({ en: 'by sub-word · 2 tokens (illustration)', zh: '按子詞 · 2 個詞元（示意）' }), parts: ['transform', 'er'], y: 360 },
      ];
      rows.forEach((r, ri) => {
        items.push({ k: 'rl' + ri, t: 'text', x: 120, y: r.y - 14, text: r.label, tone: ri === 2 ? 'a' : 'muted', size: 14.5, bold: ri === 2 });
        let x = 120;
        r.parts.forEach((p, pi) => {
          const w = Math.max(44, p.length * 15 + 22);
          items.push({ k: `rp${ri}-${pi}`, t: 'box', x, y: r.y, w, h: 44, label: p, tone: ri === 2 ? 'a' : 'ink' });
          x += w + 8;
        });
      });
    }

    if (step === 7) {
      const y = 170;
      items.push({ k: 'r1', t: 'box', x: 90, y, w: 150, h: 60, label: '“emma”', sub: t({ en: 'text', zh: '文字' }) });
      items.push({ k: 'r2', t: 'box', x: 300, y, w: 180, h: 60, label: '. e m m a .', sub: t({ en: 'tokens', zh: '詞元' }) });
      items.push({ k: 'r3', t: 'box', x: 540, y, w: 270, h: 60, label: '[0, 5, 13, 13, 1, 0]', sub: t({ en: 'ids', zh: '編號' }), tone: 'a' });
      items.push({ k: 'ra1', t: 'arrow', x1: 244, y1: y + 30, x2: 296, y2: y + 30 });
      items.push({ k: 'ra2', t: 'arrow', x1: 484, y1: y + 30, x2: 536, y2: y + 30 });
      items.push({ k: 'r4', t: 'text', x: 450, y: y + 130, text: t({ en: `${NAMES.length} names in our training list`, zh: `我們的訓練清單裡有 ${NAMES.length} 個名字` }), anchor: 'middle', tone: 'muted', size: 14 });
    }

    const tokenCount = step === 0 ? '—' : step === 6 ? '2' : String(shown.length);
    return {
      metrics: [
        { label: { en: 'Tokens', zh: '詞元數' }, value: tokenCount },
        { label: { en: 'Vocabulary', zh: '詞彙表大小' }, value: step >= 2 ? (step === 6 ? '10,000s' : '27') : '—' },
        { label: { en: 'Ids', zh: '編號序列' }, value: step >= 3 && step !== 6 ? `[${shownIds.join(', ')}]` : '—' },
        { label: { en: 'Training examples', zh: '訓練資料筆數' }, value: step === 5 || step === 7 ? '5' : '—' },
      ],
      items,
      legend: [
        { tone: 'a', text: { en: 'in use / target', zh: '使用中／目標' } },
        { tone: 'b', text: { en: 'special token', zh: '特殊詞元' } },
      ],
    };
  },
};
