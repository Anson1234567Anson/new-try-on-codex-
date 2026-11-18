import type { Lesson, Step } from './types';
import type { Item, Tone } from '../engine/scene';
import { t } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, NAMES, TRAIN_NAMES, VAL_NAMES, rng } from '../model/data';
import weights from '../model/weights.json';

// The trained model writes one name, letter by letter, starting from ".".
// The random picks come from a fixed seed, so every visit shows the same name.
const SEED = 29;

interface Pick { prefix: number[]; probs: number[]; chosen: number }
interface Data { picks: Pick[]; name: string; greedy: Pick[]; greedyName: string; sampled: string[]; count: number }

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const write = (choose: (p: number[]) => number): Pick[] => {
    const ids = [0];
    const picks: Pick[] = [];
    while (ids.length < model.cfg.ctx) {
      const probs = model.next(ids);
      const chosen = choose(probs);
      picks.push({ prefix: [...ids], probs, chosen });
      if (chosen === 0) break;
      ids.push(chosen);
    }
    return picks;
  };
  // the same weighted draw that model.sample uses
  const r = rng(SEED);
  const picks = write((p) => { let x = r(), id = 0; while (id < p.length - 1 && (x -= p[id]) > 0) id++; return id; });
  const greedy = write((p) => p.indexOf(Math.max(...p)));
  const word = (ps: Pick[]) => ps.filter((p) => p.chosen !== 0).map((p) => CHARS[p.chosen]).join('');
  const r2 = rng(7);
  const sampled = Array.from({ length: 8 }, () => model.sample(r2) || '·');
  return { picks, name: word(picks), greedy, greedyName: word(greedy), sampled, count: model.paramCount() };
})();

const pct = (v: number) => Math.round(v * 100) + '%';
const rank = (p: Pick) => [...p.probs].sort((a, b) => b - a).indexOf(p.probs[p.chosen]) + 1;
const top = (p: number[]) => p.indexOf(Math.max(...p));
const show = (ids: number[]) => ids.map((i) => CHARS[i]).join('');
const q = (s: string) => `“${s}”`;
const qz = (s: string) => `「${s}」`;

// which pick (index into data().picks) each step is about
const PICK_AT: Record<number, number> = { 1: 0, 2: 0, 3: 1 };
const pickFor = (step: number) => {
  const d = data();
  if (step in PICK_AT) return PICK_AT[step];
  if (step === 4) return d.picks.length - 2; // the last letter
  return d.picks.length - 1; // the stop
};

export const languageModels: Lesson = {
  id: 'language-models',
  title: { en: 'What a Language Model Does', zh: '語言模型在做什麼' },
  blurb: {
    en: 'A language model does one thing: give a probability to every possible next token. Writing is that, on repeat.',
    zh: '語言模型只做一件事：替每一個可能的下一個詞元給出機率。寫作，就是不斷重複這件事。',
  },
  minutes: 8,
  level: 'beginner',
  learn: [
    { text: { en: 'The one question a language model answers', zh: '語言模型回答的唯一問題' }, step: 0 },
    { text: { en: 'What “a probability for every next token” looks like', zh: '「替每個下一個詞元給出機率」長什麼樣子' }, step: 1 },
    { text: { en: 'How writing works: pick, append, repeat', zh: '寫作怎麼運作：挑選、接上、重複' }, step: 4 },
    { text: { en: 'How the model knows when to stop', zh: '模型怎麼知道何時停下' }, step: 5 },
    { text: { en: 'Why picking at random beats always taking the top', zh: '為什麼隨機挑選比永遠選第一名好' }, step: 6 },
  ],
  get steps(): Step[] {
    const d = data();
    const first = d.picks[0], second = d.picks[1], stop = d.picks[d.picks.length - 1], lastL = d.picks[d.picks.length - 2];
    const ch = (p: Pick) => CHARS[p.chosen];
    const fav = CHARS[top(first.probs)];
    const inList = NAMES.includes(d.name);
    // the four letters that gain the most after the first pick
    const rise = second.probs.map((v, i) => ({ i, g: v - first.probs[i] })).filter((x) => x.i !== 0).sort((a, b) => b.g - a.g).slice(0, 4).map((x) => x.i);
    return [
      {
        short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
        title: { en: 'How does a machine write?', zh: '機器要怎麼寫字？' },
        body: [
          { en: `In this course we build a tiny model that invents first names. It learned from a list of about 3,000 real names (it trains on ${TRAIN_NAMES.length.toLocaleString('en-US')} of them; we keep ${VAL_NAMES.length} aside to test it), and it is the same kind of machine as a chatbot, only far smaller.`, zh: `在這門課裡，我們會打造一個會發明英文名字的小模型。它從一份約 3,000 個真實名字的清單學習（其中 ${TRAIN_NAMES.length.toLocaleString('en-US')} 個用來訓練，另外 ${VAL_NAMES.length} 個留著測試它），跟聊天機器人是同一類機器，只是小得多。` },
          { en: 'Strangely, it can answer only one question: <b>given the text so far, what comes next?</b> Here the text so far is just <code>.</code>, our mark for “a name starts here”.', zh: '奇妙的是，它只會回答一個問題：<b>根據目前為止的文字，下一個是什麼？</b>這裡目前的文字只有 <code>.</code>，也就是我們用來表示「名字從這裡開始」的記號。' },
        ],
        callout: { kind: 'insight', text: { en: 'A language model is a next-token predictor. Everything it “writes” comes from answering that one question again and again.', zh: '語言模型就是「下一個詞元」的預測器。它「寫」出的一切，都來自一次又一次回答同一個問題。' } },
      },
      {
        short: { en: 'Probabilities', zh: '機率' }, chapter: 'technique',
        title: { en: 'A probability for every possible token', zh: '每個可能的詞元都有一個機率' },
        body: [
          { en: 'The answer is not one letter. It is a <b>probability</b> for each of the 27 possible <b>tokens</b> (the pieces it writes with: “.” and a–z). Together they add up to 100%.', zh: '答案不是一個字母，而是替 27 個可能的<b>詞元</b>（它用來寫字的小塊：「.」和 a–z）各給一個<b>機率</b>，全部加起來是 100%。' },
          { en: `These are the trained model’s real numbers for the first letter. Its favourite is ${q(fav)} at ${pct(first.probs[top(first.probs)])}, but no letter is a sure thing.`, zh: `這些是訓練好的模型對第一個字母給出的真實數字。它最看好的是${qz(fav)}，機率 ${pct(first.probs[top(first.probs)])}，但沒有哪個字母是穩贏的。` },
        ],
        callout: { kind: 'note', text: { en: 'The next lesson, Tokens, explains how text is cut into tokens and numbered.', zh: '下一課「詞元」會說明文字如何被切成詞元並編號。' } },
        changed: [{ sign: '+', text: { en: '27 probabilities, one per token', zh: '27 個機率，每個詞元一個' } }],
      },
      {
        short: { en: 'Pick one', zh: '挑一個' }, chapter: 'technique',
        title: { en: 'Pick one, like drawing a lottery ticket', zh: '挑一個，就像抽籤' },
        body: [
          { en: 'To write, we pick one token at random, but weighted: a letter with 20% gets 20 tickets out of 100, a letter with 1% gets one. This is called <b>sampling</b>.', zh: '要寫字，就隨機挑一個詞元，但是有權重：機率 20% 的字母拿到 100 張籤裡的 20 張，1% 的只拿到 1 張。這叫做<b>取樣（sampling）</b>。' },
          { en: `This draw picked ${q(ch(first))}, which had only ${pct(first.probs[first.chosen])} (number ${rank(first)} on the list). Unlikely letters still win sometimes.`, zh: `這次抽中了${qz(ch(first))}，它的機率只有 ${pct(first.probs[first.chosen])}（排第 ${rank(first)} 名）。機率不高的字母，偶爾也會被抽中。` },
        ],
        changed: [{ sign: '+', text: { en: `First letter: ${q(ch(first))}`, zh: `第一個字母：${qz(ch(first))}` } }],
      },
      {
        short: { en: 'Again', zh: '再來' }, chapter: 'technique',
        title: { en: 'Append it and ask again', zh: '接上去，再問一次' },
        body: [
          { en: `Now the text so far is <code>${show(second.prefix)}</code>. We feed that back in and ask the same question. The probabilities are completely different, because the context changed.`, zh: `現在目前的文字是 <code>${show(second.prefix)}</code>。我們把它再送進模型，問同一個問題。機率完全不一樣了，因為上下文變了。` },
          { en: `After ${q(ch(first))}, a few letters jump up: ${rise.map((i) => CHARS[i]).join(', ')}, as in ${rise.map((i) => `<i>${ch(first)}${CHARS[i]}-</i>`).join(', ')}. The model has learned which letters can follow ${q(ch(first))}. The draw picks ${q(ch(second))} with ${pct(second.probs[second.chosen])}.`, zh: `在${qz(ch(first))}之後，有幾個字母一下子升高：${rise.map((i) => CHARS[i]).join('、')}，就像 ${rise.map((i) => `<i>${ch(first)}${CHARS[i]}-</i>`).join('、')}。模型學會了哪些字母可以接在${qz(ch(first))}後面。這次抽中${qz(ch(second))}，機率 ${pct(second.probs[second.chosen])}。` },
        ],
        changed: [{ sign: '~', text: { en: `Text so far: ${show(second.prefix)} → ${show([...second.prefix, second.chosen])}`, zh: `目前文字：${show(second.prefix)} → ${show([...second.prefix, second.chosen])}` } }],
      },
      {
        short: { en: 'Repeat', zh: '重複' }, chapter: 'solution',
        title: { en: 'Pick, append, repeat', zh: '挑選、接上、重複' },
        body: [
          { en: 'That is the whole writing loop. Each pass the model sees everything written so far, gives 27 probabilities, and one token is drawn and appended.', zh: '這就是整個寫作迴圈。每一輪，模型都看著目前寫好的全部內容，給出 27 個機率，然後抽出一個詞元接在後面。' },
          { en: `The row on top keeps each letter with the probability it had when it was picked. Now the text is <code>${show(lastL.prefix)}</code> and the draw adds ${q(ch(lastL))}.`, zh: `上面那一排記下了每個字母被抽中時的機率。現在文字是 <code>${show(lastL.prefix)}</code>，這次抽到${qz(ch(lastL))}。` },
        ],
        changed: [{ sign: '+', text: { en: `${d.picks.length - 1} letters written`, zh: `已寫出 ${d.picks.length - 1} 個字母` } }],
      },
      {
        short: { en: 'Stop', zh: '停止' }, chapter: 'solution',
        title: { en: 'Knowing when to stop', zh: '知道何時該停' },
        body: [
          { en: `Now the model gives ${q('.')} a probability of ${pct(stop.probs[0])}. The dot is also our “name ends here” mark, so drawing it means stop.`, zh: `現在模型給${qz('.')}的機率是 ${pct(stop.probs[0])}。這個點同時也是「名字到此結束」的記號，所以抽到它就代表停下。` },
          { en: `The result is <b>${d.name}</b>. ${inList ? 'It happens to be in the training list.' : 'It is not in the list the model learned from: it is a new name built from patterns it picked up.'}`, zh: `結果是 <b>${d.name}</b>。${inList ? '它剛好出現在訓練清單裡。' : '它不在模型學過的清單裡，是根據學到的規律拼出來的新名字。'}` },
        ],
        callout: { kind: 'insight', text: { en: 'Stopping is just another prediction. The model learned when names usually end, the same way it learned which letters follow which.', zh: '停下來也只是另一個預測。模型學會名字通常在哪裡結束，就跟它學會哪些字母接在哪些字母後面一樣。' } },
        changed: [{ sign: '+', text: { en: `Finished name: ${d.name}`, zh: `完成的名字：${d.name}` } }],
      },
      {
        short: { en: 'Always top?', zh: '總選第一？' }, chapter: 'problem',
        title: { en: 'Case: always take the likeliest letter', zh: '案例：永遠選機率最高的字母' },
        body: [
          { en: `Why gamble? Let us always take the top letter instead. This is called <b>greedy</b> decoding. It writes <b>${d.greedyName}</b>.`, zh: `為什麼要賭運氣？那就永遠選機率最高的字母吧。這叫做<b>貪婪（greedy）</b>解碼，它寫出 <b>${d.greedyName}</b>。` },
          { en: `Ask again and you get ${d.greedyName}, and again ${d.greedyName}: the same input always gives the same top letter. Drawing at random gives a fresh name every time, shown below.`, zh: `再問一次還是 ${d.greedyName}，再問一次依然是 ${d.greedyName}：同樣的輸入，第一名永遠是同一個字母。隨機抽籤則每次都會寫出新的名字，如下方所示。` },
        ],
        callout: { kind: 'note', text: { en: 'Chatbots face the same choice. The lesson From Logits to Text shows the dials (temperature, top-k) that sit between these two extremes.', zh: '聊天機器人也面臨同樣的選擇。「從 logits 到文字」那一課會介紹介於這兩個極端之間的調整旋鈕（溫度、top-k）。' } },
        changed: [{ sign: '~', text: { en: 'Pick rule: random draw → always top', zh: '挑選規則：隨機抽 → 永遠選第一' } }],
      },
      {
        short: { en: 'Chatbots', zh: '聊天機器人' }, chapter: 'technique',
        title: { en: 'Chatbots run the same loop', zh: '聊天機器人跑的是同一個迴圈' },
        body: [
          { en: 'A chatbot answering you does exactly this: read the text so far, give a probability to every token, pick one, append it, repeat, until it picks an “end” token.', zh: '聊天機器人回答你時，做的正是這件事：讀取目前的文字，替每個詞元給出機率，挑一個，接上去，重複，直到挑中「結束」詞元。' },
          { en: `What differs is size. Our model has ${d.count.toLocaleString()} <b>weights</b>: adjustable numbers that training tuned until the predictions came out right. Chatbots have billions of weights and vocabularies of tens of thousands to a few hundred thousand tokens, mostly pieces of words.`, zh: `不同的是規模。我們的模型有 ${d.count.toLocaleString()} 個<b>權重</b>：可以調整的數字，訓練會一直調整它們，直到預測變準。聊天機器人則有數十億個權重，詞彙表有數萬到數十萬個詞元，大多是單字的片段。` },
        ],
        callout: { kind: 'insight', text: { en: 'Inside, the rest of this course is about one thing: how the model turns the text so far into those probabilities.', zh: '接下來的整門課，其實只在講一件事：模型如何把目前的文字變成那些機率。' } },
      },
      {
        short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
        title: { en: 'Writing is prediction on repeat', zh: '寫作就是不斷重複的預測' },
        body: [
          { en: 'A language model gives a <b>probability</b> to every possible next token.<br><b>Sampling</b> draws one, weighted by those probabilities.<br>Append it and ask again: that loop writes text.<br>A special token means “stop”.<br>Always taking the top token gives the same text every time.', zh: '語言模型替每個可能的下一個詞元給出<b>機率</b>。<br><b>取樣</b>依照這些機率抽出一個。<br>接上去再問一次：這個迴圈就能寫出文字。<br>一個特殊詞元代表「停止」。<br>永遠選第一名，每次都會寫出一樣的東西。' },
        ],
        callout: { kind: 'next', text: { en: 'Next: Tokens. Before the model can give probabilities, text must be cut into tokens and turned into numbers.', zh: '下一課：詞元。在模型給出機率之前，文字必須先被切成詞元、變成數字。' } },
      },
    ];
  },

  scene(step) {
    const d = data();
    const items: Item[] = [];
    const pi = pickFor(step);
    const cur = d.picks[pi];
    const row = (ps: Pick[], upto: number, showNow: boolean, y: number, greedy = false) => {
      // chips: the start dot, then every chosen token so far
      const chips: { text: string; sub?: string; tone: Tone }[] = [{ text: '.', tone: 'b' }];
      for (let i = 0; i < upto; i++) chips.push({ text: CHARS[ps[i].chosen], sub: pct(ps[i].probs[ps[i].chosen]), tone: ps[i].chosen === 0 ? 'b' : 'ink' });
      if (showNow) chips.push({ text: CHARS[ps[upto].chosen], sub: pct(ps[upto].probs[ps[upto].chosen]), tone: ps[upto].chosen === 0 ? 'b' : 'a' });
      const x0 = 450 - (chips.length * 62 - 18) / 2;
      chips.forEach((c, i) => items.push({ k: (greedy ? 'g' : 'c') + i, t: 'token', x: x0 + i * 62, y, text: c.text, sub: c.sub, tone: c.tone }));
      return x0 + chips.length * 62;
    };

    if (step === 0) {
      items.push({ k: 'c0', t: 'token', x: 200, y: 120, text: '.', tone: 'b', sub: t({ en: 'start', zh: '開頭' }) });
      items.push({ k: 'a0', t: 'arrow', x1: 256, y1: 142, x2: 342, y2: 142 });
      items.push({ k: 'box', t: 'box', x: 350, y: 105, w: 200, h: 74, label: t({ en: 'language model', zh: '語言模型' }), sub: t({ en: 'trained on names', zh: '用名字訓練過' }), tone: 'a' });
      items.push({ k: 'a1', t: 'arrow', x1: 558, y1: 142, x2: 644, y2: 142 });
      items.push({ k: 'qm', t: 'token', x: 656, y: 120, text: '?', tone: 'a', sub: t({ en: 'what next?', zh: '下一個？' }) });
      items.push({ k: 'qt', t: 'text', x: 450, y: 300, anchor: 'middle', size: 17, text: t({ en: 'the only question it can answer:', zh: '它唯一會回答的問題：' }), tone: 'muted' });
      items.push({ k: 'qq', t: 'text', x: 450, y: 340, anchor: 'middle', size: 22, bold: true, text: t({ en: 'given the text so far, what comes next?', zh: '根據目前的文字，下一個是什麼？' }) });
    }

    if (step >= 1 && step <= 5) {
      const appended = step >= 2;
      const end = row(d.picks, pi, appended, 40);
      if (!appended) items.push({ k: 'qm', t: 'token', x: end, y: 40, text: '?', tone: 'a' });
      items.push({
        k: 'bars', t: 'bars', x: 40, y: 175, w: 820, h: 250,
        items: cur.probs.map((v, i) => ({ label: CHARS[i], v, hi: appended && i === cur.chosen })),
        title: t({ en: `the model’s probability for each next token, after “${show(cur.prefix)}”`, zh: `模型在「${show(cur.prefix)}」之後，給每個下一個詞元的機率` }),
      });
      const note = !appended
        ? t({ en: 'all 27 bars add up to 100%', zh: '27 根長條加起來剛好是 100%' })
        : cur.chosen === 0
          ? t({ en: '“.” drawn → the name is finished', zh: '抽到「.」→ 名字寫完了' })
          : t({ en: `drawn: “${CHARS[cur.chosen]}” (${pct(cur.probs[cur.chosen])}) → append it`, zh: `抽中：「${CHARS[cur.chosen]}」（${pct(cur.probs[cur.chosen])}）→ 接上去` });
      items.push({ k: 'note', t: 'text', x: 450, y: 470, anchor: 'middle', size: 16, tone: appended ? 'a' : 'muted', bold: appended, text: note });
    }

    if (step === 6) {
      items.push({ k: 'gl', t: 'text', x: 450, y: 30, anchor: 'middle', size: 15, tone: 'muted', text: t({ en: 'always take the top letter (greedy)', zh: '永遠選第一名（貪婪）' }) });
      row(d.greedy, d.greedy.length - 1, true, 50, true);
      items.push({ k: 'gr', t: 'text', x: 220, y: 160, size: 14.5, tone: 'muted', text: t({ en: 'run it 8 times:', zh: '跑 8 次：' }) });
      for (let i = 0; i < 8; i++) items.push({ k: 'gn' + i, t: 'box', x: 220 + (i % 4) * 118, y: 172 + Math.floor(i / 4) * 44, w: 108, h: 36, label: d.greedyName, tone: 'muted' });
      items.push({ k: 'sl', t: 'text', x: 220, y: 300, size: 14.5, tone: 'a', bold: true, text: t({ en: 'draw at random (sampling), 8 times:', zh: '隨機抽籤（取樣），8 次：' }) });
      d.sampled.forEach((s, i) => items.push({ k: 'sn' + i, t: 'box', x: 220 + (i % 4) * 118, y: 312 + Math.floor(i / 4) * 50, w: 108, h: 40, label: s, tone: 'a' }));
    }

    if (step === 7 || step === 8) {
      const boxes = [
        { en: 'text so far', zh: '目前的文字' }, { en: 'model', zh: '模型' },
        { en: 'a probability for every token', zh: '每個詞元一個機率' }, { en: 'pick one', zh: '挑一個' },
      ];
      const pos = [{ x: 90, y: 110 }, { x: 360, y: 110 }, { x: 590, y: 110 }, { x: 360, y: 300 }];
      boxes.forEach((b, i) => items.push({ k: 'lb' + i, t: 'box', x: pos[i].x, y: pos[i].y, w: i === 2 ? 250 : 190, h: 64, label: t(b), tone: i === 1 ? 'a' : 'ink' }));
      items.push({ k: 'la0', t: 'arrow', x1: 284, y1: 142, x2: 354, y2: 142 });
      items.push({ k: 'la1', t: 'arrow', x1: 554, y1: 142, x2: 584, y2: 142 });
      items.push({ k: 'la2', t: 'arrow', x1: 715, y1: 178, x2: 556, y2: 318 });
      items.push({ k: 'la3', t: 'arrow', x1: 354, y1: 332, x2: 190, y2: 180, tone: 'a' });
      items.push({ k: 'lt', t: 'text', x: 200, y: 290, anchor: 'end', size: 14.5, tone: 'a', text: t({ en: 'append, repeat', zh: '接上、重複' }) });
      items.push({ k: 'ls', t: 'text', x: 455, y: 405, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: 'until the “end” token is picked', zh: '直到挑中「結束」詞元' }) });
      if (step === 7) {
        items.push({ k: 'us', t: 'text', x: 220, y: 470, anchor: 'middle', size: 15, text: t({ en: `ours: ${d.count.toLocaleString()} weights · 27 tokens`, zh: `我們的：${d.count.toLocaleString()} 個權重 · 27 個詞元` }) });
        items.push({ k: 'them', t: 'text', x: 630, y: 470, anchor: 'middle', size: 15, tone: 'b', text: t({ en: 'chatbots: billions of weights · 10,000s–100,000s of tokens', zh: '聊天機器人：數十億個權重 · 數萬到數十萬個詞元' }) });
      } else {
        items.push({ k: 'us', t: 'text', x: 450, y: 470, anchor: 'middle', size: 16, tone: 'a', bold: true, text: t({ en: `our model wrote: ${d.name}`, zh: `我們的模型寫出：${d.name}` }) });
      }
    }

    const written = step === 0 ? '.' : step === 6 ? '.' + d.greedyName + '.' : step >= 7 ? '.' + d.name + '.' : show(step >= 2 ? [...cur.prefix, cur.chosen] : cur.prefix);
    const picked = step >= 2 && step <= 5 ? t({ en: `“${CHARS[cur.chosen]}” ${pct(cur.probs[cur.chosen])}`, zh: `「${CHARS[cur.chosen]}」 ${pct(cur.probs[cur.chosen])}` }) : '—';
    return {
      metrics: [
        { label: { en: 'Text so far', zh: '目前的文字' }, value: written },
        { label: { en: 'Possible next tokens', zh: '可能的下一個' }, value: '27' },
        { label: { en: 'Picked', zh: '抽中' }, value: picked },
        { label: { en: 'Pick rule', zh: '挑選規則' }, value: step === 6 ? t({ en: 'always top', zh: '永遠第一' }) : step >= 2 ? t({ en: 'random draw', zh: '隨機抽' }) : '—' },
      ],
      items,
      legend: step >= 1 && step <= 6
        ? [{ tone: 'a', text: { en: 'just picked', zh: '剛抽中' } }, { tone: 'b', text: { en: 'start / end mark', zh: '開頭／結尾記號' } }]
        : undefined,
    };
  },
};
