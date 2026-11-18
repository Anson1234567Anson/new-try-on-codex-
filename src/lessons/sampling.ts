import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, CTX, NAMES, decode, encode, rng } from '../model/data';
import { softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';

// The model has read ".ma". Its output is 27 scores (logits). How do we pick a letter?
// Every name in this lesson is written by continuing from ".ma", so the names match the prompt.

const PREFIX = '.ma';
const TEMPS = [0.5, 1, 1.5];
const SEED = 7;
const N = 300; // names written per setting, to measure variety
const K = 3; // top-k

const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const pct = (v: number) => Math.round(v * 100) + '%';
const pct1 = (v: number) => (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%';

// The model, and its logits for every prefix it has seen (the same prefix always gives the same logits,
// so writing 300 names at a new temperature only needs a few new forward passes).
let modelCache: GPT | undefined;
const model = () => (modelCache ??= GPT.fromJSON(weights as Weights));
const logitCache = new Map<string, number[]>();
const logitsFor = (ids: number[]): number[] => {
  const key = ids.join(',');
  let v = logitCache.get(key);
  if (!v) { v = model().forward(ids).logits.row(ids.length - 1); logitCache.set(key, v); }
  return v;
};
const topK = (p: number[], k: number): number[] => {
  if (k <= 0) return p;
  const cut = [...p].sort((a, b) => b - a)[Math.min(k, p.length) - 1];
  const kept = p.map((v) => (v >= cut ? v : 0));
  const z = kept.reduce((a, b) => a + b, 0);
  return kept.map((v) => v / z);
};
/** Write one name that continues from PREFIX: temperature, optional top-k, then a weighted die. */
const sampleName = (r: () => number, T: number, k: number): string => {
  const ids = encode(PREFIX);
  while (ids.length < CTX) {
    const p = topK(softmaxRow(logitsFor(ids), T), k);
    let x = r(), id = 0;
    while (id < p.length - 1 && (x -= p[id]) > 0) id++;
    if (id === 0) break;
    ids.push(id);
  }
  return decode(ids.slice(1));
};

interface Run { names: string[]; real: number; unique: number }
let knownCache: Set<string> | undefined;
const known = () => (knownCache ??= new Set(NAMES));
const runCache = new Map<string, Run>();
const run = (T: number, k: number): Run => {
  const key = T.toFixed(2) + '/' + k;
  let out = runCache.get(key);
  if (!out) {
    const r = rng(SEED);
    const names = Array.from({ length: N }, () => sampleName(r, T, k));
    out = { names, real: names.filter((n) => known().has(n)).length / N, unique: new Set(names).size };
    runCache.set(key, out);
  }
  return out;
};

interface Data {
  logits: number[]; probs: number[][]; // probs at each of TEMPS
  topk: number[]; // probabilities after keeping the top K (T = 1)
  runs: Run[]; greedy: Run; topkRun: Run;
  path: { ch: string; p: number }[]; // the greedy continuation, letter by letter
  greedyName: string; greedyKnown: boolean;
}
let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const logits = logitsFor(encode(PREFIX));
  const path: { ch: string; p: number }[] = [];
  const g = encode(PREFIX);
  while (g.length < CTX) {
    const p = softmaxRow(logitsFor(g));
    const id = p.indexOf(Math.max(...p));
    path.push({ ch: CHARS[id], p: p[id] });
    if (id === 0) break;
    g.push(id);
  }
  const greedyName = decode(g.slice(1));
  return {
    logits, probs: TEMPS.map((T) => softmaxRow(logits, T)), topk: topK(softmaxRow(logits), K),
    runs: TEMPS.map((T) => run(T, 0)), greedy: run(1, 1), topkRun: run(1, K), path,
    greedyName, greedyKnown: known().has(greedyName),
  };
})();

const R = CHARS.indexOf('r');
const top = (p: number[], n: number) =>
  p.map((v, i) => ({ label: CHARS[i], v })).sort((a, b) => b.v - a.v).slice(0, n).map((b) => ({ ...b, hi: b.label === 'r' }));
const show = (n: string) => n || '·';

// the 27-letter grid
const gridAt = (i: number) => ({ x: 70 + (i % 9) * 88, y: 60 + Math.floor(i / 9) * 92 });

export const sampling: Lesson = {
  id: 'sampling',
  title: { en: 'From Logits to Text', zh: '從 logits 到文字' },
  blurb: {
    en: 'The model gives a score to every letter. Temperature, greedy decoding and top-k decide which one is written.',
    zh: '模型替每個字母打分數。溫度、貪婪解碼和 top-k 決定最後寫出哪一個。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'What logits are', zh: '什麼是 logits' }, step: 0 },
    { text: { en: 'How sampling picks a letter', zh: '取樣如何選出一個字母' }, step: 2 },
    { text: { en: 'What temperature does', zh: '溫度在做什麼' }, step: 3 },
    { text: { en: 'Why always taking the top letter fails', zh: '為什麼永遠選第一名會出問題' }, step: 5 },
    { text: { en: 'How top-k trims unlikely letters', zh: 'top-k 如何剪掉不太可能的字母' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Scores are not a letter', zh: '分數還不是字母' },
      body: [
        { en: 'Our model has read <code>.ma</code>. At the very end it outputs one score for each of the 27 tokens. These raw scores are called <b>logits</b>: the number under each letter.', zh: '模型讀了 <code>.ma</code>。在最後，它替 27 個詞元各輸出一個分數。這些原始分數叫做 <b>logits</b>，也就是每個字母下面的數字。' },
        lz(() => ({ en: `Higher means “more likely next”. “r” scores highest (${data().logits[R].toFixed(2)}), as in “mar…”. But to write a name we need exactly one letter. Which one, and how?`, zh: `分數越高代表「越可能是下一個」。「r」最高（${data().logits[R].toFixed(2)}），就像「mar…」。但要寫出名字，我們只能挑一個字母。挑哪個？怎麼挑？` })),
      ],
      callout: { kind: 'insight', text: { en: 'A language model never writes text directly. It gives scores, and a separate, simple rule turns them into one choice.', zh: '語言模型從不直接寫出文字。它只給分數，再由另一條簡單的規則把分數變成一個選擇。' } },
    },
    {
      short: { en: 'Softmax', zh: 'Softmax' }, chapter: 'technique',
      title: { en: 'Scores become probabilities', zh: '分數變成機率' },
      body: [
        { en: '<b>Softmax</b> turns the logits into <b>probabilities</b>: positive numbers that add up to 100%. Bigger scores get much bigger shares.', zh: '<b>Softmax</b> 把 logits 變成<b>機率</b>：全部是正數，加起來等於 100%。分數大的，分到的比例會大很多。' },
        lz(() => ({ en: `After <code>.ma</code>, “r” gets ${pct(data().probs[1][R])}. The other 26 tokens share the rest, each with a small chance.`, zh: `在 <code>.ma</code> 之後，「r」拿到 ${pct(data().probs[1][R])}，其餘 26 個詞元分剩下的，各自都有一點小機會。` })),
      ],
      changed: [{ sign: '~', text: { en: 'Logits → probabilities', zh: 'logits → 機率' } }],
    },
    {
      short: { en: 'Sample', zh: '取樣' }, chapter: 'technique',
      title: { en: 'Roll a weighted die', zh: '擲一顆加權的骰子' },
      body: [
        lz(() => ({ en: `<b>Sampling</b> picks a letter at random, in proportion to its probability: “r” comes up ${pct(data().probs[1][R])} of the time, rarer letters sometimes. Then the letter is appended and the model runs again, until it writes the end mark <code>.</code>.`, zh: `<b>取樣（sampling）</b>就是依照機率隨機挑一個字母：「r」有 ${pct(data().probs[1][R])} 的機會被選中，比較少見的字母偶爾也會出現。選好的字母接到後面，模型再跑一次，直到寫出結束符號 <code>.</code> 為止。` })),
        { en: `These are 8 real names our model wrote this way, each one continuing from <code>.ma</code>. We fix the random seed (${SEED}) so you see the same names every time.`, zh: `這是模型用這種方法真正寫出的 8 個名字，每一個都是從 <code>.ma</code> 接著寫下去的。我們固定了隨機種子（${SEED}），所以每次看到的名字都一樣。` },
      ],
      changed: [{ sign: '+', text: { en: '8 sampled names', zh: '8 個取樣出的名字' } }],
    },
    {
      short: { en: 'Temperature', zh: '溫度' }, chapter: 'technique',
      title: { en: 'Temperature: divide before softmax', zh: '溫度：在 softmax 之前先除' },
      body: [
        { en: '<b>Temperature</b> is one number <i>T</i>. Divide every logit by <i>T</i>, then apply softmax. The grid below shows the top logits after dividing.', zh: '<b>溫度（temperature）</b>就是一個數字 <i>T</i>。把每個 logit 都除以 <i>T</i>，再做 softmax。下面的表是除完之後的前幾名 logits。' },
        lz(() => ({ en: `A low <i>T</i> stretches the gaps, so the leader takes even more: “r” ${pct(data().probs[0][R])} at 0.5. A high <i>T</i> shrinks the gaps and flattens the chances: “r” ${pct(data().probs[2][R])} at 1.5. As <i>T</i> gets close to 0, the leader takes 100%.`, zh: `<i>T</i> 小，差距被拉大，第一名拿得更多：<i>T</i> = 0.5 時「r」是 ${pct(data().probs[0][R])}。<i>T</i> 大，差距被縮小，機率變得平均：<i>T</i> = 1.5 時「r」只有 ${pct(data().probs[2][R])}。<i>T</i> 越接近 0，第一名就越接近拿走 100%。` })),
      ],
      changed: [{ sign: '+', text: { en: 'Same logits at T = 0.5, 1, 1.5', zh: '同一組 logits，在 T = 0.5、1、1.5' } }],
    },
    {
      short: { en: 'Names', zh: '名字' }, chapter: 'solution',
      title: { en: 'Temperature changes the names', zh: '溫度改變了名字' },
      body: [
        { en: 'Same model, same seed, three temperatures. Low <i>T</i> plays it safe: more of its names are ones already in our list. High <i>T</i> invents more and makes more odd spellings.', zh: '同一個模型、同一個種子、三種溫度。<i>T</i> 小時比較保守，寫出的名字有更多是清單裡本來就有的；<i>T</i> 大時更敢發明，也會出現更多奇怪的拼法。' },
        lz(() => ({ en: `Out of ${N} names each: already in our list ${TEMPS.map((T, i) => `${pct(data().runs[i].real)} at ${T}`).join(', ')}.`, zh: `各寫 ${N} 個名字，已經在清單裡的比例：${TEMPS.map((T, i) => `T = ${T} 時 ${pct(data().runs[i].real)}`).join('、')}。` })),
      ],
      callout: { kind: 'note', text: { en: 'Every name starts with “ma” because that is the start we gave the model. Some names in the same row look alike across temperatures because each temperature uses the same random numbers; the temperature changes what those numbers pick.', zh: '每個名字都以「ma」開頭，因為那是我們給模型的開頭。同一列的名字在不同溫度下有些很像，是因為每種溫度用的是同一串隨機數字；溫度改變的是這些隨機數字選中的字母。' } },
      changed: [{ sign: '+', text: { en: `${N} names at each temperature`, zh: `每種溫度各 ${N} 個名字` } }],
    },
    {
      short: { en: 'Greedy', zh: '貪婪' }, chapter: 'problem',
      title: { en: 'Case: always take the top letter', zh: '案例：永遠選第一名' },
      body: [
        lz(() => ({ en: `Why not skip the dice and always take the most likely letter? This is <b>greedy decoding</b>. Our model then continues <code>.ma</code> into “${data().greedyName}”, following the top choice at every step. It is the <i>T</i> → 0 extreme of temperature.${data().greedyKnown ? '' : ' It is not even in our list of names: always taking the top letter does not guarantee a real name.'}`, zh: `何不乾脆不擲骰子，每次都選最可能的字母？這叫做<b>貪婪解碼（greedy decoding）</b>。這樣模型會把 <code>.ma</code> 接成「${data().greedyName}」，每一步都照第一名走，也就是溫度 <i>T</i> → 0 的極端。${data().greedyKnown ? '' : '它甚至不在我們的名字清單裡：永遠選第一名，並不保證寫出真正的名字。'}` })),
        lz(() => ({ en: `And it writes that every single time: ${data().greedy.unique} different name out of ${N}. Each step’s winner is often far from certain (see the percentages), yet the losers never get a turn.`, zh: `而且每一次都寫一樣的：${N} 次只有 ${data().greedy.unique} 種名字。每一步的第一名其實常常沒那麼有把握（看看百分比），但其他選項永遠沒有機會。` })),
      ],
      callout: { kind: 'insight', text: { en: 'Greedy is predictable but repetitive. In long text it tends to loop on the same phrases.', zh: '貪婪解碼很穩定，卻很重複。在長文章裡，它常會一直繞著同樣的句子打轉。' } },
      changed: [{ sign: '~', text: { en: 'Dice → always the top letter', zh: '擲骰子 → 永遠選第一名' } }],
    },
    {
      short: { en: 'Top-k', zh: 'Top-k' }, chapter: 'technique',
      title: { en: 'Top-k: cut off the long tail', zh: 'Top-k：剪掉長長的尾巴' },
      body: [
        { en: `A middle way: keep only the <b>k</b> likeliest letters, drop the rest, and rescale the kept ones to add up to 100%. Then roll the die. Here k = ${K}.`, zh: `折衷的做法：只保留最可能的 <b>k</b> 個字母，其餘全部丟掉，再把保留下來的重新調整成加起來 100%，然後才擲骰子。這裡 k = ${K}。` },
        lz(() => ({ en: `Rare, odd letters can no longer sneak in, but there is still variety: ${data().topkRun.unique} different names out of ${N}, and ${pct(data().topkRun.real)} already in our list.`, zh: `奇怪的冷門字母再也溜不進來，但仍然有變化：${N} 個名字裡有 ${data().topkRun.unique} 種不同的，其中 ${pct(data().topkRun.real)} 是清單裡本來就有的。` })),
      ],
      changed: [{ sign: '+', text: { en: `Keep top ${K}, rescale, then sample`, zh: `保留前 ${K} 名、重新調整、再取樣` } }],
    },
    {
      short: { en: 'Try it', zh: '動手試試' }, chapter: 'solution',
      title: { en: 'Try it: turn the temperature dial yourself', zh: '動手試試：自己轉溫度旋鈕' },
      body: [
        { en: 'Try it: drag the temperature and pick a top-k, and watch the bars, the divided logits and the names. Everything is recomputed live from the course model, continuing from <code>.ma</code> with seed 7.', zh: '動手試試：拖動溫度、選一個 top-k，看看長條、除過的 logits 和名字怎麼變。全部都是用課程模型即時算出來的，從 <code>.ma</code> 接著寫，種子 7。' },
        { en: 'Notice: toward 0.1 the bars collapse onto one letter and the same few names repeat; toward 2 the bars flatten and the spellings get odd. Top-k 1 keeps only the leader, so it writes the greedy name at any temperature.', zh: '注意看：越接近 0.1，長條越集中在一個字母上，名字也一直重複那幾個；越接近 2，長條越平，拼法越奇怪。top-k 選 1 只留第一名，所以不管溫度多少，寫出的都是貪婪解碼的那個名字。' },
      ],
      changed: [{ sign: '+', text: { en: 'Temperature slider and top-k choice', zh: '溫度滑桿與 top-k 選項' } }],
    },
    {
      short: { en: 'Chat apps', zh: '聊天應用' }, chapter: 'solution',
      title: { en: 'Why chat apps have a temperature setting', zh: '為什麼聊天應用有「溫度」設定' },
      body: [
        { en: 'Every chat model works like ours: scores, then a choosing rule. Low temperature gives steady, predictable answers, good for facts and code. Higher temperature gives more varied, creative text, with more risk of going off track.', zh: '每個聊天模型都和我們的一樣：先給分數，再用一條規則來選。溫度低，回答穩定、可預期，適合查資料或寫程式；溫度高，文字更多變、更有創意，但也更容易跑偏。' },
        { en: 'That is why many model APIs expose it. OpenAI’s API, for example, accepts a temperature from 0 to 2. Our own runs show the same trade-off in miniature.', zh: '所以很多模型的 API 都開放這個設定，例如 OpenAI 的 API 接受 0 到 2 的溫度。我們自己的實驗，就是這個取捨的縮小版。' },
      ],
      changed: [{ sign: '+', text: { en: 'All settings side by side', zh: '所有設定並排比較' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'From logits to text in one line', zh: '一句話說完從 logits 到文字' },
      body: [
        { en: 'The model outputs <b>logits</b>. Divide by the <b>temperature</b>, apply <b>softmax</b>, optionally keep the <b>top-k</b>, then <b>sample</b> one token, append it, and repeat.', zh: '模型輸出 <b>logits</b>。除以<b>溫度</b>，做 <b>softmax</b>，可以選擇只留 <b>top-k</b>，然後<b>取樣</b>一個詞元，接到後面，重複下去。' },
        { en: 'Greedy decoding is the extreme case: always the top token, the same output every time.', zh: '貪婪解碼是最極端的情況：永遠選第一名，每次輸出都一樣。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: The Training Loop. You have seen the whole model at work; now see how its weights got their values.', zh: '下一課：訓練迴圈。你已經看過整個模型如何運作，接下來看看它的權重是怎麼得到這些數值的。' } },
    },
  ],

  scene(step, input = {}) {
    const d = data();
    const items: Item[] = [];
    const TRY = 7, CHAT = 8, RECAP = 9;
    const tT = Math.max(0.1, Math.min(2, Number(input.temp ?? 1) || 1));
    const tK = ['0', '5', '3', '1'].includes(String(input.k)) ? Number(input.k) : 0;

    // prefix chips
    if (step <= TRY) [...PREFIX].forEach((ch, i) => items.push({ k: 'pre' + i, t: 'token', text: ch, x: 20 + i * 50, y: 4, tone: ch === '.' ? 'b' : 'ink', dim: step >= 4 && step !== TRY }));

    if (step === 0 || step === 1) {
      const order = [...CHARS.slice(1), '.'];
      order.forEach((ch, i) => {
        const id = CHARS.indexOf(ch), at = gridAt(i);
        const v = step === 0 ? d.logits[id].toFixed(1) : pct1(d.probs[1][id]);
        items.push({ k: 'g' + id, t: 'token', text: ch, x: at.x, y: at.y + 20, sub: v, tone: id === R ? 'a' : ch === '.' ? 'b' : 'ink' });
      });
      items.push({ k: 'gl', t: 'text', x: 450, y: 370, text: step === 0 ? t({ en: 'the number under each token is its logit (score)', zh: '每個詞元下方的數字是它的 logit（分數）' }) : t({ en: 'after softmax: the probability of each token (all add up to 100%)', zh: '經過 softmax：每個詞元的機率（全部加起來 100%）' }), size: 14, tone: 'muted', anchor: 'middle' });
      if (step === 1) items.push({ k: 'bars', t: 'bars', x: 200, y: 400, w: 500, h: 110, items: top(d.probs[1], 6) });
    }

    if (step === 2) {
      items.push({ k: 'bars', t: 'bars', x: 200, y: 80, w: 500, h: 180, items: top(d.probs[1], 8), title: t({ en: 'next letter after “.ma” (T = 1)', zh: '「.ma」之後的下一個字母（T = 1）' }) });
      items.push({ k: 'sh', t: 'text', x: 450, y: 320, text: t({ en: `8 names written by sampling (seed ${SEED})`, zh: `用取樣寫出的 8 個名字（種子 ${SEED}）` }), size: 15, bold: true, anchor: 'middle' });
      d.runs[1].names.slice(0, 8).forEach((n, i) => items.push({ k: 'n1_' + i, t: 'text', x: 150 + (i % 4) * 200, y: 370 + Math.floor(i / 4) * 44, text: show(n), size: 20, mono: true, bold: true, anchor: 'middle', tone: 'a' }));
    }

    if (step === 3) {
      TEMPS.forEach((T, i) => items.push({ k: 'tb' + i, t: 'bars', x: 30 + i * 290, y: 80, w: 260, h: 200, items: top(d.probs[i], 5), title: `T = ${T}` }));
      const lead = top(d.probs[1], 5).map((b) => CHARS.indexOf(b.label));
      items.push({
        k: 'div', t: 'heat', x: 330, y: 340, cell: 52, fmt: 'num', max: Math.max(...lead.map((j) => Math.abs(d.logits[j] / 0.5))),
        m: TEMPS.map((T) => lead.map((j) => d.logits[j] / T)), rows: TEMPS.map((T) => `logit ÷ ${T}`), cols: lead.map((j) => CHARS[j]),
      });
    }

    if (step === 4) {
      TEMPS.forEach((T, i) => {
        const x = 150 + i * 300, r = d.runs[i];
        items.push({ k: 'th' + i, t: 'text', x, y: 70, text: `T = ${T}`, size: 18, bold: true, anchor: 'middle', tone: i === 0 ? 'a' : i === 2 ? 'b' : 'ink' });
        r.names.slice(0, 8).forEach((n, j) => items.push({ k: `n${i}_${j}`, t: 'text', x, y: 110 + j * 34, text: show(n), size: 18, mono: true, anchor: 'middle', tone: NAMES.includes(n) ? 'a' : 'ink', bold: NAMES.includes(n) }));
        items.push({ k: 'ts' + i, t: 'text', x, y: 410, text: t({ en: `in our list: ${pct(r.real)}`, zh: `清單裡有：${pct(r.real)}` }), size: 15, mono: true, bold: true, anchor: 'middle' });
        items.push({ k: 'tu' + i, t: 'text', x, y: 438, text: t({ en: `different: ${r.unique} / ${N}`, zh: `不同名字：${r.unique} / ${N}` }), size: 14, mono: true, anchor: 'middle', tone: 'muted' });
      });
      items.push({ k: 'tn', t: 'text', x: 450, y: 490, text: t({ en: `bold = a name already in our list · stats from ${N} names per temperature`, zh: `粗體 = 清單裡本來就有的名字 · 統計來自每種溫度 ${N} 個名字` }), size: 13, tone: 'muted', anchor: 'middle' });
    }

    if (step === 5) {
      items.push({ k: 'gh', t: 'text', x: 450, y: 90, text: t({ en: 'greedy: take the top letter at every step', zh: '貪婪解碼：每一步都選第一名' }), size: 15, bold: true, anchor: 'middle' });
      const n = d.path.length + PREFIX.length, x0 = 450 - (n * 70) / 2;
      const P0 = PREFIX.length;
      [...PREFIX].forEach((ch, i) => items.push({ k: 'gs' + i, t: 'token', text: ch, x: x0 + i * 70, y: 120, tone: ch === '.' ? 'b' : 'ink' }));
      d.path.forEach((s, i) => {
        items.push({ k: 'gp' + i, t: 'token', text: s.ch, x: x0 + (i + P0) * 70, y: 120, sub: pct(s.p), tone: s.ch === '.' ? 'b' : 'a' });
        items.push({ k: 'ga' + i, t: 'arrow', x1: x0 + (i + P0 - 1) * 70 + 46, y1: 142, x2: x0 + (i + P0) * 70 - 2, y2: 142, tone: 'muted' });
      });
      items.push({ k: 'gn', t: 'text', x: 450, y: 215, text: t({ en: 'the % is how likely the chosen letter was', zh: '百分比是被選中字母的機率' }), size: 13, tone: 'muted', anchor: 'middle' });
      d.greedy.names.slice(0, 8).forEach((nm, i) => items.push({ k: 'gr' + i, t: 'text', x: 150 + (i % 4) * 200, y: 300 + Math.floor(i / 4) * 44, text: show(nm), size: 20, mono: true, bold: true, anchor: 'middle', tone: 'bad' }));
      items.push({ k: 'gu', t: 'text', x: 450, y: 420, text: t({ en: `${d.greedy.unique} different name in ${N} tries`, zh: `試了 ${N} 次，只有 ${d.greedy.unique} 種名字` }), size: 18, bold: true, anchor: 'middle', tone: 'bad' });
      items.push({ k: 'gk', t: 'text', x: 450, y: 452, text: d.greedyKnown ? t({ en: `“${d.greedyName}” is in our list of names`, zh: `「${d.greedyName}」在我們的名字清單裡` }) : t({ en: `“${d.greedyName}” is not in our list of names`, zh: `「${d.greedyName}」不在我們的名字清單裡` }), size: 14, anchor: 'middle', tone: 'muted' });
    }

    if (step === 6) {
      items.push({ k: 'kb1', t: 'bars', x: 40, y: 80, w: 380, h: 200, items: top(d.probs[1], 8), title: t({ en: 'all tokens (T = 1)', zh: '全部詞元（T = 1）' }) });
      items.push({ k: 'kb2', t: 'bars', x: 480, y: 80, w: 380, h: 200, items: top(d.topk, 8), title: t({ en: `top-${K} kept, rescaled`, zh: `只留前 ${K} 名，重新調整` }) });
      items.push({ k: 'ka', t: 'arrow', x1: 425, y1: 180, x2: 475, y2: 180, tone: 'a' });
      items.push({ k: 'kh', t: 'text', x: 450, y: 330, text: t({ en: `names with T = 1, top-${K} (seed ${SEED})`, zh: `T = 1、top-${K} 寫出的名字（種子 ${SEED}）` }), size: 15, bold: true, anchor: 'middle' });
      d.topkRun.names.slice(0, 8).forEach((n, i) => items.push({ k: 'kn' + i, t: 'text', x: 150 + (i % 4) * 200, y: 375 + Math.floor(i / 4) * 40, text: show(n), size: 19, mono: true, bold: true, anchor: 'middle', tone: 'a' }));
    }

    if (step === TRY) {
      const p = topK(softmaxRow(d.logits, tT), tK);
      const tr = run(tT, tK);
      const kTxt = tK ? `top-${tK}` : t({ en: 'no top-k', zh: '不用 top-k' });
      items.push({ k: 'yb', t: 'bars', x: 40, y: 90, w: 520, h: 200, items: top(p, 8), title: t({ en: `next letter after “.ma” (T = ${tT.toFixed(2)}, ${kTxt})`, zh: `「.ma」之後的下一個字母（T = ${tT.toFixed(2)}，${kTxt}）` }) });
      const lead = top(softmaxRow(d.logits), 5).map((b) => CHARS.indexOf(b.label));
      items.push({ k: 'yl', t: 'text', x: 610, y: 100, text: t({ en: 'top 5 logits ÷ T', zh: '前 5 名 logits ÷ T' }), size: 14, tone: 'muted' });
      items.push({ k: 'yh', t: 'heat', x: 610, y: 140, cell: 52, fmt: 'num', max: Math.max(...lead.map((j) => Math.abs(d.logits[j] / 0.5))),
        m: [lead.map((j) => d.logits[j] / tT)], rows: [''], cols: lead.map((j) => CHARS[j]) });
      items.push({ k: 'yhn', t: 'text', x: 610, y: 230, text: t({ en: 'bigger gaps → sharper bars', zh: '差距越大 → 長條越集中' }), size: 13, tone: 'muted' });
      items.push({ k: 'ysh', t: 'text', x: 450, y: 336, text: t({ en: `8 names written now (seed ${SEED}) · bold = already in our list`, zh: `現在寫出的 8 個名字（種子 ${SEED}）· 粗體 = 清單裡本來就有` }), size: 14, tone: 'muted', anchor: 'middle' });
      tr.names.slice(0, 8).forEach((n, i) => items.push({ k: 'yn' + i, t: 'text', x: 150 + (i % 4) * 200, y: 378 + Math.floor(i / 4) * 40, text: show(n), size: 19, mono: true, anchor: 'middle', tone: known().has(n) ? 'a' : 'ink', bold: known().has(n) }));
      items.push({ k: 'yst', t: 'text', x: 450, y: 480, text: t({ en: `of ${N} names: ${pct(tr.real)} already in our list · ${tr.unique} different`, zh: `${N} 個名字裡：${pct(tr.real)} 本來就在清單裡 · ${tr.unique} 種不同` }), size: 15, mono: true, bold: true, anchor: 'middle' });
    }

    if (step === CHAT) {
      const settings: { label: string; run: Run; x: number }[] = [
        { label: t({ en: 'greedy', zh: '貪婪' }), run: d.greedy, x: 110 },
        { label: `T = 1, top-${K}`, run: d.topkRun, x: 280 },
        { label: 'T = 0.5', run: d.runs[0], x: 450 },
        { label: 'T = 1', run: d.runs[1], x: 620 },
        { label: 'T = 1.5', run: d.runs[2], x: 790 },
      ];
      items.push({ k: 'sl', t: 'arrow', x1: 60, y1: 120, x2: 860, y2: 120, tone: 'muted' });
      items.push({ k: 'sll', t: 'text', x: 60, y: 96, text: t({ en: '← steady, predictable', zh: '← 穩定、可預期' }), size: 14, tone: 'a', bold: true });
      items.push({ k: 'slr', t: 'text', x: 860, y: 96, text: t({ en: 'varied, creative →', zh: '多變、有創意 →' }), size: 14, tone: 'b', bold: true, anchor: 'end' });
      settings.forEach((s, i) => {
        items.push({ k: 'sb' + i, t: 'box', x: s.x - 75, y: 150, w: 150, h: 46, label: s.label, tone: i === 0 ? 'bad' : 'ink' });
        items.push({ k: 'sr' + i, t: 'text', x: s.x, y: 236, text: t({ en: `list: ${pct(s.run.real)}`, zh: `清單內：${pct(s.run.real)}` }), size: 15, mono: true, bold: true, anchor: 'middle' });
        items.push({ k: 'su' + i, t: 'text', x: s.x, y: 262, text: t({ en: `different: ${s.run.unique}`, zh: `不同：${s.run.unique}` }), size: 14, mono: true, anchor: 'middle', tone: 'muted' });
        s.run.names.slice(0, 3).forEach((n, j) => items.push({ k: `se${i}_${j}`, t: 'text', x: s.x, y: 310 + j * 30, text: show(n), size: 16, mono: true, anchor: 'middle', tone: i === 0 ? 'bad' : 'ink' }));
      });
      items.push({ k: 'sn', t: 'text', x: 450, y: 450, text: t({ en: `each setting: ${N} names, seed ${SEED} · “list” = share already in our list of names`, zh: `每種設定：${N} 個名字，種子 ${SEED} · 「清單內」= 本來就在名字清單裡的比例` }), size: 13, tone: 'muted', anchor: 'middle' });
    }

    if (step === RECAP) {
      const boxes: L[] = [
        { en: 'logits', zh: 'logits' }, { en: '÷ temperature', zh: '÷ 溫度' }, { en: 'softmax', zh: 'softmax' },
        { en: 'keep top-k', zh: '保留 top-k' }, { en: 'sample a token', zh: '取樣一個詞元' },
      ];
      boxes.forEach((b, i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 200, w: 150, h: 60, label: t(b), tone: i === 4 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 230, x2: bx + 170, y2: 230 });
      });
      items.push({ k: 'loop', t: 'arrow', x1: 795, y1: 264, x2: 105, y2: 264, dashed: true });
      items.push({ k: 'rt', t: 'text', x: 450, y: 300, text: t({ en: 'append the token and run the model again, until “.”', zh: '把詞元接到後面，再跑一次模型，直到寫出「.」' }), size: 14, tone: 'muted', anchor: 'middle' });
    }

    const T = step === 3 ? '0.5 / 1 / 1.5' : step === 4 ? '0.5 / 1 / 1.5' : step === 5 ? t({ en: 'greedy', zh: '貪婪' }) : step === TRY ? tT.toFixed(2) : step >= 1 && step <= 6 ? '1' : '—';
    const pr = step === 3 || step === 4 ? TEMPS.map((_, i) => pct(d.probs[i][R])).join('/') : step === 5 ? '100%' : step === 6 ? pct(d.topk[R])
      : step === TRY ? pct(topK(softmaxRow(d.logits, tT), tK)[R]) : step >= 1 && step <= 6 ? pct(d.probs[1][R]) : '—';
    const cur: Run | null = step === 2 ? d.runs[1] : step === 5 ? d.greedy : step === 6 ? d.topkRun : step === TRY ? run(tT, tK) : null;
    return {
      metrics: [
        { label: { en: 'Prefix', zh: '開頭' }, value: PREFIX },
        { label: { en: 'Temperature', zh: '溫度' }, value: T },
        { label: { en: 'Chance of “r”', zh: '「r」的機率' }, value: pr },
        { label: { en: `Different names / ${N}`, zh: `不同名字 / ${N}` }, value: cur ? String(cur.unique) : step === 4 ? d.runs.map((r) => r.unique).join('/') : '—' },
      ],
      items,
      legend: [
        { tone: 'a', text: { en: 'top choice / sampled names', zh: '第一名／取樣的名字' } },
        { tone: 'b', text: { en: 'start / end mark “.”', zh: '開頭／結尾符號「.」' } },
        { tone: 'bad', text: { en: 'greedy decoding', zh: '貪婪解碼' } },
      ],
      controls: step === TRY ? [
        { key: 'temp', kind: 'slider', label: { en: 'Temperature', zh: '溫度' }, min: 0.1, max: 2, step: 0.05, value: 1, format: (v: number) => v.toFixed(2) },
        { key: 'k', kind: 'choice', label: { en: 'Top-k', zh: 'Top-k' }, value: '0', options: [
          { value: '0', label: { en: 'off', zh: '不用' } }, { value: '5', label: { en: '5', zh: '5' } },
          { value: '3', label: { en: '3', zh: '3' } }, { value: '1', label: { en: '1', zh: '1' } },
        ] },
      ] : undefined,
    };
  },
};
