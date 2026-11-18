import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, NAMES, VAL_NAMES, encode, example } from '../model/data';
import { Mat, addRow, matmul, relu, softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';

// The trained model reads ".olivia". We follow the last token, the final "a",
// through the MLP: 16 numbers → 32 hidden neurons → 16 numbers.

const WORD = '.olivia';
const LABELS = [...WORD];
const F = LABELS.length - 1;
const VOWELS = new Set([...'aeiouy']);
const NA = 22; // a neuron that fires mostly on the first letter, mostly consonants
const FIRE = 0.5; // a neuron "fires" when its activation is above this
const NB = 2; // a neuron that fires on a vowel late in a longer name
const ROWS = ['.quinn', '.peter', '.olivia'];

// A text whose numbers are computed only when it is first shown.
const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const q = (s: string) => t({ en: `“${s}”`, zh: `「${s}」` });
const row = (v: number[]) => { const m = new Mat(1, v.length); m.d.set(v); return m; };

interface Data {
  model: GPT; x1: number[]; pre: number[]; hidden: number[]; out: number[];
  active: number; aboveZero: number; withMlp: number[]; noMlp: number[]; lossWith: number; lossWithout: number;
  counts: { tok: number; att: number; mlp: number; out: number; all: number };
  acts: number[][][]; // [row name][neuron A/B][position]
}
let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as Weights);
  const P = model.p;
  const tr = model.forward(encode(WORD));
  const x1 = tr.x1.row(F);
  const pre = addRow(matmul(row(x1), P.w1), P.b1).row(0);
  const hidden = tr.hidden.row(F);
  const logits = (x2: number[]) => addRow(matmul(row(x2), P.u), P.bu).row(0);
  // held-out loss with the MLP, and with its output switched off (x2 = x1)
  let lw = 0, lo = 0, n = 0;
  for (const name of VAL_NAMES) {
    const { x, y } = example(name);
    const t2 = model.forward(x);
    for (let i = 0; i < x.length; i++) {
      lw -= Math.log(softmaxRow(t2.logits.row(i))[y[i]]);
      lo -= Math.log(softmaxRow(logits(t2.x1.row(i)))[y[i]]);
      n++;
    }
  }
  const c = (names: string[]) => names.reduce((s, k) => s + P[k].d.length, 0);
  const counts = { tok: c(['tok', 'pos']), att: c(['q0', 'k0', 'v0', 'o']), mlp: c(['w1', 'b1', 'w2', 'b2']), out: c(['u', 'bu']), all: model.paramCount() };
  const acts = ROWS.map((w) => { const h = model.forward(encode(w)).hidden; return [NA, NB].map((j) => [...w].map((_, i) => h.d[i * h.c + j])); });
  return {
    model, x1, pre, hidden, out: tr.mlpOut.row(F), active: hidden.filter((v) => v > FIRE).length, aboveZero: hidden.filter((v) => v > 0).length,
    withMlp: softmaxRow(tr.logits.row(F)), noMlp: softmaxRow(logits(x1)),
    lossWith: lw / n, lossWithout: lo / n, counts, acts,
  };
})();

// How the two neurons behave over every position of every name in the list.
interface Stats { fires: number; total: number; firstFire: number; firstAll: number; consFire: number; consAll: number; vowNextFire: number; vowNextAll: number; endFire: number; endAll: number; lateFire: number; lateAll: number }
let statCache: Stats[] | undefined;
const stats = (): Stats[] => statCache ??= (() => {
  const { model } = data();
  const s = [NA, NB].map(() => ({ fires: 0, total: 0, firstFire: 0, firstAll: 0, consFire: 0, consAll: 0, vowNextFire: 0, vowNextAll: 0, endFire: 0, endAll: 0, lateFire: 0, lateAll: 0 }));
  for (const name of NAMES) {
    const { x, y } = example(name);
    const h = model.forward(x).hidden;
    for (let i = 0; i < x.length; i++) {
      const ch = CHARS[x[i]], nx = CHARS[y[i]];
      const cons = ch !== '.' && !VOWELS.has(ch), late = i >= 5 && VOWELS.has(ch), vowNext = VOWELS.has(nx), end = nx === '.';
      [NA, NB].forEach((j, n) => {
        const st = s[n];
        st.total++; if (i === 1) st.firstAll++; if (cons) st.consAll++; if (vowNext) st.vowNextAll++; if (end) st.endAll++; if (late) st.lateAll++;
        if (h.d[i * h.c + j] > FIRE) { st.fires++; if (i === 1) st.firstFire++; if (cons) st.consFire++; if (vowNext) st.vowNextFire++; if (end) st.endFire++; if (late) st.lateFire++; }
      });
    }
  }
  return s;
})();

const pct = (v: number) => Math.round(v * 100) + '%';
const fmt = (v: number) => v.toLocaleString('en-US');
const top = (p: number[], n: number) =>
  p.map((v, i) => ({ label: CHARS[i], v })).sort((a, b) => b.v - a.v).slice(0, n).map((b) => ({ ...b, hi: b.label === '.' }));

// fixed numbers used in the text (cheap to compute: only weight shapes)
const W1 = 16 * 32, B1 = 32, W2 = 32 * 16, B2 = 16;

export const mlp: Lesson = {
  id: 'mlp',
  title: { en: 'The MLP Layer', zh: 'MLP 層' },
  blurb: {
    en: 'After attention gathers context, each token thinks on its own: widen, keep the positives, shrink back.',
    zh: '注意力收集完上下文之後，每個詞元各自「思考」：先展開、只留正數、再縮回來。',
  },
  minutes: 10,
  level: 'beginner',
  learn: [
    { text: { en: 'Why gathering context is not enough', zh: '為什麼光收集上下文還不夠' }, step: 0 },
    { text: { en: 'How the MLP widens 16 numbers to 32 and back', zh: 'MLP 如何把 16 個數字展開成 32 個再縮回' }, step: 2 },
    { text: { en: 'What ReLU does', zh: 'ReLU 在做什麼' }, step: 3 },
    { text: { en: 'What a single neuron can detect', zh: '單一個神經元能偵測什麼' }, step: 5 },
    { text: { en: 'Where a transformer’s weights live', zh: 'Transformer 的權重都放在哪裡' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Gathering is not thinking', zh: '收集資訊不等於思考' },
      body: [
        { en: 'Attention lets each token collect a blend of what came before. But a blend is only a weighted average: it moves information around without doing much with it.', zh: '注意力讓每個詞元收集前面詞元的混合資訊。但混合只是加權平均：它把資訊搬來搬去，卻沒有真正加工。' },
        lz(() => ({
          en: `So after attention, each token passes through a second part, the <b>MLP</b>. Switch it off in the course model (our trained model) and its <b>held-out loss</b>, the average surprise at the real next letter on 615 names it never trained on (lower is better), rises from <b>${data().lossWith.toFixed(2)}</b> to <b>${data().lossWithout.toFixed(2)}</b>.`,
          zh: `所以在注意力之後，每個詞元還要經過第二個部分：<b>MLP</b>。在課程模型（我們訓練好的模型）裡把它關掉，它的<b>驗證損失</b>（在訓練時從沒看過的 615 個名字上，對真正下一個字母的平均「意外」程度，越低越好）會從 <b>${data().lossWith.toFixed(2)}</b> 升到 <b>${data().lossWithout.toFixed(2)}</b>。`,
        })),
      ],
      callout: { kind: 'insight', text: { en: 'Attention moves information between tokens. The MLP works on what each token now holds.', zh: '注意力負責在詞元之間搬運資訊，MLP 負責加工每個詞元手上的資訊。' } },
    },
    {
      short: { en: 'Per token', zh: '各自處理' }, chapter: 'technique',
      title: { en: 'Each token on its own', zh: '每個詞元各自處理' },
      body: [
        { en: 'The MLP looks at one token’s vector at a time. It never looks at the neighbours; that was attention’s job.', zh: 'MLP 一次只看一個詞元的向量，完全不看鄰居，那是注意力的工作。' },
        { en: 'Every token goes through the <i>same</i> MLP with the same weights, whatever its letter or position. We follow the last one, the final “a” of <code>.olivia</code>.', zh: '不管是哪個字母、在哪個位置，每個詞元都通過<i>同一個</i> MLP，用的是同一組權重。我們跟著最後一個詞元：<code>.olivia</code> 結尾的「a」。' },
      ],
      changed: [{ sign: '+', text: { en: '7 tokens, one shared MLP', zh: '7 個詞元，共用一個 MLP' } }],
    },
    {
      short: { en: 'Widen', zh: '展開' }, chapter: 'technique',
      title: { en: 'Widen: 16 numbers become 32', zh: '展開：16 個數字變成 32 個' },
      body: [
        { en: `The token’s 16 numbers are multiplied by a learned matrix <code>w1</code> (16 × 32 = ${W1} weights), and a learned list <code>b1</code> of 32 numbers, the <b>bias</b>, is added.`, zh: `詞元的 16 個數字乘上一個學習得來的矩陣 <code>w1</code>（16 × 32 = ${W1} 個權重），再加上一串學習得來的 32 個數字 <code>b1</code>，叫做<b>偏差（bias）</b>。` },
        { en: 'The result is 32 numbers. Think of each as a <b>neuron</b>: a little detector that gives a high number when the vector looks like a pattern it has learned.', zh: '結果是 32 個數字。可以把每一個想成一個<b>神經元（neuron）</b>：一個小偵測器，當向量看起來像它學過的某種模式時，就給出很大的數字。' },
      ],
      changed: [{ sign: '+', text: { en: '32 neuron scores for “a”', zh: '「a」的 32 個神經元分數' } }],
    },
    {
      short: { en: 'ReLU', zh: 'ReLU' }, chapter: 'technique',
      title: { en: 'ReLU: keep the positives', zh: 'ReLU：只留下正數' },
      body: [
        { en: 'Now a very simple rule called <b>ReLU</b>: a negative number becomes 0, a positive number stays as it is. Compare the two rows of numbers below.', zh: '接著套用一條很簡單的規則，叫做 <b>ReLU</b>：負數變成 0，正數維持原樣。比較下面兩列數字就知道了。' },
        lz(() => ({ en: `These outputs are called <b>activations</b>: a neuron is silent (0) or active with some strength. In this lesson a neuron <b>fires</b> when its activation is above ${FIRE}. For our “a”, ${data().aboveZero} of the 32 are above zero and ${data().active} fire.`, zh: `這些輸出叫做<b>活化值（activation）</b>：神經元不是安靜（0），就是以某個強度活躍。在這一課，活化值大於 ${FIRE} 就算這個神經元<b>激發</b>。對我們的「a」來說，32 個神經元裡有 ${data().aboveZero} 個大於 0，其中 ${data().active} 個激發。` })),
      ],
      callout: { kind: 'note', text: { en: 'Without this bend, multiplying by w1 and then w2 would collapse into a single 16 × 16 matrix, and the 32 neurons would add nothing a single matrix could not do. ReLU is what lets it make yes/no decisions. Real models mostly use smooth versions of the same bend (GELU in GPT-2, SwiGLU in LLaMA).', zh: '如果少了這個「折彎」，先乘 w1 再乘 w2 就會合併成單一個 16 × 16 矩陣，32 個神經元做不到任何單一矩陣做不到的事。正是 ReLU 讓它能做出「是／否」的判斷。真實的模型多半用同一種折彎的平滑版本（GPT-2 用 GELU，LLaMA 用 SwiGLU）。' } },
      changed: [{ sign: '~', text: { en: 'Negative scores → 0', zh: '負的分數 → 0' } }],
    },
    {
      short: { en: 'Shrink', zh: '縮回' }, chapter: 'technique',
      title: { en: 'Shrink back to 16', zh: '縮回 16 個數字' },
      body: [
        { en: `The 32 activations are multiplied by a second matrix <code>w2</code> (32 × 16 = ${W2} weights) and a bias <code>b2</code> (16) is added. Out come 16 numbers again, the same size as the token’s vector.`, zh: `這 32 個活化值再乘上第二個矩陣 <code>w2</code>（32 × 16 = ${W2} 個權重），加上偏差 <code>b2</code>（16 個），又得到 16 個數字，大小和詞元的向量一樣。` },
        { en: 'Each firing neuron adds its own learned pattern to this output; silent neurons add nothing. The output is then added onto the token’s vector (The Residual Stream lesson explains why <i>added</i>).', zh: '每個激發的神經元都會把自己學到的模式加進輸出，安靜的神經元什麼也不加。這個輸出接著會被加回詞元的向量上（為什麼是「加上去」，「殘差流」那一課會說明）。' },
      ],
      changed: [{ sign: '+', text: { en: '16 output numbers', zh: '16 個輸出數字' } }],
    },
    {
      short: { en: 'Neuron #22', zh: '22 號神經元' }, chapter: 'solution',
      title: { en: 'A neuron for the first letter', zh: '一個認得「第一個字母」的神經元' },
      body: [
        { en: 'Nobody told any neuron what to detect, yet some learn readable patterns. We checked all 32 over every letter of every name in our list. Neuron #22 is one of the clearest.', zh: '沒有人告訴神經元要偵測什麼，但有些還是學出了看得懂的模式。我們把 32 個神經元在清單裡每個名字的每個字母上都檢查了一遍，22 號是最清楚的之一。' },
        lz(() => { const st = stats()[0]; return { en: `The numbers under each letter are its real activations. It fires on ${pct(st.firstFire / st.firstAll)} of first letters and rarely later, and ${pct(st.consFire / st.fires)} of its firings are on consonants. It is not a pure consonant detector: it also fires on the “u” of quinn. When it fires, a vowel usually comes next. (We count y as a vowel.)`, zh: `每個字母下方的數字是它真實的活化值。名字的第一個字母有 ${pct(st.firstFire / st.firstAll)} 會讓它激發，後面的字母就很少；它激發的時候，有 ${pct(st.consFire / st.fires)} 是在子音上。它不是純粹的子音偵測器：quinn 的「u」也會讓它激發。它一激發，下一個字母通常是母音。（這裡把 y 算作母音。）` }; }),
      ],
      changed: [{ sign: '+', text: { en: 'Real activations of neuron #22', zh: '22 號神經元的真實活化值' } }],
    },
    {
      short: { en: 'Neuron #2', zh: '2 號神經元' }, chapter: 'solution',
      title: { en: 'A neuron for “near the end”', zh: '一個代表「快結束了」的神經元' },
      body: [
        { en: 'Neuron #2 does something different. It is silent on early letters and fires on a vowel late in a longer name, such as the final “a” of <code>.olivia</code>.', zh: '2 號神經元做的事不一樣。它在前面的字母上都很安靜，只在較長名字後段的母音上激發，例如 <code>.olivia</code> 最後的「a」。' },
        { en: 'When it fires, the name ends right after far more often than usual (right chart). Other neurons are messier: most respond to a mix of things, so we only claim what the counts show.', zh: '它一激發，名字緊接著就結束的比例比平常高很多（右邊的圖）。其他神經元就比較雜亂，多半同時對好幾種東西有反應，所以我們只說數字能證明的部分。' },
      ],
      callout: { kind: 'note', text: { en: 'A neuron firing together with an ending does not prove it causes the ending. The MLP’s output mixes all firing neurons at once.', zh: '某個神經元和「名字結束」一起出現，並不證明是它造成結束。MLP 的輸出是所有激發神經元一起混出來的。' } },
      changed: [{ sign: '~', text: { en: 'Same names, neuron #2', zh: '同樣的名字，換看 2 號神經元' } }],
    },
    {
      short: { en: 'Weights', zh: '權重' }, chapter: 'problem',
      title: { en: 'Case: where do the weights live?', zh: '案例：權重都放在哪裡？' },
      body: [
        lz(() => ({ en: `In our model the MLP holds ${W1} + ${B1} + ${W2} + ${B2} = ${fmt(data().counts.mlp)} weights, about the same as attention (${fmt(data().counts.att)}). That is because our hidden layer is only 2× wider than the vector (32 vs 16).`, zh: `在我們的模型裡，MLP 有 ${W1} + ${B1} + ${W2} + ${B2} = ${fmt(data().counts.mlp)} 個權重，跟注意力（${fmt(data().counts.att)} 個）差不多。這是因為我們的隱藏層只比向量寬 2 倍（32 對 16）。` })),
        { en: 'Many well-known designs widen 4×: GPT-2 small turns 768 numbers into 3,072. Then the MLP has about twice the weights of attention in every block, so most of a big transformer’s weights sit in its MLPs.', zh: '許多知名的設計會展開 4 倍：GPT-2 small 把 768 個數字展開成 3,072 個，這樣每個區塊裡 MLP 的權重大約是注意力的兩倍，所以大型 Transformer 的權重大多放在 MLP 裡。' },
      ],
      changed: [{ sign: '+', text: { en: 'Weight count per part', zh: '各部分的權重數量' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'The MLP in one line', zh: '一句話說完 MLP' },
      body: [
        { en: 'For each token on its own: multiply by <code>w1</code> and add <code>b1</code> (16 → 32), apply <b>ReLU</b> (keep positives), multiply by <code>w2</code> and add <code>b2</code> (32 → 16), and add the result onto the token’s vector.', zh: '對每個詞元各自做：乘 <code>w1</code> 加 <code>b1</code>（16 → 32），套用 <b>ReLU</b>（只留正數），乘 <code>w2</code> 加 <code>b2</code>（32 → 16），再把結果加回詞元的向量上。' },
        { en: 'The 32 hidden numbers are neurons. Some learn readable detectors, like #22 for “first letter, usually a consonant” and #2 for “late vowel”.', zh: '中間那 32 個數字就是神經元。有些學成了看得懂的偵測器，例如偵測「第一個字母，通常是子音」的 22 號，和偵測「後段母音」的 2 號。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Mixture of Experts. What if each block had several MLPs, and a router sent each token to just one of them?', zh: '下一課：混合專家模型。如果每個區塊有好幾個 MLP，再由一個路由器把每個詞元只送到其中一個，會怎樣？' } },
    },
  ],

  scene(step) {
    const d = data();
    const items: Item[] = [];
    const focus = step >= 2 && step <= 4;

    // the token row
    if (step === 0 || focus) {
      LABELS.forEach((ch, i) => items.push({
        k: 'tok' + i, t: 'token', text: ch, x: 250 + i * 58, y: 20,
        tone: i === F ? 'a' : ch === '.' ? 'b' : 'ink', dim: focus && i !== F,
      }));
    }
    if (step === 1) {
      const ry = (i: number) => 40 + i * 64;
      items.push({ k: 'mlpbox', t: 'box', x: 380, y: 30, w: 140, h: 6 * 64 + 50, label: t({ en: 'same MLP', zh: '同一個 MLP' }), sub: t({ en: 'for every token', zh: '每個詞元都用' }), tone: 'a' });
      const tr = d.model.forward(encode(WORD));
      LABELS.forEach((ch, i) => {
        items.push({ k: 'tok' + i, t: 'token', text: ch, x: 40, y: ry(i), tone: i === F ? 'a' : ch === '.' ? 'b' : 'ink' });
        items.push({ k: 'in' + i, t: 'vec', x: 110, y: ry(i) + 11, v: tr.x1.row(i), cell: 14, dim: i !== F });
        items.push({ k: 'ia' + i, t: 'arrow', x1: 340, y1: ry(i) + 22, x2: 376, y2: ry(i) + 22, tone: i === F ? 'a' : 'muted' });
        items.push({ k: 'oa' + i, t: 'arrow', x1: 524, y1: ry(i) + 22, x2: 560, y2: ry(i) + 22, tone: i === F ? 'a' : 'muted' });
        items.push({ k: 'ou' + i, t: 'vec', x: 570, y: ry(i) + 11, v: tr.mlpOut.row(i), cell: 14, dim: i !== F });
      });
      items.push({ k: 'h1', t: 'text', x: 110, y: 26, text: t({ en: 'after attention', zh: '經過注意力後' }), size: 13, tone: 'muted' });
      items.push({ k: 'h2', t: 'text', x: 570, y: 26, text: t({ en: 'MLP output', zh: 'MLP 輸出' }), size: 13, tone: 'muted' });
    }

    if (step === 0) {
      items.push({ k: 'att', t: 'box', x: 250, y: 100, w: 392, h: 50, label: t({ en: 'attention: gather from earlier tokens', zh: '注意力：從前面的詞元收集資訊' }) });
      LABELS.forEach((_, i) => items.push({ k: 'ga' + i, t: 'arrow', x1: 272 + i * 58, y1: 68, x2: 272 + i * 58, y2: 98, tone: i === F ? 'a' : 'muted' }));
      items.push({ k: 'x1', t: 'vec', x: 358, y: 200, v: d.x1, cell: 11 });
      items.push({ k: 'x1l', t: 'text', x: 546, y: 216, text: t({ en: '← vector of the last “a”', zh: '← 最後那個「a」的向量' }), size: 13, tone: 'muted' });
      items.push({ k: 'a0', t: 'arrow', x1: 446, y1: 152, x2: 446, y2: 194, tone: 'a' });
      items.push({ k: 'mlp0', t: 'box', x: 346, y: 250, w: 200, h: 46, label: t({ en: 'MLP: on or off?', zh: 'MLP：開或關？' }), tone: 'b' });
      items.push({ k: 'a1', t: 'arrow', x1: 446, y1: 228, x2: 446, y2: 248, tone: 'muted' });
      items.push({ k: 'lw', t: 'text', x: 280, y: 380, text: d.lossWith.toFixed(2), size: 34, bold: true, mono: true, anchor: 'middle', tone: 'a' });
      items.push({ k: 'lwl', t: 'text', x: 280, y: 410, text: t({ en: 'held-out loss, MLP on', zh: '驗證損失，MLP 開' }), size: 14, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'lo', t: 'text', x: 620, y: 380, text: d.lossWithout.toFixed(2), size: 34, bold: true, mono: true, anchor: 'middle', tone: 'bad' });
      items.push({ k: 'lol', t: 'text', x: 620, y: 410, text: t({ en: 'held-out loss, MLP off', zh: '驗證損失，MLP 關' }), size: 14, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'lnote', t: 'text', x: 450, y: 460, text: t({ en: 'lower is better · measured on 615 names the model never trained on', zh: '越低越好 · 用模型從沒練過的 615 個名字測量' }), size: 13, tone: 'muted', anchor: 'middle' });
    }

    if (focus) {
      const xIn = 258, cell = 24;
      items.push({ k: 'x1', t: 'vec', x: xIn, y: step === 4 ? 100 : 100, v: d.x1, cell, label: t({ en: '16 numbers in (vector of “a”)', zh: '輸入 16 個數字（「a」的向量）' }) });
      if (step === 2 || step === 3) {
        items.push({ k: 'w1a', t: 'arrow', x1: 450, y1: 130, x2: 450, y2: 188, tone: 'a' });
        items.push({ k: 'w1l', t: 'text', x: 462, y: 164, text: t({ en: `× w1 (${W1} weights) + b1 (${B1})`, zh: `× w1（${W1} 個權重）+ b1（${B1} 個）` }), size: 14, tone: 'a' });
        items.push({ k: 'pre', t: 'vec', x: 66, y: 210, v: d.pre, cell, label: t({ en: '32 neuron scores, before ReLU', zh: '32 個神經元分數（ReLU 之前）' }) });
      }
      if (step === 3 || step === 4) {
        const hy = step === 3 ? 320 : 210;
        if (step === 3) {
          items.push({ k: 'rla', t: 'arrow', x1: 450, y1: 240, x2: 450, y2: 298, tone: 'a' });
          items.push({ k: 'rll', t: 'text', x: 462, y: 274, text: t({ en: 'ReLU: negative → 0', zh: 'ReLU：負數 → 0' }), size: 14, tone: 'a' });
        }
        items.push({ k: 'hid', t: 'vec', x: 66, y: hy, v: d.hidden, cell, label: t({ en: `activations: ${d.active} of 32 neurons fire (> ${FIRE})`, zh: `活化值：32 個神經元中有 ${d.active} 個激發（> ${FIRE}）` }) });
      }
      if (step === 3) {
        const cols = Array.from({ length: 10 }, (_, j) => j);
        items.push({
          k: 'nums', t: 'heat', x: 250, y: 400, cell: 46, fmt: 'num', max: Math.max(...d.pre.map(Math.abs)),
          m: [cols.map((j) => d.pre[j]), cols.map((j) => d.hidden[j])],
          rows: [t({ en: 'before', zh: '之前' }), t({ en: 'after', zh: '之後' })], cols: cols.map((j) => '#' + j),
        });
        items.push({ k: 'numsl', t: 'text', x: 40, y: 452, text: t({ en: 'first 10 neurons:', zh: '前 10 個神經元：' }), size: 13, tone: 'muted' });
      }
      if (step === 4) {
        items.push({ k: 'w2a', t: 'arrow', x1: 450, y1: 240, x2: 450, y2: 318, tone: 'a' });
        items.push({ k: 'w2l', t: 'text', x: 462, y: 284, text: t({ en: `× w2 (${W2} weights) + b2 (${B2})`, zh: `× w2（${W2} 個權重）+ b2（${B2} 個）` }), size: 14, tone: 'a' });
        items.push({ k: 'out', t: 'vec', x: xIn, y: 340, v: d.out, cell, label: t({ en: '16 numbers out', zh: '輸出 16 個數字' }) });
        items.push({ k: 'outn', t: 'text', x: 450, y: 410, text: t({ en: '→ added onto the vector of “a”', zh: '→ 加回「a」的向量上' }), size: 14, tone: 'muted', anchor: 'middle' });
      }
    }

    if (step === 5 || step === 6) {
      const n = step === 5 ? 0 : 1;
      const st = stats()[n];
      ROWS.forEach((w, r) => {
        const y = 70 + r * 130;
        items.push({ k: 'rn' + r, t: 'text', x: 30, y: y - 12, text: w, size: 14, tone: 'muted', mono: true });
        [...w].forEach((ch, i) => {
          const a = d.acts[r][n][i];
          items.push({ k: `c${r}_${i}`, t: 'token', text: ch, x: 30 + i * 56, y, sub: a.toFixed(2), tone: a > FIRE ? 'a' : ch === '.' ? 'b' : 'ink', dim: a <= FIRE });
        });
      });
      items.push({ k: 'nh', t: 'text', x: 30, y: 30, text: t({ en: `neuron #${n === 0 ? NA : NB}: activation under each letter`, zh: `${n === 0 ? NA : NB} 號神經元：每個字母下方是活化值` }), size: 15, bold: true, tone: 'a' });
      const b1 = n === 0
        ? { title: t({ en: 'current letter is a consonant', zh: '目前的字母是子音' }), all: st.consAll / st.total, fire: st.consFire / st.fires }
        : { title: t({ en: 'a vowel, 5th letter or later', zh: '第 5 個字母以後的母音' }), all: st.lateAll / st.total, fire: st.lateFire / st.fires };
      const b2 = n === 0
        ? { title: t({ en: 'next letter is a vowel', zh: '下一個字母是母音' }), all: st.vowNextAll / st.total, fire: st.vowNextFire / st.fires }
        : { title: t({ en: 'the name ends next', zh: '名字接著就結束' }), all: st.endAll / st.total, fire: st.endFire / st.fires };
      const L = { all: t({ en: 'all letters', zh: '所有字母' }), fire: t({ en: 'it fires', zh: '激發時' }) };
      items.push({ k: 'bars1', t: 'bars', x: 480, y: 90, w: 180, h: 210, title: b1.title, items: [{ label: L.all, v: b1.all }, { label: L.fire, v: b1.fire, hi: true }] });
      items.push({ k: 'bars2', t: 'bars', x: 690, y: 90, w: 180, h: 210, title: b2.title, items: [{ label: L.all, v: b2.all }, { label: L.fire, v: b2.fire, hi: true }] });
      items.push({ k: 'bn', t: 'text', x: 680, y: 350, text: t({ en: `“it fires” = activation > ${FIRE} (${fmt(st.fires)} of ${fmt(st.total)} letters)`, zh: `「激發時」＝活化值 > ${FIRE}（${fmt(st.total)} 個字母中的 ${fmt(st.fires)} 個）` }), size: 13, tone: 'muted', anchor: 'middle' });
      items.push({ k: 'bn2', t: 'text', x: 680, y: 374, text: t({ en: `counted over all ${fmt(NAMES.length)} names`, zh: `統計全部 ${fmt(NAMES.length)} 個名字` }), size: 13, tone: 'muted', anchor: 'middle' });
      if (n === 1) {
        items.push({ k: 'pe', t: 'text', x: 680, y: 430, text: t({ en: `model after “.olivia”: end “.” ${pct(d.withMlp[0])}`, zh: `模型在「.olivia」之後：結束「.」${pct(d.withMlp[0])}` }), size: 14, anchor: 'middle', tone: 'a', bold: true });
      }
    }

    if (step === 7) {
      const c = d.counts;
      items.push({
        k: 'share', t: 'bars', x: 150, y: 80, w: 600, h: 260, title: t({ en: `share of our model’s ${fmt(c.all)} weights`, zh: `我們模型 ${fmt(c.all)} 個權重的分布` }),
        items: [
          { label: t({ en: 'embeddings', zh: '嵌入' }), v: c.tok / c.all },
          { label: t({ en: 'attention', zh: '注意力' }), v: c.att / c.all },
          { label: 'MLP', v: c.mlp / c.all, hi: true },
          { label: t({ en: 'output', zh: '輸出' }), v: c.out / c.all },
        ],
      });
      [c.tok, c.att, c.mlp, c.out].forEach((v, i) => items.push({ k: 'cn' + i, t: 'text', x: 150 + 75 + i * 150, y: 370, text: fmt(v), size: 15, mono: true, bold: i === 2, tone: i === 2 ? 'a' : 'ink', anchor: 'middle' }));
      items.push({ k: 'gpt2', t: 'box', x: 150, y: 410, w: 600, h: 60, tone: 'b', label: t({ en: 'GPT-2 small: 768 → 3,072 → 768 (4× wider)', zh: 'GPT-2 small：768 → 3,072 → 768（寬 4 倍）' }), sub: t({ en: 'MLP ≈ 2 × attention weights in each block', zh: '每個區塊裡，MLP 權重 ≈ 注意力的 2 倍' }) });
    }

    if (step === 8) {
      const boxes = [
        { en: '16 numbers in', zh: '輸入 16 個數字' }, { en: '× w1 + b1 → 32', zh: '× w1 + b1 → 32' }, { en: 'ReLU', zh: 'ReLU' },
        { en: '× w2 + b2 → 16', zh: '× w2 + b2 → 16' }, { en: 'add onto vector', zh: '加回向量' },
      ];
      boxes.forEach((b, i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 200, w: 150, h: 60, label: t(b), tone: i === 2 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 230, x2: bx + 170, y2: 230 });
      });
      items.push({ k: 'rt', t: 'text', x: 450, y: 320, text: t({ en: `${fmt(d.counts.mlp)} weights · same for every token`, zh: `${fmt(d.counts.mlp)} 個權重 · 每個詞元共用` }), size: 15, tone: 'muted', anchor: 'middle' });
    }

    return {
      metrics: [
        { label: { en: 'Following', zh: '追蹤對象' }, value: step >= 1 && step <= 4 ? t({ en: 'last “a”', zh: '最後的「a」' }) : step === 5 || step === 6 ? (step === 5 ? '#22' : '#2') : '—' },
        { label: { en: 'Numbers per token', zh: '每個詞元的數字' }, value: step === 2 || step === 3 ? '16 → 32' : step >= 4 ? '16 → 32 → 16' : '16' },
        { label: { en: 'Neurons firing', zh: '激發的神經元' }, value: step >= 3 && step <= 4 ? `${d.active} / 32` : '—' },
        { label: { en: 'MLP weights', zh: 'MLP 權重' }, value: step >= 2 ? fmt(d.counts.mlp) : '—' },
        { label: { en: 'Loss without MLP', zh: '關掉 MLP 的損失' }, value: `${d.lossWithout.toFixed(2)} vs ${d.lossWith.toFixed(2)}` },
      ],
      items,
      legend: [
        { tone: 'pos', text: { en: 'positive number', zh: '正數' } },
        { tone: 'neg', text: { en: 'negative number', zh: '負數' } },
        { tone: 'a', text: { en: `neuron fires (> ${FIRE})`, zh: `神經元激發（> ${FIRE}）` } },
      ],
    };
  },
};
