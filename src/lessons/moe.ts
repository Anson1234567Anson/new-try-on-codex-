import type { Control, Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, VAL_NAMES, encode, example } from '../model/data';
import { Mat, addRow, matmul, relu } from '../model/tensor';
import weights from '../model/weights-moe.json';
import experiments from '../model/experiments.json';

// The shipped 4-expert model (top-1 routing) reads ".baylee". We look at the
// router's real probabilities, the expert each token is sent to, and what the
// experts ended up specializing in over all held-out names.

const WORD = '.baylee';
const LABELS = [...WORD];
const F = 3; // we follow the "y"
const E = 4;
const EX = experiments.moe;
const VOWELS = new Set([...'aeiouy']);
const POS = 8;

// A text whose numbers are computed only when it is first shown.
const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const row = (v: number[]) => { const m = new Mat(1, v.length); m.d.set(v); return m; };
const fmt = (n: number) => n.toLocaleString('en-US');
const pct = (v: number) => Math.round(v * 100) + '%';

interface Data {
  model: GPT;
  router: number[][]; chosen: number[];
  outs: number[][]; gated: number[]; // each expert's output for the "y", and the combined output
  w: { total: number; shared: number; router: number; expert: number; active: number; emb: number; att: number; out: number };
  posTab: number[][]; posN: number[]; // [position][expert] counts over held-out names
  chTab: Record<string, number[]>; tokens: number;
  use: number[]; // share of all held-out tokens per expert
  unsure: number; // share of held-out tokens whose chosen expert has under 50%
}
let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const model = GPT.fromJSON(weights as unknown as Weights);
  const P = model.p;
  const tr = model.forward(encode(WORD));
  const router = LABELS.map((_, i) => tr.router!.row(i));
  const chosen = tr.chosen!.map((c) => c[0]);
  const x1 = row(tr.x1.row(F));
  const outs = Array.from({ length: E }, (_, e) => {
    const s = '_e' + e;
    const hid = relu(addRow(matmul(x1, P['w1' + s]), P['b1' + s]));
    return addRow(matmul(hid, P['w2' + s]), P['b2' + s]).row(0);
  });
  const expert = ['w1', 'b1', 'w2', 'b2'].reduce((s, n) => s + P[n + '_e0'].d.length, 0);
  const total = model.paramCount(), rw = P.router.d.length;
  const shared = total - rw - E * expert;
  const c = (names: string[]) => names.reduce((s, n) => s + P[n].d.length, 0);
  const posTab = Array.from({ length: POS }, () => new Array(E).fill(0));
  const chTab: Record<string, number[]> = {};
  const use = new Array(E).fill(0);
  let tokens = 0, unsure = 0;
  for (const name of VAL_NAMES) {
    const { x } = example(name);
    const tr2 = model.forward(x);
    tr2.chosen!.forEach((c, i) => {
      const e = c[0];
      if (tr2.router!.d[i * E + e] < 0.5) unsure++;
      posTab[i][e]++; use[e]++; tokens++;
      (chTab[CHARS[x[i]]] ??= new Array(E).fill(0))[e]++;
    });
  }
  return {
    model, router, chosen, outs, gated: tr.mlpOut.row(F),
    w: { total, shared, router: rw, expert, active: shared + rw + expert, emb: c(['tok', 'pos']), att: c(['q0', 'k0', 'v0', 'o']), out: c(['u', 'bu']) },
    posTab, posN: posTab.map((r) => r.reduce((a, b) => a + b, 0)),
    chTab, tokens, use: use.map((v) => v / tokens), unsure: unsure / tokens,
  };
})();

// Facts about what each expert does, measured over all held-out names.
const spec = () => {
  const d = data();
  const at = (p: number, e: number) => d.posTab[p][e] / d.posN[p];
  const ch = (c: string, e: number) => d.chTab[c][e] / d.chTab[c].reduce((a, b) => a + b, 0);
  const e0all = d.posTab.reduce((s, r) => s + r[0], 0);
  const e0vow = Object.entries(d.chTab).reduce((s, [c, r]) => s + (VOWELS.has(c) ? r[0] : 0), 0);
  return {
    start: at(0, 1), second: at(1, 1), third: at(2, 2), mid: [at(3, 3), at(4, 3), at(5, 3)],
    e: ch('e', 0), y: ch('y', 0), e0vow: e0vow / e0all, e0early: at(1, 0), e0late: at(6, 0),
    // how well one clue alone predicts the chosen expert: always guess the usual expert for that position, or for that letter
    byPos: d.posTab.reduce((a, r) => a + Math.max(...r), 0) / d.tokens,
    byLetter: Object.values(d.chTab).reduce((a, r) => a + Math.max(...r), 0) / d.tokens,
  };
};

const last = (v: number[]) => v[v.length - 1];
const VAL = { dense: last(EX.dense.val), e4: last(EX.experts4.val), top2: last(EX.top2.val), e8: last(EX.experts8.val) };
const DENSE_W = EX.dense.params;

const EXPERT_X = (e: number) => 45 + e * 210;

// the "route your own name" step
const TRY = 7;
const tryControls: Control[] = [{ key: 'name', kind: 'text', label: { en: 'Name', zh: '名字' }, value: 'james', maxLength: 7, pattern: 'a-z', placeholder: { en: 'a–z', zh: 'a–z' } }];
const routeCache = new Map<string, { router: number[][]; chosen: number[] }>();
const route = (name: string) => {
  let r = routeCache.get(name);
  if (!r) {
    const tr = data().model.forward(encode('.' + name));
    r = { router: tr.router!.d.length ? Array.from({ length: name.length + 1 }, (_, i) => tr.router!.row(i)) : [], chosen: tr.chosen!.map((c) => c[0]) };
    if (routeCache.size > 200) routeCache.clear();
    routeCache.set(name, r);
  }
  return r;
};

export const moe: Lesson = {
  id: 'moe',
  title: { en: 'Mixture of Experts', zh: '混合專家模型（MoE）' },
  blurb: {
    en: 'Several MLPs per block and a router that sends each token to just one: more weights, without more work per token.',
    zh: '每個區塊放好幾個 MLP，再由路由器把每個詞元只送去其中一個：權重變多，每個詞元的工作量卻沒變多。',
  },
  minutes: 11,
  level: 'intermediate',
  learn: [
    { text: { en: 'Why a normal (dense) model makes every token pay for every weight', zh: '為什麼一般（密集）模型讓每個詞元都要用到每個權重' }, step: 0 },
    { text: { en: 'How a router picks an expert for each token', zh: '路由器如何替每個詞元挑選專家' }, step: 3 },
    { text: { en: 'Total weights versus weights used per token', zh: '總權重數和每個詞元實際用到的權重數' }, step: 5 },
    { text: { en: 'What the experts specialized in', zh: '各個專家分別專精什麼' }, step: 6 },
    { text: { en: 'What MoE gains at our size, and why it grows with scale', zh: 'MoE 在我們的規模下有多少好處，以及為什麼規模越大越有用' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Every token pays for every weight', zh: '每個詞元都要用上每個權重' },
      body: [
        { en: 'More weights can hold more knowledge. But in the model so far, a normal (<b>dense</b>) model, every token goes through the <i>whole</i> MLP and uses all 1,072 of its weights, whatever the letter.', zh: '權重越多，能記住的知識就越多。但在目前的模型（一般的<b>密集</b>模型）裡，每個詞元都要通過<i>整個</i> MLP：不管是哪個字母，都要用上它全部 1,072 個權重。' },
        lz(() => ({ en: `Make the MLP 4× bigger (${fmt(4 * data().w.expert)} weights) and every token does 4× the MLP work. Most of that work is probably wasted: a start token and a final “e” need different knowledge.`, zh: `把 MLP 放大 4 倍（${fmt(4 * data().w.expert)} 個權重），每個詞元的 MLP 工作量也跟著變成 4 倍。其中大部分可能是白做的：名字開頭的詞元和結尾的「e」需要的知識並不一樣。` })),
      ],
      callout: { kind: 'insight', text: { en: 'Idea: store many weights, but let each token use only the part it needs.', zh: '想法：存很多權重，但每個詞元只用它需要的那一部分。' } },
    },
    {
      short: { en: 'Experts', zh: '專家' }, chapter: 'technique',
      title: { en: 'Several experts and a router', zh: '好幾個專家，加上一個路由器' },
      body: [
        lz(() => ({ en: `Replace the one MLP with 4 <b>experts</b>. Each expert is an ordinary MLP of the same size (16 → 32 → 16, ${fmt(data().w.expert)} weights), with its own weights.`, zh: `把原本的一個 MLP 換成 4 個<b>專家（expert）</b>。每個專家都是一個普通的 MLP，大小和原本一樣（16 → 32 → 16，${fmt(data().w.expert)} 個權重），各有各的權重。` })),
        lz(() => ({ en: `A small <b>router</b> (16 × 4 = ${data().w.router} weights) looks at each token’s vector and decides which expert it goes to. This is a <b>mixture of experts</b>, or <b>MoE</b>.`, zh: `再加一個小小的<b>路由器（router）</b>（16 × 4 = ${data().w.router} 個權重），看每個詞元的向量，決定要把它送給哪個專家。這就是<b>混合專家模型（mixture of experts，MoE）</b>。` })),
      ],
      changed: [{ sign: '~', text: { en: '1 MLP → 4 experts', zh: '1 個 MLP → 4 個專家' } }, { sign: '+', text: { en: 'router', zh: '路由器' } }],
    },
    {
      short: { en: 'Router', zh: '路由器' }, chapter: 'technique',
      title: { en: 'The router scores every expert', zh: '路由器替每個專家打分數' },
      body: [
        { en: 'The router multiplies a token’s 16 numbers by its 16 × 4 matrix: one score per expert. Softmax turns the 4 scores into probabilities that add up to 100%.', zh: '路由器把詞元的 16 個數字乘上它的 16 × 4 矩陣，每個專家得到一個分數。softmax 再把這 4 個分數變成加起來是 100% 的機率。' },
        { en: 'These are the real probabilities for every token of <code>.baylee</code> from our trained 4-expert model (the 4-expert comparison run below, saved after its 1,500 steps). Like everything else, the router learned them during training.', zh: '這些是我們訓練好的 4 專家模型（就是後面比較實驗裡的 4 專家模型，訓練 1,500 步後存下來的），對 <code>.baylee</code> 每個詞元給出的真實機率。路由器和其他部分一樣，是在訓練中學會這樣打分數的。' },
      ],
      changed: [{ sign: '+', text: { en: 'router probabilities (7 × 4)', zh: '路由器機率（7 × 4）' } }],
    },
    {
      short: { en: 'Top-1', zh: '選一個' }, chapter: 'technique',
      title: { en: 'Each token goes to its top expert', zh: '每個詞元送去分數最高的專家' },
      body: [
        { en: 'Each token is sent to the expert with the highest probability. Choosing the best <i>k</i> experts is called <b>top-k routing</b>; our model uses <b>top-1</b>.', zh: '每個詞元都送去機率最高的那個專家。挑出最好的 <i>k</i> 個專家叫做 <b>top-k 路由</b>；我們的模型用的是 <b>top-1</b>。' },
        lz(() => { const d = data(); const p = d.router[F][d.chosen[F]]; return { en: `Follow the “y”: the router gives E${d.chosen[F]} ${pct(p)}, so “y” goes to E${d.chosen[F]}. The router is often unsure: over the ${VAL_NAMES.length} held-out names (names the model never trained on), ${pct(d.unsure)} of tokens get a winning expert under 50%.`, zh: `跟著「y」看：路由器給 E${d.chosen[F]} ${pct(p)}，所以「y」被送到 E${d.chosen[F]}。路由器常常不太篤定：在驗證集的 ${VAL_NAMES.length} 個名字（模型訓練時從沒看過）裡，有 ${pct(d.unsure)} 的詞元，勝出專家的機率不到 50%。` }; }),
      ],
      changed: [{ sign: '+', text: { en: 'chosen expert per token', zh: '每個詞元選中的專家' } }],
    },
    {
      short: { en: 'Combine', zh: '組合' }, chapter: 'technique',
      title: { en: 'Scale the expert’s output', zh: '把專家的輸出乘上機率' },
      body: [
        lz(() => { const d = data(); const e = d.chosen[F]; return { en: `Expert E${e} turns the “y” vector into 16 numbers, and they are multiplied by its router probability, ${d.router[F][e].toFixed(2)}. That is the layer’s output, added onto the vector as before.`, zh: `專家 E${e} 把「y」的向量變成 16 個數字，再乘上它的路由器機率 ${d.router[F][e].toFixed(2)}。這就是這一層的輸出，和以前一樣加回向量上。` }; }),
        { en: 'Multiplying by the probability matters for training: it lets training tell whether giving this expert more or less weight would have helped, so the router can learn which expert helps which token (the Gradients lesson explains how).', zh: '乘上機率這一步對訓練很重要：它讓訓練能判斷「多給或少給這個專家一點比重，會不會更好」，路由器才學得會哪個專家對哪個詞元有幫助（「梯度與反向傳播」那一課會說明怎麼做到）。' },
      ],
      callout: { kind: 'note', text: { en: 'For clarity our code runs all 4 experts on every token and multiplies the unchosen outputs by 0 (dimmed). Real systems run only the chosen expert; the result is the same.', zh: '為了好懂，我們的程式讓每個詞元都跑過 4 個專家，再把沒被選中的輸出乘上 0（變暗的部分）。真實系統只會跑被選中的專家，結果完全一樣。' } },
      changed: [{ sign: '+', text: { en: 'gated output for “y”', zh: '「y」乘上機率後的輸出' } }],
    },
    {
      short: { en: 'Weights', zh: '權重' }, chapter: 'solution',
      title: { en: 'Twice the weights, almost the same work', zh: '權重多一倍，工作量幾乎不變' },
      body: [
        lz(() => { const w = data().w; return { en: `Our MoE model stores ${fmt(w.total)} weights, about twice the ${fmt(DENSE_W)} of the normal model. But one token only uses the shared parts (${fmt(w.shared)}), the router (${w.router}) and one expert (${fmt(w.expert)}).`, zh: `我們的 MoE 模型存了 ${fmt(w.total)} 個權重，大約是一般模型 ${fmt(DENSE_W)} 個的兩倍。但一個詞元只會用到共用的部分（${fmt(w.shared)} 個）、路由器（${w.router} 個）和一個專家（${fmt(w.expert)} 個）。` }; }),
        lz(() => { const w = data().w; return { en: `That is ${fmt(w.shared)} + ${w.router} + ${fmt(w.expert)} = <b>${fmt(w.active)}</b> weights per token, only ${w.active - DENSE_W} more than the normal model, while the other 3 experts hold extra knowledge.`, zh: `也就是每個詞元 ${fmt(w.shared)} + ${w.router} + ${fmt(w.expert)} = <b>${fmt(w.active)}</b> 個權重，只比一般模型多 ${w.active - DENSE_W} 個，另外 3 個專家則存著額外的知識。` }; }),
      ],
      callout: { kind: 'insight', text: { en: 'MoE separates two things a dense model ties together: how much the model stores, and how much work each token costs.', zh: 'MoE 把密集模型綁在一起的兩件事拆開了：模型存了多少東西，以及每個詞元要花多少工作量。' } },
      changed: [{ sign: '+', text: { en: 'stored vs used per token', zh: '總儲存量 vs 每個詞元用到的量' } }],
    },
    {
      short: { en: 'Specialists', zh: '專長' }, chapter: 'solution',
      title: { en: 'What did the experts specialize in?', zh: '專家們各自專精什麼？' },
      body: [
        lz(() => { const s = spec(); return { en: `Nobody assigned roles. Over all ${VAL_NAMES.length} held-out names, <b>position</b> is a strong clue: E1 takes ${pct(s.start)} of the start “.” and ${pct(s.second)} of 2nd letters, E2 takes ${pct(s.third)} of 3rd letters, E3 ${pct(Math.min(...s.mid))}–${pct(Math.max(...s.mid))} of letters 4–6.`, zh: `沒有人分配角色。統計驗證集全部 ${VAL_NAMES.length} 個名字，<b>位置</b>是很強的線索：E1 接下 ${pct(s.start)} 的開頭「.」和 ${pct(s.second)} 的第 2 個字母，E2 接下 ${pct(s.third)} 的第 3 個字母，E3 則接下第 4～6 個字母的 ${pct(Math.min(...s.mid))}～${pct(Math.max(...s.mid))}。` }; }),
        lz(() => { const s = spec(); return { en: `E0 mixes both: it takes more of the later positions (${pct(s.e0early)} of 2nd letters, ${pct(s.e0late)} of 7th) and especially vowels there: ${pct(s.e)} of all “e” and ${pct(s.y)} of all “y”. ${pct(s.e0vow)} of its tokens are vowels (counting y).`, zh: `E0 兩者都有：位置越後面，它接下的越多（第 2 個字母的 ${pct(s.e0early)}，第 7 個的 ${pct(s.e0late)}），而且特別是那裡的母音：所有「e」有 ${pct(s.e)}、所有「y」有 ${pct(s.y)} 被送到它這裡。它收到的詞元有 ${pct(s.e0vow)} 是母音（y 也算）。` }; }),
      ],
      callout: { kind: 'note', text: lz(() => { const s = spec(); return { en: `Neither clue explains everything. Guessing each token’s expert from its position alone is right for ${pct(s.byPos)} of held-out tokens; guessing from its letter alone, ${pct(s.byLetter)}. The router uses both, and experts are rarely neat “topic experts”.`, zh: `兩種線索都解釋不了全部。只看位置來猜每個詞元的專家，在驗證集的詞元上猜對 ${pct(s.byPos)}；只看字母來猜，猜對 ${pct(s.byLetter)}。路由器兩者都用，專家也很少是整整齊齊的「主題專家」。` }; }) },
      changed: [{ sign: '+', text: { en: 'routing counted over 615 names', zh: '統計 615 個名字的路由結果' } }],
    },
    {
      short: { en: 'Try it', zh: '動手試試' }, chapter: 'solution',
      title: { en: 'Route your own name', zh: '替你自己的名字分派專家' },
      body: [
        { en: 'Try it: type a name (up to 7 letters) and watch the router’s real probabilities for every token, and the expert each one goes to.', zh: '動手試試：輸入一個名字（最多 7 個字母），看看路由器對每個詞元給出的真實機率，以及每個詞元被送去哪個專家。' },
        lz(() => ({ en: `The right column shows the expert most tokens at that position go to, over all held-out names. On average ${pct(spec().byPos)} of tokens follow it, about two in three. Notice where your name breaks from it, often a vowel late in the name pulled to E0.`, zh: `最右邊那一欄，是驗證集所有名字在那個位置上最常被送去的專家。平均有 ${pct(spec().byPos)} 的詞元跟它一樣，大約三個裡有兩個。看看你的名字在哪裡不一樣，常常是名字後段的母音被拉去 E0。` })),
      ],
      changed: [{ sign: '+', text: { en: 'any name, routed live', zh: '任何名字，即時分派' } }],
    },
    {
      short: { en: 'Result', zh: '結果' }, chapter: 'problem',
      title: { en: 'Case: is it better?', zh: '案例：真的比較好嗎？' },
      body: [
        { en: `In this set of comparison runs, both models were trained the same way for 1,500 steps. <b>Held-out loss</b> (the average surprise at the real next letter on the ${VAL_NAMES.length} held-out names; lower is better): normal model <b>${VAL.dense.toFixed(3)}</b>, 4 experts <b>${VAL.e4.toFixed(3)}</b>. A small gain, from one training run each, so it could partly be luck.`, zh: `在這組比較實驗裡，兩個模型用同樣的方式各訓練 1,500 步。<b>驗證損失</b>（在驗證集 ${VAL_NAMES.length} 個名字上，對真正下一個字母的平均「意外」程度，越低越好）：一般模型 <b>${VAL.dense.toFixed(3)}</b>，4 個專家 <b>${VAL.e4.toFixed(3)}</b>。進步很小，而且各只訓練一次，有一部分可能只是運氣。` },
        { en: 'That is honest for our size: names are a tiny task, and one 1,072-weight MLP already holds most of what there is to learn. MoE pays off when there is far more knowledge than one MLP can hold, as in large language models.', zh: '以我們的規模來說這很正常：名字是很小的任務，一個 1,072 個權重的 MLP 就已經裝得下大部分能學的東西。當要學的知識遠遠超過一個 MLP 能裝的量（例如大型語言模型），MoE 才真正划算。' },
      ],
      callout: { kind: 'note', text: { en: `These runs use their own recipe (warmup, then a cosine-shaped decay from a peak learning rate of 0.01), so the normal model scores ${VAL.dense.toFixed(3)} here, not the ${last(experiments.variants.base.val).toFixed(3)} of the constant-rate runs in the Multi-Head lesson. Compare only within one set of runs: that recipe change alone moved the normal model by more than MoE did.`, zh: `這組實驗有自己的訓練方式（先暖身，再從最高學習率 0.01 依餘弦曲線遞減），所以一般模型在這裡是 ${VAL.dense.toFixed(3)}，而不是「多頭注意力」那一課固定學習率實驗的 ${last(experiments.variants.base.val).toFixed(3)}。只能在同一組實驗裡互相比較：光是換訓練方式，一般模型的變化就比 MoE 帶來的還大。` } },
      changed: [{ sign: '+', text: { en: 'held-out loss, dense vs MoE', zh: '驗證損失：密集 vs MoE' } }],
    },
    {
      short: { en: 'Variants', zh: '變化' }, chapter: 'problem',
      title: { en: 'Case: two experts per token, or eight experts', zh: '案例：每個詞元用兩個專家，或用八個專家' },
      body: [
        { en: `With <b>top-2</b>, each token goes to its 2 best experts and their outputs are added: loss ${VAL.top2.toFixed(3)}. With <b>8 experts</b> (top-1): ${VAL.e8.toFixed(3)}, the best here, but the differences are all within about 0.02.`, zh: `用 <b>top-2</b> 時，每個詞元送去最好的 2 個專家，再把它們的輸出相加：損失 ${VAL.top2.toFixed(3)}。用 <b>8 個專家</b>（top-1）：${VAL.e8.toFixed(3)}，是這裡最好的，但所有差距都在 0.02 左右以內。` },
        { en: 'Large modern models use the same recipe at a vastly bigger size. Mixtral 8x7B, for example, has 8 experts per layer and sends each token to 2 of them.', zh: '現代的大型模型也用同一套做法，只是規模大得多。例如 Mixtral 8x7B 每層有 8 個專家，每個詞元送給其中 2 個。' },
      ],
      callout: { kind: 'note', text: { en: 'Mixtral 8x7B stores about 47 billion parameters but uses about 13 billion per token (Mistral AI’s paper, 2024).', zh: 'Mixtral 8x7B 存了大約 470 億個參數，但每個詞元只用到大約 130 億個（Mistral AI 的論文，2024）。' } },
      changed: [{ sign: '+', text: { en: 'top-2 and 8-expert runs', zh: 'top-2 和 8 個專家的訓練結果' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Mixture of experts in one line', zh: '一句話說完混合專家模型' },
      body: [
        { en: 'Replace the MLP with several experts. A router gives each token a probability per expert; the token goes to its top-k experts, and their outputs are added, each multiplied by its probability.', zh: '把 MLP 換成好幾個專家。路由器替每個詞元算出每個專家的機率；詞元送去前 k 名的專家，再把它們的輸出各乘上機率後相加。' },
        lz(() => ({ en: `Our model stores ${fmt(data().w.total)} weights but uses ${fmt(data().w.active)} per token. The experts learned to split the work by position and by letter: position alone predicts the expert for ${pct(spec().byPos)} of tokens.`, zh: `我們的模型存了 ${fmt(data().w.total)} 個權重，每個詞元卻只用 ${fmt(data().w.active)} 個。專家們學會的分工同時看位置和字母：光看位置，就能猜對 ${pct(spec().byPos)} 的詞元該去哪個專家。` })),
      ],
      callout: { kind: 'next', text: { en: 'Next: The Residual Stream. Why the attention and MLP (or expert) outputs are <i>added</i> onto each token’s vector.', zh: '下一課：殘差流。為什麼注意力和 MLP（或專家）的輸出，都是「加」到每個詞元的向量上。' } },
    },
  ],

  scene(step, input) {
    const d = data();
    const items: Item[] = [];
    const eTone = (e: number, on: boolean): Tone => (on ? 'a' : 'ink');

    // token row (steps 0, 1)
    if (step <= 1) {
      LABELS.forEach((ch, i) => items.push({
        k: 'tok' + i, t: 'token', text: ch, x: 248 + i * 58, y: 20,
        tone: step === 1 && i === F ? 'a' : ch === '.' ? 'b' : 'ink', dim: step === 1 && i !== F,
      }));
    }

    if (step === 0) {
      items.push({ k: 'mlp', t: 'box', x: 248, y: 130, w: 392, h: 70, label: t({ en: `one MLP: ${fmt(d.w.expert)} weights`, zh: `一個 MLP：${fmt(d.w.expert)} 個權重` }), sub: t({ en: 'every token uses all of them', zh: '每個詞元全部都要用' }), tone: 'a' });
      LABELS.forEach((_, i) => items.push({ k: 'ga' + i, t: 'arrow', x1: 270 + i * 58, y1: 68, x2: 270 + i * 58, y2: 128, tone: 'a' }));
      items.push({ k: 'big', t: 'box', x: 150, y: 290, w: 600, h: 70, label: t({ en: `4× bigger MLP: ${fmt(4 * d.w.expert)} weights`, zh: `放大 4 倍的 MLP：${fmt(4 * d.w.expert)} 個權重` }), sub: t({ en: '→ 4× the MLP work for every single token', zh: '→ 每個詞元的 MLP 工作量都變成 4 倍' }), tone: 'bad' });
      items.push({ k: 'bigq', t: 'text', x: 450, y: 260, text: t({ en: 'want more knowledge?', zh: '想要更多知識？' }), size: 15, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'n0', t: 'text', x: 450, y: 420, text: t({ en: `our model: ${fmt(DENSE_W)} weights, every token uses all ${fmt(DENSE_W)}`, zh: `我們的模型：${fmt(DENSE_W)} 個權重，每個詞元 ${fmt(DENSE_W)} 個全部都用` }), size: 15, anchor: 'middle', tone: 'ink' });
    }

    if (step === 1) {
      items.push({ k: 'router', t: 'box', x: 330, y: 120, w: 240, h: 56, label: t({ en: 'router', zh: '路由器' }), sub: t({ en: `${d.w.router} weights`, zh: `${d.w.router} 個權重` }), tone: 'b' });
      items.push({ k: 'ra', t: 'arrow', x1: 248 + F * 58 + 22, y1: 68, x2: 420, y2: 118, tone: 'a' });
      for (let e = 0; e < E; e++) {
        const on = e === d.chosen[F];
        items.push({ k: 'ex' + e, t: 'box', x: EXPERT_X(e), y: 280, w: 180, h: 70, label: t({ en: `expert E${e}`, zh: `專家 E${e}` }), sub: t({ en: `${fmt(d.w.expert)} weights`, zh: `${fmt(d.w.expert)} 個權重` }), tone: eTone(e, on), active: on });
        items.push({ k: 'rx' + e, t: 'arrow', x1: 450, y1: 178, x2: EXPERT_X(e) + 90, y2: 276, tone: on ? 'a' : 'muted', dashed: !on });
      }
      items.push({ k: 'n1', t: 'text', x: 450, y: 410, text: t({ en: `“y” is sent to one expert; the other three do nothing for it`, zh: `「y」只被送到一個專家，另外三個不替它做事` }), size: 15, anchor: 'middle' });
      items.push({ k: 'n2', t: 'text', x: 450, y: 440, text: t({ en: 'each expert = a full MLP, 16 → 32 → 16, with its own weights', zh: '每個專家 = 一個完整的 MLP，16 → 32 → 16，各有自己的權重' }), size: 14, anchor: 'middle', tone: 'muted' });
    }

    if (step === 2 || step === 3) {
      const cell = 52, hx = 330, hy = 110;
      items.push({ k: 'router', t: 'box', x: hx - 4, y: 22, w: E * cell + 4, h: 50, label: t({ en: 'router', zh: '路由器' }), sub: t({ en: `16 × 4 = ${d.w.router} weights`, zh: `16 × 4 = ${d.w.router} 個權重` }), tone: 'b' });
      LABELS.forEach((ch, i) => items.push({ k: 'tok' + i, t: 'token', text: ch, x: 250, y: hy + i * cell + 2, tone: i === F ? 'a' : ch === '.' ? 'b' : 'ink' }));
      items.push({
        k: 'heat', t: 'heat', x: hx, y: hy, cell, fmt: 'pct', max: 1, hi: step === 3 ? F : undefined,
        m: d.router, rows: LABELS.map(() => ''), cols: ['E0', 'E1', 'E2', 'E3'],
      });
      if (step === 2) {
        items.push({ k: 'h2a', t: 'text', x: 580, y: 200, text: t({ en: 'each row: one token', zh: '每一列：一個詞元' }), size: 15 });
        items.push({ k: 'h2b', t: 'text', x: 580, y: 228, text: t({ en: 'its 4 probabilities add up to 100%', zh: '4 個機率加起來是 100%' }), size: 14, tone: 'muted' });
        items.push({ k: 'h2c', t: 'text', x: 580, y: 300, text: t({ en: 'darker = the router prefers', zh: '顏色越深 = 路由器越偏好' }), size: 14, tone: 'muted' });
        items.push({ k: 'h2d', t: 'text', x: 580, y: 324, text: t({ en: 'that expert for this token', zh: '這個詞元交給那個專家' }), size: 14, tone: 'muted' });
      } else {
        items.push({ k: 'chh', t: 'text', x: 580, y: 96, text: t({ en: 'goes to', zh: '送往' }), size: 13, tone: 'muted' });
        LABELS.forEach((_, i) => {
          const e = d.chosen[i];
          items.push({ k: 'cha' + i, t: 'arrow', x1: 548, y1: hy + i * cell + 24, x2: 574, y2: hy + i * cell + 24, tone: i === F ? 'a' : 'muted' });
          items.push({ k: 'ch' + i, t: 'text', x: 580, y: hy + i * cell + 30, text: `E${e}  (${pct(d.router[i][e])})`, size: 16, mono: true, bold: i === F, tone: i === F ? 'a' : 'ink' });
        });
      }
    }

    if (step === 4) {
      const ce = d.chosen[F];
      items.push({ k: 'tok' + F, t: 'token', text: 'y', x: 428, y: 14, tone: 'a' });
      for (let e = 0; e < E; e++) {
        const on = e === ce;
        items.push({ k: 'ex' + e, t: 'box', x: EXPERT_X(e), y: 90, w: 180, h: 56, label: t({ en: `expert E${e}`, zh: `專家 E${e}` }), sub: `p = ${d.router[F][e].toFixed(2)}`, tone: eTone(e, on), active: on });
        items.push({ k: 'rx' + e, t: 'arrow', x1: 450, y1: 62, x2: EXPERT_X(e) + 90, y2: 86, tone: on ? 'a' : 'muted', dashed: !on });
        items.push({ k: 'ov' + e, t: 'vec', x: EXPERT_X(e) + 2, y: 186, v: d.outs[e], cell: 11, dim: !on, max: Math.max(...d.outs.flat().map(Math.abs)) });
        items.push({ k: 'ovl' + e, t: 'text', x: EXPERT_X(e) + 90, y: 236, text: on ? `× ${d.router[F][e].toFixed(2)}` : '× 0', size: 15, anchor: 'middle', mono: true, bold: on, tone: on ? 'a' : 'muted' });
      }
      items.push({ k: 'ovh', t: 'text', x: 30, y: 176, text: t({ en: 'each expert’s 16 output numbers for “y”', zh: '每個專家對「y」輸出的 16 個數字' }), size: 13, tone: 'muted' });
      items.push({ k: 'ga', t: 'arrow', x1: EXPERT_X(ce) + 90, y1: 248, x2: 450, y2: 316, tone: 'a' });
      items.push({ k: 'gated', t: 'vec', x: 362, y: 330, v: d.gated, cell: 11, max: Math.max(...d.outs.flat().map(Math.abs)) });
      items.push({ k: 'gl', t: 'text', x: 450, y: 384, text: t({ en: `layer output = ${d.router[F][ce].toFixed(2)} × E${ce}’s output`, zh: `這一層的輸出 = ${d.router[F][ce].toFixed(2)} × E${ce} 的輸出` }), size: 15, anchor: 'middle', tone: 'a', bold: true });
      items.push({ k: 'gl2', t: 'text', x: 450, y: 412, text: t({ en: '→ added onto the vector of “y”', zh: '→ 加回「y」的向量上' }), size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'gl3', t: 'text', x: 450, y: 470, text: t({ en: 'dimmed: computed by our code, then multiplied by 0 (a real system skips them)', zh: '變暗的部分：我們的程式算了但乘上 0（真實系統直接跳過）' }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 5) {
      const w = d.w, X = 70, S = 760 / w.total;
      const seg = (k: string, x: number, y: number, n: number, label: string, tone: Tone, dim = false) =>
        items.push({ k, t: 'box', x, y, w: Math.max(4, n * S - 3), h: 46, label, tone, active: !dim && tone === 'a' });
      items.push({ k: 'r1', t: 'text', x: X, y: 70, text: t({ en: `MoE model stores ${fmt(w.total)} weights`, zh: `MoE 模型存了 ${fmt(w.total)} 個權重` }), size: 15, bold: true });
      seg('s_sh', X, 84, w.shared, t({ en: 'shared', zh: '共用' }), 'ink');
      seg('s_ro', X + w.shared * S, 84, w.router, '', 'b');
      for (let e = 0; e < E; e++) seg('s_e' + e, X + (w.shared + w.router + e * w.expert) * S, 84, w.expert, `E${e}`, 'ink');
      items.push({ k: 'r2', t: 'text', x: X, y: 200, text: t({ en: `one token (“y”) uses ${fmt(w.active)}`, zh: `一個詞元（「y」）用到 ${fmt(w.active)} 個` }), size: 15, bold: true, tone: 'a' });
      seg('u_sh', X, 214, w.shared, t({ en: 'shared', zh: '共用' }), 'a');
      seg('u_ro', X + w.shared * S, 214, w.router, '', 'b');
      const ce = d.chosen[F];
      seg('u_e', X + (w.shared + w.router) * S, 214, w.expert, `E${ce}`, 'a');
      items.push({ k: 'r3', t: 'text', x: X, y: 330, text: t({ en: `normal model: stores and uses ${fmt(DENSE_W)}`, zh: `一般模型：存 ${fmt(DENSE_W)} 個，也用 ${fmt(DENSE_W)} 個` }), size: 15, bold: true, tone: 'muted' });
      seg('d_all', X, 344, DENSE_W, t({ en: 'shared + one MLP', zh: '共用 + 一個 MLP' }), 'ink');
      items.push({ k: 'rlab', t: 'text', x: X + w.shared * S + 4, y: 158, text: t({ en: `↑ router (${w.router})`, zh: `↑ 路由器（${w.router}）` }), size: 13, tone: 'b' });
      items.push({ k: 'note5', t: 'text', x: 450, y: 450, text: t({ en: `shared = embeddings ${fmt(w.emb)} + attention ${fmt(w.att)} + output ${fmt(w.out)} = ${fmt(w.shared)}`, zh: `共用 = 嵌入 ${fmt(w.emb)} + 注意力 ${fmt(w.att)} + 輸出 ${fmt(w.out)} = ${fmt(w.shared)}` }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 6) {
      const s = spec();
      const cell = 44;
      const ord = (p: number) => t({ en: ['1st “.”', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'][p], zh: ['第 1 個「.」', '第 2 個', '第 3 個', '第 4 個', '第 5 個', '第 6 個', '第 7 個', '第 8 個'][p] });
      items.push({ k: 'sph', t: 'text', x: 40, y: 34, text: t({ en: 'where each position’s tokens go', zh: '每個位置的詞元被送去哪裡' }), size: 15, bold: true, tone: 'a' });
      items.push({
        k: 'spec', t: 'heat', x: 140, y: 80, cell, fmt: 'pct', max: 1,
        m: d.posTab.map((r, p) => r.map((v) => v / d.posN[p])), rows: Array.from({ length: POS }, (_, p) => ord(p)), cols: ['E0', 'E1', 'E2', 'E3'],
      });
      items.push({ k: 'sbars', t: 'bars', x: 560, y: 100, w: 300, h: 230, title: t({ en: 'share sent to E0', zh: '送到 E0 的比例' }), items: [
        { label: t({ en: 'all tokens', zh: '所有詞元' }), v: d.use[0] },
        { label: '“e”', v: s.e, hi: true },
        { label: '“y”', v: s.y, hi: true },
      ] });
      items.push({ k: 'sn1', t: 'text', x: 710, y: 380, text: t({ en: `${fmt(VAL_NAMES.length)} held-out names, ${fmt(d.tokens)} tokens`, zh: `${fmt(VAL_NAMES.length)} 個驗證集名字，共 ${fmt(d.tokens)} 個詞元` }), size: 13, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'sn2', t: 'text', x: 710, y: 404, text: t({ en: 'each row of the grid adds up to 100%', zh: '左表每一列加起來是 100%' }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    let controls: Control[] | undefined;
    let tryName = '';
    if (step === TRY) {
      controls = tryControls;
      const name = String(input?.name ?? 'james').toLowerCase().replace(/[^a-z]/g, '').slice(0, 7);
      tryName = name;
      const chars = ['.', ...name];
      const cell = 46, hx = 330, hy = 90;
      items.push({ k: 'router', t: 'box', x: hx - 4, y: 14, w: E * cell + 4, h: 46, label: t({ en: 'router', zh: '路由器' }), sub: t({ en: 'real probabilities', zh: '真實機率' }), tone: 'b' });
      const r = route(name);
      const usual = d.posTab.map((row) => row.indexOf(Math.max(...row)));
      chars.forEach((ch, i) => items.push({ k: 'tok' + i, t: 'token', text: ch, x: 250, y: hy + i * cell + 1, tone: ch === '.' ? 'b' : 'ink' }));
      items.push({ k: 'heat', t: 'heat', x: hx, y: hy, cell, fmt: 'pct', max: 1, m: r.router, rows: chars.map(() => ''), cols: ['E0', 'E1', 'E2', 'E3'] });
      items.push({ k: 'chh', t: 'text', x: 560, y: hy - 10, text: t({ en: 'goes to', zh: '送往' }), size: 13, tone: 'muted' });
      items.push({ k: 'usu', t: 'text', x: 740, y: hy - 10, text: t({ en: 'usual at this position', zh: '這個位置通常送往' }), size: 13, tone: 'muted' });
      chars.forEach((_, i) => {
        const e = r.chosen[i], same = e === usual[i];
        items.push({ k: 'ch' + i, t: 'text', x: 560, y: hy + i * cell + 28, text: `E${e}  (${pct(r.router[i][e])})`, size: 16, mono: true, bold: !same, tone: same ? 'ink' : 'a' });
        items.push({ k: 'us' + i, t: 'text', x: 740, y: hy + i * cell + 28, text: `E${usual[i]}`, size: 16, mono: true, tone: 'muted' });
      });
      if (!name) items.push({ k: 'hint', t: 'text', x: 450, y: 300, anchor: 'middle', size: 16, tone: 'muted', text: t({ en: 'type a name (letters a–z) in the box above', zh: '請在上方的方框輸入一個名字（字母 a–z）' }) });
      const off = chars.filter((_, i) => r.chosen[i] !== usual[i]).length;
      items.push({ k: 'tsum', t: 'text', x: 450, y: 505, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: `${chars.length - off} of ${chars.length} tokens go to the usual expert for their position (teal: the others)`, zh: `${chars.length} 個詞元中有 ${chars.length - off} 個送往該位置通常的專家（青綠色是例外）` }) });
    }

    if (step === 8) {
      items.push({ k: 'chart', t: 'chart', x: 80, y: 60, w: 500, h: 360, ymin: 2.15, ymax: 2.6, xmax: EX.steps, every: EX.dense.every,
        series: [{ pts: EX.dense.val, tone: 'muted' }, { pts: EX.experts4.val, tone: 'a' }],
        xlabel: t({ en: 'training step', zh: '訓練步數' }), ylabel: t({ en: 'held-out loss (lower is better)', zh: '驗證損失（越低越好）' }) });
      items.push({ k: 'vd', t: 'text', x: 740, y: 150, text: VAL.dense.toFixed(3), size: 32, bold: true, mono: true, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'vdl', t: 'text', x: 740, y: 178, text: t({ en: `normal MLP · ${fmt(DENSE_W)} weights`, zh: `一般 MLP · ${fmt(DENSE_W)} 個權重` }), size: 13, anchor: 'middle', tone: 'muted' });
      items.push({ k: 've', t: 'text', x: 740, y: 260, text: VAL.e4.toFixed(3), size: 32, bold: true, mono: true, anchor: 'middle', tone: 'a' });
      items.push({ k: 'vel', t: 'text', x: 740, y: 288, text: t({ en: `4 experts · ${fmt(d.w.total)} weights`, zh: `4 個專家 · ${fmt(d.w.total)} 個權重` }), size: 13, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'vdiff', t: 'text', x: 740, y: 350, text: t({ en: `difference: ${(VAL.dense - VAL.e4).toFixed(3)}`, zh: `差距：${(VAL.dense - VAL.e4).toFixed(3)}` }), size: 15, anchor: 'middle', bold: true });
      items.push({ k: 'vnote', t: 'text', x: 740, y: 376, text: t({ en: 'small: a tiny task for a tiny model', zh: '很小：小模型做小任務' }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 9) {
      const w = d.w, r8 = 16 * 8;
      const runs = [
        { name: t({ en: 'normal MLP', zh: '一般 MLP' }), total: DENSE_W, used: DENSE_W, val: VAL.dense },
        { name: t({ en: '4 experts, top-1', zh: '4 個專家，top-1' }), total: EX.experts4.params, used: w.active, val: VAL.e4 },
        { name: t({ en: '4 experts, top-2', zh: '4 個專家，top-2' }), total: EX.top2.params, used: w.shared + w.router + 2 * w.expert, val: VAL.top2 },
        { name: t({ en: '8 experts, top-1', zh: '8 個專家，top-1' }), total: EX.experts8.params, used: w.shared + r8 + w.expert, val: VAL.e8 },
      ];
      const cx = [60, 400, 570, 760];
      [t({ en: 'model', zh: '模型' }), t({ en: 'stored', zh: '儲存' }), t({ en: 'used per token', zh: '每個詞元用到' }), t({ en: 'held-out loss', zh: '驗證損失' })]
        .forEach((h, j) => items.push({ k: 'th' + j, t: 'text', x: cx[j], y: 50, text: h, size: 14, tone: 'muted', anchor: j ? 'middle' : 'start' }));
      runs.forEach((r, i) => {
        const y = 92 + i * 52, hi = i >= 2;
        items.push({ k: 'tr' + i, t: 'box', x: 40, y: y - 30, w: 820, h: 44, tone: hi ? 'a' : 'ink', active: hi });
        items.push({ k: 'tn' + i, t: 'text', x: cx[0], y, text: r.name, size: 15, bold: hi });
        items.push({ k: 'tt' + i, t: 'text', x: cx[1], y, text: fmt(r.total), size: 15, mono: true, anchor: 'middle' });
        items.push({ k: 'tu' + i, t: 'text', x: cx[2], y, text: fmt(r.used), size: 15, mono: true, anchor: 'middle' });
        items.push({ k: 'tv' + i, t: 'text', x: cx[3], y, text: r.val.toFixed(3), size: 15, mono: true, anchor: 'middle', bold: true, tone: hi ? 'a' : 'ink' });
      });
      items.push({ k: 'tnote', t: 'text', x: 450, y: 310, text: t({ en: 'all trained 1,500 steps the same way · one run each', zh: '全部用同樣方式訓練 1,500 步 · 各只跑一次' }), size: 13, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'mix', t: 'box', x: 120, y: 360, w: 660, h: 84, tone: 'b', label: t({ en: 'Mixtral 8x7B: 8 experts per layer, top-2', zh: 'Mixtral 8x7B：每層 8 個專家，top-2' }), sub: t({ en: '≈ 47 billion parameters stored, ≈ 13 billion used per token', zh: '儲存約 470 億個參數，每個詞元約用 130 億個' }) });
    }

    if (step === 10) {
      const boxes = [
        { en: 'token vector', zh: '詞元向量' }, { en: 'router → probabilities', zh: '路由器 → 機率' }, { en: 'top-k experts', zh: '前 k 名專家' },
        { en: '× probability, add', zh: '× 機率，相加' }, { en: 'add onto vector', zh: '加回向量' },
      ];
      boxes.forEach((b, i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 190, w: 150, h: 60, label: t(b), tone: i === 1 || i === 2 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 220, x2: bx + 170, y2: 220 });
      });
      items.push({ k: 'rt', t: 'text', x: 450, y: 310, text: t({ en: `stores ${fmt(d.w.total)} weights · uses ${fmt(d.w.active)} per token`, zh: `存 ${fmt(d.w.total)} 個權重 · 每個詞元用 ${fmt(d.w.active)} 個` }), size: 15, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'rt2', t: 'text', x: 450, y: 340, text: t({ en: `held-out loss ${VAL.e4.toFixed(3)} vs ${VAL.dense.toFixed(3)} (normal MLP)`, zh: `驗證損失 ${VAL.e4.toFixed(3)} vs ${VAL.dense.toFixed(3)}（一般 MLP）` }), size: 15, anchor: 'middle', tone: 'muted' });
    }

    const moeOn = step >= 1;
    return {
      metrics: [
        ...(step === TRY ? [{ label: { en: 'Name', zh: '名字' }, value: '.' + tryName }] : []),
        { label: { en: 'MLPs per block', zh: '每區塊的 MLP' }, value: step === 9 ? '1 / 4 / 8' : moeOn ? t({ en: '4 experts', zh: '4 個專家' }) : '1' },
        { label: { en: 'Weights stored', zh: '儲存的權重' }, value: fmt(moeOn ? d.w.total : DENSE_W) },
        { label: { en: 'Used per token', zh: '每個詞元用到' }, value: fmt(moeOn ? d.w.active : DENSE_W) },
        ...(step === TRY ? [] : [{ label: { en: '“y” goes to', zh: '「y」送往' }, value: step >= 3 ? `E${d.chosen[F]} (${pct(d.router[F][d.chosen[F]])})` : '—' },
        { label: { en: 'Held-out loss', zh: '驗證損失' }, value: step >= 8 ? `${VAL.e4.toFixed(3)} vs ${VAL.dense.toFixed(3)}` : '—' }]),
      ],
      items,
      controls,
      legend: step === TRY ? [{ tone: 'heat', text: { en: 'router probability', zh: '路由器機率' } }, { tone: 'a', text: { en: 'not the usual expert for its position', zh: '不是該位置通常的專家' } }]
        : step === 4 ? [{ tone: 'pos', text: { en: 'positive number', zh: '正數' } }, { tone: 'neg', text: { en: 'negative number', zh: '負數' } }, { tone: 'a', text: { en: 'chosen expert', zh: '被選中的專家' } }]
        : step >= 2 && step <= 6 ? [{ tone: 'heat', text: { en: 'share / probability', zh: '比例／機率' } }, { tone: 'a', text: { en: 'the token we follow / used', zh: '追蹤的詞元／用到的部分' } }] : undefined,
    };
  },
};
