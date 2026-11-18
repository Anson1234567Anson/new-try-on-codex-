import type { Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t, type L } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, VAL_NAMES, example, rng } from '../model/data';
import { Mat, addRow, matmul, softmaxRow } from '../model/tensor';
import wBal from '../model/weights-moe.json';
import wNob from '../model/weights-moe-nobalance.json';
import experiments from '../model/experiments.json';

// Two shipped 4-expert models, trained the same way except for the load-balancing
// loss (weight 0.01 vs 0). We route one real batch of 32 seeded held-out names
// through both, and measure balance, the balancing loss, and capacity overflow.

const EX = experiments.moe;
const E = 4;
const NB = 32;
const CFS = [1, 1.25, 2];
const ALPHA = 0.01;

const lz = (f: () => L): L => ({ get en() { return f().en; }, get zh() { return f().zh; } });
const fmt = (n: number) => n.toLocaleString('en-US');
const pct = (v: number) => Math.round(v * 100) + '%';
const pct1 = (v: number) => (v * 100).toFixed(1) + '%';
const last = <T>(v: T[]) => v[v.length - 1];

interface Run {
  counts: number[]; f: number[]; P: number[]; balance: number;
  cap: { cf: number; cap: number; dropped: number; loss: number }[]; lossAll: number;
  route: { ch: string; e: number }[][]; // the chosen expert of every token, per name
}
interface Data { names: string[]; tokens: number; bal: Run; nob: Run }

function measure(w: Weights, batch: { x: number[]; y: number[] }[]): Run {
  const m = GPT.fromJSON(w);
  const P = m.p;
  const counts = new Array(E).fill(0), psum = new Array(E).fill(0);
  // every token in batch order: its expert and its loss with and without the expert's output
  const toks: { e: number; lossIn: number; lossOut: number }[] = [];
  const route: { ch: string; e: number }[][] = [];
  const ce = (logits: number[], y: number) => -Math.log(softmaxRow(logits)[y]);
  for (const ex of batch) {
    const tr = m.forward(ex.x);
    const r: { ch: string; e: number }[] = [];
    ex.x.forEach((id, i) => {
      const e = tr.chosen![i][0];
      counts[e]++;
      for (let j = 0; j < E; j++) psum[j] += tr.router!.d[i * E + j];
      const x1 = new Mat(1, 16); x1.d.set(tr.x1.row(i));
      const skip = addRow(matmul(x1, P.u), P.bu).row(0); // the token skips the expert: only the residual
      toks.push({ e, lossIn: ce(tr.logits.row(i), ex.y[i]), lossOut: ce(skip, ex.y[i]) });
      r.push({ ch: CHARS[id], e });
    });
    route.push(r);
  }
  const T = toks.length;
  const f = counts.map((c) => c / T), Pm = psum.map((p) => p / T);
  const cap = CFS.map((cf) => {
    const c = Math.floor((cf * T) / E);
    const used = new Array(E).fill(0);
    let dropped = 0, loss = 0;
    for (const tk of toks) {
      if (used[tk.e] < c) { used[tk.e]++; loss += tk.lossIn; } else { dropped++; loss += tk.lossOut; }
    }
    return { cf, cap: c, dropped, loss: loss / T };
  });
  return {
    counts, f, P: Pm, balance: E * f.reduce((s, v, e) => s + v * Pm[e], 0),
    cap, lossAll: toks.reduce((s, tk) => s + tk.lossIn, 0) / T, route,
  };
}

let cache: Data | undefined;
const data = (): Data => cache ??= (() => {
  const r = rng(7);
  const names = Array.from({ length: NB }, () => VAL_NAMES[Math.floor(r() * VAL_NAMES.length)]);
  const batch = names.map(example);
  return {
    names, tokens: batch.reduce((s, b) => s + b.x.length, 0),
    bal: measure(wBal as unknown as Weights, batch), nob: measure(wNob as unknown as Weights, batch),
  };
})();

const busiest = (u: number[][]) => u.map((r) => Math.max(...r));
const U4 = EX.experts4.extra as number[][];
const UN = EX.noBalance.extra as number[][];
const VB = last(EX.experts4.val), VN = last(EX.noBalance.val);
const even = () => data().tokens / E;

export const moeTraining: Lesson = {
  id: 'moe-training',
  title: { en: 'Training Mixture-of-Experts Models', zh: '訓練混合專家模型' },
  blurb: {
    en: 'Left alone, a router piles tokens onto a few experts. A balancing loss, a capacity limit and clever placement keep every expert busy.',
    zh: '放著不管，路由器會把詞元都丟給少數幾個專家。平衡損失、容量上限和巧妙的擺放方式，讓每個專家都有事做。',
  },
  minutes: 12,
  level: 'intermediate',
  learn: [
    { text: { en: 'Why routers become unbalanced', zh: '為什麼路由器會分配不均' }, step: 1 },
    { text: { en: 'How the load-balancing loss measures imbalance', zh: '負載平衡損失如何衡量分配不均' }, step: 2 },
    { text: { en: 'Why imbalance costs speed more than quality', zh: '為什麼分配不均傷的是速度，而不是品質' }, step: 5 },
    { text: { en: 'What expert capacity is, and which tokens get dropped', zh: '什麼是專家容量，哪些詞元會被丟掉' }, step: 6 },
    { text: { en: 'How experts are spread over many GPUs', zh: '專家如何分散到許多 GPU 上' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'One expert gets most of the work', zh: '一個專家包辦了大部分工作' },
      body: [
        { en: 'We trained a 4-expert model with nothing to encourage sharing. On 200 held-out names, the router ended up sending each expert this share of the tokens.', zh: '我們訓練了一個 4 專家模型，完全沒有任何鼓勵「分工」的機制。在 200 個保留名字上，路由器最後送給每個專家的詞元比例如下。' },
        { en: `Expert E2 gets <b>${last(UN)[2]}%</b> and E1 only ${last(UN)[1]}%. If the work were split evenly, each would get 25%.`, zh: `專家 E2 拿到 <b>${last(UN)[2]}%</b>，E1 只有 ${last(UN)[1]}%。如果平均分配，每個專家應該各拿 25%。` },
      ],
      callout: { kind: 'insight', text: { en: 'An expert that gets few tokens also gets little training, so it stays weak, and the router has even less reason to pick it.', zh: '拿到很少詞元的專家，得到的訓練也很少，所以一直很弱，路由器就更沒有理由選它。' } },
    },
    {
      short: { en: 'Over time', zh: '隨時間' }, chapter: 'technique',
      title: { en: 'How the imbalance grows', zh: '分配不均是怎麼變嚴重的' },
      body: [
        { en: `The chart follows the busiest expert’s share during training. Without balancing (red) it climbs and stays high, ending at ${Math.max(...last(UN))}%. With a <b>load-balancing loss</b> (teal) it ends at ${Math.max(...last(U4))}%.`, zh: `圖上追蹤的是訓練過程中「最忙的專家」拿到的比例。沒有平衡機制時（紅線）它一路升高、居高不下，最後是 ${Math.max(...last(UN))}%。加上<b>負載平衡損失（load-balancing loss）</b>後（青綠線），最後是 ${Math.max(...last(U4))}%。` },
        { en: 'This runaway is called <b>router collapse</b> when it goes all the way: nearly every token goes to one expert, and the others are wasted weights.', zh: '如果一路惡化到底，就叫做<b>路由崩潰（router collapse）</b>：幾乎所有詞元都送給同一個專家，其他專家就成了白白存著的權重。' },
      ],
      changed: [{ sign: '+', text: { en: 'busiest expert over 1,500 steps', zh: '1,500 步中最忙的專家' } }],
    },
    {
      short: { en: 'Balance loss', zh: '平衡損失' }, chapter: 'technique',
      title: { en: 'A loss that measures imbalance', zh: '一個衡量分配不均的損失' },
      body: [
        { en: 'For each expert take two numbers on a batch: <b>f</b>, the share of tokens actually sent to it, and <b>P</b>, the router’s average probability for it. The loss is 4 × the sum of f × P over the 4 experts.', zh: '對每個專家，在一個批次上取兩個數字：<b>f</b> 是實際送給它的詞元比例，<b>P</b> 是路由器給它的平均機率。損失 = 4 × 把 4 個專家的 f × P 加起來。' },
        lz(() => ({ en: `If everything is even (f = P = 25%), it equals exactly 1. On our real batch of ${NB} held-out names (${data().tokens} tokens), the balanced model scores <b>${data().bal.balance.toFixed(3)}</b>.`, zh: `如果完全平均（f = P = 25%），結果剛好是 1。在我們真實的 ${NB} 個保留名字批次上（${data().tokens} 個詞元），有平衡的模型得到 <b>${data().bal.balance.toFixed(3)}</b>。` })),
      ],
      callout: { kind: 'note', text: { en: 'f alone cannot be trained (picking the top expert has no gradient), but P can. Pairing them pushes probability away from overloaded experts.', zh: '單靠 f 沒辦法訓練（「挑最高分」這個動作沒有梯度），但 P 可以。把兩者配在一起，就能把機率從太忙的專家身上推開。' } },
      changed: [{ sign: '+', text: { en: 'f and P on a real batch', zh: '真實批次上的 f 和 P' } }],
    },
    {
      short: { en: 'Both models', zh: '兩個模型' }, chapter: 'technique',
      title: { en: 'The same batch, without balancing', zh: '同一個批次，換成沒有平衡的模型' },
      body: [
        lz(() => { const n = data().nob; return { en: `The unbalanced model sends ${pct(n.f[2])} of this batch to E2, and gives E2 the highest probability too, so f × P is large there: the loss is <b>${n.balance.toFixed(3)}</b>.`, zh: `沒有平衡的模型把這個批次的 ${pct(n.f[2])} 送給 E2，給 E2 的機率也最高，所以那一格 f × P 很大：損失是 <b>${n.balance.toFixed(3)}</b>。` }; }),
        { en: 'During training this loss is multiplied by a small weight and added to the prediction loss. Our balanced model used <b>0.01</b>, the value chosen in the Switch Transformer paper.', zh: '訓練時，這個損失會乘上一個小權重，再加到預測損失上。我們有平衡的模型用的是 <b>0.01</b>，也就是 Switch Transformer 論文所選的值。' },
      ],
      callout: { kind: 'note', text: { en: 'Our training code computes this for each name and averages; here we pool the whole batch, which reads more easily.', zh: '我們的訓練程式是對每個名字各算一次再平均；這裡把整個批次合起來算，比較好讀。' } },
      changed: [{ sign: '+', text: { en: 'unbalanced model, same batch', zh: '沒有平衡的模型，同一個批次' } }],
    },
    {
      short: { en: 'Quality', zh: '品質' }, chapter: 'problem',
      title: { en: 'Case: does imbalance hurt quality?', zh: '案例：分配不均會傷害品質嗎？' },
      body: [
        { en: `Not at our size. Held-out loss after 1,500 steps: balanced <b>${VB.toFixed(3)}</b>, unbalanced <b>${VN.toFixed(3)}</b>. Practically the same.`, zh: `在我們的規模下不會。訓練 1,500 步後的保留資料損失：有平衡 <b>${VB.toFixed(3)}</b>，沒有平衡 <b>${VN.toFixed(3)}</b>，幾乎一樣。` },
        { en: 'Our task is so small that even a lopsided split has enough expert capacity. The real cost of imbalance shows up elsewhere: on the hardware.', zh: '我們的任務太小了，就算分配很偏，專家的能力還是夠用。分配不均真正的代價出現在別的地方：硬體上。' },
      ],
      changed: [{ sign: '+', text: { en: 'held-out loss, both models', zh: '兩個模型的保留資料損失' } }],
    },
    {
      short: { en: 'Hardware', zh: '硬體' }, chapter: 'technique',
      title: { en: 'The busiest expert sets the speed', zh: '最忙的專家決定了速度' },
      body: [
        { en: 'Big MoE models put different experts on different chips (GPUs). All chips work at the same time, and the next step waits for the slowest one: the chip whose expert got the most tokens.', zh: '大型 MoE 模型會把不同的專家放在不同的晶片（GPU）上。所有晶片同時工作，下一步要等最慢的那一個做完，也就是分到最多詞元的那個專家所在的晶片。' },
        lz(() => { const d = data(); const b = Math.max(...d.bal.counts), n = Math.max(...d.nob.counts); return { en: `On our batch, an even split is ${even()} tokens per expert. The busiest expert gets ${b} (balanced) or ${n} (unbalanced): if time grows with tokens, that is <b>${(b / even()).toFixed(2)}×</b> vs <b>${(n / even()).toFixed(2)}×</b> the even time.`, zh: `在我們的批次上，平均分配是每個專家 ${even()} 個詞元。最忙的專家分到 ${b} 個（有平衡）或 ${n} 個（沒平衡）：如果時間跟詞元數成正比，就是平均情況的 <b>${(b / even()).toFixed(2)} 倍</b>對 <b>${(n / even()).toFixed(2)} 倍</b>。` }; }),
      ],
      callout: { kind: 'insight', text: { en: 'With imbalance, the chips holding quiet experts sit idle while they wait. Balance is about keeping every chip busy.', zh: '分配不均時，放著冷門專家的晶片只能閒著等。平衡的目的，就是讓每顆晶片都有事做。' } },
      changed: [{ sign: '+', text: { en: 'tokens per expert, real batch', zh: '真實批次中每個專家的詞元數' } }],
    },
    {
      short: { en: 'Capacity', zh: '容量' }, chapter: 'technique',
      title: { en: 'Capacity: a limit per expert', zh: '容量：每個專家的上限' },
      body: [
        lz(() => { const c = data().bal.cap[0]; return { en: `To keep each step’s time predictable, each expert gets a fixed <b>capacity</b>: capacity factor × (tokens ÷ experts). With factor 1.0: 1.0 × ${data().tokens} ÷ 4 → <b>${c.cap}</b> tokens.`, zh: `為了讓每一步的時間可以預估，每個專家都有固定的<b>容量（capacity）</b>：容量係數 × （詞元數 ÷ 專家數）。係數 1.0 時：1.0 × ${data().tokens} ÷ 4 → <b>${c.cap}</b> 個詞元。` }; }),
        lz(() => { const d = data(); return { en: `Tokens beyond the limit <b>overflow</b>: they are dropped from the expert and only pass along the residual connection, unchanged. Here ${d.bal.cap[0].dropped} tokens overflow in the balanced model, ${d.nob.cap[0].dropped} in the unbalanced one (red).`, zh: `超過上限的詞元會<b>溢出（overflow）</b>：它們不進專家，只沿著殘差連線原封不動往下傳。這裡有平衡的模型溢出 ${d.bal.cap[0].dropped} 個詞元，沒平衡的模型溢出 ${d.nob.cap[0].dropped} 個（紅色）。` }; }),
      ],
      changed: [{ sign: '+', text: { en: 'capacity line (factor 1.0)', zh: '容量線（係數 1.0）' } }, { sign: '+', text: { en: 'overflow in red', zh: '紅色是溢出' } }],
    },
    {
      short: { en: 'Factor', zh: '係數' }, chapter: 'solution',
      title: { en: 'Choosing the capacity factor', zh: '選擇容量係數' },
      body: [
        lz(() => { const d = data(); return { en: `A bigger factor drops fewer tokens but reserves more room (and memory) that may go unused. At 1.25 the balanced model drops ${d.bal.cap[1].dropped}; the unbalanced one still drops ${d.nob.cap[1].dropped}, and ${d.nob.cap[2].dropped} even at 2.0.`, zh: `係數越大，丟掉的詞元越少，但要預留更多可能用不到的空間（和記憶體）。係數 1.25 時，有平衡的模型丟掉 ${d.bal.cap[1].dropped} 個；沒平衡的模型還是丟掉 ${d.nob.cap[1].dropped} 個，連 2.0 時都還丟掉 ${d.nob.cap[2].dropped} 個。` }; }),
        { en: 'Dropped tokens miss the expert’s help, so the loss on this batch rises (right columns). Balancing and capacity work together: a balanced router lets a small factor drop almost nothing.', zh: '被丟掉的詞元得不到專家的幫忙，所以這個批次的損失會上升（右邊幾欄）。平衡和容量是一起發揮作用的：路由器平衡了，小小的係數就幾乎不會丟掉詞元。' },
      ],
      callout: { kind: 'note', text: { en: 'Our models were trained without a capacity limit; here we apply one afterwards to the same batch, to see its effect.', zh: '我們的模型訓練時沒有容量上限；這裡是事後在同一個批次上套用上限，看看它的影響。' } },
      changed: [{ sign: '+', text: { en: 'dropped tokens at 1.0, 1.25, 2.0', zh: '係數 1.0、1.25、2.0 時丟掉的詞元' } }],
    },
    {
      short: { en: 'Parallel', zh: '平行' }, chapter: 'technique',
      title: { en: 'Expert parallelism: send tokens to their expert', zh: '專家平行：把詞元送到它的專家那裡' },
      body: [
        { en: 'In <b>expert parallelism</b>, each GPU holds one expert (or a few) and a slice of the batch. After the router decides, every GPU sends each token to the GPU of its expert, and the results are sent back.', zh: '在<b>專家平行（expert parallelism）</b>中，每個 GPU 放一個（或幾個）專家，以及批次的一部分。路由器決定之後，每個 GPU 把每個詞元送到它的專家所在的 GPU，算完再送回來。' },
        { en: 'Because every GPU may send to every other, this exchange is called <b>all-to-all</b>. It happens twice per MoE layer, and on big systems it takes a real share of the time. The routes shown are real; the GPUs are an illustration.', zh: '因為每個 GPU 都可能傳給其他每個 GPU，這種交換叫做 <b>all-to-all（全對全）</b>。每個 MoE 層要做兩次，在大型系統上會佔掉相當多的時間。圖中的路由是真實的，GPU 則是示意。' },
      ],
      callout: { kind: 'note', text: { en: 'Google’s GShard (2020) trained a 600-billion-parameter translation model this way, with experts spread over 2,048 TPU cores and top-2 routing.', zh: 'Google 的 GShard（2020）就是用這種方式，把專家分散到 2,048 個 TPU 核心上，以 top-2 路由訓練出一個 6,000 億參數的翻譯模型。' } },
      changed: [{ sign: '+', text: { en: '4 GPUs, one expert each (illustration)', zh: '4 個 GPU，各放一個專家（示意）' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Training MoE in one line', zh: '一句話說完 MoE 的訓練' },
      body: [
        { en: 'Routers drift toward a few experts. A load-balancing loss (4 × Σ f × P, weight 0.01) pushes back. Each expert has a capacity; overflow tokens skip it and ride the residual.', zh: '路由器會往少數專家靠攏，負載平衡損失（4 × Σ f × P，權重 0.01）把它拉回來。每個專家都有容量上限，溢出的詞元跳過專家，沿著殘差往下傳。' },
        { en: 'At our size balance barely changes quality. Its real value is speed: experts live on different GPUs, tokens travel all-to-all, and the busiest expert sets the pace.', zh: '在我們的規模下，平衡幾乎不影響品質。它真正的價值在速度：專家放在不同的 GPU 上，詞元透過 all-to-all 傳送，而最忙的專家決定了整體的步調。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Build a Tiny GPT. Put every piece together into one small model you can follow from letters to prediction.', zh: '下一課：打造一個迷你 GPT。把所有零件組合成一個小模型，從字母一路追到預測。' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const lbl = { bal: t({ en: 'balanced (0.01)', zh: '有平衡（0.01）' }), nob: t({ en: 'no balancing', zh: '沒有平衡' }) };
    const usageBars = (k: string, x: number, u: number[], title: string) => items.push({
      k, t: 'bars', x, y: 90, w: 340, h: 260, title,
      items: u.map((v, e) => ({ label: `E${e}`, v: v / 100, hi: v === Math.max(...u) })),
    });

    if (step === 0) {
      usageBars('ubn', 280, last(UN), t({ en: `no balancing · share of tokens per expert`, zh: '沒有平衡 · 每個專家拿到的詞元比例' }));
      items.push({ k: 'ev', t: 'text', x: 450, y: 400, text: t({ en: 'even split would be 25% each', zh: '平均分配的話每個 25%' }), size: 15, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'evn', t: 'text', x: 450, y: 428, text: t({ en: `measured on ${EX.usageNames} held-out names after ${fmt(EX.steps)} steps`, zh: `訓練 ${fmt(EX.steps)} 步後，用 ${EX.usageNames} 個保留名字測量` }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 1) {
      items.push({ k: 'uc', t: 'chart', x: 80, y: 70, w: 480, h: 340, ymin: 0, ymax: 100, xmax: EX.steps, every: EX.usageEvery,
        series: [{ pts: busiest(UN), tone: 'bad' }, { pts: busiest(U4), tone: 'a' }],
        xlabel: t({ en: 'training step', zh: '訓練步數' }), ylabel: t({ en: 'busiest expert’s share of tokens (%)', zh: '最忙的專家拿到的詞元比例（%）' }),
        guide: { y: 25, label: t({ en: 'even: 25%', zh: '平均：25%' }) } });
      items.push({ k: 'unv', t: 'text', x: 740, y: 160, text: Math.max(...last(UN)) + '%', size: 32, bold: true, mono: true, anchor: 'middle', tone: 'bad' });
      items.push({ k: 'unl', t: 'text', x: 740, y: 188, text: lbl.nob, size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'u4v', t: 'text', x: 740, y: 270, text: Math.max(...last(U4)) + '%', size: 32, bold: true, mono: true, anchor: 'middle', tone: 'a' });
      items.push({ k: 'u4l', t: 'text', x: 740, y: 298, text: lbl.bal, size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'uh', t: 'text', x: 740, y: 350, text: t({ en: 'busiest expert at the end', zh: '最後最忙的專家' }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 2 || step === 3) {
      const d = data();
      const panel = (k: string, x: number, r: Run, title: string, tone: Tone) => {
        items.push({ k: k + 'h', t: 'text', x: x - 40, y: 50, text: title, size: 15, bold: true, tone });
        items.push({ k: k + 'm', t: 'heat', x, y: 100, cell: 58, fmt: 'pct', max: 0.65, m: r.f.map((f, e) => [f, r.P[e]]), rows: ['E0', 'E1', 'E2', 'E3'], cols: ['f', 'P'] });
        items.push({ k: k + 'ph', t: 'text', x: x + 150, y: 90, text: 'f × P', size: 13, tone: 'muted', anchor: 'middle' });
        r.f.forEach((f, e) => items.push({ k: k + 'p' + e, t: 'text', x: x + 150, y: 100 + e * 58 + 33, text: (f * r.P[e]).toFixed(3), size: 15, mono: true, anchor: 'middle' }));
        items.push({ k: k + 'sum', t: 'text', x: x + 60, y: 370, text: `4 × ${r.f.reduce((s, f, e) => s + f * r.P[e], 0).toFixed(3)} = ${r.balance.toFixed(3)}`, size: 20, bold: true, mono: true, anchor: 'middle', tone });
        items.push({ k: k + 'sl', t: 'text', x: x + 60, y: 398, text: t({ en: 'load-balancing loss', zh: '負載平衡損失' }), size: 13, anchor: 'middle', tone: 'muted' });
      };
      panel('pb', step === 2 ? 360 : 140, d.bal, lbl.bal, 'a');
      if (step === 3) panel('pn', 580, d.nob, lbl.nob, 'bad');
      items.push({ k: 'bnote', t: 'text', x: 450, y: 450, text: t({ en: `${NB} held-out names, ${d.tokens} tokens · perfectly even = 1.000 · all on one expert = up to 4`, zh: `${NB} 個保留名字，${d.tokens} 個詞元 · 完全平均 = 1.000 · 全給同一個專家最多到 4` }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 4) {
      items.push({ k: 'vc', t: 'chart', x: 80, y: 60, w: 500, h: 360, ymin: 2.15, ymax: 2.6, xmax: EX.steps, every: EX.experts4.every,
        series: [{ pts: EX.noBalance.val, tone: 'bad' }, { pts: EX.experts4.val, tone: 'a' }],
        xlabel: t({ en: 'training step', zh: '訓練步數' }), ylabel: t({ en: 'held-out loss (lower is better)', zh: '保留資料損失（越低越好）' }) });
      items.push({ k: 'unv', t: 'text', x: 740, y: 160, text: VN.toFixed(3), size: 32, bold: true, mono: true, anchor: 'middle', tone: 'bad' });
      items.push({ k: 'unl', t: 'text', x: 740, y: 188, text: lbl.nob, size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'u4v', t: 'text', x: 740, y: 270, text: VB.toFixed(3), size: 32, bold: true, mono: true, anchor: 'middle', tone: 'a' });
      items.push({ k: 'u4l', t: 'text', x: 740, y: 298, text: lbl.bal, size: 14, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'uh', t: 'text', x: 740, y: 350, text: t({ en: 'the two lines almost overlap', zh: '兩條線幾乎重疊' }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 5 || step === 6) {
      const d = data();
      const base = 420, S = 270 / Math.max(...d.nob.counts, 1);
      const groups: [string, Run, string, number][] = [['b', d.bal, lbl.bal, 60], ['n', d.nob, lbl.nob, 490]];
      for (const [g, r, title, gx] of groups) {
        items.push({ k: 'gt' + g, t: 'text', x: gx + 175, y: 40, text: title, size: 15, bold: true, anchor: 'middle', tone: g === 'b' ? 'a' : 'bad' });
        const cap = r.cap[0].cap;
        r.counts.forEach((c, e) => {
          const x = gx + e * 90, keep = step === 6 ? Math.min(c, cap) : c;
          items.push({ k: `g${g}k${e}`, t: 'box', x, y: base - keep * S, w: 70, h: Math.max(4, keep * S), tone: c === Math.max(...r.counts) ? (g === 'b' ? 'a' : 'bad') : 'ink' });
          if (step === 6 && c > cap) items.push({ k: `g${g}o${e}`, t: 'box', x, y: base - c * S, w: 70, h: (c - cap) * S - 2, tone: 'bad', active: true });
          const line = step === 6 ? cap : even();
          const near = c < line && (line - c) * S < 18;
          items.push({ k: `g${g}c${e}`, t: 'text', x: x + 35, y: base - c * S + (near ? 20 : -8), text: step === 6 && c > cap ? `${c} (+${c - cap})` : String(c), size: 14, mono: true, anchor: 'middle', bold: true });
          items.push({ k: `g${g}l${e}`, t: 'text', x: x + 35, y: base + 22, text: step === 5 ? `GPU ${e} · E${e}` : `E${e}`, size: 13, anchor: 'middle', tone: 'muted' });
        });
        const ly = base - (step === 6 ? cap : even()) * S;
        items.push({ k: 'gline' + g, t: 'box', x: gx - 10, y: ly - 1, w: 360, h: 3, tone: step === 6 ? 'bad' : 'muted' });
        items.push({ k: 'gll' + g, t: 'text', x: gx - 8, y: ly - 8, text: step === 6 ? t({ en: `capacity ${cap}`, zh: `容量 ${cap}` }) : t({ en: `even: ${even()}`, zh: `平均：${even()}` }), size: 13, anchor: 'start', tone: step === 6 ? 'bad' : 'muted' });
      }
      items.push({ k: 'gnote', t: 'text', x: 450, y: 490, text: step === 5
        ? t({ en: `tokens per expert in our batch of ${NB} names · GPUs: illustration`, zh: `我們 ${NB} 個名字的批次中，每個專家的詞元數 · GPU：示意` })
        : t({ en: `capacity = 1.0 × ${d.tokens} ÷ 4, rounded down · red = overflow, dropped`, zh: `容量 = 1.0 × ${d.tokens} ÷ 4（無條件捨去）· 紅色 = 溢出、被丟掉` }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 7) {
      const d = data();
      const cx = [70, 270, 400, 600, 730];
      const head = [t({ en: 'capacity factor', zh: '容量係數' }), t({ en: 'dropped', zh: '丟掉' }), t({ en: 'loss', zh: '損失' }), t({ en: 'dropped', zh: '丟掉' }), t({ en: 'loss', zh: '損失' })];
      items.push({ k: 'thb', t: 'text', x: 335, y: 40, text: lbl.bal, size: 15, bold: true, anchor: 'middle', tone: 'a' });
      items.push({ k: 'thn', t: 'text', x: 665, y: 40, text: lbl.nob, size: 15, bold: true, anchor: 'middle', tone: 'bad' });
      head.forEach((h, j) => items.push({ k: 'th' + j, t: 'text', x: cx[j], y: 72, text: h, size: 13, tone: 'muted', anchor: j ? 'middle' : 'start' }));
      const rows = [...CFS.map((_, i) => ({
        name: `${CFS[i].toFixed(2)} (${t({ en: 'cap', zh: '上限' })} ${d.bal.cap[i].cap})`,
        bd: d.bal.cap[i].dropped, bl: d.bal.cap[i].loss, nd: d.nob.cap[i].dropped, nl: d.nob.cap[i].loss,
      })), { name: t({ en: 'no limit', zh: '不設上限' }), bd: 0, bl: d.bal.lossAll, nd: 0, nl: d.nob.lossAll }];
      rows.forEach((r, i) => {
        const y = 118 + i * 58;
        items.push({ k: 'tr' + i, t: 'box', x: 40, y: y - 32, w: 820, h: 48, tone: 'ink' });
        items.push({ k: 'tn' + i, t: 'text', x: cx[0], y, text: r.name, size: 15, mono: true });
        const dropTxt = (n: number) => `${n} (${pct1(n / d.tokens)})`;
        items.push({ k: 'tbd' + i, t: 'text', x: cx[1], y, text: dropTxt(r.bd), size: 15, mono: true, anchor: 'middle', tone: r.bd ? 'bad' : 'good', bold: true });
        items.push({ k: 'tbl' + i, t: 'text', x: cx[2], y, text: r.bl.toFixed(3), size: 15, mono: true, anchor: 'middle' });
        items.push({ k: 'tnd' + i, t: 'text', x: cx[3], y, text: dropTxt(r.nd), size: 15, mono: true, anchor: 'middle', tone: r.nd ? 'bad' : 'good', bold: true });
        items.push({ k: 'tnl' + i, t: 'text', x: cx[4], y, text: r.nl.toFixed(3), size: 15, mono: true, anchor: 'middle' });
      });
      items.push({ k: 'tnote', t: 'text', x: 450, y: 370, text: t({ en: `same batch: ${NB} held-out names, ${d.tokens} tokens · loss = prediction loss on this batch`, zh: `同一個批次：${NB} 個保留名字，${d.tokens} 個詞元 · 損失 = 這個批次的預測損失` }), size: 13, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'tsw', t: 'box', x: 150, y: 400, w: 600, h: 70, tone: 'b', label: t({ en: 'Switch Transformer: top-1 routing, capacity factor', zh: 'Switch Transformer：top-1 路由，加上容量係數' }), sub: t({ en: 'overflow tokens skip the expert and pass through the residual', zh: '溢出的詞元跳過專家，經由殘差連線傳下去' }) });
    }

    if (step === 8) {
      const d = data();
      const gx = (g: number) => 30 + g * 215;
      for (let g = 0; g < E; g++) {
        items.push({ k: 'gpu' + g, t: 'box', x: gx(g), y: 40, w: 195, h: 380, tone: 'ink' });
        items.push({ k: 'gpl' + g, t: 'text', x: gx(g) + 97, y: 64, text: `GPU ${g}`, size: 15, bold: true, anchor: 'middle' });
        items.push({ k: 'gex' + g, t: 'box', x: gx(g) + 30, y: 340, w: 135, h: 56, label: t({ en: `expert E${g}`, zh: `專家 E${g}` }), tone: 'a', active: true });
        // a few real tokens of one name from this GPU's slice of the batch (letters 2–4)
        const ni = g * (NB / E);
        const toks = d.bal.route[ni].slice(1, 4);
        items.push({ k: 'gnm' + g, t: 'text', x: gx(g) + 97, y: 92, text: `${d.names[ni]} …`, size: 13, mono: true, anchor: 'middle', tone: 'muted' });
        toks.forEach((tk, j) => {
          const x = gx(g) + 18 + j * 56;
          items.push({ k: `gt${g}_${j}`, t: 'token', text: tk.ch, sub: `→ E${tk.e}`, x, y: 110 });
          items.push({ k: `ga${g}_${j}`, t: 'arrow', x1: x + 22, y1: 176, x2: gx(tk.e) + 97 + (j - 1) * 30, y2: 336, tone: tk.e === g ? 'muted' : 'a' });
        });
      }
      items.push({ k: 'comb', t: 'text', x: 450, y: 456, text: t({ en: 'arrows = dispatch (all-to-all) · after the expert, results go back the same way (all-to-all again)', zh: '箭頭 = 派送（all-to-all）· 專家算完，結果再沿原路送回（又一次 all-to-all）' }), size: 13, anchor: 'middle', tone: 'a' });
      items.push({ k: 'ill', t: 'text', x: 450, y: 495, text: t({ en: 'illustration: the routes come from the balanced model; the GPUs are drawn for explanation', zh: '示意圖：路由來自有平衡的模型，GPU 是為了說明而畫的' }), size: 13, anchor: 'middle', tone: 'muted' });
    }

    if (step === 9) {
      const boxes = [
        { en: 'router picks experts', zh: '路由器挑專家' }, { en: '+ 0.01 × balance loss', zh: '+ 0.01 × 平衡損失' }, { en: 'capacity per expert', zh: '每個專家的容量' },
        { en: 'all-to-all to GPUs', zh: 'all-to-all 送到各 GPU' }, { en: 'results back', zh: '結果送回' },
      ];
      boxes.forEach((b, i) => {
        const bx = 30 + i * 172;
        items.push({ k: 'rb' + i, t: 'box', x: bx, y: 190, w: 150, h: 60, label: t(b), tone: i === 1 || i === 2 ? 'a' : 'ink' });
        if (i < 4) items.push({ k: 'rba' + i, t: 'arrow', x1: bx + 152, y1: 220, x2: bx + 170, y2: 220 });
      });
      const d = data();
      items.push({ k: 'rt', t: 'text', x: 450, y: 310, text: t({ en: `busiest expert: ${Math.max(...last(U4))}% (balanced) vs ${Math.max(...last(UN))}% (not)`, zh: `最忙的專家：${Math.max(...last(U4))}%（有平衡）vs ${Math.max(...last(UN))}%（沒平衡）` }), size: 15, anchor: 'middle', tone: 'muted' });
      items.push({ k: 'rt2', t: 'text', x: 450, y: 340, text: t({ en: `dropped at factor 1.25: ${d.bal.cap[1].dropped} vs ${d.nob.cap[1].dropped} of ${d.tokens} tokens`, zh: `係數 1.25 時丟掉：${d.tokens} 個詞元中 ${d.bal.cap[1].dropped} 個 vs ${d.nob.cap[1].dropped} 個` }), size: 15, anchor: 'middle', tone: 'muted' });
    }

    const d = step >= 2 ? data() : undefined;
    return {
      metrics: [
        { label: { en: 'Busiest expert', zh: '最忙的專家' }, value: step >= 1 ? `${Math.max(...last(U4))}% / ${Math.max(...last(UN))}%` : `${Math.max(...last(UN))}%` },
        { label: { en: 'Balance loss', zh: '平衡損失' }, value: d ? (step === 2 ? d.bal.balance.toFixed(3) : `${d.bal.balance.toFixed(2)} / ${d.nob.balance.toFixed(2)}`) : '—' },
        { label: { en: 'Held-out loss', zh: '保留資料損失' }, value: step >= 4 ? `${VB.toFixed(3)} / ${VN.toFixed(3)}` : '—' },
        { label: { en: 'Dropped (×1.0)', zh: '丟掉（×1.0）' }, value: d && step >= 6 ? `${d.bal.cap[0].dropped} / ${d.nob.cap[0].dropped}` : '—' },
        { label: { en: 'Balance weight', zh: '平衡權重' }, value: step >= 1 ? `${ALPHA} / 0` : '0' },
      ],
      items,
      legend: step === 0 ? undefined : [
        { tone: 'a', text: { en: 'balanced model (weight 0.01)', zh: '有平衡的模型（權重 0.01）' } },
        { tone: 'bad', text: { en: 'no balancing / overflow', zh: '沒有平衡／溢出' } },
      ],
    };
  },
};
