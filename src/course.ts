import type { L } from './i18n';
import type { Lesson } from './lessons/types';
// Every file in src/lessons/ that exports a Lesson is picked up automatically.
const found = import.meta.glob<Record<string, unknown>>('./lessons/*.ts', { eager: true });

const isLesson = (v: unknown): v is Lesson =>
  typeof v === 'object' && v !== null && 'steps' in v && 'scene' in v && 'id' in v;

/** Lessons that are written, by id. Anything else in the outline is shown as "soon". */
export const LESSONS: Record<string, Lesson> = {};
for (const mod of Object.values(found)) for (const v of Object.values(mod)) if (isLesson(v)) LESSONS[v.id] = v;

export interface Module { title: L; lessons: { id: string; title: L }[] }

const l = (id: string, en: string, zh: string) => ({ id, title: { en, zh } });

export const MODULES: Module[] = [
  { title: { en: 'Foundations', zh: '基礎' }, lessons: [
    l('language-models', 'What a Language Model Does', '語言模型在做什麼'),
    l('tokens', 'Tokens', '詞元（Token）'),
    l('embeddings', 'Embeddings', '嵌入'),
    l('next-token', 'Predicting the Next Token', '預測下一個詞元'),
  ] },
  { title: { en: 'Attention', zh: '注意力' }, lessons: [
    l('self-attention', 'Self-Attention, Step by Step', '自注意力，一步一步來'),
    l('multi-head', 'Multi-Head Attention', '多頭注意力'),
    l('positions', 'Positional Information', '位置資訊'),
  ] },
  { title: { en: 'The Transformer Block', zh: 'Transformer 區塊' }, lessons: [
    l('mlp', 'The MLP Layer', 'MLP 層'),
    l('residual', 'The Residual Stream', '殘差流'),
    l('layer-norm', 'Layer Normalization', '層正規化'),
    l('stacking', 'Stacking Blocks', '堆疊區塊'),
    l('sampling', 'From Logits to Text', '從 logits 到文字'),
  ] },
  { title: { en: 'Training', zh: '訓練' }, lessons: [
    l('training-loop', 'The Training Loop', '訓練迴圈'),
    l('gradients', 'Gradients and Backpropagation', '梯度與反向傳播'),
    l('optimizers', 'Optimizers: SGD and Adam', '最佳化器：SGD 與 Adam'),
    l('learning-rate', 'Learning Rate and Warmup', '學習率與暖身'),
    l('overfitting', 'Overfitting and Validation', '過度擬合與驗證'),
  ] },
  { title: { en: 'Training at Scale', zh: '大規模訓練' }, lessons: [
    l('data-pipeline', 'Data and Tokenizer Pipeline', '資料與分詞流程'),
    l('compute', 'Compute and Mixed Precision', '運算資源與混合精度'),
    l('evaluation', 'Checkpoints and Evaluation', '檢查點與評估'),
    l('fine-tuning', 'Pretraining vs Fine-Tuning', '預訓練與微調'),
    l('lora', 'LoRA', 'LoRA'),
  ] },
  { title: { en: 'Whole Systems', zh: '完整系統' }, lessons: [
    l('tiny-gpt', 'Build a Tiny GPT', '打造一個迷你 GPT'),
    l('train-live', 'Train It Live', '現場訓練它'),
    l('real-run', 'Read a Real Training Run', '讀懂一次真實的訓練'),
  ] },
];

export const BUILT_ORDER: string[] = MODULES.flatMap((m) => m.lessons.map((x) => x.id)).filter((id) => id in LESSONS);
export const TOTAL = MODULES.reduce((s, m) => s + m.lessons.length, 0);
