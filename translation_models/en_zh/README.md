# 本地翻译模型：英 → 中（en_zh）

单文件 MarianMT ONNX，配合 Venera-SSR `feature/translation-embed` 的 `MarianMtEngine` 使用。

## 文件

| 文件 | 说明 |
|---|---|
| `model.onnx` | **单文件** seq2seq 模型。输入 `input_ids`+`attention_mask`(int64)，输出 `output_ids`(int64 token id)。贪心解码 + EOS 早停已烤进图（max_len=100）。约 448 MB（fp32，词表占大头）。 |
| `vocab.json` | `MarianTokenizer` 用的 token→id 映射（`Helsinki-NLP/opus-mt-en-zh`）。 |
| `source.spm` / `target.spm` | SentencePiece 模型（源/目标），用于**正确的分词**（见下）。 |
| `tokenizer_config.json` | 分词器配置（参考）。 |

## 引擎如何加载

`MarianMtEngine` 在「Model Directory」下找 `model.onnx`（或 `encoder_model.onnx`）+ `vocab.json`。
把本目录内容放进你在设置里填的目录即可，例如：

```
/sdcard/Android/data/.../translation/marian/
├── model.onnx
├── vocab.json
├── source.spm
└── target.spm
```

## ⚠️ 必须修的集成点（否则翻译质量退化）

1. **分词器必须是真正的 SentencePiece，不能用引擎当前的朴素空格分词。**
   引擎 `MarianMtEngine.MarianTokenizer` 现在是「空格切分 + `▁` 前缀查表」，
   而 MarianMT 用的是 SentencePiece（Unigram）。朴素分词会把
   "manga" 等词整体查不到 → 退化（实测 "I love reading manga." → "我喜欢看书" 而非 "我喜欢看漫画"）。
   正确分词（用 `source.spm`）下模型输出完美（"你好,世界,这是个测试" / "我喜欢看漫画" / "今天天气不错"）。
   → 需用 `source.spm` 实现 SentencePiece Unigram 分词替换 `MarianTokenizer.encode`。

2. **`>>zh<<` 语言标签无害。** 该专用模型 vocab 里没有 `>>zh<<`（会变 UNK），但实测带不带前缀翻译都正确，引擎现有 `toMarianSourceTag()` 可保留。

## 验证记录（Python + onnxruntime，CPU）

| 英文输入 | ONNX 输出（正确分词） |
|---|---|
| Hello world, this is a test. | 你好,世界,这是个测试 |
| I love reading manga. | 我喜欢看漫画 |
| The weather is nice today. | 今天天气不错 |

模型由 `Helsinki-NLP/opus-mt-en-zh` 经 PyTorch 导出：encoder + decoder 展开成单图，
`lm_head` 后加 `final_logits_bias`，逐 token 贪心 argmax，预测 EOS(0) 即停。

## 关于体积 / int8

当前为 fp32（~448 MB）。onnxruntime 的 `quantize_dynamic` 对本图会膨胀
（embedding 走 `Gather` 不被量化，266 MB fp32 词表保留 + 其余加 int8/scale → 更大），
故未采用。如需更小体积，建议改走「引擎侧解码 + onnx-community 的
`opus-mt-en-zh` 分体 int8（`encoder_model_int8.onnx` + `decoder_model_merged_int8.onnx`，合计约 246 MB）」，
由引擎做自回归循环 + beam search；代价是 `MarianMtEngine` 需改为加载分体模型并补 KV-cache 循环。
