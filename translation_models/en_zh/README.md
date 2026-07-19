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

## 关于体积 / int8（实测结论，2026-07-19）

当前分支只提交 **`model.onnx`（fp32，469 MB）**，因为它**唯一经过验证且运行时稳定**：
CPU 加载 ~43s、5 句翻译逐字正确、无 OOM。

做过的两版 int8 与结果：

| 方案 | 大小 | 加载 | 推理 | 翻译正确性 | 结论 |
|---|---|---|---|---|---|
| fp32（已提交） | 469 MB | ~43s | 正常 | ✅ 正确 | **推荐，直接用** |
| ORT `quantize_dynamic`（QOperator/MatMulInteger） | 466 MB | ~41s | 正常 | ✅ 与 fp32 逐字一致 | 正确但**几乎没变小** |
| 手工 `DequantizeLinear`（per-axis） | 137 MB | 慢(~8min) | **OOM** | 未跑出结果 | 磁盘小但**运行时不可用** |

**为什么 ORT 动态 int8 几乎没变小（466≈469）？**
动态量化只量化 `MatMul`/`Gemm` 权重，而本模型最大的一块是
**嵌入表（65001×512 ≈ 133 MB，走 `Gather` 不被量化）**，且 `lm_head` 与嵌入
权重共享（tied）→ 整块被跳过，仍以 fp32 保留。量化掉的只是注意力/FFN 那部分
（省约 77 MB），所以总量几乎不变。

**为什么官方 onnx-community 的 int8 更小（encoder 53MB + decoder 193MB ≈ 246MB）？**
1. 官方把**嵌入表也量化了**（他们的导出/量化流水线覆盖 `Gather` 与 tied 权重），
   133 MB 那块能压到 ~33 MB；
2. 用 ORT 原生融合 int8（`MatMulInteger`），运行时零展开；
3. **拆分成 encoder / decoder 多个文件**，单文件自然小；解码循环由调用方
   （transformers.js 或你的 Kotlin）在外层用 for 循环 + KV-cache 跑。

**手工 `DequantizeLinear` 为啥会 OOM？**
`DequantizeLinear` 没被 ORT 融合，跑 100 步展开解码图时每个 `MatMul` 都要临时
展开成 fp32 权重 → 100 份拷贝撑爆内存（实测在 8 GB 上限被 kill）。

### 想要「小 + 能跑」的可行路径
- **路径 A（最小改动）**：继续用 fp32 单文件（469 MB），把 `MarianMtEngine` 的
  朴素空格分词换成真正 SentencePiece（用 `source.spm`）即可。体积大但稳。
- **路径 B（最小体积）**：改用官方分体 int8（`encoder_model_int8.onnx` +
  `decoder_with_past_model_int8.onnx`），代价是 `MarianMtEngine` 要改成
  「加载分体模型 + 自回归循环 + KV-cache + 真 SentencePiece」。
- **路径 C（单文件 + 小）**：在本单文件基础上**专门量化嵌入表**（转成
  `MatMul` + `MatMulNBits`/int8 融合），其余用 ORT 原生 int8，可压到 ~140–200 MB
  且融合不 OOM；需额外处理 `Gather`→量化路径，工作量中等。

> 量化脚本保留在 `quantize_int8.py`（手工）与 `quantize_ort.py`（ORT 原生），
> 供参考；当前未提交任何 int8 模型文件。
