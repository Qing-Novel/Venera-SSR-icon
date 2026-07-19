import torch, json, os, numpy as np
import onnxruntime as ort
from transformers import MarianMTModel, MarianTokenizer

MODEL = "Helsinki-NLP/opus-mt-en-zh"
MAX_LEN = 100
OUTDIR = "/workspace/Venera-SSR/translation_models/en_zh"
os.makedirs(OUTDIR, exist_ok=True)

print(">> loading model + tokenizer")
tok = MarianTokenizer.from_pretrained(MODEL)
model = MarianMTModel.from_pretrained(MODEL)
model.eval()
bos = int(model.config.decoder_start_token_id)
eos_id = int(model.config.eos_token_id or tok.eos_token_id)
pad_id = int(model.config.pad_token_id or tok.pad_token_id)
unk_id = int(tok.unk_token_id)
print("   decoder_start=%d eos=%d pad=%d unk=%d vocab=%d" % (bos, eos_id, pad_id, unk_id, model.config.vocab_size))

tests = ["Hello world, this is a test.", "I love reading manga.", "The weather is nice today."]

# [A] 标准参考输出
print("\n>> [A] 标准 MarianMT 管线 (正确分词) 参考输出:")
for s in tests:
    ids = tok(s, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=60, num_beams=4)
    print(f"  ZH: {tok.decode(out[0], skip_special_tokens=True)}")

# 朴素分词器 (复刻 MarianMtEngine.MarianTokenizer.encode)
tok2id = tok.get_vocab()
print("\n>> '>>zh<<' in vocab?", ">>zh<<" in tok2id, " | '▁Hello' in vocab?", "▁Hello" in tok2id)

def naive_encode(text):
    SP = "\u2581"
    ids = []
    for idx, w in enumerate(text.split()):
        if not w:
            continue
        pref = w if idx == 0 else SP + w
        ids.append(tok2id.get(pref, tok2id.get(w, unk_id)))
    ids.append(eos_id)
    return ids

def naive_decode(ids):
    SP = "\u2581"
    id2tok = {v: k for k, v in tok2id.items()}
    txt = ""
    for i in ids:
        t = id2tok.get(int(i), "")
        if t.startswith(SP):
            txt += (" " if txt else "") + t[len(SP):]
        else:
            txt += t
    return txt.strip()

# 包装器: 贪心解码 + EOS 早停, 展开进图
class Wrapper(torch.nn.Module):
    def __init__(self, m, max_len, bos, eos, pad):
        super().__init__()
        self.m = m; self.max_len = max_len; self.bos = bos; self.eos = eos; self.pad = pad
    def forward(self, input_ids, attention_mask):
        ehs = self.m.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dec_input = torch.full((1, 1), self.bos, dtype=torch.long)
        stopped = torch.zeros(1, dtype=torch.bool)
        pad_t = torch.tensor([[self.pad]], dtype=torch.long)
        eos_t = torch.tensor([[self.eos]], dtype=torch.long)
        outs = []
        for _ in range(self.max_len):
            d = self.m.model.decoder(input_ids=dec_input, encoder_hidden_states=ehs,
                                     encoder_attention_mask=attention_mask, use_cache=False)
            logits = self.m.lm_head(d.last_hidden_state) + self.m.final_logits_bias
            pre = logits[:, -1, :].argmax(-1, keepdim=True)
            is_eos = (pre == eos_t)
            stopped = stopped | is_eos
            nxt = torch.where(stopped, pad_t, pre)
            outs.append(nxt)
            dec_input = torch.cat([dec_input, nxt], dim=1)
        return torch.cat(outs, dim=1)

print("\n>> exporting single-file model.onnx (max_len=%d) ..." % MAX_LEN)
wrapper = Wrapper(model, MAX_LEN, bos, eos_id, pad_id).eval()
dummy_ids = torch.zeros((1, 8), dtype=torch.long)
dummy_mask = torch.ones((1, 8), dtype=torch.long)
torch.onnx.export(
    wrapper, (dummy_ids, dummy_mask),
    os.path.join(OUTDIR, "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["output_ids"],
    dynamic_axes={"input_ids": {1: "S"}, "attention_mask": {1: "S"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print("   exported ->", os.path.join(OUTDIR, "model.onnx"))

# [B] onnxruntime 验证 (朴素分词, 模拟引擎)
print("\n>> [B] 单文件 ONNX + 朴素分词 (复刻引擎行为) 实测:")
sess = ort.InferenceSession(os.path.join(OUTDIR, "model.onnx"))
for s in tests:
    tagged = ">>zh<< " + s
    ids = naive_encode(tagged)
    ids_t = np.array([ids], dtype=np.int64)
    mask_t = np.ones((1, len(ids)), dtype=np.int64)
    out = sess.run(["output_ids"], {"input_ids": ids_t, "attention_mask": mask_t})[0][0]
    gen = [i for i in out if int(i) not in (eos_id, pad_id)]
    print(f"  EN: {s}")
    print(f"  ZH(naive): {naive_decode(gen)}")

# 抽取 tokenizer 文件
print("\n>> saving tokenizer files")
tok.save_pretrained(OUTDIR)
with open(os.path.join(OUTDIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(tok2id, f, ensure_ascii=False)
print("   files:", sorted(os.listdir(OUTDIR)))
print("\nDONE")
