import torch, json, os, numpy as np
import onnxruntime as ort
from transformers import MarianTokenizer

OUTDIR = "/workspace/Venera-SSR/translation_models/en_zh"
tok = MarianTokenizer.from_pretrained(OUTDIR)
tok2id = tok.get_vocab()
id2tok = {v: k for k, v in tok2id.items()}
eos_id = int(tok.eos_token_id); pad_id = int(tok.pad_token_id)
print("eos=%d pad=%d" % (eos_id, pad_id))

sess = ort.InferenceSession(os.path.join(OUTDIR, "model.onnx"))

def naive_decode(ids):
    SP = "\u2581"; txt = ""
    for i in ids:
        t = id2tok.get(int(i), "")
        if t.startswith(SP): txt += (" " if txt else "") + t[len(SP):]
        else: txt += t
    return txt.strip()

tests = ["Hello world, this is a test.", "I love reading manga.", "The weather is nice today."]

print("\n>> [C] ONNX + 正确 SentencePiece 编码 (无标签) -> 中文:")
for s in tests:
    ids = tok(s, return_tensors="pt").input_ids[0].tolist()   # 正确 SP ids, 无标签
    ids_t = np.array([ids], dtype=np.int64)
    mask = np.ones((1, len(ids)), dtype=np.int64)
    out = sess.run(["output_ids"], {"input_ids": ids_t, "attention_mask": mask})[0][0]
    gen = [i for i in out if int(i) not in (eos_id, pad_id)]
    print(f"  EN: {s}")
    print(f"  ZH: {naive_decode(gen)}")

print("\n>> [D] ONNX + 正确 SP 编码 (带 '>>zh<< ' 标签) -> 中文 (看标签是否破坏):")
for s in tests:
    ids = tok(">>zh<< " + s, return_tensors="pt").input_ids[0].tolist()
    ids_t = np.array([ids], dtype=np.int64)
    mask = np.ones((1, len(ids)), dtype=np.int64)
    out = sess.run(["output_ids"], {"input_ids": ids_t, "attention_mask": mask})[0][0]
    gen = [i for i in out if int(i) not in (eos_id, pad_id)]
    print(f"  EN: {s}")
    print(f"  ZH: {naive_decode(gen)}")
print("\nDONE-C")
