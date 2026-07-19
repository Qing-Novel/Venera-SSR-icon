import os, json, numpy as np, time
import onnxruntime as ort
import sentencepiece as spm

OUTDIR = "/workspace/Venera-SSR/translation_models/en_zh"
sp_src = spm.SentencePieceProcessor(model_file=os.path.join(OUTDIR, "source.spm"))
tok2id = json.load(open(os.path.join(OUTDIR, "vocab.json")))
id2tok = {v: k for k, v in tok2id.items()}
eos_id = 0  # Marian </s>
print("loaded spm + vocab (%d entries)" % len(id2tok), flush=True)

print("loading int8...", flush=True)
t = time.time()
sess = ort.InferenceSession(os.path.join(OUTDIR, "model.int8.onnx"), providers=["CPUExecutionProvider"])
print("loaded in %.1fs" % (time.time() - t), flush=True)

def detok(ids):
    SP = "▁"
    txt = ""
    for i in ids:
        i = int(i)
        if i == eos_id:
            continue
        t = id2tok.get(i, "")
        if t.startswith(SP):
            txt += (" " if txt else "") + t[len(SP):]
        else:
            txt += t
    return txt.strip()

tests = ["Hello world, this is a test.", "I love reading manga.", "The weather is nice today.",
         "She is a talented programmer.", "The book on the table is mine."]

print("\n>> int8 单文件 (spm 编码, 无标签, 末尾补 </s>):", flush=True)
for s in tests:
    ids = sp_src.encode(s, out_type=int) + [eos_id]
    ids_t = np.array([ids], dtype=np.int64)
    mask = np.ones((1, len(ids)), dtype=np.int64)
    out = sess.run(["output_ids"], {"input_ids": ids_t, "attention_mask": mask})[0][0]
    print(f"  EN: {s}", flush=True)
    print(f"  ZH: {detok(out)}", flush=True)
print("\nDONE", flush=True)
