import onnx, numpy as np
from onnx import numpy_helper, helper

SRC = "/workspace/Venera-SSR/translation_models/en_zh/model.onnx"
DST = "/workspace/Venera-SSR/translation_models/en_zh/model.int8.onnx"

m = onnx.load(SRC)
g = m.graph
inits = {i.name: i for i in g.initializer}

# 找需要量化的权重: 被 MatMul/Gemm/Gather/Conv 使用且 ndim>=2 且较大
weight_names = set()
for node in g.node:
    if node.op_type in ("MatMul", "Gemm", "Gather", "Conv"):
        for inp in node.input:
            if inp in inits:
                arr = numpy_helper.to_array(inits[inp])
                if arr.ndim >= 2 and arr.size > 4096:
                    weight_names.add(inp)
print("待量化权重数:", len(weight_names))

new_inits = list(g.initializer)
new_nodes = list(g.node)
for wn in weight_names:
    arr = numpy_helper.to_array(inits[wn]).astype(np.float32)
    maxabs = np.max(np.abs(arr), axis=1).astype(np.float32)
    maxabs[maxabs == 0] = 1.0
    scale = (maxabs / 127.0).astype(np.float32)          # (dim0,)
    q = np.round(arr / scale[:, None]).clip(-127, 127).astype(np.int8)
    zp = np.zeros(q.shape[0], dtype=np.int8)
    qn, sn, zpn, dn = wn + "_q", wn + "_s", wn + "_zp", wn + "_dq"
    new_inits.append(numpy_helper.from_array(q, qn))
    new_inits.append(numpy_helper.from_array(scale, sn))
    new_inits.append(numpy_helper.from_array(zp, zpn))
    new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn, zpn], [dn], axis=0))
    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp == wn:
                node.input[i] = dn

final = [i for i in new_inits if i.name not in weight_names]
g.ClearField("initializer"); g.initializer.extend(final)
g.ClearField("node"); g.node.extend(new_nodes)
onnx.save(m, DST)
print("saved", DST)
import os
print("fp32:", round(os.path.getsize(SRC)/1e6, 1), "MB | int8:", round(os.path.getsize(DST)/1e6, 1), "MB")
