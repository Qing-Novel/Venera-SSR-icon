import os, time
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

SRC = "/workspace/Venera-SSR/translation_models/en_zh/model.onnx"
DST = "/workspace/Venera-SSR/translation_models/en_zh/model.int8.onnx"

t = time.time()
quantize_dynamic(
    model_input=SRC,
    model_output=DST,
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["MatMul", "Gemm"],
)
print("quantized in %.1fs" % (time.time() - t))
print("fp32: %.1f MB | int8: %.1f MB" % (os.path.getsize(SRC)/1e6, os.path.getsize(DST)/1e6))
