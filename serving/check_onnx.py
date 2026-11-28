import numpy as np, tensorflow as tf, onnxruntime as ort

X = np.load("/tmp/check_X.npy")
stored = np.load("/tmp/check_scores.npy")

sess = ort.InferenceSession("model_repo/triton/ranker/1/model.onnx", providers=["CPUExecutionProvider"])
print("inputs: ", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
onnx_out = sess.run(None, {sess.get_inputs()[0].name: X})[0].ravel()

infer = tf.saved_model.load("model_repo/ranker_wd/1/model.savedmodel").signatures["serving_default"]
key = list(infer.structured_input_signature[1].keys())[0]
tf_out = list(infer(**{key: tf.constant(X)}).values())[0].numpy().ravel()

print("max |ONNX - SavedModel|:        ", float(np.abs(onnx_out - tf_out).max()))
print("max |SavedModel - stored score|:", float(np.abs(tf_out - stored).max()))