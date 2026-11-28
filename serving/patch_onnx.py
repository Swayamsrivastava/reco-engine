p = "serving/recommender.py"
s = open(p).read()

def swap(old, new):
    global s
    assert s.count(old) == 1, f"expected exactly one match for: {old[:60]}"
    s = s.replace(old, new)

swap("import time, numpy as np, pandas as pd, faiss, tensorflow as tf",
     "import os, time, numpy as np, pandas as pd, faiss, tensorflow as tf")
swap("out = self.infer(**{self.in_key: tf.constant(X)})", "out = None")
swap("s = list(out.values())[0].numpy().ravel()", "s = self._score(X)")
swap("self.in_key = list(self.infer.structured_input_signature[1].keys())[0]",
     "self.in_key = list(self.infer.structured_input_signature[1].keys())[0]\n"
     "        # optional: score with the exported ONNX ranker (set RANKER_BACKEND=onnx)\n"
     "        self._ort = None\n"
     "        if os.environ.get('RANKER_BACKEND') == 'onnx':\n"
     "            import onnxruntime as ort\n"
     "            self._ort = ort.InferenceSession('model_repo/triton/ranker/1/model.onnx',\n"
     "                                             providers=['CPUExecutionProvider'])\n"
     "            self._ort_in = self._ort.get_inputs()[0].name")
swap("    def _describe(self, ids):",
     "    def _score(self, X):\n"
     "        if self._ort is not None:\n"
     "            return self._ort.run(None, {self._ort_in: X})[0].ravel()\n"
     "        out = self.infer(**{self.in_key: tf.constant(X)})\n"
     "        return list(out.values())[0].numpy().ravel()\n"
     "\n"
     "    def _describe(self, ids):")
open(p, "w").write(s)
print("patched")