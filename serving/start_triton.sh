#!/bin/bash
docker rm -f triton 2>/dev/null
docker run -d --name triton -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /workspaces/reco-engine/model_repo/triton:/models \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  tritonserver --model-repository=/models