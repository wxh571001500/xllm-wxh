#!/bin/bash
pkill -9 xllm 2>/dev/null
sleep 2

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.9
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:$LD_PRELOAD
export MINDIE_LOG_LEVEL=INFO
export MINDIE_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export PROFILING_MODE=dynamic
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export XLLM_DSPARK_ACCURACY_DUMP_DIR=/export/home/wangxiaohan17/wangxiaohan/xllm-dspark-dump
export XLLM_DSPARK_ACCURACY_TRIGGER_FILE=/export/home/wangxiaohan17/wangxiaohan/dspark-trace
export XLLM_DSPARK_ACCURACY_MAX_CALLS=100

cd /export/home/wangxiaohan17/wangxiaohan/xllm-wxh-k3
mkdir -p logs

for i in 0 1 2 3; do
  PORT=$((13636+i))
  setsid build/xllm/core/server/xllm serve \
    --model=/export/home/models/kimi-k3-wxh \
    --model_id=kimi \
    --backend=vlm \
    --host=11.87.191.104 \
    --port=$PORT \
    --master_node_addr=11.87.191.104:29936 \
    --nnodes=4 \
    --node_rank=$i \
    --draft_model=/export/home/models/Kimi-K3-DSpark \
    --num_speculative_tokens=7 \
    --speculative_algorithm=DSpark \
    --max_memory_utilization=0.9 \
    --max_cache_size=4294967296 \
    --max_linear_state_cache_slots=32 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=20 \
    --block_size=128 \
    --communication_backend=hccl \
    --model_impl=python \
    --python_model_path=/export/home/wangxiaohan17/wangxiaohan/xllm-wxh-k3 \
    --enable_graph=false \
    --python_graph_backend=off \
    --enable_prefix_cache=true \
    --enable_flashcomm1=true \
    --dp_size=1 \
    --ep_size=4 \
    --enable_schedule_overlap=true \
    --enable_chunked_prefill=true \
    --max_tokens_per_chunk_for_prefill=2048 \
    --reasoning_parser=kimi_k3 \
    --tool_call_parser=kimi_k3 \
    > logs/xllm_node_$i.log 2>&1 &
  echo "node $i PID: $!"
done
echo "All xLLM nodes launched"
