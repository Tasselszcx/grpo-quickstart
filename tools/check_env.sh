#!/usr/bin/env bash
# 环境检查脚本：验证 Search-R1 所有依赖是否就绪
echo "=== Search-R1 环境检查 ==="

cd $HOME  # 避免本地 verl 0.1 遮蔽系统 verl 0.9

echo -n "[1] 系统 verl 版本: "
python3 -c "import verl; print(open(verl.__file__.replace('__init__.py','')+'version/version').read().strip())" 2>/dev/null || echo "❌ 未找到"

echo -n "[2] sglang 版本: "
python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "❌ 未找到"

echo -n "[3] sglang weight_sync: "
python3 -c "from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights; print('OK')" 2>/dev/null || echo "❌ 需要 sglang>=0.4.9"

echo -n "[4] vllm 版本: "
python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "❌ 未找到"

echo -n "[5] transformers 版本: "
python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "❌ 未找到"

echo -n "[6] Qwen3-8B 模型: "
MODEL=/home/hadoop-efficient-llm/dolphinfs_ssd_hadoop-efficient-llm/models/Qwen/Qwen3-8B
[ -f "$MODEL/config.json" ] && echo "OK ($MODEL)" || echo "❌ 未找到"

echo -n "[7] 检索服务 (port 8000): "
curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && echo "OK" || echo "❌ 未启动 (运行: python mock_retrieval_server.py)"

echo -n "[8] 训练数据: "
TRAIN=/home/hadoop-efficient-llm/projects/Search-R1/data/searchr1/train.parquet
[ -f "$TRAIN" ] && python3 -c "import pandas as pd; df=pd.read_parquet('$TRAIN'); print(f'OK ({len(df)} 条)')" || echo "❌ 未找到 (运行: python scripts/prepare_searchr1_data.py)"

echo -n "[9] GPU 状态: "
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -2 | awk -F', ' '{print "GPU"$1": "$2"/"$3}'

echo ""
echo "如果 [3] 报错，请运行:"
echo "  pip install sgl-kernel --upgrade --proxy http://10.217.148.40:8080"
echo "  pip install sglang==0.4.9.post6 --proxy http://10.217.148.40:8080"
