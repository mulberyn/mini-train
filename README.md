# miniLLM-engine

个人开发的从零开始的轻量级 LLM 引擎，旨在进行轻量且快捷的 LLM 训练及推理。

## 项目结构

```txt
miniLLM-engine/
├── trainer/          # 训练
├── inference/        # 推理（ModelRunner / KV Cache / Attention / 缓存模型）
├── serving/          # FastAPI 服务
├── profiler/         # 性能分析（kv_cache / attention）
├── benchmark/        # Benchmark 脚本和结果（kv_cache / attention）
├── docs/             # 文档和架构图
├── tests/            # 单元测试
├── examples/         # 示例
└── configs/          # 配置文件
```

## 训练框架

可以看作课程 [cs336 Spring 2026](https://cs336.stanford.edu/) 的拓展，还在学习/开发中。

## 推理框架

已实现 **KV Cache → Paged Attention** 完整链路（见 `docs/kv_cache.md`）：

```text
ModelRunner
    ├── NaiveKVCache        # torch.cat 逐 token 拼接（参考实现）
    ├── StaticKVCache       # 一次性预分配 [B, H, max_seq_len, D]，原地写入
    ├── DynamicKVCache      # 容量按需倍增（O(log T) 次分配）
    └── PagedKVCache        # 物理块池 + 每序列 block table（vLLM 风格）
            └── BlockManager # 空闲/已分配块管理 + 引用计数 + 复用
    └── Paged Attention     # Python 参考实现：按 block table 逐块 gather 计算
```

### 统一 KVCache 接口（`inference/kv_cache/base.py`）

所有 cache 实现共享同一接口，ModelRunner / attention / benchmark 无需感知具体策略：

```python
kv_cache.update(layer_idx, key, value, positions)   # key/value: [B, H, T_new, D]
k, v = kv_cache.get(layer_idx)                      # [B, H, T, D]
kv_cache.reset()
kv_cache.memory_usage()                             # 当前占用字节数
kv_cache.allocation_count                           # K/V tensor 分配次数
```

### 缓存推理 API（`inference/model_runner.py`）

```python
runner = ModelRunner(model, tokenizer, device="cuda")
cache = runner.build_kv_cache(cache_type="paged", max_seq_len=2048, block_size=16)

logits = runner.prefill(input_ids, cache)          # 一次写入整个 prompt
logits = runner.decode_step(next_token, cache)     # 单 token 解码（position 自动）
tokens  = runner.generate_with_cache(prompt, cache, max_new_tokens=128)
```

核心正确性保证（`tests/inference/test_kv_generation.py`）：

```text
cache generation == no-cache generation   （token 完全一致）
prefill + decode 的逐步 logits == 全序列 forward 的 logits（assert_close）
```

### Paged Attention（`inference/attention/paged_attention.py`）

统一接口 + 可插拔实现（`implementation=`）：

```python
out = paged_attention(
    query, key_pool, value_pool, block_tables, context_lengths,
    block_size=16, num_kv_heads=8, implementation="torch",  # loop | torch | triton
)
```

| implementation | 说明 | batch 扩展性 |
| -------------- | ---- | ----------- |
| `"loop"`  | 逐序列 Python 循环（正确性参考，O(B) 次 kernel launch） | O(B) ❌ |
| `"torch"` | 批量向量化：一次 `key_cache[block_tables]` gather + 批量 matmul（默认） | ~flat ✅ |
| `"triton"`| 原生 Triton kernel：每 (batch, head) 一个 program，flash-attention 在线 softmax（Linux+CUDA，本机自动跳过） | ~flat ✅ |

与稠密 attention（`dense_attention` / `F.scaled_dot_product_attention`）逐项对比测试通过
（`tests/inference/test_paged_attention.py`，含 GQA 与 fuzz）。

### Paged Attention 性能优化（docs/paged_attention.md）

核心问题：Python 逐序列循环导致 `runtime ≈ O(B)`。批量向量化后（RTX 5070 Ti, ctx=2048）：

```text
   B    ctx       loop        torch       dense
   1   2048      2.628       0.427       2.889
   8   2048     20.495       0.988       2.805
  64   2048    175.274       6.360       5.052
```

loop 呈 O(B) 线性恶化（67x），torch 向量化实现接近持平 —— 消除 batch 并行瓶颈。

系统级实验（`benchmark/attention/`）：

- `benchmark_batch_scaling.py` — 实验 A：B = 1..64，ctx = 2048
- `benchmark_page_size.py` — 实验 B：page = 8..256（展示 sweet spot 与 KV 碎片化）
- `benchmark_memory.py` — 实验 C：Dense vs Paged 显存（ctx=512 时 paged 仅用 6.2%）
- `benchmark_heterogeneous.py` — 实验 D：异长序列 Dense+padding vs Paged（2–5.6x）
- `benchmark_paged_attention.py` — 三种实现 vs Dense SDPA 全矩阵

profiler（`profiler/attention/profile_paged_attention.py`）直接展示 kernel launch 数：
loop 为逐序列 kernel（CPU 162ms），torch 为单次 gather（CPU 8.6ms）。

### 快速验证

```bash
python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp_$PID   # 515 passed
python -m benchmark.kv_cache.benchmark_compare --device cuda
python -m benchmark.attention.benchmark_batch_scaling --device cuda
python -m benchmark.attention.benchmark_page_size --device cuda
python -m benchmark.attention.benchmark_heterogeneous --device cuda
python -m profiler.kv_cache.profile_kv_cache --device cuda
```

> 注：在 DSH 沙箱下运行 pytest 时需使用 `--basetemp=<workspace 内唯一路径>`（沙箱会把
> `mode=0o700` 的目录转成不可枚举的 ACL，且不允许删除其他进程创建的目录，`conftest.py`
> 已兼容 mode 问题，basetemp 建议每次用唯一名字如 `.pytest_tmp_$PID`）。
