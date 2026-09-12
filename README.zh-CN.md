# DeepSeek-V4.1-Flash（SGLang）· 4× DGX Spark TP4 · 无交换机环网

在 **4× NVIDIA DGX Spark（GB10）无交换机 RoCE 环网**上以 **SGLang TP4** 部署
**deepseek-ai/DeepSeek-V4.1-Flash**（552B MoE、8B/16B 激活、MXFP4 experts、
1M 上下文、DSpark 投机解码）的生产方案。

**四机环网实测（1M ctx / 4M KV 池）：**

| 指标 | 数值 |
|---|---|
| decode 峰/均（code，temp 0） | **89.6 / 72.1** tok/s |
| 散文 OFF / ON | 30.8 / 33.6 tok/s |
| prefill 8K / 32K / 100K | **2814 / 3149 / 3119** t/s |
| 聚合 c1 / c4 / c8 / c12 | 74 / 210 / 264 / 271 tok/s |
| 470K needle / 900K 冷 prefill | ✅ / 1271 t/s（738s） |
| 质量门禁 | needle 30K-470K ✅ · corruption 0/0/0 · 终止性 18/18+18/18 · code-gate 12/12 · GSM8K n=50 **1.00** |
| 冷启动 | 约 9 分钟，无冷启动惩罚 |

六方案横向总表（LuZ / Vision-Exp / GLM 对照）：[docs/4DGX-dsv41-基准测试-横向对比-20260912.md](docs/4DGX-dsv41-基准测试-横向对比-20260912.md)。

---

> **姊妹项目：** [DeepSeek-V4-Flash-Vision-Exp TP4 无交换机环网](https://github.com/ntxf31415/deepseek-v4-vision-exp-dgxspark-tp4-switchless-ring)（vLLM，同一环网底座）· [GLM-5.3-Flash NVFP4 TP4 无交换机环网](https://github.com/ntxf31415/glm-5.3-flash-nvfp4-4x-dgx-spark-switchless)（同底座配方）。

## 这是什么

上游 SGLang 配方的**环网适配 + 运维层**。仓库包含启动脚本、SGLang 补丁
（Engram NVMe 行存储、MXFP8 b12x、prefill 缓存钩子）、自愈监控与基准门禁
套件。**不含权重、镜像、NCCL 二进制。**

| 组件 | 来源 | 许可证 |
|---|---|---|
| SGLang 配方（boot、适配器、Engram 行存储、DSpark 设置） | [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | AGPL-3.0-or-later |
| 配方谱系 / 基准方法 | [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) | MIT |
| 模型权重 | `deepseek-ai/DeepSeek-V4.1-Flash`（Hugging Face） | 见模型卡 |

## 环网适配差异（相对上游 TP4 profile）

- ring-only NCCL 2.30.7 + libncclpin 核绑定 shim（`LD_PRELOAD`）；`NCCL_IB_GID_INDEX=-1` 铁律；四边接线的 per-rank `PEER_HCA`
- 节点本地权重（无 NFS）、loopback 引擎 + 并发代理、多别名 served name（旧名保留，消费端零改动）
- static verify 模式 + 实测 SPS 表（上游 compact 模式在并发短请求下触发 engram target-verify 断言）
- 本地打包 Engram 分片（约 47 GiB/节点），权重加载 49s → 24s
- 补丁全表：部署方案 §三

## 脱敏说明

内部 IP/主机名已占位符化、API key 已移除；站点 `.env.tp4` 不入库。本 fork
保留上游 `main` 分支不动，适配内容在 `4dgx-ring` 分支。
