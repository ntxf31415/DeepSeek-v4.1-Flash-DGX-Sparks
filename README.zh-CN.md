# DeepSeek-V4.1-Flash（SGLang）· 4× DGX Spark TP4 · 无交换机环网

在 **4× NVIDIA DGX Spark（GB10）无交换机 RoCE 环网**上以 **SGLang TP4** 部署
**deepseek-ai/DeepSeek-V4.1-Flash**（552B MoE、8B/16B 激活、MXFP4 experts、
1M 上下文、DSpark 投机解码）的生产方案。

**四机环网实测（1M ctx / 5M KV 池）。思考模式写作 `OFF · ON`；数字对应的构建见 [BUILD-IDENTITY.md](BUILD-IDENTITY.md)。**

| 指标 | 数值 |
|---|---|
| decode 峰/均（code，temp 0） | **100.3 / 81.4** · 99.8 / 81.2 tok/s |
| 散文 OFF · ON | **33.3 · 36.6** tok/s |
| prefill 8K / 32K / 100K | **3102 / 3443 / 3253** t/s |
| 聚合 c1 / c4 / c8 / c12 | 82 / 223 / 295 / **398** tok/s（ON：83 / 225 / 298 / 403） |
| 质量门禁 | needle 30K-470K ✅ · corruption 0/0/0 · 终止性 18/18+18/18 · code-gate 12/12 · GSM8K n=50 **1.00** |
| DSpark 接受 | 3.98 tok/step，rate 0.595（数学流打满 6.0） |
| 冷启动 | 约 9 分钟，无冷启动惩罚 |

**长上下文**（冷 prefill，needle 校验）：

| 深度 | 结果 |
|---|---|
| 470K | ✅ 211.7s · 2144 t/s |
| 600K | ✅ 341.7s · 内存地板 4.92 GB |
| **900K** | ⚠️ **失败**——Engram 行缓存与共享专家 pad 缓冲吃掉约 2GB 深上下文余量；600K 及以下不受影响 |

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
| 宿主 ring-only NCCL 2.30.7 构建 + `libncclpin` 核绑定 shim（宿主侧，仓库不含） | [luxingcom/aicad-nccl-optimization](https://github.com/luxingcom/aicad-nccl-optimization)（LuZ 谱系） | **未声明许可证** |
| 模型权重 | `deepseek-ai/DeepSeek-V4.1-Flash`（Hugging Face） | 见模型卡 |

## 环网适配差异（相对上游 TP4 profile）

**传输 / 拓扑**

- ring-only NCCL 2.30.7 + libncclpin 核绑定 shim（`LD_PRELOAD`）；`NCCL_IB_GID_INDEX=-1` 铁律；四边接线的 per-rank `PEER_HCA`（`./start-tp4.sh ncclcheck` 可自检 ring-only 是否真的生效）。该库是 **LuZ 谱系**构建（靠 NCCL 算法矩阵 `Tree=0 / Ring=1` 实现 ring-only），**不是** SparkRing 的补丁库——见 [BUILD-IDENTITY.md](BUILD-IDENTITY.md)
- 节点本地权重（无 NFS）、loopback 引擎 + 并发代理、多别名 served name（旧名保留，消费端零改动）

**为环网做的调优**（每项单独 A/B，`[measured]` 数据见配置示例头部注释）

| 设置 | 理由 |
|---|---|
| `EP_SIZE=2`（原 4） | 消除专家并行 straggler；500K 长档 prefill 595s → 345s |
| `MAX_RUNNING_REQUESTS=12`（原 8） | 配合 `--min-free-slots-delay 1` 第 12 路才真正并行：c12 271 → 398 |
| `DSV41_CACHE_GIB=1` / 16-way | Engram 行缓存：命中率 0 → 99.1%，c12 +6%，prefill 100K +10.5% |
| `DSV41_SHARED_PAD_K=1` | 上游 PR#17：让共享专家 K=576 的形状重回 b12x（逐位无损） |
| static verify | 上游 compact/ragged 模式在 V4.1 上触发 engram target-verify 断言（sgl-project/sglang#39173） |

试后回退：`--enable-deepseek-v4-fp4-indexer`（500K 长档 −11%、多吃 6.4GB 统一内存，c12 无增益）。唯一保留的回退项：c6 260 → 236（EP2 的副作用，但 c8/c12 涨幅远大于此）。

## 仓库内容

- `start.sh / start-tp4.sh / stop.sh / boot.py` — 编排、锁修订版下载、冒烟 + 预热
- `adapter/` — SGLang 补丁（Engram 行存储 C++、MXFP8 后端、共享专家 K padding、prefill 缓存钩子）
- `scripts/` — SSH 助手、verify/ 探针集、自愈监控 + systemd 单元、`gate.sh`、`nccl_selfcheck.sh`
- `bench/` — 门禁套件、vision 门禁、事件矩阵 + 重叠窗分析、散文、GSM8K、第三方形状并发扫描
- `.env.tp4.ring.example` — **本仓库实际运行的配置**（脱敏，每处偏离上游都带实测理由）
- `BUILD-IDENTITY.md` — 镜像 ID、SGLang commit、组件版本、逐文件 md5
- `docs/` — 部署方案、上游 ISSUE/PR 调研、基准横向对比

## 脱敏说明

内部 IP/主机名已占位符化、API key 已移除；站点 `.env.tp4` 不入库。本 fork
保留上游 `main` 分支不动，适配内容在 `4dgx-ring` 分支。
