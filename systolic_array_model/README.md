# 模型仿真器

面向 LLM 推理 GEMM 的脉动阵列周期级仿真器，提供 Web 可视化界面。选择模型与算子后自动填入 M×K×N 参数，展示 Logic Die → PPU → Systolic Core 三层视图及 SRAM 访存统计。

## 硬件层次

```
Logic Die (4×8 = 32 PPU)
  └── PPU (4×4 = 16 Core)
        └── Systolic Core (16×16 MAC)
```

- **OS（部分和驻留）**：M ≤ 16，K 流式输入，N 在 Core / PPU 间切分；单 wave 最多 8192 列，N 更大时自动分 wave 串行计算
- **WS（权重驻留）**：4×4 Tensor Core 阵列，K ≤ 64，N ≤ 16
- **存储**：所有 Load/Store 均经 SRAM，不建模 DDR 层级

## Web 界面

左侧面板选择 **模型** 与 **GEMM 算子**，数据流默认 OS，可切换 WS。点击算子即开始仿真；右侧状态检查器展示利用率、SRAM 带宽（W/A/C 分项）及 MAC 单元寄存器状态。

访存符号约定（GEMM **C = A × W**）：

| 符号 | 含义 | 方向 |
|------|------|------|
| W | 权重 | 读 |
| A | 激活 | 读 |
| C | 输出 | 写 |

## 内置模型

### Llama3-8B（decode，batch=1，seq=1，tp=1）

| 算子 | 典型形状 M×K×N | 说明 |
|------|----------------|------|
| Q/O_Proj | 1×4096×4096 | Q_Proj · O_Proj |
| K/V_Proj | 1×4096×1024 | K_Proj · V_Proj |
| Gate/Up_Proj | 1×4096×14336 | Gate_Proj · Up_Proj |
| Down_Proj | 1×14336×4096 | |
| LM_Head | 1×4096×128256 | |

N 超过 8192 时按 wave 分批，每 wave 最多占用 32 PPU，各 wave 串行执行；总周期为各 wave 周期之和。

## 启动

```bash
# 构建前端（首次或前端改动后）
cd systolic_array_model/webui/frontend && npm install && npm run build

# 启动服务（默认端口 8766）
bash systolic_array_model/webui/run.sh
```

浏览器访问 http://127.0.0.1:8766

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models` | 模型与 GEMM 目录 |
| POST | `/api/simulate/model` | 按模型算子仿真，body: `{ model_id, gemm_id, dataflow }` |
| GET | `/api/simulate/{sim_id}/snapshots` | 宏观周期快照 |
| GET | `/api/simulate/{sim_id}/micro/{cycle}` | Core 级微观快照 |

## 目录结构

```
systolic_array_model/
├── hierarchy.py          # Logic Die 切分与约束
├── logic_die_engine.py   # OS 多 PPU 仿真引擎
├── mac.py / array.py     # MAC 单元与脉动阵列
├── memory.py             # SRAM 访存模型
├── models/registry.py    # 模型 GEMM 参数表
└── webui/
    ├── server.py         # FastAPI 服务
    ├── run.sh
    └── frontend/         # React 可视化（MacGrid 等）
```
