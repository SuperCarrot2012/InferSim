# systolic_array_model

基于 `systolic_array` 的模型驱动脉动阵列仿真器：按 LLM 推理 GEMM 算子选择参数，可视化 OS/WS 数据流。

## 与 systolic_array 的差异

| 项 | systolic_array | systolic_array_model |
|---|---|---|
| 标题 | InferSim 脉动阵列仿真器 | **脉动阵列仿真器** |
| 左侧面板 | 手动输入 M/K/N | **模型 + GEMM 按钮**（当前 Llama3-8B） |
| 默认数据流 | — | **部分和驻留 (OS)**，可下拉切换 |
| Logic Die PPU | 48 (6×8) | **32 (4×8)** |
| 存储模型 | 通用访存统计 | **SRAM only**（不含 DDR） |

## Llama3-8B GEMM 列表

Decode 单步 (batch=1, seq=1, tp=1)：

- Q_Proj、K_Proj、V_Proj、O_Proj
- Gate_Proj、Up_Proj、Down_Proj
- LM_Head

部分大 N 算子（Gate/Up/LM_Head）超出单 Die 32 PPU 容量时会报错；UI 会标注所需 PPU 数。

## 启动

```bash
# 构建前端（首次或前端改动后）
cd systolic_array_model/webui/frontend && npm install && npm run build

# 启动服务（默认端口 8766）
bash systolic_array_model/webui/run.sh
```

浏览器打开 http://127.0.0.1:8766 ，选择 GEMM 算子即开始仿真。

## API

- `GET /api/models` — 模型与 GEMM 目录
- `POST /api/simulate/model` — `{ model_id, gemm_id, dataflow }`
