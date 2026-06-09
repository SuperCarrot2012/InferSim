# Systolic Array Simulator

自研 GPU 脉动阵列（16×16，Weight-Stationary）的 cycle 级 dataflow 仿真器，附带 React Web UI 可视化。

仅模拟数据流动（W/A/P 矩阵元素坐标），不做数值计算与结果校验。

Phase 1 约束：`K ≤ 16`，`N ≤ 16`，`M` 任意（A 的每一行依次流过阵列）。

---

## 1. CLI 快速验证

在项目根目录执行，无需额外依赖（Python 标准库即可）：

```bash
cd /root/code/InferSim

# 4×4 GEMM 示例
python -m systolic_array.cli --m 4 --k 4 --n 4

# 16×16 满阵列示例
python -m systolic_array.cli --m 16 --k 16 --n 16
```

常用参数：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--m` | 矩阵 A 行数 (M) | 4 |
| `--k` | 内维 K | 4 |
| `--n` | 矩阵 B 列数 (N) | 4 |
| `--rows` | 阵列行数 | 16 |
| `--cols` | 阵列列数 | 16 |
| `--json` | 输出完整 JSON（含每 cycle snapshot） | 关闭 |

输出示例：

```
Array: 16x16
GEMM: A[4x4] x B[4x4]
Total cycles: 133
Snapshots: 133
```

---

## 2. Web UI（生产模式）

使用已构建的 React 静态资源 + FastAPI 后端，一条命令启动：

```bash
cd /root/code/InferSim
bash systolic_array/webui/run.sh
```

浏览器打开：**http://127.0.0.1:8765**

在界面输入 M / K / N，点击「开始仿真」后才会绘制 K×N 的 PE 网格。

可通过环境变量修改端口：

```bash
PORT=9000 bash systolic_array/webui/run.sh
```

### 依赖

Web 服务需要安装 FastAPI 相关包（仿真核心本身无第三方依赖）：

```bash
pip install -r systolic_array/webui/requirements.txt
```

若前端尚未构建，需先执行：

```bash
cd systolic_array/webui/frontend
npm install
npm run build
```

---

## 3. Web UI（开发模式）

前后端分离启动，前端支持热更新，适合改 UI 时使用。

**终端 1 — 后端 API（默认 8765）：**

```bash
cd /root/code/InferSim
export PYTHONPATH=.
python -m uvicorn systolic_array.webui.server:app --host 0.0.0.0 --port 8765
```

**终端 2 — 前端 dev server（默认 5173）：**

```bash
cd /root/code/InferSim/systolic_array/webui/frontend
npm install   # 首次需要
npm run dev
```

浏览器打开：**http://127.0.0.1:5173**

前端会将 `/api` 请求代理到 `http://127.0.0.1:8765`，因此两个进程需同时运行。

---

## 4. Python API 调用

```python
from systolic_array import run_simulation

result = run_simulation(m=4, k=4, n=4, rows=16, cols=16)

print(result.total_cycles)   # cycle 总数
print(result.dims)           # {"m": 4, "k": 4, "n": 4}
print(result.snapshots[0])   # 首 cycle 的 dataflow 快照
```

---

## 目录结构

```
systolic_array/
├── simulator.py          # 对外入口 run_simulation()
├── pe.py / array.py      # PE 与阵列
├── engine.py             # cycle 引擎
├── dataflow/             # Weight-Stationary 数据流
├── cli.py                # 命令行工具
└── webui/
    ├── server.py         # FastAPI 服务
    ├── run.sh            # 一键启动脚本
    ├── requirements.txt
    └── frontend/         # React + Vite 前端
```
