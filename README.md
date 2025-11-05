# RAG-LangGraph 楼宇问答系统

## 一、项目概述

系统通过 **LangGraph** 将多智能体编排为“自然语言 → SPARQL → 执行 → 分析/回答”的闭环。采用 **Brick Schema** 作为语义底座，支持**拓扑类**问题（房间/设备/点位关系）与**时序类**问题（均值、极值、趋势、指定时刻值）。



## 二、系统架构（LangGraph）
![alt text](<pic.png>)
> 说明：**回退链**是在“执行结果为 0 行”时**动态触发**。



## 三、核心节点（Agents）

| 模块 | 主要职责 |
|---|---|
| `app/nodes/rag_agent.py` | 向量检索 **TTL数据**，构造 `context`；必要时把 `context + hints` 喂给 LLM 生成 SPARQL。 |
| `app/nodes/normalize_time_agent.py` | 将“昨天/最近6小时/2025-10-20 14:00”等时间描述语句解析为标准的 UTC 时间窗/时点。 |
| `app/nodes/sparql_agent.py` | 生成 SPARQL：模板直出（拓扑类）/ LLM 直出 / 携带 `context` 的 RAG+LLM。 |
| `app/nodes/sparql_exec.py` | 执行 SPARQL，返回 `rows`、错误信息、耗时。 |
| `app/nodes/analysis_agent.py` | 时序统计（均值/极值/趋势/时点值）与拓扑计数等。 |
| `app/nodes/answer_agent.py` | 将结果组织为自然语言回答（中英双语），并做行数限制/空结果提示。 |
| `app/nodes/fewshot_agent.py` | **few-shot（HNSW/FAISS/Naive）示例检索**，支持多训练集 `train_data_*.json`。 |
| `app/graph.py` | **LangGraph 编排与路由**：执行主路 → 判空 → 逐级回退。 |
| `app/web_app.py` | Streamlit 前端：输入问题、展示 trace、SPARQL 历史与可视化流程。 |


## 四、回退策略（触发条件：执行 0 行）

- **0 级（首轮，非回退）**：模板/首轮生成（拓扑题优先模板；时序题用时序生成器）。  
- **第 1 级**：**LLM 直出**（不加上下文、不用示例）。
- **第 2 级**：**RAG + LLM**（检索 **知识片段** → 组装上下文 → 生成）。
- **第 3 级**：**few-shot（HNSW 挑示例）**  
  - `copy-nearest`：直接拷贝最近邻示例的 SPARQL；  
  - `few-shot + LLM`：将 Top-K *(问题→SPARQL)* 作为示例喂给 LLM 复写。  



## 五、数据准备（生成 `topology.ttl` & `timeseries.csv`）

目录：`data_generator/`。

### 1) 生成

```bash
cd data_generator
python data_generator.py 
```

输出：`data/topology.ttl`、`data/timeseries.csv`  。

可选：`--points-per-day`、`--days-back`。

### 2) 校验

```bash
cd data_generator
python data_validate.py
```

检查 SHACL、单位/量纲、拓扑规模。

---

## 六、Experiments（Schemes 与 Graph 回退的对应）

**Schemes 是静态流程；Graph 回退是动态触发。** 两者方法库一致，编排不同。

### 6.1 映射表

| Scheme | 流程/策略 | 对应 Graph 回退级别 |
|---|---|---|
| **Scheme 3 — template** | 模板/规则直出 | **0 级（首轮）** |
| **Scheme 4 — llm** | LLM 直出 | **第 1 级** |
| **Scheme 5 — rag_llm** | RAG + LLM（检索 TTL 片段 → 拼上下文 → 生成） | **第 2 级** |
| **Scheme 1 — fewshot_faiss** | few-shot（Top-K 示例；可拷最近邻/LLM 复写） | **第 3 级（实际选用了效果更好的Scheme2）** |
| **Scheme 2 — fewshot_hnsw** | few-shot + **HNSW/FAISS** 检索加速 | **第 3 级** |


### 6.2 示例结果

| Scheme | 严格 F1（macro） | 有结果率（nonempty） | 备注 |
|---|---:|---:|---|
| fewshot_faiss（全量, 无 LLM） | ~0.150 | ~0.383 | 朴素召回/FAISS，覆盖广，语法稳 | 
| fewshot_hnsw（全量, 无 LLM） | ~0.152 | ~0.389 | **HNSW** 加速选例 |
| fewshot_hnsw（抽样100, +LLM） | **~0.880** | ~0.78 | 多文件合并，count 题 Acc≈1.0 |
| template（全量） | ~0.00 | — | 覆盖面小 |
| llm（抽样100） | ~0.01 | — | 无上下文/示例 |
| rag_llm（抽样50） | ~0.34 | ~0.42 | 依赖检索片段质量 |



### 6.3 Scheme 2 细化结果（K=100, `--use_llm`, `--random_test`, `--seed 42`，后端：FAISS HNSW，M=32/efC=300/efS=256）

| 测试集 | 抽样K | 有效n | 解析错率 | 有结果率 | 严格F1(macro) | Overall严格 | Overall宽松 |   
|---|---:|---:|---:|---:|---:|---:|---:|
| test_data_1 | 80 | 75 | 0.0625 | 0.9250 | 0.9867 | 0.9867 | 0.9867 |   
| test_data_2 | 80 | 75 | 0.0625 | 0.9375 | 1.0000 | 1.0000 | 1.0000 |   
| test_data_3 | 100 | 79 | 0.2100 | 0.7600 | 0.9620 | 0.9620 | 0.9620 |   
| test_data_4 | 100 | 86 | 0.1400 | 0.8600 | 0.9651 | 0.9651 | 1.0000 |    
| test_data_5 | 100 | 78 | 0.2200 | 0.7600 | 0.9744 | 0.9744 | 0.9744 |   
| test_data_6 | 100 | 89 | 0.1100 | 0.5000 | 0.4494 | **0.4494** | 0.4719 |   
| 合并 | 560 | 482 | 0.1393 | 0.7804 | 0.8575 | 0.8797 | 0.8900 |  

> 说明：具体数值会随采样与提示词变化而波动；
---

## 七、快速开始

```bash
# 启动 Web UI
streamlit run app/web_app.py
```

示例提问：
- “整栋楼有多少个房间？”
- “305 房昨天平均温度与趋势？”
- “10 月 20 日 14:00 305 房温度是多少？”
- “哪些房间没有CO2浓度传感器？”
---
