# 🎰 彩票智能选号系统 V5

技能 **lottery-analysis-smart-v5** —— 中国彩票（福彩3D / 排列三 / 大乐透）数据分析 + 智能选号。

## ⚠️ 数学声明（必读）

**彩票开奖是独立随机事件。** 每期开奖号码与历史数据无关，任何分析都不能提高命中率。

本项目的价值在于**选号质量**而非预测：让号码组合符合概率分布，避免极端组合，并在多模型之间做有依据的取舍。**不构成任何投注建议。**

## V5 是什么：一次「合并 + 纠错」

V5 把两条并行开发线合并为一条，并修掉了其中几个会实质性降低推荐质量的缺陷：

| 来源 | 贡献 |
|:---|:---|
| 本地 V3.2 线 | Dirichlet-Multinomial 共轭后验、精确期望遗漏、金码/银码去重与 0-9 覆盖校正 |
| QClaw V4 线 | V4 模块（3D/排三、大乐透）、Ensemble 集成、回测框架 |
| V5 新增修复 | 见下 |

### V5 修复的关键缺陷

1. **Ensemble 长期「假装用 V4」** —— 集成器调用的方法名写错（`.recommend()` vs `.recommend_v4()`），且把 V4 返回的 dict 当 list 解析，异常被静默吞掉，实际一直在单跑 V3。修复后三彩种均显示「来源：v3+v4」。
2. **大乐透前后区联合优化** —— 前区（35选5）与后区（12选2）是**独立抽取**的，把它们绑在同一个个体里做遗传优化并不合理。改为前区交给 GA、后区对全部 66 个组合独立评分后取 top-k。后区 0 中次数由 15/20 降到 3/40，V4 首次稳定优于 V3。
3. **遗传算法种群塌缩** —— 5 注曾出现前区 4 注雷同、后区 4 注相同。加入适应度共享（fitness sharing）、自适应变异与灾变重启。
4. **大乐透报告表格全 0** —— AC / 跨度 / 奇偶 / 三区 / 分数未计算，改为缺失时按号码现算。
5. **文案串台** —— 大乐透报告尾部误用 3D 的「组六 72%、和值 7-20」口径，改为按彩种分别输出。
6. **金码银码重复** —— V3 路径把「金码/银码」正反各渲染一遍，出现「金码 128/629、银码 629/128」，已改为各显示单号码。

## 三档模型

| 版本 | 说明 | 单次耗时 |
|:---|:---|:---|
| `v3`（默认） | Dirichlet 后验 + 指数衰减 + 复隔中 + 多策略共识 | ~0.03s |
| `v4` | Markov1/2、联合分布、几何遗漏、ACF、Regime、Bootstrap；大乐透另含 Copula 共现、PageRank、后区 Markov、遗传算法 | ~3s |
| `ensemble` | V3 + V4 各自生成候选，Borda 计数 + 归一化得分融合 | ~3s |

## 使用方法

```bash
cd scripts

# V3 + V4 集成（推荐默认）
python run_analysis.py --games 3d,35,85 --count 100 --version ensemble

# 指定单版本
python run_analysis.py --games 85 --count 100 --version v4

# 三档模型横向回测
python run_backtest.py --games 3d,35 --count 200 --train 100 --test 1 --step 5
python run_backtest.py --games 85 --count 300 --step 1 --min-tests 40 --seed 42
```

| 彩种代码 | 彩种 | 数据源 |
|:---:|:---|:---|
| `3d` | 福彩3D | cwl.gov.cn |
| `35` | 排列三 | sporttery.cn |
| `85` | 大乐透 | sporttery.cn |

## 已知限制

- 大乐透官方接口最多返回约 **117 期**，回测时脚本会自动收缩训练窗口以保证测试点数。
- 3D/排三在 20 个测试点的样本下，三档模型的差异仍在噪声范围内 —— 这符合彩票的数学性质。
- 回测 ROI 恒为负是正常现象；样本 <30 时奖金类指标不可信，应看命中分布这类连续指标。

## 目录结构

```
├── SKILL.md                    # skill 定义（输出模板、自检机制、API pitfall）
├── scripts/
│   ├── run_analysis.py         # 入口：数据获取 + 分析 + 格式化输出
│   ├── run_backtest.py         # 三档模型滚动回测对比
│   ├── lottery_analysis.py     # 数据获取（fetch_3d / fetch_pl3 / fetch_dlt）
│   ├── smart_selector.py       # 3D/排三 V3（Dirichlet 后验，V3.2 数学）
│   ├── smart_selector_v4.py    # 3D/排三 V4
│   ├── dlt_analysis.py         # 大乐透 V3（15 维评分）
│   ├── dlt_analysis_v4.py      # 大乐透 V4（Copula / PageRank / GA / 后区独立选优）
│   ├── ensemble.py             # V3+V4 Borda 集成
│   ├── backtest.py             # 回测框架
│   └── _archive/               # 历次升级的 patch 脚本归档
├── references/                 # 数据源、分析维度、选号策略参考文档
├── templates/output_format.md  # 输出格式模板
└── reports/                    # 历次分析报告存档
```

## License

MIT
