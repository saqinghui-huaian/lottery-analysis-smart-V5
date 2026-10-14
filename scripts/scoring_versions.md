# 彩票评分系统版本历史

## 当前版本

### V1 (2026-08-12) - 高级版，16策略+信息熵+贝叶斯 ⭐当前使用
- **文件**: `scoring_v1.py`
- **特点**:
  - 和值处理：5维分析（奇偶/大小/质合/012路/振幅）
  - 跨度处理：5维分析（质合/奇偶/大小/振幅/012路）
  - 高级统计: 信息熵、贝叶斯、时间序列
  - 用户偏好支持: 保守、激进、平衡三种风格
  - 16种候选策略
  - 总分范围0-100，区分度更高

## 保存版本

### V2 (GitHub原始版本) - 专业分析师维度整合版
- **文件**: `scoring_v2.py`
- **特点**:
  - 整合中彩网专业分析师维度
  - 和值5维分析: 奇偶/大小/质合/012路/振幅
  - 跨度5维分析: 质合/奇偶/大小/振幅/012路
  - 复隔中分析
  - 卡方检验、置信区间
  - 12种候选策略
  - 总分范围0-100，top10通常在40-60之间

## 使用方法

### V1（当前版本 ⭐）
```python
from scripts.scoring_v1 import LotteryScorerV1

scorer = LotteryScorerV1(data, style='balanced')
recommendations = scorer.recommend(10)
```

### V2（保存版本）
```python
from scripts.scoring_v2 import LotteryScorerV2

scorer = LotteryScorerV2(data)
recommendations = scorer.recommend(10)
```

## 版本对比

| 版本 | 和值处理 | 跨度处理 | 高级统计 | 候选策略 | 状态 |
|------|----------|----------|----------|----------|------|
| V1 | 5维分析 | 5维分析 | 信息熵/贝叶斯/时间序列 | 16种 | ⭐当前 |
| V2 | 5维分析 | 5维分析 | 卡方检验/置信区间 | 12种 | 保存 |

## 文件位置

```
C:\Users\21920\AppData\Local\hermes\skills\data-analysis\lottery-data-analysis\scripts\
├── scoring_v1.py          # V1版本 ⭐当前使用
├── scoring_v2.py          # V2版本（保存）
└── scoring_versions.md    # 本文件
```
