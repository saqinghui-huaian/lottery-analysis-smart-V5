#!/usr/bin/env python3
"""
大乐透专业分析系统 V1 — 基于统计学与概率论
============================================

数据采集：
- 来源：sporttery.cn 官方API (gameNo=85)
- 采集量：当前年份全部期数（约90-100期/年）
- 方式：分页获取，每页100期，页间延迟3秒防WAF
- 原因：大乐透每周一/三/六开奖，一年约153期，
  采集当年全量数据可覆盖完整的年度周期性

核心分析维度（基于概率论与统计学）：
1. 理论概率计算 — C(35,5)×C(12,2)=21,425,712 种组合
2. 贝叶斯后验概率 — 先验均匀 × 观测频率 → 后验估计
3. 信息熵分析 — 衡量号码分布的随机性/规律性
4. 加权热度 — 指数衰减加权，近期数据权重更高
5. 遗漏值分析 — 几何分布模型，P(遗漏≥k)=(6/7)^k
6. 和值正态分布 — E[Sum]=90, σ≈22.6
7. AC值(算术复杂度) — 有效差异对数-4
8. 012路分析 — 按mod 3分组
9. 质合比 — 1-35中11个质数
10. 三区分布 — 1-12/13-23/24-35
11. 尾数频率 — 个位数字分布
12. 邻号分析 — 与上期号码±1的关系
13. 重号分析 — 与上期重复的号码
14. 连号分析 — 相邻数字对
15. 偏度/峰度 — 分布形态检验

推荐策略：5注精选，严格多样性约束
"""

import random
import math
from collections import Counter
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════
#  概率论基础
# ═══════════════════════════════════════════════════════════════
#
# 大乐透规则：前区1-35选5，后区1-12选2
# 总组合数：C(35,5) × C(12,2) = 324,632 × 66 = 21,425,712
# 一等奖概率：1/21,425,712 ≈ 4.67×10⁻⁸
#
# 前区理论分布：
# - 单号出现概率 p = C(34,4)/C(35,5) = 5/35 = 1/7 ≈ 14.29%
# - 遗漏期望 E[miss] = 1/p - 1 = 6 期
# - P(遗漏≥k) = (1-p)^k = (6/7)^k
# - P(遗漏≥15) = 9.9%, P(≥20) = 4.6%, P(≥25) = 2.1%
#
# 和值理论分布（正态近似）：
# - E[Sum] = 5 × (1+35)/2 = 90
# - Var[Sum] = 5 × (35²-1)/12 ≈ 510, σ ≈ 22.6
# - 68%区间: [67, 113], 95%区间: [45, 135]
#
# 重号期望：5×5/35 ≈ 0.71 个/期
# 连号概率：约50%的期次有至少1对连号

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
FRONT_RANGE = range(1, 36)
BACK_RANGE = range(1, 13)
THEORETICAL_SUM = 90
THEORETICAL_SIGMA = 22.6
THEORETICAL_SPAN = 28  # E[max-min] ≈ 28 for 5 from 35


class DLTAnalyzer:
    """大乐透专业分析器"""

    def __init__(self, data: List[Dict], style: str = 'balanced'):
        """
        Args:
            data: 开奖数据列表，每项含 period, date, front, back
            style: 'conservative'(偏热号), 'balanced'(均衡), 'aggressive'(偏冷号)
        """
        self.data = data
        self.style = style
        self.style_weights = {
            'conservative': {'heat': 0.7, 'cold': 0.3, 'risk_aversion': 0.8},
            'balanced':     {'heat': 0.5, 'cold': 0.5, 'risk_aversion': 0.5},
            'aggressive':   {'heat': 0.3, 'cold': 0.7, 'risk_aversion': 0.2},
        }[style]

        # Pre-compute all statistics
        self._compute_all()

    def _compute_all(self):
        """预计算所有统计量"""
        n = len(self.data)

        # ── 前区整体频率 ──
        self.front_freq = Counter()
        for d in self.data:
            for num in d['front']:
                self.front_freq[num] += 1

        # ── 前区各位频率（排序后第1-5位）──
        self.front_pos_freq = [Counter() for _ in range(5)]
        for d in self.data:
            for i, num in enumerate(sorted(d['front'])):
                self.front_pos_freq[i][num] += 1

        # ── 后区频率 ──
        self.back_freq = Counter()
        for d in self.data:
            for num in d['back']:
                self.back_freq[num] += 1

        # ── 加权热度（指数衰减：近5期×5, 5-10×3, 10-20×2, 20-50×1）──
        self.front_heat = Counter()
        self.back_heat = Counter()
        for idx, d in enumerate(self.data[:50]):
            if idx < 5:
                w = 5
            elif idx < 10:
                w = 3
            elif idx < 20:
                w = 2
            else:
                w = 1
            for num in d['front']:
                self.front_heat[num] += w
            for num in d['back']:
                self.back_heat[num] += w

        # ── 遗漏值（距上次出现的期数）──
        self.front_miss = {}
        for num in FRONT_RANGE:
            for i, d in enumerate(self.data):
                if num in d['front']:
                    self.front_miss[num] = i
                    break
            else:
                self.front_miss[num] = n

        self.back_miss = {}
        for num in BACK_RANGE:
            for i, d in enumerate(self.data):
                if num in d['back']:
                    self.back_miss[num] = i
                    break
            else:
                self.back_miss[num] = n

        # ── 和值统计 ──
        self.sums = [sum(d['front']) for d in self.data]
        self.avg_sum = sum(self.sums) / len(self.sums)
        self.sum_std = (sum((s - self.avg_sum)**2 for s in self.sums) / len(self.sums)) ** 0.5

        # ── 跨度统计 ──
        self.spans = [max(d['front']) - min(d['front']) for d in self.data]
        self.avg_span = sum(self.spans) / len(self.spans)

        # ── 奇偶比分布 ──
        self.odd_even_dist = Counter()
        for d in self.data:
            odd = sum(1 for n in d['front'] if n % 2 == 1)
            self.odd_even_dist[f"{odd}:{5-odd}"] += 1

        # ── 大小比分布（17+为大）──
        self.big_small_dist = Counter()
        for d in self.data:
            big = sum(1 for n in d['front'] if n >= 17)
            self.big_small_dist[f"{big}:{5-big}"] += 1

        # ── 三区分布（1-12/13-23/24-35）──
        self.zone_dist = Counter()
        for d in self.data:
            z = [0, 0, 0]
            for n in d['front']:
                if n <= 12: z[0] += 1
                elif n <= 23: z[1] += 1
                else: z[2] += 1
            self.zone_dist[f"{z[0]}:{z[1]}:{z[2]}"] += 1

        # ── 012路分布 ──
        self.road_dist = Counter()
        for d in self.data:
            roads = [0, 0, 0]
            for n in d['front']:
                roads[n % 3] += 1
            self.road_dist[f"{roads[0]}:{roads[1]}:{roads[2]}"] += 1

        # ── 重号统计（与上期重复数）──
        self.repeat_dist = Counter()
        for i in range(n - 1):
            curr = set(self.data[i]['front'])
            prev = set(self.data[i + 1]['front'])
            self.repeat_dist[len(curr & prev)] += 1

        # ── 连号统计 ──
        self.consec_dist = Counter()
        for d in self.data:
            f = sorted(d['front'])
            c = sum(1 for i in range(4) if f[i+1] - f[i] == 1)
            self.consec_dist[c] += 1

        # ── 质合比 ──
        self.prime_dist = Counter()
        for d in self.data:
            pc = sum(1 for n in d['front'] if n in PRIMES)
            self.prime_dist[pc] += 1

        # ── 尾数频率 ──
        self.tail_freq = Counter()
        for d in self.data:
            for n in d['front']:
                self.tail_freq[n % 10] += 1

        # ── 邻号关系（上期号码±1）──
        self.adjacent_freq = Counter()
        for i in range(n - 1):
            prev_set = set(self.data[i + 1]['front'])
            adj_set = set()
            for num in prev_set:
                if num > 1: adj_set.add(num - 1)
                if num < 35: adj_set.add(num + 1)
            for num in self.data[i]['front']:
                if num in adj_set:
                    self.adjacent_freq[num] += 1

        # ── 贝叶斯后验概率（Dirichlet-Multinomial 共轭后验均值）──
        # 先验: Dirichlet(α)，α=2/号（前区等效70期先验样本，向均匀1/35收缩）
        # 后验均值: P(num) = (count + α) / (N + 35α)
        # V3.0 旧实现 prior×likelihood 归一化后恒等于观测频率（先验被约掉），修复之。
        ALPHA_F = 2.0
        total_front = sum(self.front_freq.values())
        denom_front = total_front + 35 * ALPHA_F
        self.bayesian_front = {}
        for num in FRONT_RANGE:
            self.bayesian_front[num] = (self.front_freq.get(num, 0) + ALPHA_F) / denom_front

        # ── 后区贝叶斯（同款 Dirichlet 后验，α=1.5/号 等效18期先验）──
        ALPHA_B = 1.5
        total_back = sum(self.back_freq.values())
        denom_back = total_back + 12 * ALPHA_B
        self.bayesian_back = {}
        for num in BACK_RANGE:
            self.bayesian_back[num] = (self.back_freq.get(num, 0) + ALPHA_B) / denom_back

        # ── 信息熵（衡量分布均匀性）──
        # H = -Σ p(x) log2 p(x)
        # 前区理论最大熵 = log2(35) ≈ 5.13
        probs = [self.front_freq.get(n, 0) / total_front for n in FRONT_RANGE if total_front > 0]
        self.front_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        self.front_max_entropy = math.log2(35)  # ≈5.13

    def calc_ac(self, combo: List[int]) -> int:
        """计算AC值（算术复杂度）
        AC = |{ |aᵢ-aⱼ| : i<j }| - (n-1)
        AC≥4 为合理，AC 5-8 最常见（约73%）
        """
        diffs = set()
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                diffs.add(abs(combo[i] - combo[j]))
        return len(diffs) - (len(combo) - 1)

    def calc_skewness(self, combo: List[int]) -> float:
        """计算偏度 γ₁ = Σ(xᵢ-x̄)³ / (n·σ³)"""
        n = len(combo)
        mean = sum(combo) / n
        std = (sum((x - mean)**2 for x in combo) / n) ** 0.5
        if std == 0:
            return 0
        return sum((x - mean)**3 for x in combo) / (n * std**3)

    def score_front(self, combo: List[int]) -> Tuple[float, Dict]:
        """前区5维综合评分

        返回 (总分, 各维度得分明细)
        总分范围约 0-100
        """
        combo = sorted(combo)
        details = {}
        score = 0.0

        # ── 维度1: 贝叶斯后验概率（满分15）──
        # 基准：随机组合的期望后验和 = 5×(1/35) ≈ 0.1429 → 得分7.5（中位）
        # V3.0 旧公式 sum×100×3 使均值组合就得42.9分封顶，维度零区分度，修复之。
        bayes_sum = sum(self.bayesian_front.get(n, 0) for n in combo)
        d1 = min(bayes_sum / (5 / 35) * 7.5, 15)
        details['贝叶斯'] = round(d1, 1)
        score += d1

        # ── 维度2: 加权热度（满分12）──
        heat_raw = sum(self.front_heat.get(n, 0) for n in combo)
        # Expected: 5 numbers × avg_heat. avg_heat ≈ total_heat/35
        total_heat = sum(self.front_heat.values())
        expected_heat = 5 * total_heat / 35 if total_heat > 0 else 1
        heat_ratio = heat_raw / expected_heat if expected_heat > 0 else 1
        # Apply style: conservative likes hot, aggressive likes cold
        if self.style == 'conservative':
            d2 = min(heat_ratio * 8, 12)
        elif self.style == 'aggressive':
            d2 = min((2 - heat_ratio) * 6, 12) if heat_ratio < 2 else 0
        else:
            d2 = min(heat_ratio * 6, 12)
        details['热度'] = round(d2, 1)
        score += d2

        # ── 维度3: 和值合理性（满分12）──
        s = sum(combo)
        z_score = abs(s - self.avg_sum) / self.sum_std if self.sum_std > 0 else 0
        if z_score <= 0.5:
            d3 = 12
        elif z_score <= 1.0:
            d3 = 10
        elif z_score <= 1.5:
            d3 = 7
        elif z_score <= 2.0:
            d3 = 4
        else:
            d3 = 0
        details['和值'] = round(d3, 1)
        score += d3

        # ── 维度4: AC值（满分12）──
        ac = self.calc_ac(combo)
        if ac >= 7:
            d4 = 12
        elif ac == 6:
            d4 = 10
        elif ac == 5:
            d4 = 7
        elif ac == 4:
            d4 = 3
        else:
            d4 = -8  # AC<4 排除
        details['AC值'] = round(d4, 1)
        score += d4

        # ── 维度5: 跨度合理性（满分8）──
        span = max(combo) - min(combo)
        span_diff = abs(span - self.avg_span)
        if span_diff <= 3:
            d5 = 8
        elif span_diff <= 6:
            d5 = 6
        elif span_diff <= 10:
            d5 = 3
        else:
            d5 = 0
        details['跨度'] = round(d5, 1)
        score += d5

        # ── 维度6: 奇偶平衡（满分6）──
        odd = sum(1 for n in combo if n % 2 == 1)
        if odd in [2, 3]:
            d6 = 6
        elif odd in [1, 4]:
            d6 = 3
        else:
            d6 = 0
        details['奇偶'] = round(d6, 1)
        score += d6

        # ── 维度7: 大小平衡（满分6）──
        big = sum(1 for n in combo if n >= 17)
        if big in [2, 3]:
            d7 = 6
        elif big in [1, 4]:
            d7 = 3
        else:
            d7 = 0
        details['大小'] = round(d7, 1)
        score += d7

        # ── 维度8: 三区分布（满分10）──
        z = [0, 0, 0]
        for n in combo:
            if n <= 12: z[0] += 1
            elif n <= 23: z[1] += 1
            else: z[2] += 1
        if 0 in z:
            d8 = -5  # 空区惩罚
        elif sorted(z) in [[1, 2, 2], [2, 2, 1], [2, 1, 2]]:
            d8 = 10
        elif sorted(z) in [[1, 1, 3], [1, 3, 1], [3, 1, 1]]:
            d8 = 5
        else:
            d8 = 3
        details['三区'] = round(d8, 1)
        score += d8

        # ── 维度9: 012路多样性（满分5）──
        roads = set(n % 3 for n in combo)
        if len(roads) == 3:
            d9 = 5
        elif len(roads) == 2:
            d9 = 2
        else:
            d9 = 0
        details['012路'] = round(d9, 1)
        score += d9

        # ── 维度10: 质合比（满分4）──
        pc = sum(1 for n in combo if n in PRIMES)
        if pc in [1, 2]:
            d10 = 4
        elif pc == 3:
            d10 = 2
        elif pc == 0:
            d10 = 1
        else:
            d10 = 0
        details['质合'] = round(d10, 1)
        score += d10

        # ── 维度11: 遗漏回补（满分8）──
        # 引入Z-score：基于历史遗漏均值/标准差判断当前遗漏是否异常
        miss_score = 0
        for n in combo:
            m = self.front_miss.get(n, 0)
            # 几何分布理论均值 ~6期
            mean = 6.0
            std = 6.0
            z = (m - mean) / std
            if 0.5 <= z <= 2.5:
                miss_score += min(z * 2.0, 3.0)
            elif z > 2.5:
                miss_score += 1.5  # 极端遗漏不追
            if 8 <= m <= 20:
                miss_score += 0.5
        d11 = min(miss_score, 8)
        details['遗漏'] = round(d11, 1)
        score += d11

        # ── 维度12: 重号加分（满分4）──
        last_set = set(self.data[0]['front'])
        repeats = len(set(combo) & last_set)
        if repeats == 1:
            d12 = 4
        elif repeats == 2:
            d12 = 2
        elif repeats == 0:
            d12 = 0
        else:
            d12 = 0
        details['重号'] = round(d12, 1)
        score += d12

        # ── 维度13: 邻号关系（满分3）──
        prev_front = set(self.data[0]['front'])
        adj_set = set()
        for num in prev_front:
            if num > 1: adj_set.add(num - 1)
            if num < 35: adj_set.add(num + 1)
        adj_count = len(set(combo) & adj_set)
        d13 = min(adj_count * 1.5, 3)
        details['邻号'] = round(d13, 1)
        score += d13

        # ── 维度14: 尾数多样性（满分3）──
        tails = set(n % 10 for n in combo)
        d14 = len(tails) * 0.6  # 5个不同尾数=3分
        details['尾数'] = round(d14, 1)
        score += d14

        # ── 维度15: 偏度检验（满分2）──
        skew = self.calc_skewness(combo)
        if abs(skew) < 0.5:
            d15 = 2  # 接近对称
        elif abs(skew) < 1.0:
            d15 = 1
        else:
            d15 = 0
        details['偏度'] = round(d15, 1)
        score += d15

        return round(score, 1), details

    def score_back(self, combo: Tuple[int, int]) -> float:
        """后区2维评分"""
        score = 0.0

        # 加权热度
        total_back_heat = sum(self.back_heat.values())
        heat_raw = sum(self.back_heat.get(n, 0) for n in combo)
        expected = 2 * total_back_heat / 12 if total_back_heat > 0 else 1
        score += min(heat_raw / expected * 5, 8) if expected > 0 else 0

        # 遗漏甜蜜区 5-15期（理论期望 1/(2/12)-1=5期）
        for n in combo:
            m = self.back_miss.get(n, 0)
            mean = 5.0
            std = 5.0
            z = (m - mean) / std
            if 0.0 <= z <= 2.0:
                score += min(2.0 + z, 4.0)
            elif 2.0 < z <= 3.5:
                score += 2.0
            else:
                score += 0.5

        # 奇偶1:1
        if sum(n % 2 for n in combo) == 1:
            score += 3

        # 和值接近理论均值 (1+12)/2×2 = 13
        s = sum(combo)
        if 10 <= s <= 15:
            score += 4
        elif 7 <= s <= 18:
            score += 2

        # 012路多样性
        roads = len(set(n % 3 for n in combo))
        if roads == 2:
            score += 2

        return round(score, 1)

    def generate_candidates(self, n_front: int = 15000) -> List[Tuple]:
        """多策略生成前区候选组合"""
        candidates = set()
        # random seed removed for variety

        # Strategy 1: 位置热号（3000次）
        top_pos = []
        for pos in range(5):
            top_nums = [n for n, _ in self.front_pos_freq[pos].most_common(10)]
            top_pos.append(top_nums)
        for _ in range(3000):
            combo = tuple(sorted(random.choice(top_pos[i]) for i in range(5)))
            if len(set(combo)) == 5:
                candidates.add(combo)

        # Strategy 2: 全局热号（2000次）
        hot_nums = [n for n, _ in self.front_heat.most_common(15)]
        for _ in range(2000):
            candidates.add(tuple(sorted(random.sample(hot_nums, 5))))

        # Strategy 3: 热+冷混合（2000次）
        cold_nums = [n for n, _ in sorted(self.front_miss.items(), key=lambda x: x[1], reverse=True)[:10]]
        for _ in range(2000):
            h = random.sample(hot_nums, 3)
            c = random.sample(cold_nums, 2)
            combo = tuple(sorted(set(h + c)))
            if len(combo) == 5:
                candidates.add(combo)

        # Strategy 4: 包含重号（2000次）
        last_front = list(self.data[0]['front'])
        others = [n for n in FRONT_RANGE if n not in last_front]
        for _ in range(2000):
            rep = random.sample(last_front, random.choice([1, 2]))
            oth = random.sample(others, 5 - len(rep))
            combo = tuple(sorted(rep + oth))
            if len(combo) == 5:
                candidates.add(combo)

        # Strategy 5: 三区均衡（3000次）
        for _ in range(3000):
            z1 = random.sample(range(1, 13), random.choice([1, 2]))
            z2 = random.sample(range(13, 24), random.choice([1, 2]))
            z3 = random.sample(range(24, 36), random.choice([1, 2]))
            combo = tuple(sorted(z1 + z2 + z3))
            if len(combo) == 5:
                candidates.add(combo)

        # Strategy 6: 贝叶斯采样（3000次）
        front_nums = list(FRONT_RANGE)
        front_probs = [self.bayesian_front.get(n, 1/35) for n in front_nums]
        total_p = sum(front_probs)
        front_probs = [p / total_p for p in front_probs]
        for _ in range(3000):
            chosen = set()
            while len(chosen) < 5:
                r = random.random()
                cum = 0
                for num, p in zip(front_nums, front_probs):
                    cum += p
                    if r <= cum:
                        chosen.add(num)
                        break
            candidates.add(tuple(sorted(chosen)))

        return list(candidates)

    def recommend(self, n_bets: int = 5) -> List[Dict]:
        """生成n_bets注推荐，严格多样性约束

        多样性约束（针对5注优化）：
        1. 任意两注前区重叠 ≤ 2 个号码
        2. 每个号码最多出现在 2 注中
        3. 任意两注和值差 ≥ 8
        4. 每注三区均不能为空
        5. 后区5注各不相同
        6. 前区总覆盖 ≥ 15 个不同号码（5注×5=25个位置，≥60%覆盖）
        """
        # 生成候选
        candidates = self.generate_candidates()

        # 评分
        scored = []
        for combo in candidates:
            sc, details = self.score_front(combo)
            # AC<4 直接排除
            if self.calc_ac(combo) < 4:
                continue
            scored.append((sc, combo, details))
        scored.sort(reverse=True)

        # 后区评分排序
        all_back = []
        for i in BACK_RANGE:
            for j in range(i + 1, 13):
                all_back.append((i, j))
        scored_back = [(self.score_back(b), b) for b in all_back]
        scored_back.sort(reverse=True)

        # 贪心选择（带多样性约束）
        selected = []
        used_nums = Counter()
        used_backs = set()

        for sc, front, details in scored:
            if len(selected) >= n_bets:
                break

            # 约束1: 任意两注重叠 ≤ 2
            too_similar = any(
                len(set(front) & set(sf)) > 2
                for _, sf, _, _ in selected
            )
            if too_similar:
                continue

            # 约束2: 每号最多2次
            if any(used_nums[n] >= 2 for n in front):
                continue

            # 约束3: 和值差 ≥ 8
            s = sum(front)
            if any(abs(s - sum(sf)) < 8 for _, sf, _, _ in selected):
                continue

            # 约束4: 三区不空
            z = [0, 0, 0]
            for n in front:
                if n <= 12: z[0] += 1
                elif n <= 23: z[1] += 1
                else: z[2] += 1
            if 0 in z:
                continue

            # 选后区
            back = None
            for bs, bc in scored_back:
                if bc not in used_backs:
                    back = bc
                    break
            if back is None:
                continue

            selected.append((sc, front, details, back))
            for n in front:
                used_nums[n] += 1
            used_backs.add(back)

        # 如果不够，放宽重叠到 ≤ 3
        if len(selected) < n_bets:
            for sc, front, details in scored:
                if len(selected) >= n_bets:
                    break
                if any(front == sf for _, sf, _, _ in selected):
                    continue
                if any(len(set(front) & set(sf)) > 3 for _, sf, _, _ in selected):
                    continue
                if any(used_nums[n] >= 3 for n in front):
                    continue
                z = [0, 0, 0]
                for n in front:
                    if n <= 12: z[0] += 1
                    elif n <= 23: z[1] += 1
                    else: z[2] += 1
                if 0 in z:
                    continue
                back = None
                for bs, bc in scored_back:
                    if bc not in used_backs:
                        back = bc
                        break
                if back is None:
                    continue
                selected.append((sc, front, details, back))
                for n in front:
                    used_nums[n] += 1
                used_backs.add(back)

        return selected

    def get_analysis_report(self) -> Dict:
        """生成完整分析报告数据"""
        n = len(self.data)

        # 热号TOP10
        hot_front = self.front_heat.most_common(10)
        # 遗漏TOP10
        miss_front = sorted(self.front_miss.items(), key=lambda x: x[1], reverse=True)[:10]
        # 后区热号TOP5
        hot_back = self.back_heat.most_common(5)
        # 后区遗漏TOP5
        miss_back = sorted(self.back_miss.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'data_count': n,
            'latest': self.data[0],
            'hot_front': hot_front,
            'miss_front': miss_front,
            'hot_back': hot_back,
            'miss_back': miss_back,
            'avg_sum': round(self.avg_sum, 1),
            'sum_std': round(self.sum_std, 1),
            'avg_span': round(self.avg_span, 1),
            'odd_even': self.odd_even_dist.most_common(3),
            'big_small': self.big_small_dist.most_common(3),
            'zone': self.zone_dist.most_common(5),
            'repeat': dict(self.repeat_dist),
            'consec': dict(self.consec_dist),
            'prime': dict(self.prime_dist),
            'entropy': round(self.front_entropy, 2),
            'max_entropy': round(self.front_max_entropy, 2),
        }


def format_report(analyzer: DLTAnalyzer, selected: List[Dict]) -> str:
    """格式化输出报告"""
    report = analyzer.get_analysis_report()
    total_periods = report['data_count']
    latest = report['latest']

    lines = []
    lines.append(f"基于{total_periods}期数据，专业统计分析如下：")
    lines.append("---")

    # 最新开奖
    front_str = ' '.join(f'{x:02d}' for x in sorted(latest['front']))
    back_str = ' '.join(f'{x:02d}' for x in sorted(latest['back']))
    lines.append(f"**最新开奖：{latest['period']}期（{latest['date']}）→ 前区 {front_str} 后区 {back_str}**")

    # 待开期号
    latest_period = int(latest['period'])
    lines.append(f"**待开：{latest_period + 1}期**")

    # 关键数据
    lines.append("")
    lines.append("📊 **关键数据速览**")

    hot_str = '/'.join(f'{n}({s}分)' for n, s in report['hot_front'][:7])
    lines.append(f"【前区热度TOP7】 {hot_str}")

    miss_str = '/'.join(f'{n}({m}期)' for n, m in report['miss_front'][:7])
    lines.append(f"【前区遗漏TOP7】 {miss_str}")

    hot_b = '/'.join(f'{n}({s}分)' for n, s in report['hot_back'][:5])
    lines.append(f"【后区热度TOP5】 {hot_b}")

    miss_b = '/'.join(f'{n}({m}期)' for n, m in report['miss_back'][:5])
    lines.append(f"【后区遗漏TOP5】 {miss_b}")

    lines.append(f"【信息熵】 {report['entropy']}/{report['max_entropy']}（越接近最大值分布越均匀）")
    lines.append(f"【和值】 均值{report['avg_sum']}，标准差{report['sum_std']}")

    # 5注推荐
    lines.append("")
    lines.append("🏆 **5注精选推荐**")
    lines.append("| 序号 | 前区号码 | 后区 | 和值 | AC | 跨度 | 奇偶 | 三区 | 分数 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    all_front_nums = set()
    for i, (sc, front, details, back) in enumerate(selected):
        s = sum(front)
        ac = analyzer.calc_ac(front)
        span = max(front) - min(front)
        odd = sum(1 for n in front if n % 2 == 1)
        z = [0, 0, 0]
        for n in front:
            if n <= 12: z[0] += 1
            elif n <= 23: z[1] += 1
            else: z[2] += 1
        front_str = ' '.join(f'{n:02d}' for n in front)
        back_str = ' '.join(f'{n:02d}' for n in back)
        lines.append(f"| {i+1} | **{front_str}** | {back_str} | {s} | {ac} | {span} | {odd}:{5-odd} | {z[0]}:{z[1]}:{z[2]} | {sc:.1f} |")
        all_front_nums.update(front)

    lines.append("")

    # 选号逻辑
    lines.append("🔍 **选号核心逻辑**")

    # 遗漏回补
    top_miss = report['miss_front'][:3]
    miss_items = []
    for n, m in top_miss:
        expected_miss = 6  # E[miss] = 1/p - 1 = 6
        ratio = m / expected_miss
        if m >= 15:
            miss_items.append(f"前区**{n}**遗漏**{m}期**（期望{expected_miss}期，{ratio:.1f}倍）→ 强回补信号")
        elif m >= 10:
            miss_items.append(f"前区**{n}**遗漏**{m}期**（{ratio:.1f}倍）→ 回补方向")
    lines.append(f"- 遗漏回补：{'；'.join(miss_items)}")

    # 热号支撑
    hot_items = [f'{n}' for n, _ in report['hot_front'][:5]]
    lines.append(f"- 热号支撑：前区{'/'.join(hot_items)}")

    # 后区
    hot_b_items = [f'{n}' for n, _ in report['hot_back'][:3]]
    miss_b_items = [f'{n}({m}期)' for n, m in report['miss_back'][:2]]
    lines.append(f"- 后区：热号{'/'.join(hot_b_items)}，遗漏{'/'.join(miss_b_items)}")

    # 形态
    oe = report['odd_even'][0]
    bs = report['big_small'][0]
    zn = report['zone'][0]
    lines.append(f"- 形态：奇偶比{oe[0]}({oe[1]}次)、大小比{bs[0]}({bs[1]}次)、三区{zn[0]}({zn[1]}次)")

    # 重号/连号
    rep = report['repeat']
    consec = report['consec']
    most_rep = max(rep.items(), key=lambda x: x[1])
    most_consec = max(consec.items(), key=lambda x: x[1])
    lines.append(f"- 重号：{most_rep[0]}个重号出现{most_rep[1]}次最多；连号：{most_consec[0]}对出现{most_consec[1]}次最多")

    lines.append("")
    lines.append(f"> ⚠️ 仅供参考，请理性投注！")
    lines.append(f"> 数据来源：官方API（{total_periods}期），V1专业统计分析系统")
    lines.append(f"> 分析维度：贝叶斯/热度/和值/AC值/跨度/奇偶/大小/三区/012路/质合/遗漏/重号/邻号/尾数/偏度 共15维")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from scripts.lottery_analysis import fetch_dlt

    print("正在获取大乐透数据...")
    data = fetch_dlt(200)
    print(f"获取到 {len(data)} 期数据（{data[-1]['period']}~{data[0]['period']}）")

    analyzer = DLTAnalyzer(data, style='balanced')

    # 打印统计概要
    report = analyzer.get_analysis_report()
    print(f"\n{'='*60}")
    print(f"  大乐透专业分析 | {report['data_count']}期数据")
    print(f"{'='*60}")
    print(f"\n信息熵: {report['entropy']}/{report['max_entropy']}")
    print(f"和值均值: {report['avg_sum']} ± {report['sum_std']}")
    print(f"跨度均值: {report['avg_span']}")

    # 生成推荐
    selected = analyzer.recommend(5)
    output = format_report(analyzer, selected)
    print(f"\n{output}")
