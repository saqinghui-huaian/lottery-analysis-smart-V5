#!/usr/bin/env python3
"""
大乐透专业分析系统 V4 — 网络共现、马尔可夫配对与遗传优化
==========================================================

在 V3 的 15 维概率统计评分基础上，V4 引入以下增强：

1. 前区 Copula / 共现网络
   - 构造 35×35 共现衰减矩阵，近期开奖权重更高
   - 用 PageRank 识别网络中的“核心号码”

2. 后区联合分布
   - 12×12 后区共现矩阵
   - 后区第一球 → 第二球的条件概率（Markov 配对）

3. 形态模式量化
   - 同尾对频率（tail_pattern）
   - 相邻号码差值分布（diff_pattern）
   - 连号条件概率（consecutive_cond）

4. Bootstrap 统计
   - 对前区和值、后区和值、AC 值做有放回重采样
   - 输出均值、标准差与分位数，用于评估推荐组合的统计位置

5. 遗传算法推荐
   - 编码：前区 5 码 + 后区 2 码
   - 适应度综合前区评分、后区评分、种群多样性、区间/尾数覆盖
   - 锦标赛选择、单点交叉、5% 变异、精英保留

重要声明：
-----------
大乐透每次开奖均为独立随机事件，总组合数 C(35,5)×C(12,2)=21,425,712。
本分析器仅基于历史观测数据优化“选号质量”——即让推荐组合在统计分布上
更接近历史形态，避免极端、拥挤、低覆盖的组合，**绝不声称能够预测未来开奖结果**。
"""

import random
import math
from collections import Counter
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════
#  常量与概率论基础
# ═══════════════════════════════════════════════════════════════

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
FRONT_RANGE = range(1, 36)
BACK_RANGE = range(1, 13)
THEORETICAL_SUM = 90
THEORETICAL_SIGMA = 22.6
THEORETICAL_SPAN = 28

# 近期衰减权重：与 V3 保持一致，便于横向对比
DECAY_WEIGHTS = [(5, 5), (10, 3), (20, 2), (50, 1)]


def _decay_weight(idx: int) -> int:
    """根据距当前期数的索引返回衰减权重。"""
    for bound, w in DECAY_WEIGHTS:
        if idx < bound:
            return w
    return 0


class DLTAnalyzerV4:
    """大乐透 V4 分析器：共现网络 + 遗传优化。"""

    def __init__(self, data: List[Dict], style: str = 'adaptive'):
        """
        Args:
            data: 开奖数据列表，每项含 period, date, front, back。
                  默认 data[0] 为最新一期，按时间倒序排列。
            style: 'adaptive' | 'conservative' | 'balanced' | 'aggressive'。
                   adaptive 会根据近期熵自动选择保守/均衡/激进风格。
        """
        self.data = data
        self.style_input = style
        self.style = style

        # 预计算所有统计量
        self._compute_all()

        # 自适应风格在统计计算完成后确定
        if self.style == 'adaptive':
            self.style = self._detect_style()

        self.style_weights = {
            'conservative': {'heat': 0.7, 'cold': 0.3, 'risk_aversion': 0.8},
            'balanced':     {'heat': 0.5, 'cold': 0.5, 'risk_aversion': 0.5},
            'aggressive':   {'heat': 0.3, 'cold': 0.7, 'risk_aversion': 0.2},
        }[self.style]

    # ─────────────────────────────────────────────────────────────
    #  统计量计算
    # ─────────────────────────────────────────────────────────────

    def _compute_all(self):
        """预计算 V3 全部统计量以及 V4 新增统计量。"""
        n = len(self.data)
        lookback = min(n, 50)

        # ── 基础频率 ──
        self.front_freq = Counter()
        self.front_pos_freq = [Counter() for _ in range(5)]
        self.back_freq = Counter()
        for d in self.data:
            front_sorted = sorted(d['front'])
            for num in front_sorted:
                self.front_freq[num] += 1
            for i, num in enumerate(front_sorted):
                self.front_pos_freq[i][num] += 1
            for num in d['back']:
                self.back_freq[num] += 1

        # ── 加权热度（指数衰减）──
        self.front_heat = Counter()
        self.back_heat = Counter()
        for idx, d in enumerate(self.data[:lookback]):
            w = _decay_weight(idx)
            if w == 0:
                continue
            for num in d['front']:
                self.front_heat[num] += w
            for num in d['back']:
                self.back_heat[num] += w

        # ── 遗漏值 ──
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

        # ── 和值、跨度、形态分布 ──
        self.sums = [sum(d['front']) for d in self.data]
        self.avg_sum = sum(self.sums) / len(self.sums) if self.sums else THEORETICAL_SUM
        self.sum_std = self._std(self.sums) or THEORETICAL_SIGMA

        self.spans = [max(d['front']) - min(d['front']) for d in self.data]
        self.avg_span = sum(self.spans) / len(self.spans) if self.spans else THEORETICAL_SPAN

        self.front_sum_skewness = self._skewness(self.sums)
        self.front_sum_kurtosis = self._kurtosis(self.sums)

        self.odd_even_dist = Counter()
        self.big_small_dist = Counter()
        self.zone_dist = Counter()
        self.road_dist = Counter()
        self.repeat_dist = Counter()
        self.consec_dist = Counter()
        self.prime_dist = Counter()
        self.tail_freq = Counter()

        for idx, d in enumerate(self.data):
            front = sorted(d['front'])
            back = sorted(d['back'])

            odd = sum(1 for x in front if x % 2 == 1)
            self.odd_even_dist[f"{odd}:{5 - odd}"] += 1

            big = sum(1 for x in front if x >= 17)
            self.big_small_dist[f"{big}:{5 - big}"] += 1

            z = [0, 0, 0]
            for x in front:
                if x <= 12:
                    z[0] += 1
                elif x <= 23:
                    z[1] += 1
                else:
                    z[2] += 1
            self.zone_dist[f"{z[0]}:{z[1]}:{z[2]}"] += 1

            roads = [0, 0, 0]
            for x in front:
                roads[x % 3] += 1
            self.road_dist[f"{roads[0]}:{roads[1]}:{roads[2]}"] += 1

            pc = sum(1 for x in front if x in PRIMES)
            self.prime_dist[pc] += 1

            for x in front:
                self.tail_freq[x % 10] += 1

            c = sum(1 for i in range(4) if front[i + 1] - front[i] == 1)
            self.consec_dist[c] += 1

            if idx < n - 1:
                curr = set(front)
                prev = set(sorted(self.data[idx + 1]['front']))
                self.repeat_dist[len(curr & prev)] += 1

        # ── 邻号关系 ──
        self.adjacent_freq = Counter()
        for i in range(n - 1):
            prev_set = set(self.data[i + 1]['front'])
            adj_set = set()
            for num in prev_set:
                if num > 1:
                    adj_set.add(num - 1)
                if num < 35:
                    adj_set.add(num + 1)
            for num in self.data[i]['front']:
                if num in adj_set:
                    self.adjacent_freq[num] += 1

        # ── 贝叶斯后验 ──
        self.bayesian_front = self._bayesian(self.front_freq, 35)
        self.bayesian_back = self._bayesian(self.back_freq, 12)

        # ── 信息熵 ──
        total_front = sum(self.front_freq.values())
        probs = [self.front_freq.get(x, 0) / total_front for x in FRONT_RANGE if total_front > 0]
        self.front_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        self.front_max_entropy = math.log2(35)

        # ── V4 新增统计 ──
        self._build_front_copula()
        self._build_pagerank()
        self._build_back_copula()
        self._build_back_markov()
        self._build_tail_pattern()
        self._build_diff_pattern()
        self._build_consecutive_cond()
        self._build_bootstrap_stats()

    @staticmethod
    def _std(values: List[float]) -> float:
        """样本标准差（分母 N）。"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    @staticmethod
    def _skewness(values: List[float]) -> float:
        """偏度 γ₁。"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        std = DLTAnalyzerV4._std(values)
        if std == 0:
            return 0.0
        return sum((x - mean) ** 3 for x in values) / (len(values) * std ** 3)

    @staticmethod
    def _kurtosis(values: List[float]) -> float:
        """超额峰度。"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        std = DLTAnalyzerV4._std(values)
        if std == 0:
            return 0.0
        return sum((x - mean) ** 4 for x in values) / (len(values) * std ** 4) - 3

    def _bayesian(self, freq: Counter, total_numbers: int) -> Dict[int, float]:
        """基于均匀先验与观测频率的贝叶斯后验概率。"""
        prior = 1.0 / total_numbers
        total = sum(freq.values())
        result = {}
        rng = FRONT_RANGE if total_numbers == 35 else BACK_RANGE
        for num in rng:
            likelihood = freq.get(num, 0) / total if total > 0 else 1 / total_numbers
            result[num] = prior * likelihood
        total_bayes = sum(result.values())
        if total_bayes > 0:
            for num in result:
                result[num] /= total_bayes
        return result

    def _detect_style(self) -> str:
        """根据近 20 期的熵集中程度自动选择风格。

        原理：当近期号码高度集中于少数号码时（熵低），历史热号延续概率在
        经验上更值得关注，选择 conservative；当号码分布接近均匀（熵高）时，
        冷号回补机会相对增加，选择 aggressive。
        """
        recent_n = min(len(self.data), 20)
        recent_freq = Counter()
        for d in self.data[:recent_n]:
            for num in d['front']:
                recent_freq[num] += 1
        total = sum(recent_freq.values())
        if total == 0:
            return 'balanced'
        entropy = -sum((c / total) * math.log2(c / total) for c in recent_freq.values())
        ratio = entropy / math.log2(35)
        if ratio < 0.85:
            return 'conservative'
        if ratio > 0.95:
            return 'aggressive'
        return 'balanced'

    # ─────────────────────────────────────────────────────────────
    #  V4 新增：共现网络与马尔可夫结构
    # ─────────────────────────────────────────────────────────────

    def _build_front_copula(self):
        """构造前区 35×35 共现衰减矩阵。

        数学说明：
        C[i][j] = Σ w(t)·I{号码 i 与 j 在第 t 期同时出现}
        其中 w(t) 为距当前期数的衰减权重（近 5 期 5，5-10 期 3，10-20 期 2，20-50 期 1）。
        该矩阵可视为经验 Copula 的离散近似，描述号码间的联合出现倾向。
        """
        size = 36  # 使用 1-based 索引
        self.front_copula = [[0.0] * size for _ in range(size)]
        lookback = min(len(self.data), 50)

        for idx, d in enumerate(self.data[:lookback]):
            w = _decay_weight(idx)
            if w == 0:
                continue
            front = sorted(d['front'])
            for i in range(5):
                a = front[i]
                for j in range(i + 1, 5):
                    b = front[j]
                    self.front_copula[a][b] += w
                    self.front_copula[b][a] += w

        # 计算非对角线总权重与均值，供评分归一化使用
        total = 0.0
        count = 0
        for i in FRONT_RANGE:
            for j in FRONT_RANGE:
                if i < j:
                    total += self.front_copula[i][j]
                    count += 1
        self.front_copula_avg = total / count if count > 0 else 1.0
        self.front_copula_total = total

    def _build_pagerank(self):
        """基于 front_copula 计算 PageRank。

        将共现矩阵视为无向加权图，迭代公式：
            r^{(t+1)} = α·M·r^{(t)} + (1-α)·v
        其中 α=0.85 为阻尼系数，v 为均匀随机冲浪向量，M 为行随机化转移矩阵。
        PageRank 值高的号码可理解为在共现网络中处于“中心位置”。
        """
        n = 35
        alpha = 0.85
        iterations = 50

        row_sum = [0.0] * (n + 1)
        for i in FRONT_RANGE:
            row_sum[i] = sum(self.front_copula[i][j] for j in FRONT_RANGE)

        # 初始化
        rank = [1.0 / n] * (n + 1)
        teleport = (1.0 - alpha) / n

        for _ in range(iterations):
            new_rank = [0.0] * (n + 1)
            for j in FRONT_RANGE:
                s = 0.0
                for i in FRONT_RANGE:
                    if row_sum[i] > 0:
                        s += rank[i] * self.front_copula[i][j] / row_sum[i]
                    else:
                        # 孤立节点：均匀跳转
                        s += rank[i] / n
                new_rank[j] = alpha * s + teleport
            rank = new_rank

        # 归一化
        total = sum(rank[j] for j in FRONT_RANGE)
        self.pagerank = {j: rank[j] / total if total > 0 else 1.0 / n for j in FRONT_RANGE}

    def _build_back_copula(self):
        """构造后区 12×12 共现衰减矩阵。"""
        size = 13
        self.back_copula = [[0.0] * size for _ in range(size)]
        lookback = min(len(self.data), 50)

        for idx, d in enumerate(self.data[:lookback]):
            w = _decay_weight(idx)
            if w == 0:
                continue
            back = sorted(d['back'])
            a, b = back[0], back[1]
            self.back_copula[a][b] += w
            self.back_copula[b][a] += w

        total = 0.0
        count = 0
        for i in BACK_RANGE:
            for j in BACK_RANGE:
                if i < j:
                    total += self.back_copula[i][j]
                    count += 1
        self.back_copula_avg = total / count if count > 0 else 1.0

    def _build_back_markov(self):
        """后区第一球 → 第二球的条件概率。

        后区两球按升序排列，记第一球为 i，第二球为 j。
        P(j | i) = count(i, j) / count(i 作为第一球出现)。
        该条件分布描述了两球之间的配对偏好。
        """
        size = 13
        trans = [[0.0] * size for _ in range(size)]
        row_sum = [0.0] * size

        for d in self.data:
            back = sorted(d['back'])
            a, b = back[0], back[1]
            trans[a][b] += 1.0
            row_sum[a] += 1.0

        self.back_markov = {}
        for i in BACK_RANGE:
            self.back_markov[i] = {}
            for j in BACK_RANGE:
                if i < j and row_sum[i] > 0:
                    self.back_markov[i][j] = trans[i][j] / row_sum[i]

    def _build_tail_pattern(self):
        """同尾对频率统计。"""
        self.tail_pair_dist = Counter()
        self.tail_pair_digit_freq = Counter()
        for d in self.data:
            front = sorted(d['front'])
            tails = [x % 10 for x in front]
            pair_count = 0
            for i in range(5):
                for j in range(i + 1, 5):
                    if tails[i] == tails[j]:
                        pair_count += 1
                        self.tail_pair_digit_freq[tails[i]] += 1
            self.tail_pair_dist[pair_count] += 1

        total_pairs = sum(k * v for k, v in self.tail_pair_dist.items())
        total_draws = sum(self.tail_pair_dist.values())
        self.tail_pair_mean = total_pairs / total_draws if total_draws > 0 else 0.5

    def _build_diff_pattern(self):
        """前区相邻号码差值分布。"""
        self.diff_freq = Counter()
        diffs = []
        for d in self.data:
            front = sorted(d['front'])
            for i in range(4):
                diff = front[i + 1] - front[i]
                self.diff_freq[diff] += 1
                diffs.append(diff)
        self.avg_diff = sum(diffs) / len(diffs) if diffs else 7.0
        self.diff_std = self._std(diffs) or 3.5

    def _build_consecutive_cond(self):
        """连号条件概率。

        对每期前区升序号码 f[0..4]，若 f[k+1] - f[k] == 1，则称 f[k] 产生一次
        连号 successor。号码 i 的条件概率定义为：
            P(i+1 出现 | i 出现) = count(i 产生连号) / count(i 出现)。
        """
        occur = Counter()
        succ = Counter()
        for d in self.data:
            front = sorted(d['front'])
            for i, num in enumerate(front):
                occur[num] += 1
                if i < 4 and front[i + 1] - num == 1:
                    succ[num] += 1
        self.consecutive_cond = {}
        for num in FRONT_RANGE:
            if occur[num] > 0:
                self.consecutive_cond[num] = succ[num] / occur[num]
            else:
                self.consecutive_cond[num] = 0.0

    def _build_bootstrap_stats(self, n_boot: int = 1000, seed: int = 42):
        """Bootstrap 统计。

        从历史数据中有放回重采样 n_boot 次，计算前区和值、后区和值、AC 值的
        均值、标准差及 5%/95% 分位数，用于定位候选组合的统计位置。
        """
        rng = random.Random(seed)
        n = len(self.data)
        indices = list(range(n))

        front_sums = []
        back_sums = []
        ac_values = []

        for _ in range(n_boot):
            idx = rng.choice(indices)
            d = self.data[idx]
            front_sums.append(sum(d['front']))
            back_sums.append(sum(d['back']))
            ac_values.append(self.calc_ac(d['front']))

        def summary(vals: List[float]) -> Dict:
            s = sorted(vals)
            return {
                'mean': sum(vals) / len(vals),
                'std': self._std(vals),
                'p5': s[int(0.05 * (len(s) - 1))],
                'p95': s[int(0.95 * (len(s) - 1))],
            }

        self.bootstrap_stats = {
            'front_sum': summary(front_sums),
            'back_sum': summary(back_sums),
            'ac': summary(ac_values),
            'n_boot': n_boot,
        }

    # ─────────────────────────────────────────────────────────────
    #  公共工具函数
    # ─────────────────────────────────────────────────────────────

    def calc_ac(self, combo: List[int]) -> int:
        """计算 AC 值（算术复杂度）。

        AC = |{ |aᵢ - aⱼ| : i < j }| - (n - 1)
        对于前区 5 码，AC≥4 为合理，5-8 最常见。
        """
        diffs = set()
        combo = sorted(combo)
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                diffs.add(combo[j] - combo[i])
        return len(diffs) - (len(combo) - 1)

    def calc_skewness(self, combo: List[int]) -> float:
        """计算单组号码的偏度。"""
        n = len(combo)
        mean = sum(combo) / n
        std = self._std(combo)
        if std == 0:
            return 0.0
        return sum((x - mean) ** 3 for x in combo) / (n * std ** 3)

    # ─────────────────────────────────────────────────────────────
    #  V4 评分函数
    # ─────────────────────────────────────────────────────────────

    def score_front_v4(self, nums: List[int]) -> Tuple[float, Dict]:
        """前区 V4 综合评分。

        在 V3 的 15 维评分基础上新增：
        - 共现网络加分（copula_bonus）
        - PageRank 加分（pagerank_bonus）
        - 同尾对适度奖励/惩罚（tail）
        - 差值分布奖励（diff）

        Args:
            nums: 5 个前区号码（无需排序）。

        Returns:
            (总分, 各维度明细字典)
        """
        combo = sorted(nums)
        details = {}
        score = 0.0

        # ── 维度 1：贝叶斯后验概率（满分 15）──
        bayes_score = sum(self.bayesian_front.get(n, 0) for n in combo) * 100
        d1 = min(bayes_score * 3, 15)
        details['贝叶斯'] = round(d1, 1)
        score += d1

        # ── 维度 2：加权热度（满分 12）──
        heat_raw = sum(self.front_heat.get(n, 0) for n in combo)
        total_heat = sum(self.front_heat.values())
        expected_heat = 5 * total_heat / 35 if total_heat > 0 else 1
        heat_ratio = heat_raw / expected_heat if expected_heat > 0 else 1
        if self.style == 'conservative':
            d2 = min(heat_ratio * 8, 12)
        elif self.style == 'aggressive':
            d2 = min((2 - heat_ratio) * 6, 12) if heat_ratio < 2 else 0
        else:
            d2 = min(heat_ratio * 6, 12)
        details['热度'] = round(d2, 1)
        score += d2

        # ── 维度 3：和值合理性（满分 12）──
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

        # ── 维度 4：AC 值（满分 12）──
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
            d4 = -8
        details['AC值'] = round(d4, 1)
        score += d4

        # ── 维度 5：跨度合理性（满分 8）──
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

        # ── 维度 6：奇偶平衡（满分 6）──
        odd = sum(1 for n in combo if n % 2 == 1)
        d6 = 6 if odd in [2, 3] else (3 if odd in [1, 4] else 0)
        details['奇偶'] = round(d6, 1)
        score += d6

        # ── 维度 7：大小平衡（满分 6）──
        big = sum(1 for n in combo if n >= 17)
        d7 = 6 if big in [2, 3] else (3 if big in [1, 4] else 0)
        details['大小'] = round(d7, 1)
        score += d7

        # ── 维度 8：三区分布（满分 10）──
        z = [0, 0, 0]
        for n in combo:
            if n <= 12:
                z[0] += 1
            elif n <= 23:
                z[1] += 1
            else:
                z[2] += 1
        if 0 in z:
            d8 = -5
        elif sorted(z) in [[1, 2, 2], [2, 2, 1], [2, 1, 2]]:
            d8 = 10
        elif sorted(z) in [[1, 1, 3], [1, 3, 1], [3, 1, 1]]:
            d8 = 5
        else:
            d8 = 3
        details['三区'] = round(d8, 1)
        score += d8

        # ── 维度 9：012 路多样性（满分 5）──
        roads = set(n % 3 for n in combo)
        d9 = 5 if len(roads) == 3 else (2 if len(roads) == 2 else 0)
        details['012路'] = round(d9, 1)
        score += d9

        # ── 维度 10：质合比（满分 4）──
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

        # ── 维度 11：遗漏回补（满分 8）──
        miss_score = 0.0
        for n in combo:
            m = self.front_miss.get(n, 0)
            mean = 6.0
            std = 6.0
            z = (m - mean) / std
            if 0.5 <= z <= 2.5:
                miss_score += min(z * 2.0, 3.0)
            elif z > 2.5:
                miss_score += 1.5
            if 8 <= m <= 20:
                miss_score += 0.5
        d11 = min(miss_score, 8)
        details['遗漏'] = round(d11, 1)
        score += d11

        # ── 维度 12：重号加分（满分 4）──
        last_set = set(self.data[0]['front']) if self.data else set()
        repeats = len(set(combo) & last_set)
        d12 = 4 if repeats == 1 else (2 if repeats == 2 else 0)
        details['重号'] = round(d12, 1)
        score += d12

        # ── 维度 13：邻号关系（满分 3）──
        prev_front = set(self.data[0]['front']) if self.data else set()
        adj_set = set()
        for num in prev_front:
            if num > 1:
                adj_set.add(num - 1)
            if num < 35:
                adj_set.add(num + 1)
        adj_count = len(set(combo) & adj_set)
        d13 = min(adj_count * 1.5, 3)
        details['邻号'] = round(d13, 1)
        score += d13

        # ── 维度 14：尾数多样性（满分 3）──
        tails = set(n % 10 for n in combo)
        d14 = len(tails) * 0.6
        details['尾数'] = round(d14, 1)
        score += d14

        # ── 维度 15：偏度检验（满分 2）──
        skew = self.calc_skewness(combo)
        d15 = 2 if abs(skew) < 0.5 else (1 if abs(skew) < 1.0 else 0)
        details['偏度'] = round(d15, 1)
        score += d15

        # ── V4 新增 1：共现网络加分（满分 6）──
        pair_sum = 0.0
        for i in range(5):
            for j in range(i + 1, 5):
                pair_sum += self.front_copula[combo[i]][combo[j]]
        pairs = 10
        ratio = pair_sum / (pairs * self.front_copula_avg) if self.front_copula_avg > 0 else 1
        d_copula = min(math.log1p(ratio) * 4, 6)
        details['共现'] = round(d_copula, 1)
        score += d_copula

        # ── V4 新增 2：PageRank 加分（满分 4）──
        top5_rank = sum(sorted(self.pagerank.values(), reverse=True)[:5]) / 5
        rank_sum = sum(self.pagerank.get(n, 0) for n in combo)
        ratio_pr = rank_sum / top5_rank if top5_rank > 0 else 1
        d_pr = min(ratio_pr * 2.5, 4)
        details['PageRank'] = round(d_pr, 1)
        score += d_pr

        # ── V4 新增 3：同尾对适度奖励/惩罚（满分 4）──
        tail_counts = Counter(n % 10 for n in combo)
        same_tail_pairs = sum(c * (c - 1) // 2 for c in tail_counts.values())
        mean_pairs = self.tail_pair_mean
        if same_tail_pairs == 0:
            d_tail = 1.0 if mean_pairs < 0.5 else 0.0
        elif same_tail_pairs == 1:
            d_tail = 4.0
        elif same_tail_pairs == 2:
            d_tail = 2.0
        else:
            d_tail = -2.0
        details['同尾'] = round(d_tail, 1)
        score += d_tail

        # ── V4 新增 4：差值分布奖励（满分 4）──
        diffs = [combo[i + 1] - combo[i] for i in range(4)]
        mean_z = sum(abs(d - self.avg_diff) / self.diff_std for d in diffs) / 4 if self.diff_std > 0 else 0
        d_diff = max(0, 4 - mean_z)
        details['差值'] = round(d_diff, 1)
        score += d_diff

        return round(score, 1), details

    def score_back_v4(self, nums: List[int]) -> Tuple[float, Dict]:
        """后区 V4 综合评分。

        在 V3 后区评分基础上新增：
        - 后区共现矩阵加分（back_copula_bonus）
        - 后区 Markov 配对加分（back_markov_bonus）

        Args:
            nums: 2 个后区号码（无需排序）。

        Returns:
            (总分, 各维度明细字典)
        """
        combo = sorted(nums)
        a, b = combo[0], combo[1]
        details = {}
        score = 0.0

        # ── 加权热度（满分 8）──
        total_back_heat = sum(self.back_heat.values())
        heat_raw = sum(self.back_heat.get(n, 0) for n in combo)
        expected = 2 * total_back_heat / 12 if total_back_heat > 0 else 1
        d_heat = min(heat_raw / expected * 5, 8) if expected > 0 else 0
        details['热度'] = round(d_heat, 1)
        score += d_heat

        # ── 遗漏甜蜜区（满分约 8）──
        d_miss = 0.0
        for n in combo:
            m = self.back_miss.get(n, 0)
            mean = 5.0
            std = 5.0
            z = (m - mean) / std
            if 0.0 <= z <= 2.0:
                d_miss += min(2.0 + z, 4.0)
            elif 2.0 < z <= 3.5:
                d_miss += 2.0
            else:
                d_miss += 0.5
        d_miss = min(d_miss, 8)
        details['遗漏'] = round(d_miss, 1)
        score += d_miss

        # ── 奇偶 1:1（满分 3）──
        d_odd = 3 if sum(n % 2 for n in combo) == 1 else 0
        details['奇偶'] = round(d_odd, 1)
        score += d_odd

        # ── 和值接近理论均值（满分 4）──
        s = sum(combo)
        if 10 <= s <= 15:
            d_sum = 4
        elif 7 <= s <= 18:
            d_sum = 2
        else:
            d_sum = 0
        details['和值'] = round(d_sum, 1)
        score += d_sum

        # ── 012 路多样性（满分 2）──
        roads = len(set(n % 3 for n in combo))
        d_road = 2 if roads == 2 else 0
        details['012路'] = round(d_road, 1)
        score += d_road

        # ── V4 新增 1：后区共现加分（满分 4）──
        cop = self.back_copula[a][b]
        d_copula = min(cop / (self.back_copula_avg + 1e-9) * 2, 4)
        details['共现'] = round(d_copula, 1)
        score += d_copula

        # ── V4 新增 2：后区 Markov 配对加分（满分 4）──
        prob = self.back_markov.get(a, {}).get(b, 0.0)
        d_markov = min(prob * 5, 4)
        details['Markov'] = round(d_markov, 1)
        score += d_markov

        return round(score, 1), details

    # ─────────────────────────────────────────────────────────────
    #  遗传算法推荐
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_sample_without_replacement(population: List[int],
                                              weights: List[float],
                                              k: int) -> List[int]:
        """按权重无放回采样。"""
        pool = list(population)
        w = list(weights)
        result = []
        while len(result) < k and pool:
            total = sum(w)
            if total <= 0:
                # 权重失效时退化为均匀随机
                choice = random.choice(pool)
            else:
                r = random.uniform(0, total)
                cum = 0.0
                choice = pool[-1]
                for idx, (val, wt) in enumerate(zip(pool, w)):
                    cum += wt
                    if r <= cum:
                        choice = val
                        pop_idx = idx
                        break
                else:
                    pop_idx = len(pool) - 1
            result.append(choice)
            # 移除已选元素
            if choice in pool:
                idx = pool.index(choice)
                pool.pop(idx)
                w.pop(idx)
            else:
                break
        return result

    def _random_individual(self) -> Dict:
        """生成一个随机合法个体。"""
        front = sorted(random.sample(list(FRONT_RANGE), 5))
        back = sorted(random.sample(list(BACK_RANGE), 2))
        front_score, front_details = self.score_front_v4(front)
        back_score, back_details = self.score_back_v4(back)
        return {
            'front': front,
            'back': back,
            'front_score': front_score,
            'back_score': back_score,
            'front_details': front_details,
            'back_details': back_details,
        }

    def _encode(self, ind: Dict) -> List[int]:
        """编码为 7 维向量：前区 5 码 + 后区 2 码。"""
        return list(ind['front']) + list(ind['back'])

    def _decode(self, encoded: List[int]) -> Dict:
        """解码并评分。"""
        front = sorted(encoded[:5])
        back = sorted(encoded[5:7])
        front_score, front_details = self.score_front_v4(front)
        back_score, back_details = self.score_back_v4(back)
        return {
            'front': front,
            'back': back,
            'front_score': front_score,
            'back_score': back_score,
            'front_details': front_details,
            'back_details': back_details,
        }

    def _repair(self, encoded: List[int]) -> List[int]:
        """修复交叉或变异后可能出现的重复号码，保持前 5 / 后 2 分区。"""
        front = list(encoded[:5])
        back = list(encoded[5:7])

        # 修复前区
        missing_front = [n for n in FRONT_RANGE if n not in front]
        seen = set()
        new_front = []
        for n in front:
            if n not in seen and 1 <= n <= 35:
                new_front.append(n)
                seen.add(n)
            else:
                if missing_front:
                    # 优先按贝叶斯后验概率采样缺失号码
                    weights = [self.bayesian_front.get(x, 1e-6) for x in missing_front]
                    replacement = self._weighted_sample_without_replacement(
                        missing_front, weights, 1)[0]
                    new_front.append(replacement)
                    missing_front.remove(replacement)
                    seen.add(replacement)
        while len(new_front) < 5 and missing_front:
            weights = [self.bayesian_front.get(x, 1e-6) for x in missing_front]
            replacement = self._weighted_sample_without_replacement(
                missing_front, weights, 1)[0]
            new_front.append(replacement)
            missing_front.remove(replacement)

        # 修复后区
        missing_back = [n for n in BACK_RANGE if n not in back]
        seen_back = set()
        new_back = []
        for n in back:
            if n not in seen_back and 1 <= n <= 12:
                new_back.append(n)
                seen_back.add(n)
            else:
                if missing_back:
                    weights = [self.bayesian_back.get(x, 1e-6) for x in missing_back]
                    replacement = self._weighted_sample_without_replacement(
                        missing_back, weights, 1)[0]
                    new_back.append(replacement)
                    missing_back.remove(replacement)
                    seen_back.add(replacement)
        while len(new_back) < 2 and missing_back:
            weights = [self.bayesian_back.get(x, 1e-6) for x in missing_back]
            replacement = self._weighted_sample_without_replacement(
                missing_back, weights, 1)[0]
            new_back.append(replacement)
            missing_back.remove(replacement)

        return sorted(new_front) + sorted(new_back)

    def _mutate(self, encoded: List[int], mutation_rate: float = 0.05) -> List[int]:
        """按 mutation_rate 对每个基因尝试变异。"""
        encoded = list(encoded)
        front = encoded[:5]
        back = encoded[5:7]

        # 前区基因
        for i in range(5):
            if random.random() < mutation_rate:
                missing = [n for n in FRONT_RANGE if n not in front]
                if missing:
                    front[i] = random.choice(missing)

        # 后区基因
        for i in range(2):
            if random.random() < mutation_rate:
                missing = [n for n in BACK_RANGE if n not in back]
                if missing:
                    back[i] = random.choice(missing)

        return self._repair(front + back)

    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        """单点交叉并修复，产生一个合法后代。"""
        e1 = self._encode(p1)
        e2 = self._encode(p2)
        cut = random.randint(1, 6)
        child_encoded = e1[:cut] + e2[cut:]
        child_encoded = self._repair(child_encoded)
        return self._decode(child_encoded)

    def _diversity_and_coverage(self, population: List[Dict]) -> Tuple[List[float], List[float]]:
        """计算种群中每个个体的多样性奖励与覆盖奖励。"""
        sets = [set(self._encode(ind)) for ind in population]
        n = len(population)
        diversities = []
        coverages = []
        for i, ind in enumerate(population):
            # 多样性：与种群其他个体的平均 Jaccard 距离
            dists = []
            for j in range(n):
                if i == j:
                    continue
                inter = len(sets[i] & sets[j])
                dists.append(1 - inter / 7.0)
            diversity = sum(dists) / len(dists) if dists else 0.0

            # 覆盖性：前区三区 + 尾数覆盖度
            zones = set()
            tails = set()
            for num in ind['front']:
                if num <= 12:
                    zones.add(1)
                elif num <= 23:
                    zones.add(2)
                else:
                    zones.add(3)
                tails.add(num % 10)
            coverage = (len(zones) / 3.0) * 2.0 + (len(tails) / 5.0) * 2.0

            diversities.append(diversity)
            coverages.append(coverage)
        return diversities, coverages

    def _population_diversity(self, population: List[Dict]) -> float:
        """种群多样性：唯一「前区+后区」组合数占种群规模的比例。

        取值 0~1，越接近 1 表示种群越分散；接近 0 说明已塌缩到少数几个组合。
        """
        if not population:
            return 0.0
        uniq = {(tuple(ind['front']), tuple(ind['back'])) for ind in population}
        return len(uniq) / len(population)

    def _shared_fitness(self, population: List[Dict], fitness: List[float],
                        sigma: float = 0.6, window: int = 24) -> List[float]:
        """适应度共享（fitness sharing）。

        与邻居高度相似的个体会被互相削减适应度，从而在多个「小生境」里
        各自保留优秀个体，避免整个种群塌缩到同一个局部最优 —— 这正是
        原先 5 注高度雷同的根因。

        Args:
            population: 当前种群。
            fitness: 原始适应度列表。
            sigma: 共享半径（Jaccard 距离小于该值的个体视为同一小生境）。
            window: 仅与邻近 window 个个体比较，控制 O(n·window) 开销。

        Returns:
            共享后的适应度列表（用于排序与选择；不改变原始得分语义）。
        """
        n = len(population)
        fronts = [set(ind['front']) for ind in population]
        shared: List[float] = [0.0] * n
        for i in range(n):
            share = 0.0
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                union = fronts[i] | fronts[j]
                if not union:
                    continue
                dist = 1.0 - len(fronts[i] & fronts[j]) / len(union)  # Jaccard 距离
                if dist < sigma:
                    share += 1.0 - dist / sigma
            denom = 1.0 + share
            shared[i] = fitness[i] / denom if denom > 0 else fitness[i]
        return shared

    def genetic_recommend(self, n: int = 5, pop_size: int = 200,
                          generations: int = 50, elite: float = 0.1) -> List[Dict]:
        """使用遗传算法生成 n 注推荐。

        编码：前区 5 个号码（已排序） + 后区 2 个号码（已排序），共 7 维。
        适应度：
            fitness = w1·score_front + w2·score_back + w3·diversity_bonus + w4·coverage_bonus
        操作：
            - 适应度共享（fitness sharing），维持多个小生境
            - 锦标赛选择（tournament size=3）
            - 单点交叉 + 重复修复
            - 变异率自适应：种群塌缩时提高，必要时灾变重启注入新鲜基因
            - 每代保留 top elite 个体

        Args:
            n: 最终返回推荐注数。
            pop_size: 种群规模。
            generations: 迭代代数。
            elite: 精英保留比例。

        Returns:
            每个元素为字典，包含 front, back, front_sum, back_sum, ac, span,
            odd_even, three_zone, score, details 等字段。
        """
        if not self.data:
            return []

        # 权重配置
        w1, w2, w3, w4 = 1.0, 2.0, 3.0, 1.0

        # 初始化种群
        population = [self._random_individual() for _ in range(pop_size)]

        elite_count = max(1, int(pop_size * elite))

        # 变异率自适应：种群塌缩时提高；仍不足则灾变重启
        base_mutation = 0.05
        mutation_rate = base_mutation

        for generation in range(generations):
            diversities, coverages = self._diversity_and_coverage(population)

            # 计算原始适应度
            raw_fitness = []
            for i, ind in enumerate(population):
                fit = (w1 * ind['front_score'] +
                       w2 * ind['back_score'] +
                       w3 * diversities[i] +
                       w4 * coverages[i])
                raw_fitness.append(fit)

            # 适应度共享：抑制同类个体扎堆，维持小生境
            fitness = self._shared_fitness(population, raw_fitness)

            # 按共享适应度排序
            ranked = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)

            # 精英保留
            next_population = [ind for _, ind in ranked[:elite_count]]

            # 锦标赛选择填充种群
            while len(next_population) < pop_size:
                # 选择父代 1
                contestants = random.sample(ranked, min(3, len(ranked)))
                parent1 = max(contestants, key=lambda x: x[0])[1]
                contestants = random.sample(ranked, min(3, len(ranked)))
                parent2 = max(contestants, key=lambda x: x[0])[1]

                child = self._crossover(parent1, parent2)
                child = self._mutate(self._encode(child), mutation_rate=mutation_rate)
                child = self._decode(child)
                next_population.append(child)

            population = next_population

            # 每 10 代检测一次种群多样性
            if (generation + 1) % 10 == 0:
                div = self._population_diversity(population)
                if div < 0.30:
                    # 灾变：只保留本代精英，其余重新随机注入新鲜基因
                    survivors = population[:elite_count]
                    population = survivors + [
                        self._random_individual()
                        for _ in range(pop_size - len(survivors))
                    ]
                    mutation_rate = base_mutation
                elif div < 0.55:
                    mutation_rate = min(0.30, mutation_rate * 1.6)

        # 最终评估：同样使用共享适应度排序，保证头部候选彼此有差异
        diversities, coverages = self._diversity_and_coverage(population)
        raw_final = []
        for i, ind in enumerate(population):
            fit = (w1 * ind['front_score'] +
                   w2 * ind['back_score'] +
                   w3 * diversities[i] +
                   w4 * coverages[i])
            raw_final.append(fit)
        shared_final = self._shared_fitness(population, raw_final)

        final = [(s, r, ind) for s, r, ind in zip(shared_final, raw_final, population)]
        final.sort(key=lambda x: x[0], reverse=True)

        # 去重并取前 n 个
        seen = set()
        result = []
        # 遗传算法容易收敛，导致 n 注高度雷同（前区几乎一样、后区重复），
        # 覆盖率反而下降。这里做三重多样性约束：同一后区次数、同一前区次数、
        # 前区互相重叠上限。若约束过严凑不满 n 注，则放宽再补一轮。
        max_same_back = max(1, (n + 2) // 3)
        max_same_front = max(1, (n + 2) // 3)
        back_counts: Dict[tuple, int] = {}
        front_counts: Dict[tuple, int] = {}
        selected_fronts: List[tuple] = []

        def _overlaps_existing(fkey: tuple) -> bool:
            """两注前区最多重合 len-2 个号码（5 选 5 时即最多重合 3 个）。"""
            cap = len(fkey) - 2
            return any(len(set(fkey) & set(s)) > cap for s in selected_fronts)

        for strict in (True, False):
            for s_fit, fit, ind in final:
                if len(result) >= n:
                    break
                key = (tuple(ind['front']), tuple(ind['back']))
                if key in seen:
                    continue
                fkey = tuple(sorted(ind['front']))
                bkey = tuple(sorted(ind['back']))
                if strict:
                    if back_counts.get(bkey, 0) >= max_same_back:
                        continue
                    if front_counts.get(fkey, 0) >= max_same_front:
                        continue
                    if _overlaps_existing(fkey):
                        continue
                seen.add(key)
                back_counts[bkey] = back_counts.get(bkey, 0) + 1
                front_counts[fkey] = front_counts.get(fkey, 0) + 1
                selected_fronts.append(fkey)
                front = ind['front']
                back = ind['back']
                ac = self.calc_ac(front)
                span = max(front) - min(front)
                odd = sum(1 for x in front if x % 2 == 1)
                z = [0, 0, 0]
                for num in front:
                    if num <= 12:
                        z[0] += 1
                    elif num <= 23:
                        z[1] += 1
                    else:
                        z[2] += 1
                details = {
                    'front': ind['front_details'],
                    'back': ind['back_details'],
                }
                result.append({
                    'front': front,
                    'back': back,
                    'front_sum': sum(front),
                    'back_sum': sum(back),
                    'ac': ac,
                    'span': span,
                    'odd_even': f"{odd}:{5 - odd}",
                    'three_zone': f"{z[0]}:{z[1]}:{z[2]}",
                    'score': round(fit, 2),
                    'front_score': ind['front_score'],
                    'back_score': ind['back_score'],
                    'details': details,
                })
                if len(result) >= n:
                    break
            if len(result) >= n:
                break

        # 后区多样性兜底：种群收敛时严格约束会凑不满 n 注而放宽，
        # 结果后区可能仍高度集中。这里用「按后区评分排序的候选池」
        # 把超限的重复后区替换成尚未使用的高分后区。
        if len(result) > 1:
            back_pool: Dict[tuple, float] = {}
            for _s, _r, ind in final:
                bkey = tuple(sorted(ind['back']))
                score = ind.get('back_score', 0.0)
                if bkey not in back_pool or score > back_pool[bkey]:
                    back_pool[bkey] = score
            ranked_backs = sorted(back_pool.items(), key=lambda kv: -kv[1])

            used: Dict[tuple, int] = {}
            for r in result:
                bkey = tuple(sorted(r['back']))
                used[bkey] = used.get(bkey, 0) + 1
                if used[bkey] <= max_same_back:
                    continue
                for cand, _ in ranked_backs:
                    if used.get(cand, 0) < max_same_back:
                        r['back'] = list(cand)
                        r['back_sum'] = sum(cand)
                        used[cand] = used.get(cand, 0) + 1
                        break

        return result

    # ─────────────────────────────────────────────────────────────
    #  对外推荐接口
    # ─────────────────────────────────────────────────────────────

    def _top_back_combinations(self, k: int = 5) -> List[List[int]]:
        """独立评估全部 66 个后区组合，返回评分最高的 k 个。

        大乐透的前区（35 选 5）与后区（12 选 2）是**独立抽取**的，
        两者之间没有统计关联，因此把它们绑在同一个个体里做联合遗传优化
        并不合理 —— 后区应单独按共现矩阵与一球→二球条件概率选优。
        """
        scored = []
        for a in range(1, 13):
            for b in range(a + 1, 13):
                try:
                    sc, _ = self.score_back_v4([a, b])
                except Exception:
                    sc = 0.0
                scored.append((float(sc), [a, b]))
        scored.sort(key=lambda kv: -kv[0])
        return [combo for _, combo in scored[:k]]

    def recommend_v4(self, n: int = 5) -> Dict:
        """生成 V4 推荐结果。

        Returns:
            包含 gold/silver/top5/context/style_used/backtest_hint 的字典。
        """
        top5 = self.genetic_recommend(n=n)

        # 后区独立选优：前区由遗传算法负责，后区改用独立评分的 top-k 组合。
        # 这样既保证 n 注后区互不相同（覆盖更广），又避免后区被前区优化带偏。
        try:
            best_backs = self._top_back_combinations(k=max(n, 2))
            for i, item in enumerate(top5):
                if i < len(best_backs):
                    item['back'] = list(best_backs[i])
                    item['back_sum'] = sum(best_backs[i])
        except Exception:
            pass

        gold = top5[0] if top5 else None
        silver = top5[1] if len(top5) > 1 else None

        context = {
            'data_count': len(self.data),
            'avg_sum': round(self.avg_sum, 1),
            'sum_std': round(self.sum_std, 1),
            'avg_span': round(self.avg_span, 1),
            'entropy': round(self.front_entropy, 2),
            'max_entropy': round(self.front_max_entropy, 2),
            'hot_front': self.front_heat.most_common(7),
            'miss_front': sorted(self.front_miss.items(), key=lambda x: x[1], reverse=True)[:7],
            'hot_back': self.back_heat.most_common(5),
            'miss_back': sorted(self.back_miss.items(), key=lambda x: x[1], reverse=True)[:5],
            'bootstrap': {
                'front_sum': {
                    'mean': round(self.bootstrap_stats['front_sum']['mean'], 1),
                    'std': round(self.bootstrap_stats['front_sum']['std'], 1),
                    'p5': self.bootstrap_stats['front_sum']['p5'],
                    'p95': self.bootstrap_stats['front_sum']['p95'],
                },
                'back_sum': {
                    'mean': round(self.bootstrap_stats['back_sum']['mean'], 1),
                    'std': round(self.bootstrap_stats['back_sum']['std'], 1),
                    'p5': self.bootstrap_stats['back_sum']['p5'],
                    'p95': self.bootstrap_stats['back_sum']['p95'],
                },
                'ac': {
                    'mean': round(self.bootstrap_stats['ac']['mean'], 2),
                    'std': round(self.bootstrap_stats['ac']['std'], 2),
                    'p5': self.bootstrap_stats['ac']['p5'],
                    'p95': self.bootstrap_stats['ac']['p95'],
                },
            },
            'copula_top_pair': self._top_copula_pair(),
            'pagerank_top3': [n for n, _ in sorted(self.pagerank.items(), key=lambda x: x[1], reverse=True)[:3]],
        }

        backtest_hint = (
            "彩票开奖为独立随机事件，总组合数 C(35,5)×C(12,2)=21,425,712。"
            "V4 模型基于历史统计分布（共现网络、PageRank、Markov 配对、遗传优化）"
            "提升推荐组合的统计合理性，但不提高绝对命中率。建议理性投注、量力而行。"
        )

        return {
            'gold': gold,
            'silver': silver,
            'top5': top5,
            'context': context,
            'style_used': self.style,
            'backtest_hint': backtest_hint,
        }

    def _top_copula_pair(self) -> Tuple[Tuple[int, int], float]:
        """返回共现权重最高的前区号码对。"""
        best = ((1, 2), 0.0)
        for i in FRONT_RANGE:
            for j in FRONT_RANGE:
                if i < j and self.front_copula[i][j] > best[1]:
                    best = ((i, j), self.front_copula[i][j])
        return best[0]


# ═══════════════════════════════════════════════════════════════
#  简单自测
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 构造 60 期模拟数据用于快速验证接口
    def mock_data(count: int = 60) -> List[Dict]:
        data = []
        for i in range(count):
            front = sorted(random.sample(range(1, 36), 5))
            back = sorted(random.sample(range(1, 13), 2))
            data.append({
                'period': str(2026001 + i),
                'date': '2026-01-01',
                'front': front,
                'back': back,
            })
        return data

    analyzer = DLTAnalyzerV4(mock_data(80), style='adaptive')
    rec = analyzer.recommend_v4(5)
    print("Style used:", rec['style_used'])
    print("Gold:", rec['gold'])
    print("Silver:", rec['silver'])
    print("Top5 count:", len(rec['top5']))
