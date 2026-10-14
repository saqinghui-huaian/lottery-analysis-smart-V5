#!/usr/bin/env python3
"""
智能选号系统 V3.1 (2026-09-25)

核心理念：基于理论概率分布的智能选号，不是预测

⚠️ 数学基础（最重要）：
彩票开奖是独立随机事件。每期开奖号码与历史数据无关。

V2.5 明确删除的方法（独立随机事件无意义）：
- 马尔可夫链转移概率（独立随机事件无意义）
- 信息熵（描述历史，不预测未来）
- 贝叶斯推断（独立随机事件退化为无用）
- 自相关分析（小样本假阳性）
- 时间序列特征（独立随机事件无意义）

V3.1 保留的有效方法：
- 和值概率分布（理论概率：7-20占83.2%）
- 跨度概率分布（理论概率：4-7最常见）
- 形态分布（组六72%、组三27%、豹子1%）
- 奇偶平衡（1:2或2:1各37.5%）
- 大小平衡（1:2或2:1各37.5%）
- 012路平衡（0路40%、1路30%、2路30%）
- 质合比（质数2,3,5,7占40%）
- 蒙特卡洛+多维过滤（基于理论概率）
- 用户偏好调整（保守/激进）

V3.1 改进（相比V2.5）：
1. 更精确的理论概率计算（已验证）
2. 蒙特卡洛候选生成（更高效）
3. 更好的覆盖面约束
4. 更清晰的代码结构
"""

import random
from collections import Counter
from typing import List, Dict, Tuple, Optional


class SmartNumberSelector:
    """
    智能选号系统 V3.1
    
    核心原则：
    1. 彩票是独立随机事件，历史数据不能预测未来
    2. 选号策略基于理论概率分布，不是"模式识别"
    3. 选择"更合理"的号码组合，不是"更可能中"的组合
    4. 合理 = 符合概率分布，避免极端组合
    """
    
    # 理论概率分布（枚举计算，精确值）
    SUM_PROBS = {}  # 和值 -> 概率
    SPAN_PROBS = {}  # 跨度 -> 概率
    SHAPE_PROBS = {'baozi': 0.01, 'zusan': 0.27, 'zuliu': 0.72}
    
    # 0-9的质数集合
    PRIMES = {2, 3, 5, 7}
    
    def __init__(self):
        self._compute_theoretical_distributions()
    
    def _compute_theoretical_distributions(self):
        """计算理论概率分布（精确枚举）"""
        sum_count = Counter()
        span_count = Counter()
        
        for b in range(10):
            for s in range(10):
                for g in range(10):
                    hz = b + s + g
                    span = max(b, s, g) - min(b, s, g)
                    sum_count[hz] += 1
                    span_count[span] += 1
        
        # 归一化为概率
        total = 1000
        for hz, cnt in sum_count.items():
            self.SUM_PROBS[hz] = cnt / total
        for span, cnt in span_count.items():
            self.SPAN_PROBS[span] = cnt / total
    
    def compute_context(self, data: List[Dict]) -> Dict:
        """
        计算统计上下文（只保留有用的维度）
        
        有用的维度：
        - 各位频率（用于用户偏好：热号/冷号）
        - 遗漏值（作为参考维度，不是主要依据）
        - 近期和值/跨度分布（用于候选生成）
        - 形态分布（用于覆盖面约束）
        - 奇偶/大小分布（用于覆盖面约束）
        - 012路频率（用于覆盖面约束）
        """
        ctx = {}
        n = len(data)
        
        # 1. 各位频率（近20/50期）
        ctx['freq20'] = [Counter(), Counter(), Counter()]
        ctx['freq50'] = [Counter(), Counter(), Counter()]
        for d in data[:20]:
            for pos, key in enumerate(['h', 't', 'u']):
                ctx['freq20'][pos][d[key]] += 1
        for d in data[:50]:
            for pos, key in enumerate(['h', 't', 'u']):
                ctx['freq50'][pos][d[key]] += 1
        
        # 2. 加权热度（近20期×3，20-50期×1）
        ctx['weighted'] = Counter()
        for d in data[:20]:
            for x in [d['h'], d['t'], d['u']]:
                ctx['weighted'][x] += 3
        for d in data[20:50]:
            for x in [d['h'], d['t'], d['u']]:
                ctx['weighted'][x] += 1
        
        # 3. 遗漏值（用于遗漏回补参考）
        ctx['miss'] = [{}, {}, {}]
        for digit in range(10):
            for pos, key in enumerate(['h', 't', 'u']):
                for i, d in enumerate(data):
                    if d[key] == digit:
                        ctx['miss'][pos][digit] = i
                        break
                else:
                    ctx['miss'][pos][digit] = n
        
        # 4. 近期和值/跨度分布
        ctx['recent_sums'] = [d['h'] + d['t'] + d['u'] for d in data[:20]]
        ctx['recent_spans'] = [max(d['h'], d['t'], d['u']) - min(d['h'], d['t'], d['u']) for d in data[:20]]
        
        # 5. 近期形态分布
        ctx['recent_shapes'] = Counter()
        for d in data[:20]:
            unique = len(set([d['h'], d['t'], d['u']]))
            if unique == 1:
                ctx['recent_shapes']['baozi'] += 1
            elif unique == 2:
                ctx['recent_shapes']['zusan'] += 1
            else:
                ctx['recent_shapes']['zuliu'] += 1
        
        # 6. 和值均值（用于候选生成）
        sums = [d['h'] + d['t'] + d['u'] for d in data[:50]]
        ctx['hz_avg'] = sum(sums) / len(sums) if sums else 13.5
        
        # 7. 012路频率（用于候选生成）
        ctx['road20'] = [Counter(), Counter(), Counter()]
        for d in data[:20]:
            for pos, key in enumerate(['h', 't', 'u']):
                ctx['road20'][pos][d[key] % 3] += 1
        
        # 8. 形态统计（近50期）
        ctx['shape_cnt'] = Counter()
        for d in data[:50]:
            unique = len(set([d['h'], d['t'], d['u']]))
            if unique == 1:
                ctx['shape_cnt']['豹子'] += 1
            elif unique == 2:
                ctx['shape_cnt']['组三'] += 1
            else:
                ctx['shape_cnt']['组六'] += 1
        
        # 9. 奇偶/大小分布（近20期）
        ctx['odd_even_cnt'] = Counter()
        ctx['big_small_cnt'] = Counter()
        for d in data[:20]:
            odd = sum(1 for x in [d['h'], d['t'], d['u']] if x % 2 == 1)
            ctx['odd_even_cnt'][f'{odd}:{3-odd}'] += 1
            big = sum(1 for x in [d['h'], d['t'], d['u']] if x >= 5)
            ctx['big_small_cnt'][f'{big}:{3-big}'] += 1
        
        return ctx
    
    def score_candidate(self, b: int, s: int, g: int, ctx: Dict, style: str = 'balanced') -> Tuple[float, Dict]:
        """
        候选评分（基于理论概率，不是"模式识别"）
        
        评分维度：
        1. 和值合理性（基于理论概率：7-20占83.2%）
        2. 跨度合理性（基于理论概率：4-7占56.4%）
        3. 形态加分（组六72% > 组三27% > 豹子1%）
        4. 奇偶平衡（1:2或2:1各37.5%）
        5. 大小平衡（1:2或2:1各37.5%）
        6. 012路多样性（0路40%、1路30%、2路30%）
        7. 质合平衡（质数2,3,5,7占40%）
        8. 用户偏好调整（保守/激进）
        
        注意：不包含以下"伪科学"维度：
        - 贝叶斯后验（独立随机事件退化为无用）
        - 指数衰减（暗示近期热号更可能出现，是赌徒谬误）
        - 复隔中分析（"上期号码"下期概率仍是10%）
        - 伴随号分析（小样本伪模式）
        - 信息熵（描述历史，不预测未来）
        """
        details = {}
        score = 0.0
        
        # ── 维度1: 和值合理性（满分20）──
        hz = b + s + g
        hz_prob = self.SUM_PROBS.get(hz, 0)
        # 和值7-20占83.2%，这些和值的概率较高
        if 7 <= hz <= 20:
            d1 = 20
        elif 5 <= hz <= 22:
            d1 = 15
        elif 3 <= hz <= 24:
            d1 = 8
        else:
            d1 = 0
        details['和值'] = round(d1, 1)
        score += d1
        
        # ── 维度2: 跨度合理性（满分15）──
        kd = max(b, s, g) - min(b, s, g)
        if 4 <= kd <= 7:
            d2 = 15
        elif 3 <= kd <= 8:
            d2 = 10
        elif 2 <= kd <= 9:
            d2 = 5
        else:
            d2 = 1
        details['跨度'] = round(d2, 1)
        score += d2
        
        # ── 维度3: 形态加分（满分15）──
        unique = len(set([b, s, g]))
        if unique == 3:
            d3 = 15  # 组六72%
        elif unique == 2:
            d3 = 8   # 组三27%
        else:
            d3 = 1   # 豹子1%
        details['形态'] = round(d3, 1)
        score += d3
        
        # ── 维度4: 奇偶平衡（满分10）──
        odd_count = sum(1 for x in [b, s, g] if x % 2 == 1)
        if odd_count in [1, 2]:
            d4 = 10
        elif odd_count == 0 or odd_count == 3:
            d4 = 2
        else:
            d4 = 5
        details['奇偶'] = round(d4, 1)
        score += d4
        
        # ── 维度5: 大小平衡（满分10）──
        big_count = sum(1 for x in [b, s, g] if x >= 5)
        if big_count in [1, 2]:
            d5 = 10
        elif big_count == 0 or big_count == 3:
            d5 = 2
        else:
            d5 = 5
        details['大小'] = round(d5, 1)
        score += d5
        
        # ── 维度6: 012路多样性（满分8）──
        roads = set([b % 3, s % 3, g % 3])
        if len(roads) == 3:
            d6 = 8
        elif len(roads) == 2:
            d6 = 4
        else:
            d6 = 0
        details['012路'] = round(d6, 1)
        score += d6
        
        # ── 维度7: 质合平衡（满分5）──
        prime_count = sum(1 for x in [b, s, g] if x in self.PRIMES)
        if prime_count in [1, 2]:
            d7 = 5
        elif prime_count == 0 or prime_count == 3:
            d7 = 1
        else:
            d7 = 3
        details['质合'] = round(d7, 1)
        score += d7
        
        # ── 维度8: 用户偏好调整（满分10）──
        if style == 'conservative':
            # 保守型：偏好热号（近期出现频率高的数字）
            heat = ctx['weighted'].get(b, 0) + ctx['weighted'].get(s, 0) + ctx['weighted'].get(g, 0)
            d8 = min(heat * 0.5, 10)
        elif style == 'aggressive':
            # 激进型：偏好冷号（近期出现频率低的数字）
            cold_bonus = 0
            for pos, digit in enumerate([b, s, g]):
                if ctx['freq20'][pos].get(digit, 0) <= 1:
                    cold_bonus += 3
            d8 = min(cold_bonus, 10)
        else:
            d8 = 0
        details['偏好'] = round(d8, 1)
        score += d8
        
        return round(score, 1), details
    
    def generate_candidates(self, data: List[Dict], ctx: Dict, style: str = 'balanced', n_candidates: int = 500) -> List[Tuple]:
        """
        多策略候选生成（基于理论概率）
        
        策略1: 和值目标采样（和值7-20占83.2%）
        策略2: 跨度目标采样（跨度4-7占56.4%）
        策略3: 形态目标采样（组六72%、组三27%）
        策略4: 蒙特卡洛+多维过滤
        策略5: 热号组合（用户偏好）
        """
        candidates = set()
        
        # ── 策略1: 和值目标采样（150次）──
        target_hz = int(ctx['hz_avg'])
        for _ in range(150):
            hz_target = target_hz + random.randint(-3, 3)
            b = random.randint(0, 9)
            s = random.randint(0, 9)
            g_needed = hz_target - b - s
            if 0 <= g_needed <= 9:
                candidates.add((b, s, g_needed))
        
        # ── 策略2: 跨度目标采样（120次）──
        for _ in range(120):
            target_span = random.choice([4, 5, 6, 7])  # 高频跨度
            b = random.randint(0, 9)
            lo = max(0, b - target_span)
            hi = min(9, b + target_span)
            if hi - lo >= target_span:
                g = lo + target_span
                s = random.randint(lo, g)
                candidates.add((b, s, g))
        
        # ── 策略3: 形态目标采样（100次）──
        for _ in range(100):
            # 72%概率生成组六，27%概率生成组三
            if random.random() < 0.72:
                # 组六：三位不同
                b, s, g = random.sample(range(10), 3)
            else:
                # 组三：两位相同
                b = random.randint(0, 9)
                s = random.randint(0, 9)
                g = b if random.random() < 0.5 else s
            candidates.add((b, s, g))
        
        # ── 策略4: 蒙特卡洛+多维过滤（200次）──
        for _ in range(300):
            b, s, g = random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
            odd = sum(1 for x in [b, s, g] if x % 2 == 1)
            big = sum(1 for x in [b, s, g] if x >= 5)
            
            # 多维过滤（基于理论概率）
            if (7 <= hz <= 20 and  # 和值合理（83.2%）
                4 <= kd <= 7 and   # 跨度合理（56.4%）
                odd in [1, 2] and  # 奇偶平衡（75%）
                big in [1, 2]):    # 大小平衡（75%）
                candidates.add((b, s, g))
        
        # ── 策略5: 热号组合（50次，用户偏好）──
        if style == 'conservative':
            for _ in range(50):
                combo = []
                for pos in range(3):
                    hot_nums = [n for n, _ in ctx['freq20'][pos].most_common(5)]
                    combo.append(random.choice(hot_nums))
                candidates.add(tuple(combo))
        
        return list(candidates)
    
    def select_top10_with_coverage(self, candidates: List[Tuple], ctx: Dict, style: str = 'balanced') -> List[Dict]:
        """
        选择10注号码，保证覆盖面
        
        多样性约束：
        1. 跨度覆盖：至少4个不同跨度值
        2. 形态覆盖：组六7-8注，组三2-3注
        3. 和值覆盖：覆盖7-13和14-20两个区间
        4. 奇偶覆盖：1奇2偶和2奇1偶各至少3注
        5. 大小覆盖：1大2小和2大1小各至少3注
        """
        # 评分
        scored = []
        for b, s, g in candidates:
            sc, details = self.score_candidate(b, s, g, ctx, style)
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
            unique = len(set([b, s, g]))
            shape = 'baozi' if unique == 1 else ('zusan' if unique == 2 else 'zuliu')
            odd = sum(1 for x in [b, s, g] if x % 2 == 1)
            big = sum(1 for x in [b, s, g] if x >= 5)
            scored.append({
                'score': sc, 'b': b, 's': s, 'g': g,
                'hz': hz, 'kd': kd, 'shape': shape,
                'odd': odd, 'big': big, 'details': details
            })
        
        scored.sort(key=lambda x: -x['score'])
        
        selected = []
        seen = set()
        
        # 覆盖面计数器
        hz_mid = 0   # 7-13
        hz_big = 0   # 14-20
        span_covered = set()
        span_count = Counter()
        shape_covered = Counter()
        odd_covered = Counter()
        big_covered = Counter()
        selected_sums = []
        
        def get_hz_range(hz):
            if hz <= 6: return 'small'
            elif hz <= 13: return 'mid'
            elif hz <= 20: return 'big'
            else: return 'large'
        
        # 第零轮：强制选2注组三
        zusan_scored = [item for item in scored if item['shape'] == 'zusan']
        random.shuffle(zusan_scored)
        for item in zusan_scored:
            if shape_covered['zusan'] >= 2:
                break
            combo = (item['b'], item['s'], item['g'])
            if combo not in seen:
                selected.append(item)
                seen.add(combo)
                span_covered.add(item['kd'])
                span_count[item['kd']] += 1
                shape_covered['zusan'] += 1
                odd_covered[item['odd']] += 1
                big_covered[item['big']] += 1
                selected_sums.append(item['hz'])
                hz_range = get_hz_range(item['hz'])
                if hz_range == 'mid': hz_mid += 1
                elif hz_range == 'big': hz_big += 1
        
        # 第一轮：强制保证覆盖面
        for item in scored[:200]:
            b, s, g = item['b'], item['s'], item['g']
            combo = (b, s, g)
            
            if combo in seen:
                continue
            
            # 跨度多样性硬约束
            if span_count[item['kd']] >= 2:
                continue
            
            # 和值分散约束
            too_close = any(abs(item['hz'] - s) < 3 for s in selected_sums)
            if too_close and len(selected) >= 5:
                continue
            
            need = False
            hz_range = get_hz_range(item['hz'])
            
            # 跨度：必须覆盖至少4种
            if item['kd'] in [4, 5, 6, 7] and item['kd'] not in span_covered:
                need = True
            if len(span_covered) < 4 and item['kd'] not in span_covered:
                need = True
            
            # 形态：组三至少2注
            if item['shape'] == 'zusan' and shape_covered['zusan'] < 2:
                need = True
            
            # 和值：mid和big各至少3注
            if hz_range == 'mid' and hz_mid < 3:
                need = True
            if hz_range == 'big' and hz_big < 3:
                need = True
            
            # 奇偶：1奇和2奇各至少3注
            if item['odd'] == 1 and odd_covered[1] < 3:
                need = True
            if item['odd'] == 2 and odd_covered[2] < 3:
                need = True
            
            # 大小：1大和2大各至少3注
            if item['big'] == 1 and big_covered[1] < 3:
                need = True
            if item['big'] == 2 and big_covered[2] < 3:
                need = True
            
            if need:
                selected.append(item)
                seen.add(combo)
                span_covered.add(item['kd'])
                span_count[item['kd']] += 1
                shape_covered[item['shape']] += 1
                odd_covered[item['odd']] += 1
                big_covered[item['big']] += 1
                selected_sums.append(item['hz'])
                if hz_range == 'mid': hz_mid += 1
                elif hz_range == 'big': hz_big += 1
                
                if len(selected) >= 10:
                    break
        
        # 第二轮：补充剩余名额
        remaining_candidates = [item for item in scored[:50] if (item['b'], item['s'], item['g']) not in seen]
        random.shuffle(remaining_candidates)
        
        for item in remaining_candidates:
            if len(selected) >= 10:
                break
            combo = (item['b'], item['s'], item['g'])
            if span_count[item['kd']] >= 3:
                continue
            if item['shape'] == 'zusan' and shape_covered['zusan'] >= 3:
                continue
            selected.append(item)
            seen.add(combo)
            span_covered.add(item['kd'])
            span_count[item['kd']] += 1
            shape_covered[item['shape']] += 1
        
        # 第三轮：放宽约束
        final_candidates = [item for item in scored if (item['b'], item['s'], item['g']) not in seen]
        random.shuffle(final_candidates)
        
        for item in final_candidates:
            if len(selected) >= 10:
                break
            combo = (item['b'], item['s'], item['g'])
            if combo not in seen:
                selected.append(item)
                seen.add(combo)
        
        return selected[:10]
    
    def generate_gold_silver(self, data: List[Dict], ctx: Dict, style: str = 'balanced') -> Tuple[Tuple, Tuple]:
        """
        生成金码和银码
        
        金码：基于理论概率+保守策略（偏热号）的最高评分
        银码：基于理论概率+激进策略（偏冷号）的最高评分
        """
        # 金码：综合评分最高（保守策略）
        gold_candidates = []
        for b in range(10):
            for s in range(10):
                for g in range(10):
                    hz = b + s + g
                    kd = max(b, s, g) - min(b, s, g)
                    if 7 <= hz <= 20 and 4 <= kd <= 7:
                        score, details = self.score_candidate(b, s, g, ctx, 'conservative')
                        gold_candidates.append((score, b, s, g, details))
        gold_candidates.sort(key=lambda x: -x[0])
        gold = gold_candidates[0][1:4]
        
        # 银码：综合评分最高（激进策略）
        silver_candidates = []
        for b in range(10):
            for s in range(10):
                for g in range(10):
                    if (b, s, g) == gold:
                        continue
                    hz = b + s + g
                    kd = max(b, s, g) - min(b, s, g)
                    if 7 <= hz <= 20 and 4 <= kd <= 7:
                        score, details = self.score_candidate(b, s, g, ctx, 'aggressive')
                        silver_candidates.append((score, b, s, g, details))
        silver_candidates.sort(key=lambda x: -x[0])
        silver = silver_candidates[0][1:4]
        
        return gold, silver
    
    def analyze_and_recommend(self, data: List[Dict], style: str = 'balanced', top_n: int = 10) -> Tuple[List[Dict], Dict]:
        """完整分析流程"""
        ctx = self.compute_context(data)
        candidates = self.generate_candidates(data, ctx, style=style)
        selected = self.select_top10_with_coverage(candidates, ctx, style=style)
        return selected, ctx
    
    def get_analysis_report(self, data: List[Dict], style: str = 'balanced') -> Dict:
        """生成分析报告"""
        ctx = self.compute_context(data)
        
        miss_stats = {}
        for pos, key in enumerate(['h', 't', 'u']):
            miss_stats[key] = {}
            for digit in range(10):
                miss_stats[key][digit] = ctx['miss'][pos].get(digit, 0)
        
        report = {
            'data_periods': len(data),
            'style': style,
            'version': 'V3.1',
            'method': '基于理论概率分布的智能选号（蒙特卡洛+多维过滤）',
            'statistics': {
                'theoretical_sum_mean': 13.5,
                'theoretical_span_mean': 4.5,
                'shape_distribution': self.SHAPE_PROBS,
                'recent_shapes': dict(ctx['recent_shapes']),
                'recent_sum_avg': ctx['hz_avg'],
                'miss_stats': miss_stats,
            },
            'hot_numbers': {
                'bai': [x[0] for x in ctx['freq20'][0].most_common(3)],
                'shi': [x[0] for x in ctx['freq20'][1].most_common(3)],
                'ge': [x[0] for x in ctx['freq20'][2].most_common(3)]
            },
            'cold_numbers': {
                'bai': [x[0] for x in ctx['freq20'][0].most_common()[-3:]],
                'shi': [x[0] for x in ctx['freq20'][1].most_common()[-3:]],
                'ge': [x[0] for x in ctx['freq20'][2].most_common()[-3:]]
            },
            'disclaimer': '基于理论概率分布的智能选号，不是预测。命中率≈1%（与随机选号相同），只提高选号质量。'
        }
        
        return report


if __name__ == '__main__':
    # 测试用
    print("SmartNumberSelector V3.1")
    print("请通过 run_analysis.py 调用")
