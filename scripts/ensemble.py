#!/usr/bin/env python3
"""
彩票分析系统集成与风格自适应模块

包含：
- StyleAdaptivePicker：根据近期熵、热号集中度、regime 自动选择风格
- EnsembleRecommender3D：V3/V4 3D 选择器 Borda 计数集成
- EnsembleRecommenderDLT：V3/V4 大乐透选择器 Borda 计数集成
- historical_similarity：历史同态匹配

仅使用标准库，并对 V4 模块进行优雅降级。
"""

import math
import random
import sys
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional

# 添加脚本目录到路径，便于本地导入
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 尝试导入 V3 选择器（必须存在）
from smart_selector import SmartNumberSelector
from dlt_analysis import DLTAnalyzer

# 尝试导入 V4 选择器（可能由其他子代理同时实现）
try:
    from smart_selector_v4 import SmartNumberSelectorV4
    _V4_3D_AVAILABLE = True
except Exception:
    SmartNumberSelectorV4 = None
    _V4_3D_AVAILABLE = False

try:
    from dlt_analysis_v4 import DLTAnalyzerV4
    _V4_DLT_AVAILABLE = True
except Exception:
    DLTAnalyzerV4 = None
    _V4_DLT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _entropy_of(counter: Counter, total: Optional[int] = None) -> float:
    """计算离散分布的香农熵（以 2 为底）。"""
    if total is None:
        total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for v in counter.values():
        if v > 0:
            p = v / total
            entropy -= p * math.log2(p)
    return entropy


def _regime_of(data: List[Dict], window: int = 10) -> str:
    """
    判断近期 regime：
    - hot：近 window 期出现的热号高度集中
    - cold：近 window 期重复度低，新号较多
    - mixed：介于两者之间
    """
    if not data or len(data) < window:
        return 'mixed'
    recent = data[:window]
    all_digits = []
    for d in recent:
        all_digits.extend([d.get('h', d.get('b', 0)),
                           d.get('t', d.get('s', 0)),
                           d.get('u', d.get('g', 0))])
    counter = Counter(all_digits)
    max_freq = max(counter.values())
    distinct = len(counter)
    # 30 个位置，10 个数字；若某个数字出现 >= 6 次认为高度集中
    if max_freq >= 6 or distinct <= 6:
        return 'hot'
    if max_freq <= 2 and distinct >= 9:
        return 'cold'
    return 'mixed'


class StyleAdaptivePicker:
    """
    自适应风格选择器。

    根据近期信息熵、热号集中度、regime 自动选择：
    - conservative（保守，追热号）
    - balanced（均衡）
    - aggressive（激进，追冷号/遗漏）
    """

    def __init__(self, data: List[Dict], game: str = '3d'):
        """
        Args:
            data: 开奖数据列表，3D 含 h/t/u 或 b/s/g；大乐透含 front/back。
            game: '3d' 或 'dlt'。
        """
        self.data = data
        self.game = game.lower()

    def _entropy_score(self) -> float:
        """返回近期熵与最大熵的比值，越接近 1 越分散。"""
        if not self.data:
            return 1.0
        window = min(20, len(self.data))
        if self.game == 'dlt':
            counter = Counter()
            for d in self.data[:window]:
                counter.update(d.get('front', []))
            max_entropy = math.log2(35)
        else:
            counter = Counter()
            for d in self.data[:window]:
                counter.update([d.get('h', d.get('b', 0)),
                                d.get('t', d.get('s', 0)),
                                d.get('u', d.get('g', 0))])
            max_entropy = math.log2(10)
        if max_entropy <= 0:
            return 1.0
        return _entropy_of(counter) / max_entropy

    def _heat_concentration(self) -> float:
        """返回热号集中度，0-1，越大越集中。"""
        if not self.data:
            return 0.5
        window = min(20, len(self.data))
        if self.game == 'dlt':
            counter = Counter()
            for d in self.data[:window]:
                counter.update(d.get('front', []))
            total = sum(counter.values())
        else:
            counter = Counter()
            for d in self.data[:window]:
                counter.update([d.get('h', d.get('b', 0)),
                                d.get('t', d.get('s', 0)),
                                d.get('u', d.get('g', 0))])
            total = sum(counter.values())
        if total <= 0:
            return 0.5
        top3 = sum(v for _, v in counter.most_common(3))
        return top3 / total

    def pick_style(self) -> str:
        """
        根据近期 entropy、热号集中度、regime 自动选择风格。

        Returns:
            'conservative' | 'balanced' | 'aggressive'
        """
        entropy_ratio = self._entropy_score()
        heat_ratio = self._heat_concentration()
        regime = _regime_of(self.data, window=10)

        # 熵低且热号集中 -> 追热，保守
        if entropy_ratio < 0.85 and heat_ratio > 0.45 and regime == 'hot':
            return 'conservative'
        # 熵高且热号分散 -> 追冷/遗漏，激进
        if entropy_ratio > 0.93 and heat_ratio < 0.35 and regime == 'cold':
            return 'aggressive'
        # 中间态默认 balanced
        return 'balanced'


class EnsembleRecommender3D:
    """
    3D/排三集成推荐器。

    同时加载 V3 与 V4 选择器（如果 V4 存在），独立生成候选后通过 Borda 计数合并，
    输出 top n。若 V4 不存在，优雅降级为 V3。
    """

    def __init__(self, data: List[Dict], style: str = 'adaptive'):
        """
        Args:
            data: 3D/排三开奖数据列表。
            style: 'conservative' | 'balanced' | 'aggressive' | 'adaptive'。
        """
        self.data = data
        if style == 'adaptive':
            picker = StyleAdaptivePicker(data, game='3d')
            self.style = picker.pick_style()
        else:
            self.style = style

        self.v3_selector = SmartNumberSelector()
        self.v4_selector = SmartNumberSelectorV4() if _V4_3D_AVAILABLE else None

    def _v3_candidates(self) -> Dict[Tuple[int, int, int], float]:
        """生成 V3 候选及其得分。"""
        ctx = self.v3_selector.compute_context(self.data)
        candidates = self.v3_selector.generate_candidates_v3(self.data, ctx, style=self.style)
        selected = self.v3_selector.select_top10_with_coverage(candidates, ctx, style=self.style)
        scores: Dict[Tuple[int, int, int], float] = {}
        for item in selected:
            combo = (item['b'], item['s'], item['g'])
            scores[combo] = item['score']
        # 若候选不足，对全部候选评分
        if len(scores) < 5:
            for combo in candidates[:50]:
                sc, _ = self.v3_selector.score_candidate_v3(*combo, ctx, self.style, self.data)
                scores[combo] = scores.get(combo, 0.0) + sc
        return scores

    def _v4_candidates(self) -> Dict[Tuple[int, int, int], float]:
        """生成 V4 候选及其得分（若不可用返回空）。"""
        if not self.v4_selector:
            return {}
        try:
            # 注意：V4 选择器的入口方法是 recommend_v4（不是 recommend），
            # 且返回 dict：{'gold','silver','top10':[{'num':(b,s,g),'score':..}, ...]}
            rec = self.v4_selector.recommend_v4(self.data, n=20, include_group=True)
            items: List[Dict] = []
            if isinstance(rec, dict):
                items = [x for x in (rec.get('top10') or []) if isinstance(x, dict)]
            elif isinstance(rec, (list, tuple)):
                items = [x for x in rec if isinstance(x, dict)]

            scores: Dict[Tuple[int, int, int], float] = {}
            for item in items:
                num = item.get('num')
                if isinstance(num, (tuple, list)) and len(num) >= 3:
                    combo = (int(num[0]), int(num[1]), int(num[2]))
                elif all(k in item for k in ('b', 's', 'g')):
                    combo = (int(item['b']), int(item['s']), int(item['g']))
                else:
                    continue
                try:
                    score = float(item.get('score') or 50.0)
                except (TypeError, ValueError):
                    score = 50.0
                scores[combo] = max(scores.get(combo, 0.0), score)
            return scores
        except Exception:
            return {}

    def recommend(self, n: int = 10) -> Dict:
        """
        集成推荐，返回 top n 注。

        Returns:
            Dict，包含：
            - 'bets': List[Dict]，每项含 b/s/g/score/style/source
            - 'style': 当前风格
            - 'sources': 参与集成的来源列表
        """
        v3_scores = self._v3_candidates()
        v4_scores = self._v4_candidates()

        sources = ['v3']
        if v4_scores:
            sources.append('v4')

        all_combos = set(v3_scores.keys()) | set(v4_scores.keys())
        if not all_combos:
            # 兜底：随机生成 n 注组六
            bets = []
            seen = set()
            while len(bets) < n:
                combo = tuple(sorted(random.sample(range(10), 3)))
                if combo not in seen:
                    seen.add(combo)
                    bets.append({'b': combo[0], 's': combo[1], 'g': combo[2],
                                 'score': 0.0, 'style': self.style, 'source': 'fallback'})
            return {'bets': bets, 'style': self.style, 'sources': sources}

        # Borda 计数：对每个来源单独排序后给分，再求和
        borda: Dict[Tuple[int, int, int], float] = {combo: 0.0 for combo in all_combos}

        def _add_borda(scores: Dict[Tuple[int, int, int], float], weight: float = 1.0):
            sorted_combos = sorted(scores.items(), key=lambda x: -x[1])
            for rank, (combo, _) in enumerate(sorted_combos):
                borda[combo] += weight * max(0, len(sorted_combos) - rank)

        _add_borda(v3_scores, weight=1.0)
        if v4_scores:
            _add_borda(v4_scores, weight=1.0)

        # 综合原始得分（标准化到 0-100）作为 tie-breaker
        def _normalize(scores: Dict[Tuple[int, int, int], float]) -> Dict[Tuple[int, int, int], float]:
            if not scores:
                return {}
            mx = max(scores.values())
            mn = min(scores.values())
            if mx <= mn:
                return {k: 50.0 for k in scores}
            return {k: (v - mn) / (mx - mn) * 100 for k, v in scores.items()}

        norm_v3 = _normalize(v3_scores)
        norm_v4 = _normalize(v4_scores)

        final_scores: Dict[Tuple[int, int, int], float] = {}
        for combo in all_combos:
            score = borda.get(combo, 0.0)
            if combo in norm_v3:
                score += norm_v3[combo] * 0.2
            if combo in norm_v4:
                score += norm_v4[combo] * 0.2
            final_scores[combo] = score

        top = sorted(final_scores.items(), key=lambda x: -x[1])[:n]
        bets = []
        for combo, score in top:
            src = []
            if combo in v3_scores:
                src.append('v3')
            if combo in v4_scores:
                src.append('v4')
            bets.append({'b': combo[0], 's': combo[1], 'g': combo[2],
                         'score': round(score, 2),
                         'style': self.style,
                         'source': '+'.join(src)})

        return {'bets': bets, 'style': self.style, 'sources': sources}


class EnsembleRecommenderDLT:
    """
    大乐透集成推荐器。

    同时加载 V3 与 V4 选择器（如果 V4 存在），独立生成候选后通过 Borda 计数合并，
    输出 top n。若 V4 不存在，优雅降级为 V3。
    """

    def __init__(self, data: List[Dict], style: str = 'adaptive'):
        """
        Args:
            data: 大乐透开奖数据列表。
            style: 'conservative' | 'balanced' | 'aggressive' | 'adaptive'。
        """
        self.data = data
        if style == 'adaptive':
            picker = StyleAdaptivePicker(data, game='dlt')
            self.style = picker.pick_style()
        else:
            self.style = style

        self.v3_selector = DLTAnalyzer(data, style=self.style)
        self.v4_selector = DLTAnalyzerV4(data, style=self.style) if _V4_DLT_AVAILABLE else None

    def _v3_candidates(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]:
        """生成 V3 候选。"""
        selected = self.v3_selector.recommend(n_bets=10)
        result = []
        for sc, front, details, back in selected:
            result.append((tuple(front), tuple(back), float(sc)))
        return result

    def _v4_candidates(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]:
        """生成 V4 候选（若不可用返回空）。"""
        if not self.v4_selector:
            return []
        try:
            # 注意：V4 分析器的入口方法是 recommend_v4（不是 recommend），
            # 且返回 dict：{'gold','silver','top5':[{'front','back',...}, ...]}
            rec = self.v4_selector.recommend_v4(n=10)
            items: List[Dict] = []
            if isinstance(rec, dict):
                items = [x for x in (rec.get('top5') or []) if isinstance(x, dict)]
            elif isinstance(rec, (list, tuple)):
                items = [x for x in rec if isinstance(x, dict)]

            result = []
            for item in items:
                front = tuple(item.get('front', []))
                back = tuple(item.get('back', []))
                if len(front) != 5 or len(back) != 2:
                    continue
                try:
                    score = float(item.get('score') or 50.0)
                except (TypeError, ValueError):
                    score = 50.0
                result.append((front, back, score))
            return result
        except Exception:
            return []

    def recommend(self, n: int = 5) -> Dict:
        """
        集成推荐，返回 top n 注。

        Returns:
            Dict，包含：
            - 'bets': List[Dict]，每项含 front/back/score/style/source
            - 'style': 当前风格
            - 'sources': 参与集成的来源列表
        """
        v3_list = self._v3_candidates()
        v4_list = self._v4_candidates()

        sources = ['v3']
        if v4_list:
            sources.append('v4')

        if not v3_list and not v4_list:
            # 兜底：随机生成 n 注
            bets = []
            for _ in range(n):
                front = tuple(sorted(random.sample(range(1, 36), 5)))
                back = tuple(sorted(random.sample(range(1, 13), 2)))
                bets.append({'front': list(front), 'back': list(back),
                             'score': 0.0, 'style': self.style, 'source': 'fallback'})
            return {'bets': bets, 'style': self.style, 'sources': sources}

        # 以前区为主键合并；后区作为独立属性
        # 若前区相同但后区不同，视为不同候选
        all_candidates: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        for front, back, score in v3_list:
            all_candidates[(front, back)] = all_candidates.get((front, back), 0.0) + score
        for front, back, score in v4_list:
            all_candidates[(front, back)] = all_candidates.get((front, back), 0.0) + score

        # Borda 计数：分别对 V3、V4 排序
        borda: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {
            k: 0.0 for k in all_candidates
        }

        def _add_borda(candidates: List[Tuple[Tuple[int, ...], Tuple[int, ...], float]], weight: float = 1.0):
            score_map = {(f, b): s for f, b, s in candidates}
            items = list(score_map.items())
            items.sort(key=lambda x: -x[1])
            for rank, (combo, _) in enumerate(items):
                borda[combo] += weight * max(0, len(items) - rank)

        _add_borda(v3_list, weight=1.0)
        if v4_list:
            _add_borda(v4_list, weight=1.0)

        # 综合原始得分作为 tie-breaker
        def _normalize(candidates: List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]) -> Dict[Tuple, float]:
            if not candidates:
                return {}
            scores = [s for _, _, s in candidates]
            mx = max(scores)
            mn = min(scores)
            if mx <= mn:
                return {(f, b): 50.0 for f, b, _ in candidates}
            return {(f, b): (s - mn) / (mx - mn) * 100 for f, b, s in candidates}

        norm_v3 = _normalize(v3_list)
        norm_v4 = _normalize(v4_list)

        final_scores: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        for combo in all_candidates:
            score = borda.get(combo, 0.0)
            if combo in norm_v3:
                score += norm_v3[combo] * 0.2
            if combo in norm_v4:
                score += norm_v4[combo] * 0.2
            final_scores[combo] = score

        top = sorted(final_scores.items(), key=lambda x: -x[1])[:n]
        bets = []
        v3_set = {(f, b) for f, b, _ in v3_list}
        v4_set = {(f, b) for f, b, _ in v4_list}
        for combo, score in top:
            front, back = combo
            src = []
            if combo in v3_set:
                src.append('v3')
            if combo in v4_set:
                src.append('v4')
            bets.append({'front': list(front), 'back': list(back),
                         'score': round(score, 2),
                         'style': self.style,
                         'source': '+'.join(src)})

        return {'bets': bets, 'style': self.style, 'sources': sources}


def historical_similarity(
    data: List[Dict],
    window: int = 10,
    k: int = 3,
    game: str = '3d'
) -> List[Dict]:
    """
    历史同态匹配。

    用最近 window 期与历史所有窗口比较：
    - 3D/排三用位置数字向量（h/t/u 拼接）。
    - 大乐透用前区排序后向量（5 个号码）。

    返回最相似的 k 个历史窗口及其后续走势。

    Args:
        data: 开奖数据列表。
        window: 比较窗口大小。
        k: 返回最相似窗口数量。
        game: '3d' 或 'dlt'。

    Returns:
        List[Dict]，每项包含：
        - 'start_idx': 历史窗口起始下标（从新到旧计数）
        - 'similarity': 相似度得分（越高越相似）
        - 'window': 该历史窗口数据
        - 'next': 窗口后紧接着的走势（下一期）
    """
    game = game.lower()
    n = len(data)
    if n < window + 2:
        return []

    def _vector(item: Dict) -> Tuple[int, ...]:
        if game == 'dlt':
            return tuple(sorted(item.get('front', [])))
        return (item.get('h', item.get('b', 0)),
                item.get('t', item.get('s', 0)),
                item.get('u', item.get('g', 0)))

    def _flatten_window(window_data: List[Dict]) -> List[int]:
        vec = []
        for d in window_data:
            vec.extend(_vector(d))
        return vec

    recent = data[:window]
    recent_vec = _flatten_window(recent)
    dim = len(recent_vec)

    candidates = []
    # 历史窗口从 data[1] 开始，保证后面至少有一期
    for start in range(1, n - window):
        hist_window = data[start:start + window]
        hist_vec = _flatten_window(hist_window)
        if len(hist_vec) != dim:
            continue
        # 欧氏距离（数值差绝对值和）
        dist = sum(abs(a - b) for a, b in zip(recent_vec, hist_vec))
        max_dist = dim * (9 if game != 'dlt' else 34)
        sim = 1.0 - dist / max_dist if max_dist > 0 else 1.0
        candidates.append((start, sim, hist_window, data[start + window]))

    candidates.sort(key=lambda x: -x[1])
    result = []
    for start, sim, hist_window, nxt in candidates[:k]:
        result.append({
            'start_idx': start,
            'similarity': round(sim, 4),
            'window': hist_window,
            'next': nxt,
        })
    return result


if __name__ == '__main__':
    # 简单自测
    fake_3d = [{'h': (i * 3) % 10, 't': (i * 5) % 10, 'u': (i * 7) % 10} for i in range(30)]
    picker = StyleAdaptivePicker(fake_3d, game='3d')
    print('3d style:', picker.pick_style())

    rec3d = EnsembleRecommender3D(fake_3d, style='balanced')
    print('3d ensemble bets:', len(rec3d.recommend(n=10)['bets']))

    fake_dlt = [{'front': sorted(random.sample(range(1, 36), 5)),
                 'back': sorted(random.sample(range(1, 13), 2))} for _ in range(100)]
    recdlt = EnsembleRecommenderDLT(fake_dlt, style='balanced')
    print('dlt ensemble bets:', len(recdlt.recommend(n=5)['bets']))

    sim = historical_similarity(fake_3d, window=5, k=2, game='3d')
    print('historical similarity count:', len(sim))
