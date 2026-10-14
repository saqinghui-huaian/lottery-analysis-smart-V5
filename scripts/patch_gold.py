import re

path = r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts\smart_selector.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''    def generate_gold_silver(self, data: List[Dict], ctx: Dict, style: str = 'balanced') -> Tuple[Tuple, Tuple]:
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
                        score, details = self.score_candidate_v3(b, s, g, ctx, 'conservative')
                        gold_candidates.append((score, b, s, g, details))
        gold_candidates.sort(key=lambda x: -x[0])
        gold = gold_candidates[0][1:4]
        
        silver_candidates = []
        for b in range(10):
            for s in range(10):
                for g in range(10):
                    if (b, s, g) == gold: continue
                    hz = b + s + g
                    kd = max(b, s, g) - min(b, s, g)
                    if 7 <= hz <= 20 and 4 <= kd <= 7:
                        score, details = self.score_candidate_v3(b, s, g, ctx, 'aggressive')
                        silver_candidates.append((score, b, s, g, details))
        silver_candidates.sort(key=lambda x: -x[0])
        silver = silver_candidates[0][1:4]
        
        return gold, silver'''

new = '''    def generate_gold_silver(self, data: List[Dict], ctx: Dict, style: str = 'balanced') -> Tuple[Tuple, Tuple]:
        """
        生成金码和银码推荐：
        - 金码：全枚举 + 重号/邻号奖励，得分最高的号码
        - 银码：与金码数字重叠不超过1个、形态尽量不同的次高号码
        """
        if not data:
            return (1, 2, 3), (4, 5, 6)

        last_draw = (data[0]['h'], data[0]['t'], data[0]['u'])
        last_set = set(last_draw)
        neighbor_set = set()
        for d in last_draw:
            neighbor_set.update({(d - 1) % 10, (d + 1) % 10})
        neighbor_set -= last_set

        def full_enumerate(filter_style: str = 'balanced', exclude: set = None):
            candidates = []
            for b in range(10):
                for s in range(10):
                    for g in range(10):
                        combo = (b, s, g)
                        if exclude and combo in exclude:
                            continue
                        hz = b + s + g
                        kd = max(b, s, g) - min(b, s, g)
                        # 放宽枚举条件：只要和值不太极端
                        if 5 <= hz <= 22 and 2 <= kd <= 8:
                            score, details = self.score_candidate_v3(b, s, g, ctx, filter_style)
                            # 重号奖励：与上期号码有交集
                            overlap = len(set(combo) & last_set)
                            if overlap >= 1:
                                score += 1.5
                            # 邻号奖励：包含上期邻号
                            neighbor_overlap = len(set(combo) & neighbor_set)
                            if neighbor_overlap >= 1:
                                score += 1.0
                            # 金码偏好组六
                            if len(set(combo)) == 3:
                                score += 0.5
                            candidates.append((score, b, s, g, details))
            candidates.sort(key=lambda x: -x[0])
            return candidates

        # 金码：保守策略 + 重号/邻号奖励
        gold_candidates = full_enumerate('conservative')
        gold = gold_candidates[0][1:4]

        # 银码：与金码数字重叠 ≤1 个，且尽量不同形态
        gold_set = set(gold)
        silver_candidates = []
        for item in full_enumerate('aggressive'):
            score, b, s, g, details = item
            combo_set = set([b, s, g])
            overlap = len(combo_set & gold_set)
            if overlap <= 1:
                # 额外奖励：形态不同
                if len(combo_set) != len(gold_set):
                    score += 0.5
                silver_candidates.append((score, b, s, g, details))

        silver_candidates.sort(key=lambda x: -x[0])
        silver = silver_candidates[0][1:4] if silver_candidates else gold_candidates[1][1:4]

        return gold, silver'''

if old not in content:
    print("OLD block not found")
    # try with CRLF
    old_crlf = old.replace("\n", "\r\n")
    new_crlf = new.replace("\n", "\r\n")
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf)
        print("Replaced CRLF block")
    else:
        print("CRLF block also not found")
        exit(1)
else:
    content = content.replace(old, new)
    print("Replaced LF block")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
