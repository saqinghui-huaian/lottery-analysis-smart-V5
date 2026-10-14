path = r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts\dlt_analysis.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. 增强遗漏评分：引入 Z-score
old1 = '''        # ── 维度11: 遗漏回补（满分5）──
        # 甜蜜区: 遗漏8-20期（理论期望6期，略超期望但不过度）
        miss_score = 0
        for n in combo:
            m = self.front_miss.get(n, 0)
            if 8 <= m <= 20:
                miss_score += 1.0
            elif 20 < m <= 30:
                miss_score += 0.5  # 过度遗漏，加分递减
            elif m > 30:
                miss_score += 0.2  # 极端遗漏，几乎不加分
        d11 = min(miss_score, 5)
        details['遗漏'] = round(d11, 1)
        score += d11'''

new1 = '''        # ── 维度11: 遗漏回补（满分8）──
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
        score += d11'''

if old1 not in content:
    print("old1 not found")
    exit(1)
content = content.replace(old1, new1)

# 2. 增强AC值评分权重
old2 = '''        # ── 维度4: AC值（满分10）──
        ac = self.calc_ac(combo)
        if ac >= 6:
            d4 = 10
        elif ac == 5:
            d4 = 8
        elif ac == 4:
            d4 = 5
        else:
            d4 = -5  # AC<4 排除
        details['AC值'] = round(d4, 1)
        score += d4'''

new2 = '''        # ── 维度4: AC值（满分12）──
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
        score += d4'''

if old2 not in content:
    print("old2 not found")
    exit(1)
content = content.replace(old2, new2)

# 3. 后区加入Z-score遗漏
old3 = '''        # 遗漏甜蜜区 5-15期
        for n in combo:
            m = self.back_miss.get(n, 0)
            if 5 <= m <= 15:
                score += 3
            elif m > 20:
                score += 1'''

new3 = '''        # 遗漏甜蜜区 5-15期（理论期望 1/(2/12)-1=5期）
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
                score += 0.5'''

if old3 not in content:
    print("old3 not found")
    exit(1)
content = content.replace(old3, new3)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done dlt patches")
