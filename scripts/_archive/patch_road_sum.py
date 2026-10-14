import re

path = r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts\smart_selector.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''        # ── 7. 012路多样性（满分5）──
        roads = set([x % 3 for x in digits])
        if len(roads) == 3: d7 = 5
        elif len(roads) == 2: d7 = 3
        else: d7 = 0
        details['012路'] = round(d7, 1)
        score += d7'''

new = '''        # ── 7. 012路多样性（满分5）──
        roads = set([x % 3 for x in digits])
        road_score = 0
        if len(roads) == 3:
            road_score = 5
        elif len(roads) == 2:
            road_score = 3
        else:
            road_score = 0
        # 融合路数遗漏：若某路在当前位置连续遗漏>=5期，组合包含该路给予额外奖励
        road_missing_bonus = 0
        for pos, digit in enumerate(digits):
            road = digit % 3
            # 统计该位置该路数近30期遗漏
            road_miss = 0
            for i, d in enumerate(data[:30]):
                if d[['h','t','u'][pos]] % 3 == road:
                    road_miss = i
                    break
                road_miss = i + 1
            if road_miss >= 5:
                road_missing_bonus += min(road_miss / 5, 2)
        road_score = min(road_score + road_missing_bonus, 8)
        details['012路'] = round(road_score, 1)
        score += road_score'''

if old not in content:
    old_crlf = old.replace("\n", "\r\n")
    new_crlf = new.replace("\n", "\r\n")
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf)
        print("Replaced CRLF block 012路")
    else:
        print("012路 block not found")
        exit(1)
else:
    content = content.replace(old, new)
    print("Replaced LF block 012路")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done 012路")
