path = r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts\smart_selector.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 将路数遗漏统计改为基于 ctx 中已有数据，而不是 score 函数里的 data 参数
old = '''        # 融合路数遗漏：若某路在当前位置连续遗漏>=5期，组合包含该路给予额外奖励
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
        road_score = min(road_score + road_missing_bonus, 8)'''

new = '''        # 融合路数遗漏：若某路在当前位置连续遗漏>=5期，组合包含该路给予额外奖励
        road_missing_bonus = 0
        for pos, digit in enumerate(digits):
            road = digit % 3
            # 从 ctx 中读取该位置各路数的当前遗漏
            road_miss = ctx.get('road_miss', [{}, {}, {}])[pos].get(road, 0)
            if road_miss >= 5:
                road_missing_bonus += min(road_miss / 5, 2)
        road_score = min(road_score + road_missing_bonus, 8)'''

if old not in content:
    old_crlf = old.replace("\n", "\r\n")
    new_crlf = new.replace("\n", "\r\n")
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf)
        print("Replaced CRLF road")
    else:
        print("road block not found")
        exit(1)
else:
    content = content.replace(old, new)
    print("Replaced LF road")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done road")
