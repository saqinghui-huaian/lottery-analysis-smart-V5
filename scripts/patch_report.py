path = r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts\run_analysis.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''    # ── Bullet 4: 热号支撑（稳定模式：始终展示热号，附加012路或伴随号）──
    hot = get_hot_numbers(ctx, top_n=3)
    cold = get_cold_numbers(ctx, top_n=3)
    road = get_road_pattern(ctx)
    pairs = get_companion_pairs(data, top_n=3)
    trend = get_position_trend(ctx)
    
    # 主行：热号（始终展示）
    lines.append(f"- 热号支撑：百位{hot['百位']}，十位{hot['十位']}，个位{hot['个位']}")
    
    # 附加行：012路走势（如果有明显偏向）'''

new = '''    # ── Bullet 4: 热号+冷号+012路+伴随号（稳定模式：综合展示）──
    hot = get_hot_numbers(ctx, top_n=3)
    cold = get_cold_numbers(ctx, top_n=3)
    road = get_road_pattern(ctx)
    pairs = get_companion_pairs(data, top_n=3)
    trend = get_position_trend(ctx)
    
    # 主行：热号（始终展示）
    lines.append(f"- 热号支撑：百位{hot['百位']}，十位{hot['十位']}，个位{hot['个位']}")
    
    # 冷号行：如果存在明显冷号
    cold_parts = []
    for pos, label in enumerate(['百位', '十位', '个位']):
        # 取 freq20 末位且 miss >= 12
        cold_list = [x[0] for x in ctx['freq20'][pos].most_common()[-3:] if ctx['miss'][pos].get(x[0], 0) >= 12]
        if cold_list:
            cold_parts.append(f"{label}{'/'.join(str(x) for x in cold_list)}")
    if cold_parts:
        lines.append(f"- 冷号防守：{'，'.join(cold_parts)}")
    
    # 伴随号行（如果热门对子出现>=2次）
    if pairs and int(pairs[0].split('(')[1].split('次')[0]) >= 2:
        lines.append(f"- 伴随号：{'、'.join(pairs[:2])}")
    
    # 附加行：012路走势（如果有明显偏向）'''

if old not in content:
    old_crlf = old.replace("\n", "\r\n")
    new_crlf = new.replace("\n", "\r\n")
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf)
        print("Replaced CRLF report")
    else:
        print("report block not found")
        exit(1)
else:
    content = content.replace(old, new)
    print("Replaced LF report")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done report")
