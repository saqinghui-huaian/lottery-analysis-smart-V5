import sys, json, math, random
sys.path.insert(0, r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts")
import lottery_analysis as la
from smart_selector import SmartNumberSelector

def evaluate(lookback=100, mode='3d', test_periods=100):
    if mode == '3d':
        data = la.fetch_3d(count=lookback + test_periods + 10)
    else:
        data = la.fetch_pl3(count=lookback + test_periods + 10)
    selector = SmartNumberSelector()
    top10_hits = 0
    gold_hits = 0
    silver_hits = 0
    baseline_hits = 0
    random_hits = 0
    for i in range(test_periods):
        train = data[i+1:i+1+lookback]
        actual = data[i]['h']*100 + data[i]['t']*10 + data[i]['u']
        selected, ctx = selector.analyze_and_recommend(train, style='balanced', top_n=10)
        top_set = set(s['b']*100 + s['s']*10 + s['g'] for s in selected)
        if actual in top_set:
            top10_hits += 1
        gold, silver = selector.generate_gold_silver(train, ctx)
        gnum = gold[0]*100 + gold[1]*10 + gold[2]
        snum = silver[0]*100 + silver[1]*10 + silver[2]
        if actual == gnum:
            gold_hits += 1
        if actual == snum:
            silver_hits += 1
        baseline = set(random.randint(0,999) for _ in range(10))
        if actual in baseline:
            baseline_hits += 1
        rnum = random.randint(0,999)
        if actual == rnum:
            random_hits += 1
    return {
        'lookback': lookback,
        'mode': mode,
        'periods': test_periods,
        'top10': top10_hits,
        'top10_rate': round(top10_hits/test_periods,4),
        'gold': gold_hits,
        'gold_rate': round(gold_hits/test_periods,4),
        'silver': silver_hits,
        'silver_rate': round(silver_hits/test_periods,4),
        'baseline10_rate': round(baseline_hits/test_periods,4),
        'random_rate': round(random_hits/test_periods,4),
    }

if __name__ == '__main__':
    print(json.dumps(evaluate(50, '3d', 100), ensure_ascii=False))
    print(json.dumps(evaluate(100, '3d', 100), ensure_ascii=False))
    print(json.dumps(evaluate(150, '3d', 100), ensure_ascii=False))
