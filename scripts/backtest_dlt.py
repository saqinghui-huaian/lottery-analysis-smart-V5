import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lottery_analysis as la
from dlt_analysis import DLTAnalyzer

def evaluate(lookback=50, test_periods=30):
    data = la.fetch_dlt(count=lookback + test_periods + 10)
    front_hits = 0
    back_hits = 0
    all_hits = 0
    base_front = 0
    for i in range(test_periods):
        train = data[i+1:i+1+lookback]
        actual = data[i]
        afront = set(actual['front'])
        aback = set(actual['back'])
        analyzer = DLTAnalyzer(train, style='balanced')
        selected = analyzer.recommend(5)
        # 检查精选5注中是否有前区3+命中、后区1+命中
        for item in selected:
            fset = set(item[1])
            bset = set(item[3])
            fh = len(fset & afront)
            bh = len(bset & aback)
            if fh >= 3:
                front_hits += 1
                break
        for item in selected:
            if len(set(item[3]) & aback) >= 1:
                back_hits += 1
                break
        # 随机基线
        rfront = set(random.sample(range(1,36), 5))
        rback = set(random.sample(range(1,13), 2))
        if len(rfront & afront) >= 3:
            base_front += 1
    return {
        'lookback': lookback,
        'periods': test_periods,
        'front3plus_hits': front_hits,
        'back1plus_hits': back_hits,
        'baseline_front3plus': base_front,
    }

if __name__ == '__main__':
    print(json.dumps(evaluate(50, 30), ensure_ascii=False))
    print(json.dumps(evaluate(100, 30), ensure_ascii=False))
