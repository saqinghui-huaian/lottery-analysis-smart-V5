#!/usr/bin/env python3
"""
回测对比脚本 — V3 vs V4 vs Ensemble
=================================================================

对 3D / 排列三 / 大乐透 做滚动回测，横向对比三个模型版本的实际表现。

用法：
    python run_backtest.py --games 3d,35,85 --count 200 --train 100 --test 1 --step 10
    python run_backtest.py --games 3d --versions v3,ensemble

指标说明：
- 3D/排三：直选命中率、组选命中率、平均命中位数、ROI
- 大乐透：前区命中分布、后区命中分布、奖级分布、ROI

注意：彩票为负期望游戏，ROI 恒为负属正常。本脚本的价值在于
**横向对比不同模型的相对表现**，而非证明任何策略可以盈利。
"""

import sys
import os
import io
import json
import random
import argparse
from typing import Dict, List, Callable

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from lottery_analysis import fetch_3d, fetch_pl3, fetch_dlt
from backtest import backtest_3d, backtest_dlt

from smart_selector import SmartNumberSelector
from dlt_analysis import DLTAnalyzer

try:
    from smart_selector_v4 import SmartNumberSelectorV4
except Exception:
    SmartNumberSelectorV4 = None
try:
    from dlt_analysis_v4 import DLTAnalyzerV4
except Exception:
    DLTAnalyzerV4 = None
try:
    from ensemble import EnsembleRecommender3D, EnsembleRecommenderDLT
except Exception:
    EnsembleRecommender3D = None
    EnsembleRecommenderDLT = None


# ────────────────────────── 3D / 排三 recommender ──────────────────────────

def _norm_bets(items) -> List[Dict]:
    """把 (b,s,g) 三元组或含 num 的 dict 统一成含 b/s/g 的 dict。"""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            num = it.get('num')
            if isinstance(num, (tuple, list)) and len(num) >= 3:
                out.append({'b': num[0], 's': num[1], 'g': num[2]})
            elif all(k in it for k in ('b', 's', 'g')):
                out.append({'b': it['b'], 's': it['s'], 'g': it['g']})
        elif isinstance(it, (tuple, list)) and len(it) >= 3:
            out.append({'b': it[0], 's': it[1], 'g': it[2]})
    return out


def make_3d_recommender(version: str, n: int = 10) -> Callable[[List[Dict]], Dict]:
    """构造 3D/排三 的推荐函数。"""
    def _v3(data):
        smart = SmartNumberSelector()
        selected, _ = smart.analyze_and_recommend(data, style='balanced')
        return {'bets': _norm_bets(selected)[:n]}

    def _v4(data):
        if SmartNumberSelectorV4 is None:
            return {'bets': []}
        rec = SmartNumberSelectorV4(style='adaptive').recommend_v4(data, n=n)
        return {'bets': _norm_bets(rec.get('top10') or [])[:n]}

    def _ens(data):
        if EnsembleRecommender3D is None:
            return {'bets': []}
        raw = EnsembleRecommender3D(data, style='adaptive').recommend(n=n)
        return {'bets': _norm_bets(raw.get('bets') or [])[:n]}

    return {'v3': _v3, 'v4': _v4, 'ensemble': _ens}[version]


# ────────────────────────── 大乐透 recommender ──────────────────────────

def _norm_dlt_bets(items) -> List[Dict]:
    """把大乐透候选统一成含 front/back 的 dict。"""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            front, back = it.get('front'), it.get('back')
        elif isinstance(it, (tuple, list)) and len(it) >= 2:
            front, back = it[0], it[1]
        else:
            continue
        if front and back and len(front) == 5 and len(back) == 2:
            out.append({'front': list(front), 'back': list(back)})
    return out


def make_dlt_recommender(version: str, n: int = 5) -> Callable[[List[Dict]], Dict]:
    """构造大乐透的推荐函数。"""
    def _v3(data):
        selected = DLTAnalyzer(data, style='balanced').recommend(n_bets=n)
        # V3 返回 [(score, front, details, back), ...]
        bets = []
        for item in selected or []:
            if isinstance(item, (tuple, list)) and len(item) >= 4:
                bets.append({'front': list(item[1]), 'back': list(item[3])})
            else:
                bets.extend(_norm_dlt_bets([item]))
        return {'bets': bets[:n]}

    def _v4(data):
        if DLTAnalyzerV4 is None:
            return {'bets': []}
        rec = DLTAnalyzerV4(data, style='adaptive').recommend_v4(n=n)
        return {'bets': _norm_dlt_bets(rec.get('top5') or [])[:n]}

    def _ens(data):
        if EnsembleRecommenderDLT is None:
            return {'bets': []}
        raw = EnsembleRecommenderDLT(data, style='adaptive').recommend(n=n)
        return {'bets': _norm_dlt_bets(raw.get('bets') or [])[:n]}

    return {'v3': _v3, 'v4': _v4, 'ensemble': _ens}[version]


NAME_MAP = {'3d': '福彩3D', '35': '排列三', '85': '大乐透'}


def run_one(game_code: str, count: int, versions: List[str],
            train: int, test: int, step: int, min_tests: int = 30) -> List[Dict]:
    """对单个彩种跑多版本回测，返回结果列表。"""
    if game_code == '3d':
        data = fetch_3d(count)
    elif game_code == '35':
        data = fetch_pl3('35', count)
    else:
        data = fetch_dlt(count)

    print(f"[{NAME_MAP[game_code]}] 获取到 {len(data)} 期数据", file=sys.stderr)

    # 大乐透官方接口当前最多返回约 117 期；若训练窗口过大，滚动测试点会少到
    # 没有统计意义。这里按「目标测试点数」反推训练窗口上限并自适应收缩。
    target_tests = max(1, min_tests)
    max_train = len(data) - test - (target_tests - 1) * step
    if game_code == '85' and train > max_train:
        new_train = max(20, max_train)
        if new_train < train:
            print(f"  ! 训练窗口 {train} 过大（仅 {len(data)} 期可用），"
                  f"自动调整为 {new_train} 以保证约 {target_tests} 个测试点", file=sys.stderr)
            train = new_train

    rows = []
    for ver in versions:
        if game_code == '85':
            fn = make_dlt_recommender(ver, n=5)
            res = backtest_dlt(data, fn, train_window=train, test_window=test, step=step)
        else:
            fn = make_3d_recommender(ver, n=10)
            res = backtest_3d(data, fn, train_window=train, test_window=test, step=step)
        res['version'] = ver
        res['game'] = NAME_MAP[game_code]
        rows.append(res)
        print(f"  · {ver} 完成", file=sys.stderr)
    return rows


def render(rows: List[Dict], is_dlt: bool) -> str:
    """把回测结果渲染成 Markdown 表格。"""
    def _f(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return str(v)

    if is_dlt:
        lines = ["| 版本 | 测试点数 | 每注数 | 前区命中分布 | 后区命中分布 | 总投入 | 总奖金 | 净收益 | ROI |",
                 "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
        for r in rows:
            if 'error' in r:
                lines.append(f"| {r['version']} | - | - | 数据量不足：{r['error']} | | | | | |")
                continue
            fh = r.get('front_hit_distribution', {})
            bh = r.get('back_hit_distribution', {})
            fh_s = ' '.join(f"{k}中×{v}" for k, v in sorted(fh.items()))
            bh_s = ' '.join(f"{k}中×{v}" for k, v in sorted(bh.items()))
            lines.append(
                f"| {r['version']} | {r.get('total_tests', 0)} | {r.get('bets_per_test', 0)} | "
                f"{fh_s or '—'} | {bh_s or '—'} | {_f(r.get('total_cost', 0), 0)} | "
                f"{_f(r.get('total_prize', 0), 0)} | {_f(r.get('net_profit', 0), 0)} | {_f(r.get('roi', 0), 3)} |"
            )
        return '\n'.join(lines)

    lines = ["| 版本 | 测试点数 | 每注数 | 直选命中 | 直选率 | 组选命中 | 组选率 | 平均命中位数 | 总投入 | 总奖金 | ROI |",
             "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        if 'error' in r:
            lines.append(f"| {r['version']} | - | - | 数据量不足：{r['error']} | | | | | | |")
            continue
        # backtest_3d 返回的 *_hit_rate 已是「每个测试点」的比率，无需再除以注数
        lines.append(
            f"| {r['version']} | {r.get('total_tests', 0)} | {r.get('bets_per_test', 0)} | "
            f"{r.get('direct_hits', 0)} | {_f(r.get('direct_hit_rate', 0))} | "
            f"{r.get('group_hits', 0)} | {_f(r.get('group_hit_rate', 0))} | "
            f"{_f(r.get('avg_digit_hits', 0), 3)} | "
            f"{_f(r.get('total_cost', 0), 0)} | {_f(r.get('total_prize', 0), 0)} | {_f(r.get('roi', 0), 3)} |"
        )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='彩票推荐模型回测对比')
    parser.add_argument('--games', type=str, default='3d,35,85', help='彩种: 3d,35,85')
    parser.add_argument('--count', type=int, default=200, help='拉取期数 (默认200)')
    parser.add_argument('--versions', type=str, default='v3,v4,ensemble',
                        help='对比版本: v3,v4,ensemble')
    parser.add_argument('--train', type=int, default=100, help='训练窗口 (默认100)')
    parser.add_argument('--test', type=int, default=1, help='每次预测期数 (默认1)')
    parser.add_argument('--step', type=int, default=10, help='滚动步长 (默认10)')
    parser.add_argument('--json', action='store_true', help='输出 JSON 而非 Markdown')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子；固定后可复现对比（V4/ensemble 含随机过程）')
    parser.add_argument('--min-tests', type=int, default=30,
                        help='大乐透自适应收缩训练窗口时保证的最小测试点数（默认30）')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    aliases = {'3d': '3d', '3D': '3d', '35': '35', 'pl3': '35', '排三': '35', '排列三': '35',
               '85': '85', 'dlt': '85', '大乐透': '85'}
    codes = []
    for p in args.games.split(','):
        c = aliases.get(p.strip(), p.strip())
        if c in ('3d', '35', '85') and c not in codes:
            codes.append(c)
    codes.sort(key=lambda x: {'3d': 0, '35': 1, '85': 2}[x])

    versions = [v.strip() for v in args.versions.split(',') if v.strip()]

    all_rows = {}
    for code in codes:
        try:
            all_rows[code] = run_one(code, args.count, versions, args.train, args.test,
                                     args.step, args.min_tests)
        except Exception as e:
            print(f"[{NAME_MAP[code]}] 回测失败：{e}", file=sys.stderr)
            all_rows[code] = []

    if args.json:
        print(json.dumps(all_rows, ensure_ascii=False, indent=2))
        return

    print("## 📈 回测对比报告")
    print(f"- 数据：每彩种最近 {args.count} 期 | 训练窗口 {args.train} | 预测 {args.test} 期 | 步长 {args.step}")
    print("- 成本口径：3D/排三 2 元/注；大乐透 2 元/注")
    print()
    for code in codes:
        rows = all_rows.get(code) or []
        print(f"### {NAME_MAP[code]}")
        if not rows:
            print("（无有效结果）\n")
            continue
        print(render(rows, is_dlt=(code == '85')))
        print()
    print("> ⚠️ 彩票为负期望游戏，ROI 恒为负属正常现象。")
    print("> 本回测仅用于横向对比模型间的**相对**表现，不构成任何投注建议。")


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        os._exit(0)
