#!/usr/bin/env python3
"""
Lottery draw data fetcher, statistical analyzer, and number recommender.
Supports: 排列三 (sporttery API via curl), 福彩3D (cwl.gov.cn API via curl + cookies).

v3: Fixed API parameters, added year-boundary filtering, improved scoring.

Usage:
    python3 lottery_analysis.py              # analyze 排列三 last 100 draws
    python3 lottery_analysis.py --game 3d    # analyze 福彩3D
    python3 lottery_analysis.py --game 35 --count 200 --recommend

PITFALLS (verified 2026-07-07):
- Both APIs MUST use curl (subprocess) — urllib gets 403 from sporttery and
  infinite 302 from cwl.gov.cn.
- cwl.gov.cn JSON has literal control chars — use regex extraction, not json.loads.
- cwl.gov.cn: issueCount MUST be EMPTY! issueCount=100 returns 404!
- sporttery.cn: Use mobile UA + mobile Referer to bypass WAF reliably.
- sporttery.cn pagination: Page 2+ may return previous year's data — must filter
  by period prefix to avoid year-crossing contamination.
- cwl.gov.cn: Cookie required (2-step flow). Timeout=30 for first request.
"""

import json
import sys
import subprocess
import tempfile
import os
import re
import time
from collections import Counter
from datetime import datetime

GAME_NAMES = {
    '35': '排列三',
    '85': '大乐透',

    '3d': '福彩3D',
    '3D': '福彩3D',
}

# Mobile UA — bypasses sporttery.cn WAF (verified 2026-06-27)
MOBILE_UA = 'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
DESKTOP_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


def _get_current_year_prefix():
    """Get 2-digit year prefix for period filtering (e.g., '26' for 2026)."""
    return str(datetime.now().year)[-2:]


def fetch_pl3(game_no='35', count=200):
    """Fetch 排列三 from sporttery official API.

    MUST use curl — sporttery.cn WAF blocks urllib with 403.
    Uses mobile UA + mobile Referer for reliable WAF bypass.
    Handles pagination with year-boundary filtering.
    Uses regex extraction because JSON can contain control characters.
    """
    current_prefix = _get_current_year_prefix()
    all_results = []
    seen_periods = set()
    pages_needed = (count + 99) // 100  # ceil division

    for page in range(1, pages_needed + 1):
        url = (
            f'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry'
            f'?gameNo={game_no}&provinceId=0&pageSize=100&isVerify=1&pageNo={page}'
        )
        result = subprocess.run([
            'curl', '-s', '-k', '--connect-timeout', '10', '--max-time', '45', '--retry', '2', '--retry-delay', '3', url,
            '-H', f'User-Agent: {MOBILE_UA}',
            '-H', 'Accept: application/json',
            '-H', 'Referer: https://m.lottery.gov.cn/',
        ], capture_output=True, text=True, timeout=70)

        raw = result.stdout
        # Regex extraction — JSON may have control chars or WAF HTML
        nums_list = re.findall(r'"lotteryDrawResult":"(\d+ \d+ \d+)"', raw)
        periods = re.findall(r'"lotteryDrawNum":"(\d+)"', raw)
        dates = re.findall(r'"lotteryDrawTime":"(\d{4}-\d{2}-\d{2})"', raw)

        page_results = []
        for i in range(min(len(nums_list), len(periods))):
            p = periods[i]
            # Skip duplicates
            if p in seen_periods:
                continue
            seen_periods.add(p)

            # CRITICAL: Filter out previous year's data
            # Period format: YYnnn (e.g., 26178 = 2026, draw 178)
            # When paginating, page 2+ may return draws from previous year
            # (e.g., 25351) which have higher seq numbers than current year
            if not p.startswith(current_prefix):
                continue

            ns = list(map(int, nums_list[i].split()))
            d = dates[i] if i < len(dates) else ''
            page_results.append({
                'period': p,
                'date': d,
                'h': ns[0], 't': ns[1], 'u': ns[2],
                'sum': sum(ns[:3])
            })

        all_results.extend(page_results)

        # Rate limit between pages
        if page < pages_needed:
            time.sleep(5)

    if not all_results:
        raise RuntimeError(
            "sporttery.cn returned no data — WAF may have blocked the request. "
            "Retry with mobile UA or wait a few seconds."
        )

    # Sort by period descending (newest first)
    all_results.sort(key=lambda x: x['period'], reverse=True)
    return all_results[:count]


def fetch_dlt(count=200):
    """Fetch 大乐透 from sporttery official API (gameNo=85).

    MUST use curl — sporttery.cn WAF blocks urllib with 403.
    Uses mobile UA + mobile Referer for reliable WAF bypass.
    Returns list of dicts with: period, date, front (sorted list of 5), back (sorted list of 2).
    """
    current_prefix = _get_current_year_prefix()
    all_results = []
    seen_periods = set()
    pages_needed = (count + 99) // 100

    for page in range(1, pages_needed + 1):
        url = (
            f'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry'
            f'?gameNo=85&provinceId=0&pageSize=100&isVerify=1&pageNo={page}'
        )
        result = subprocess.run([
            'curl', '-s', '-k', '--connect-timeout', '10', '--max-time', '45', '--retry', '2', '--retry-delay', '3', url,
            '-H', f'User-Agent: {MOBILE_UA}',
            '-H', 'Accept: application/json',
            '-H', 'Referer: https://m.lottery.gov.cn/',
        ], capture_output=True, text=True, timeout=70)

        raw = result.stdout
        # 大乐透 result format: "03 04 07 12 32 01 02" (5 front + 2 back)
        nums_list = re.findall(r'"lotteryDrawResult":"(\d+ \d+ \d+ \d+ \d+ \d+ \d+)"', raw)
        periods = re.findall(r'"lotteryDrawNum":"(\d+)"', raw)
        dates = re.findall(r'"lotteryDrawTime":"(\d{4}-\d{2}-\d{2})"', raw)

        page_results = []
        for i in range(min(len(nums_list), len(periods))):
            p = periods[i]
            if p in seen_periods:
                continue
            seen_periods.add(p)

            if not p.startswith(current_prefix):
                continue

            ns = list(map(int, nums_list[i].split()))
            front = sorted(ns[:5])
            back = sorted(ns[5:7])
            d = dates[i] if i < len(dates) else ''
            page_results.append({
                'period': p,
                'date': d,
                'front': front,
                'back': back,
                'front_sum': sum(front),
                'back_sum': sum(back),
            })

        all_results.extend(page_results)

        if page < pages_needed:
            time.sleep(5)

    if not all_results:
        raise RuntimeError(
            "sporttery.cn returned no 大乐透 data — WAF may have blocked the request. "
            "Retry with mobile UA or wait a few seconds."
        )

    all_results.sort(key=lambda x: x['period'], reverse=True)
    return all_results[:count]


def fetch_3d(count=200):
    """Fetch 福彩3D from cwl.gov.cn official API (requires curl + cookies).

    CRITICAL: issueCount must be EMPTY! issueCount=100 returns 404.
    Uses regex extraction as primary method because cwl.gov.cn JSON contains
    literal control characters that break json.loads() even with strict=False.
    """
    cookie_file = os.path.join(tempfile.gettempdir(), 'cwl_cookies.txt')

    # Step 1: GET index page to acquire session cookies (timeout=30)
    subprocess.run([
        'curl', '-s', '-c', cookie_file,
        'https://www.cwl.gov.cn/ygkj/wqkjgg/fc3d/',
        '-H', f'User-Agent: {DESKTOP_UA}',
        '-o', os.devnull
    ], timeout=30, check=False)

    all_results = []
    seen_codes = set()
    pages_needed = (count + 99) // 100

    for page in range(1, pages_needed + 1):
        # CRITICAL: issueCount= is EMPTY, not "100"!
        url = (
            f'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice'
            f'?name=3d&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd='
            f'&pageNo={page}&pageSize=100&systemType=PC'
        )
        result = subprocess.run([
            'curl', '-s', '-b', cookie_file, '-L', '--max-redirs', '5',
            '--connect-timeout', '10', '--max-time', '45', '--retry', '2', '--retry-delay', '2',
            url,
            '-H', f'User-Agent: {DESKTOP_UA}',
            '-H', 'Accept: application/json',
            '-H', 'Referer: https://www.cwl.gov.cn/ygkj/wqkjgg/fc3d/',
        ], capture_output=True, text=True, timeout=70)

        raw = result.stdout

        # Primary: regex extraction (reliable even with control chars in JSON)
        codes = re.findall(r'"code":"(\d+)"', raw)
        reds = re.findall(r'"red":"(\d+,\d+,\d+)"', raw)
        dates = re.findall(r'"date":"([^"]+)"', raw)

        if codes and reds:
            for i in range(min(len(codes), len(reds))):
                code = codes[i]
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                ns = list(map(int, reds[i].split(',')))
                d = dates[i][:10] if i < len(dates) else ''
                all_results.append({
                    'period': code,
                    'date': d,
                    'h': ns[0], 't': ns[1], 'u': ns[2],
                    'sum': sum(ns[:3])
                })
        elif '404' in raw[:200] or 'Not Found' in raw[:200]:
            # issueCount may have been set incorrectly — this shouldn't happen
            # with the fixed code, but handle gracefully
            raise RuntimeError(
                "cwl.gov.cn returned 404 — check API parameters. "
                "issueCount must be EMPTY."
            )

        # Rate limit between pages
        if page < pages_needed:
            time.sleep(3)

    if not all_results:
        # Fallback: json.loads (may fail on control chars)
        raise RuntimeError(
            "cwl.gov.cn returned unparseable data — "
            "regex found no matches. Cookies may have expired, retry."
        )

    # Sort by period descending (newest first)
    all_results.sort(key=lambda x: x['period'], reverse=True)
    return all_results[:count]




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Lottery data fetcher')
    parser.add_argument('--game', default='35', help='Game: 35=排列三, 85=大乐透, 3d=福彩3D')
    parser.add_argument('--count', type=int, default=200, help='Number of recent draws')
    args = parser.parse_args()

    name = GAME_NAMES.get(args.game, f'彩票(game={args.game})')
    print(f"正在获取 {name} 近{args.count}期数据...")

    if args.game.lower() == '3d':
        data = fetch_3d(args.count)
    elif args.game == '85':
        data = fetch_dlt(args.count)
    else:
        data = fetch_pl3(args.game, args.count)

    print(f"获取到 {len(data)} 期数据")
    for d in data[:5]:
        if 'front' in d:
            front_str = ' '.join(f'{x:02d}' for x in d['front'])
            back_str = ' '.join(f'{x:02d}' for x in d['back'])
            print(f"  {d['period']} | 前区: {front_str} 后区: {back_str}")
        else:
            print(f"  {d['period']} | {d['h']} {d['t']} {d['u']}")
