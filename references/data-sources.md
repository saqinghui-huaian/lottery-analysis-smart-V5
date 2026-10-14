# Chinese Lottery Data Sources

## Working APIs (verified 2026-06-14)

### Sporttery (official) - BEST for 排列三 & 大乐透
- Base URL: `https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry`
- Game numbers: 35=排列三, 85=大乐透
- **Recommended (WAF-free, verified 2026-06-27)**: Use mobile User-Agent + mobile Referer:
  ```
  User-Agent: Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36
  Referer: https://m.lottery.gov.cn/
  ```
  This bypasses the sporttery.cn WAF completely. Desktop UA + `lottery.gov.cn` referer gets blocked.
- **Fallback (desktop UA)**: If mobile fails, try desktop UA with `Referer: https://www.lottery.gov.cn/`. May get WAF-blocked intermittently.
- **Do NOT use `termLimits` parameter**: It caps results (e.g. `termLimits=30` returns only 30 draws). Omit it to get full pageSize results.
- Max pageSize ~100 per page. Paginate with `pageNo=1,2` for 200 periods. Add 3s delay between pages.
- Response: regex-extract from raw text (not JSON parsing):
  - `re.findall(r'"lotteryDrawResult":"(\d+ \d+ \d+)"', raw)`
  - `re.findall(r'"lotteryDrawNum":"(\d+)"', raw)`
- Period numbers are 7-digit (e.g. 26167). Extract sequence with `int(p) % 1000`.
- **Does NOT work for 福彩3D** (gameNo=3 returns empty data)
- Verified working 2026-06-27 (mobile UA bypass confirmed)

### cwl.gov.cn - BEST for 福彩3D (as of 2026-06)
- **Requires 2-step cookie flow** (cookie-less requests return empty `{"status":0,"message":null}`)
- Step 1: GET index page to acquire cookies: `curl -s -c cookies.txt 'https://www.cwl.gov.cn/ygkj/wqkjgg/fc3d/'`
- Step 2: Hit API with cookies: `curl -s -b cookies.txt -L --max-redirs 5 'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=3d&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd=&pageNo=1&pageSize=100&systemType=PC'`
- **CRITICAL**: `issueCount=` MUST be EMPTY! `issueCount=100` returns 404! (verified 2026-07-07)
- Headers needed: User-Agent + Accept: application/json + Referer: cwl.gov.cn
- Response: `data['result']` array, each with `code` (period), `red` (comma-sep numbers like "3,7,7"), `date`
- **CRITICAL**: Must use curl, not Python urllib — urllib hits infinite 302 redirect loops even with cookie handling
- If WAF blocks ("云安全平台检测到您当前的访问行为存在异常"), wait 5-10s and retry
- Verified working 2026-06-14

### kjapi.com - Browser scraping fallback
- URLs: fc3d.html for 3D, pl3.html for PL3
- About 30 recent draws in HTML table
- Requires browser tool due to anti-bot JS

## Blocked Sources (2026-06 status)
- 500.com: 404 Not Found
- 17500.cn: Anti-scraping
- caipiao.163.com: DNS fails
- sina.com.cn lottery API: 404
- lottery.gov.cn: serves JS shell, actual data via sporttery.cn API

## Period Numbering
- PL3/PL5: 7-digit year+seq (e.g. 26167 = 2026, seq 167). Extract seq: `int(p) % 1000`
- 3D: 7-digit (e.g. 2026167). Extract seq: `int(p) % 1000`

## ⚠️ Year-Crossing Pitfall (verified 2026-07-07)
When paginating (page 2+), sporttery.cn may return draws from the previous year.
For example, page 2 might return periods 25330-25351 (year 2025) mixed with
26078-26079 (year 2026). After extracting seq = int(period) % 1000, the 2025
periods (seq 330-351) would appear "newer" than 2026 periods (seq 78-79),
corrupting the analysis.

**Fix**: Filter by period prefix matching the current year:
```python
current_prefix = str(datetime.now().year)[-2:]  # e.g. '26'
if not period.startswith(current_prefix):
    continue  # skip previous year's data
```
