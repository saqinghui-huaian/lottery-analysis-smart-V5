# Image + API Cross-Verification Workflow (走势图+API交叉验证)

## Why Cross-Verify?

走势图 images provide ~130 periods of data with rich statistics (frequency, max omission, avg interval).
Official APIs provide 300-400+ periods of real-time data.
Combining both sources produces more reliable analysis than either alone.

## Workflow (Step by Step)

### Step 1: Count and Identify Images
- Count every image the user sent
- Identify which lottery type each image represents (福彩3D / 排列三 / etc.)
- **NEVER add lottery types not represented by user's images**

### Step 2: Extract Image Data (parallel vision_analyze calls)
For each image, use vision_analyze with this prompt:
```
请仔细读取这张走势图的完整数据：
1. 底部号码统计表（每个数字0-9在百位、十位、个位的出现次数、最近遗漏、平均间隔）
2. 最近20期的期号、开奖号码、和值、跨度
```

Extract into structured dict:
```python
img_stats = {
    'bai': {'freq': {0:N, ...}, 'miss': {0:N, ...}, 'avg_miss': {0:N, ...}},
    'shi': {'freq': {0:N, ...}, 'miss': {0:N, ...}, 'avg_miss': {0:N, ...}},
    'ge':  {'freq': {0:N, ...}, 'miss': {0:N, ...}, 'avg_miss': {0:N, ...}},
    'hewei': {'freq': {...}, 'miss': {...}},
    'kuadu': {'freq': {...}, 'miss': {...}},
}
```

### Step 3: Fetch API Data (parallel curl calls)
- 福彩3D: cwl.gov.cn 2-step cookie flow, 3-4 pages (300-400 periods)
- 排列三: sporttery.cn mobile UA bypass, 3-4 pages (300-400 periods)

### Step 4: Cross-Verify
Check that the most recent 5-10 periods match between image and API:
```python
for img_period in img_recent[:5]:
    api_match = [d for d in api_data if d[0] == expected_period_str]
    if api_match:
        match = (api[1]==img[1] and api[2]==img[2] and api[3]==img[3])
```
- If all match → data is consistent, proceed
- If mismatch → trust API data, note discrepancy

### Step 5: Enhanced Scoring with Image Data
When scoring candidates, add image-based bonus for omission signals:
```python
# If API AND image both show high omission → stronger signal
pos_name = ['bai','shi','ge'][pos]
m_img = img_stats[pos_name]['miss'].get(digit, 0)
avg_img = img_stats[pos_name]['avg_miss'].get(digit, 10)
if m_img > avg_img * 2:
    score += 2  # cross-verified bonus
```

### Step 6: Output with Verification Markers
In the analysis logic, add "✅" when image data confirms API data:
```
- 个位**3**遗漏**43期**（均值11.3，3.8倍）→ 强力回补（走势图43期/16期=2.7倍✅）
```

## Common Trend Chart Sources

| Source | WeChat Account | Content |
|--------|---------------|---------|
| 鸿运走势图 | 鸿运走势图 | 福彩3D/排列三 开奖号码+和尾走势图 |
| 拼搏走势图 | 拼搏福彩猫 | 福彩3D/排列三 试机号走势图 |

- 鸿运走势图: Shows official draw numbers, bottom stats table with frequency/omission/avg_interval
- 拼搏走势图: Shows 试机号 (trial machine numbers), similar stats structure

## Pitfalls

- **vision_analyze may misread dense charts**: Always cross-verify with API. Trust API when they disagree.
- **Image period numbering**: Images may show 3-digit periods (043-172) while API uses 7-digit (2026043-2026172). Map accordingly.
- **Stats table scope**: Image stats cover ~130 periods (e.g., 043-172), API covers 300-400 periods. Different averages are expected.
- **Parallel execution**: Use parallel vision_analyze calls for multiple images, and parallel curl for multiple APIs. Don't serialize unnecessarily.
