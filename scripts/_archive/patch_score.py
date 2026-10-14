path = r"C:\Users\21920\.qclaw\workspace\skills\lottery-analysis-smart-v2\scripts\smart_selector.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
old = "    def score_candidate_v3(self, b: int, s: int, g: int, ctx: Dict, style: str = 'balanced') -> Tuple[float, Dict]:"
new = "    def score_candidate_v3(self, b: int, s: int, g: int, ctx: Dict, style: str = 'balanced', data: List[Dict] = None) -> Tuple[float, Dict]:"
if old not in content:
    print("not found")
    exit(1)
content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("replaced")
