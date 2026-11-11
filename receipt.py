"""Receipt field extraction via GPT (image -> JSON).
Usage: python receipt.py <image>          -> prints fields as JSON
       python receipt.py --eval [samples/real]  -> exact-match accuracy per field vs groundtruth.yaml"""
import json, os, re, sys, unicodedata
from pathlib import Path
import gpt

MODEL = os.environ.get("RECEIPT_MODEL", "gpt-6.1-terra")
# ponytail: categories seen in the labeled data; extend the enum when new expense types appear
CATEGORIES = ["식대", "교통비", "기타"]
SCHEMA = {
    "type": "object",
    "properties": {
        "store_name": {"type": ["string", "null"], "description": "가맹점(상호)명. 택시는 운수회사명"},
        "total_amount": {"type": ["integer", "null"], "description": "최종 결제 금액(원), 쉼표 없이"},
        "transaction_datetime": {"type": ["string", "null"], "description": "거래 일시, YYYY-MM-DD HH:MM:SS (초가 없으면 YYYY-MM-DD HH:MM)"},
        "category": {"type": "string", "enum": CATEGORIES,
                     "description": "식대: 음식점·카페·편의점·제과점 등 먹을거리 구매. 교통비: 택시·버스·철도·주유·주차. 그 외: 기타"},
    },
    "required": ["store_name", "total_amount", "transaction_datetime", "category"],
    "additionalProperties": False,
}
RULES = "영수증 이미지에서 항목을 추출한다. 이미지에 없거나 읽을 수 없는 값은 추측하지 말고 null로 둔다."


def recognize(image_path, model=MODEL):
    """Returns {"store_name", "total_amount", "transaction_datetime", "category"}; unreadable fields are None."""
    r = gpt.create(
        model=model, instructions=RULES,
        input=[{"role": "user", "content": [gpt.image_part(image_path)]}],
        text={"format": {"type": "json_schema", "name": "receipt", "schema": SCHEMA, "strict": True}},
    )
    return json.loads(r.output_text)


def score(pred, g):
    """Exact match per field (store name ignores spaces/punctuation, datetime compared to the minute)."""
    nfc = lambda s: unicodedata.normalize("NFC", str(s or ""))
    squash = lambda s: re.sub(r"[\s().·\-,]+", "", nfc(s)).lower()
    return {
        "amount": pred["total_amount"] == g["total_amount"],
        "datetime": nfc(pred["transaction_datetime"])[:16] == str(g["transaction_datetime"])[:16],
        "store": squash(pred["store_name"]) == squash(g["store_name"]),
        "category": pred["category"] == g["category"],
    }


assert score({"total_amount": 59000, "transaction_datetime": "2026-04-08 18:24", "store_name": "그리너(남대문점)", "category": "식대"},
             {"total_amount": 59000, "transaction_datetime": "2026-04-08 18:24:01", "store_name": "그리너 남대문점", "category": "식대"}) \
    == {"amount": True, "datetime": True, "store": True, "category": True}

if __name__ == "__main__":
    if sys.argv[1] != "--eval":
        print(json.dumps(recognize(sys.argv[1]), ensure_ascii=False, indent=1))
        sys.exit()
    import yaml
    from concurrent.futures import ThreadPoolExecutor
    root = Path(sys.argv[2] if len(sys.argv) > 2 else "samples/real")
    gts = yaml.safe_load((root / "groundtruth.yaml").read_text())["samples"]
    files = {unicodedata.normalize("NFC", p.name): p for p in root.rglob("*") if p.is_file()}
    with ThreadPoolExecutor(8) as ex:
        preds = list(ex.map(lambda g: recognize(files[unicodedata.normalize("NFC", g["file"])]), gts))
    rows = [{"file": g["file"], **score(p, g), "pred": p} for g, p in zip(gts, preds)]
    for r in rows:
        print(r["file"], {k: v for k, v in r.items() if k not in ("file", "pred")}, r["pred"])
    print(f"\n[{MODEL}] TOTAL", {k: sum(r[k] for r in rows) for k in ("amount", "datetime", "store", "category")}, "/", len(rows))
    Path(f"receipt_result_{MODEL}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
