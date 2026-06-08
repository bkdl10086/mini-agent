import os, json, hashlib, base64, urllib.request, urllib.parse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ebusiness_id = os.getenv("KDNIAO_EBUSINESS_ID")
app_key = os.getenv("KDNIAO_APP_KEY")

request_data = json.dumps({"LogisticCode": "773367326370601"})
sign_hex = hashlib.md5((request_data + app_key).encode()).hexdigest()
sign_b64 = base64.b64encode(sign_hex.encode()).decode()

body = urllib.parse.urlencode({
    "RequestData": request_data, "EBusinessID": ebusiness_id,
    "RequestType": "8002", "DataSign": sign_b64, "DataType": "2",
}).encode()

req = urllib.request.Request("https://api.kdniao.com/api/dist", data=body)
with urllib.request.urlopen(req, timeout=10) as r:
    result = json.loads(r.read().decode())
print(json.dumps(result, ensure_ascii=False, indent=2))
