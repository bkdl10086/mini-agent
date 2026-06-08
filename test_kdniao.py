import os
import json
import hashlib
import base64
import urllib.request
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ebusiness_id = os.getenv("KDNIAO_EBUSINESS_ID")
app_key = os.getenv("KDNIAO_APP_KEY")

print(f"ID: {ebusiness_id}")
print(f"Key: {app_key}\n")

# 方案1: 标准 form-urlencoded
request_data = json.dumps({"ShipperCode": "YD", "LogisticCode": "321178654600766"})
sign_raw = request_data + app_key
sign_b64 = base64.b64encode(hashlib.md5(sign_raw.encode()).digest()).decode()
sign_enc = urllib.parse.quote(sign_b64, safe='')

body = f"RequestData={urllib.parse.quote(request_data)}&EBusinessID={ebusiness_id}&RequestType=8002&DataSign={sign_enc}&DataType=2"
print("=== 方案1: 手动构建 form ===")
print(f"Body: {body[:200]}")
print(f"Sign raw: {sign_raw}")
print(f"Sign b64: {sign_b64}")

req = urllib.request.Request(
    "https://api.kdniao.com/api/dist",
    data=body.encode(),
    headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"}
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f"Result: {r.read().decode()}\n")
except Exception as e:
    print(f"Error: {e}\n")

# 方案2: 带 OrderCode
request_data2 = json.dumps({"OrderCode": "", "ShipperCode": "YD", "LogisticCode": "321178654600766"})
sign_raw2 = request_data2 + app_key
sign_b642 = base64.b64encode(hashlib.md5(sign_raw2.encode()).digest()).decode()
sign_enc2 = urllib.parse.quote(sign_b642, safe='')

body2 = f"RequestData={urllib.parse.quote(request_data2)}&EBusinessID={ebusiness_id}&RequestType=8002&DataSign={sign_enc2}&DataType=2"
print("=== 方案2: 带 OrderCode ===")
req2 = urllib.request.Request(
    "https://api.kdniao.com/api/dist",
    data=body2.encode(),
    headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"}
)
try:
    with urllib.request.urlopen(req2, timeout=10) as r:
        print(f"Result: {r.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
