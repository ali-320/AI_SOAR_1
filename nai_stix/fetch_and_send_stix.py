import requests
import json

LOGSTASH_URL = "http://localhost:5044"
MITRE_STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

CHUNK_SIZE = 100  # number of STIX objects per request

def fetch_stix():
    print("[*] Fetching STIX 2.1 data from MITRE...")
    res = requests.get(MITRE_STIX_URL)
    res.raise_for_status()
    print("[+] Data fetched.")
    return res.json()

def split_bundle(bundle, chunk_size):
    objects = bundle["objects"]
    for i in range(0, len(objects), chunk_size):
        yield {
            "type": "bundle",
            "id": f"{bundle['id']}--chunk-{i}",
            "objects": objects[i:i + chunk_size]
        }

def send_to_logstash(bundle_chunk, index):
    headers = {'Content-Type': 'application/json'}
    print(f"[*] Sending chunk {index} with {len(bundle_chunk['objects'])} objects...")
    res = requests.post(LOGSTASH_URL, headers=headers, data=json.dumps(bundle_chunk))
    if res.status_code in [200, 202]:
        print(f"[+] Chunk {index} sent.")
    else:
        print(f"[!] Failed to send chunk {index}: {res.status_code}")

def main():
    try:
        bundle = fetch_stix()
        for i, chunk in enumerate(split_bundle(bundle, CHUNK_SIZE)):
            send_to_logstash(chunk, i + 1)
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
