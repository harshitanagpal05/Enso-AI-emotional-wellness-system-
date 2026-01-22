
import os
import sys
import json
import subprocess
import re

import requests

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
try:
    from recommendations import RECOMMENDATIONS_POOL
except ImportError:
    print("Could not import recommendations")
    sys.exit(1)

def check_link(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        content = response.text
        
        if response.status_code != 200:
            return False, f"Status {response.status_code}"
        
        # Check for common "unavailable" patterns
        content_lower = content.lower()
        if "video is unavailable" in content_lower or "this video isn't available anymore" in content_lower or "unavailable_video.png" in content_lower:
            return False, "Unavailable"
        if "private video" in content_lower:
            return False, "Private"
        if "terms of service" in content_lower and "removed" in content_lower:
            return False, "Removed"
        
        # Check if we actually got a youtube page
        if "YouTube" not in content:
            return False, "Not a YouTube page"
            
        return True, "OK"
    except Exception as e:
        return False, str(e)

all_links = []
for mood, categories in RECOMMENDATIONS_POOL.items():
    for category, items in categories.items():
        for item in items:
            if 'link' in item and item['link']:
                all_links.append((item['content'], item['link']))

print(f"Checking {len(all_links)} links...")
broken = []
for content, link in all_links:
    ok, reason = check_link(link)
    if not ok:
        print(f"[BROKEN] {content}: {reason} ({link})")
        broken.append({"content": content, "link": link, "reason": reason})
    else:
        print(f"[OK] {content}")

with open('broken_links.json', 'w') as f:
    json.dump(broken, f, indent=2)

print(f"\nSummary: {len(broken)} broken links found out of {len(all_links)}")
