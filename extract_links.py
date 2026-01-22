
import sys
import os
import json

# Add backend to path to import recommendations
sys.path.append(os.path.join(os.getcwd(), 'backend'))
try:
    from recommendations import RECOMMENDATIONS_POOL
except ImportError:
    print("Could not import recommendations")
    sys.exit(1)

links = {}
for mood, categories in RECOMMENDATIONS_POOL.items():
    for category, items in categories.items():
        for item in items:
            if 'link' in item and item['link']:
                links[item['content']] = item['link']
                # Also add versions without parentheses for matching old entries
                if "(" in item['content']:
                    base_name = item['content'].split("(")[0].strip()
                    if base_name not in links:
                        links[base_name] = item['link']

print(json.dumps(links, indent=2))
