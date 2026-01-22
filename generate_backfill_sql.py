
import json

with open('links.json', 'r') as f:
    links = json.load(f)

sql = "UPDATE recommendations SET link = CASE content\n"
for content, link in links.items():
    # Escape single quotes for SQL
    safe_content = content.replace("'", "''")
    sql += f"  WHEN '{safe_content}' THEN '{link}'\n"

sql += "ELSE link END WHERE category IN ('music', 'movie');"

with open('backfill.sql', 'w') as f:
    f.write(sql)
