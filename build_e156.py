"""
Build E156 index.html from paper.json using e156_interactive_template.html.
"""

import json
import html as html_mod
import re
import sys
import io
from pathlib import Path

if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEMPLATE_PATH = Path(r"C:\E156\templates\e156_interactive_template.html")
PAPER_JSON    = Path(r"C:\Models\EvidenceSpectral\e156-submission\paper.json")
OUT_HTML      = Path(r"C:\Models\EvidenceSpectral\e156-submission\index.html")

# Load paper.json
with open(PAPER_JSON, encoding="utf-8") as f:
    article = json.load(f)

# Load template
with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template = f.read()

# Build the article dict that matches the Codex template schema
notes = article.get("notes", {})
full_article = {
    "title": article.get("title", ""),
    "summary": article.get("summary", ""),
    "type": article.get("type", "methods"),
    "primary_estimand": article.get("primary_estimand", ""),
    "study_count": article.get("study_count"),
    "participant_count": article.get("participant_count"),
    "version": article.get("version", notes.get("version", "1.0")),
    "date": article.get("date", ""),
    "certainty": article.get("certainty", notes.get("certainty", "")),
    "app": notes.get("app", ""),
    "data": notes.get("data", ""),
    "code": notes.get("code", ""),
    "doi": notes.get("doi", ""),
    "protocol": notes.get("protocol", ""),
    "source_article": notes.get("source_article", ""),
    "body": article.get("body", ""),
    "validation": article.get("validation", {"status": "pass", "checks": []}),
    "sentences": article.get("sentences", []),
    "primary_plot": article.get("primary_plot", {}),
    "studies": article.get("studies", []),
    "search_strategy": article.get("search_strategy", {}),
    "prisma": article.get("prisma", {}),
    "included_papers": article.get("included_papers", []),
    "analysis_modules": article.get("analysis_modules", []),
    "author": article.get("author", ""),
    "slug": article.get("slug", ""),
    "schema": article.get("schema", "e156-v0.2"),
}

# Inject JSON into template (safe: escape </script in JSON)
safe_json = json.dumps(full_article, indent=2, ensure_ascii=False)
safe_json = re.sub(r"</script", r"<\\/script", safe_json, flags=re.IGNORECASE)

html = template.replace("__E156_JSON__", safe_json)
html = html.replace("__TITLE__", html_mod.escape(full_article["title"]))

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"E156 index.html written: {OUT_HTML}")
print(f"Lines: {html.count(chr(10))}")

# Quick validation
words = len(full_article["body"].split())
import re as _re
sents = len(_re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', full_article["body"].strip()))
print(f"Body: {words} words, {sents} sentences")
print(f"Title: {full_article['title']}")
