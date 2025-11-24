# File: admin.py
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import json, os

LOG_PATH = "logs/simplification_requests.json"
GLOSSARY_PATH = "glossary.json"

app = Flask(name)

TEMPLATE = """
<!doctype html>
<title>Admin - Simplification Monitor</title>
<h1>Admin Dashboard</h1>
<h2>Recent Simplification Requests</h2>
{% if entries %}
  <table border=1 cellpadding=5 cellspacing=0>
    <tr><th>Timestamp (UTC)</th><th>User</th><th>Level</th><th>Model</th><th>Input (snippet)</th><th>Output (snippet)</th></tr>
    {% for e in entries[::-1] %}
      <tr>
        <td>{{ e.timestamp }}</td>
        <td>{{ e.user }}</td>
        <td>{{ e.level }}</td>
        <td>{{ e.model_used }}</td>
        <td style="max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{{ e.input_snippet }}</td>
        <td style="max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{{ e.output_snippet }}</td>
      </tr>
    {% endfor %}
  </table>
{% else %}
  <p>No entries yet.</p>
{% endif %}

<h2>Manage Glossary</h2>
<form method="POST" action="/glossary">
  <textarea name="glossary" rows="10" cols="80">{{ glossary }}</textarea><br>
  <button type="submit">Save Glossary</button>
</form>
"""

def read_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_glossary():
    if not os.path.exists(GLOSSARY_PATH):
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/")
def index():
    entries = read_json(LOG_PATH)
    glossary = json.dumps(read_glossary(), indent=2)
    return render_template_string(TEMPLATE, entries=entries, glossary=glossary)

@app.route("/glossary", methods=["POST"])
def update_glossary():
    text = request.form.get("glossary", "{}")
    try:
        parsed = json.loads(text)
        with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
    except Exception as e:
        return f"Invalid JSON: {e}", 400
    return redirect(url_for("index"))

if name == "main":
    app.run(host="0.0.0.0", port=8501)  # run on 8501 by default