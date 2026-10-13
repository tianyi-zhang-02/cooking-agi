"""Runs SQL for the counter tests on a real in-memory SQLite.

One JSON request per line on stdin: {"sql": "...", "params": [...]}.
One JSON reply per line on stdout: {"rows": [{...}]} or {"error": "..."}.
"""
import json
import sqlite3
import sys

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.row_factory = sqlite3.Row
for line in sys.stdin:
    request = json.loads(line)
    try:
        cursor = connection.execute(request["sql"], request.get("params", []))
        rows = [dict(row) for row in cursor.fetchall()]
        reply = {"rows": rows}
    except sqlite3.Error as error:
        reply = {"error": str(error)}
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
