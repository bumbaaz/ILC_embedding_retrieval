import sqlite3
import csv

with sqlite3.connect("ilc.sqlite") as conn:
    conn.text_factory = bytes
    rows = conn.execute(
        "SELECT notation, verbal FROM ilc"
    ).fetchall()

with open("notation_verbal.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")

    for notation, verbal in rows:
        notation = notation.decode("utf-8", errors="replace")
        verbal = verbal.decode("latin-1")  # or cp1252
        writer.writerow([notation, verbal])
