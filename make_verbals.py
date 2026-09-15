#!/usr/bin/env python3

import sqlite3
import csv

DB = "ilc.sqlite"
TABLE = "ilc"
OUTFILE = "verbals.tsv"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()



conn.text_factory = bytes
conn.row_factory = sqlite3.Row



# Read the whole table into memory
rows = cur.execute(f"""
    SELECT notation, verbal, synonyms, description, discipline
    FROM {TABLE}
""").fetchall()

# Index by notation
by_notation = {}
for r in rows:
    by_notation.setdefault(r["notation"], []).append(r)

# Helper
def join_nonempty(values):
    return ", ".join(
        str(v).strip()
        for v in values
        if v is not None and str(v).strip()
    )



def decode(x):
    if x is None:
        return ""

    if isinstance(x, bytes):
        return str(x.decode("latin1"))

    return str(x)

with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")

    writer.writerow([
        "notation",
        "field_1",
        "field_2",
        "field_3",
        "field_4",
    ])


    rows = [
        {k: decode(r[k]) for k in r.keys()}
        for r in rows
    ]

    for x in rows:

        notation = x["notation"]

        # Skip anything containing digits
        if any(c.isdigit() for c in notation):
            continue

        # field_1
        field_1 = x["verbal"] or ""

        # field_2
        field_2 = join_nonempty([
            x["verbal"],
            x["synonyms"],
            x["description"],
            x["discipline"],
        ])

        # field_3:
        # verbal of all records whose notation is an initial segment of X.notation
        prefixes = []

        for y in rows:
            n = y["notation"]
            if notation.startswith(n):
                prefixes.append(y["verbal"])

        field_3 = join_nonempty(prefixes)

        # field_4:
        # verbal of all records whose notation starts with X.notation
        # but is strictly longer and whose remainder contains no digits
        descendants = []

        for y in rows:
            n = y["notation"]

            if (
                n.startswith(notation)
                and n != notation
            ):
                suffix = n[len(notation):]

                if suffix and not any(c.isdigit() for c in suffix):
                    descendants.append(y["verbal"])

        field_4 = join_nonempty(descendants)

        writer.writerow([
            notation,
            field_1,
            field_2,
            field_3,
            field_4,
        ])

conn.close()
