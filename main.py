import requests
import numpy as np
import sqlite3
import csv

import os
os.environ["HF_HUB_OFFLINE"] = "1"


from sentence_transformers import SentenceTransformer
import logging
logging.basicConfig(level=logging.INFO)


MAX_DEPTH = 6

VERBAL = {}

with open("notation_verbal.tsv", encoding="utf-8") as f:
    VERBAL = dict(csv.reader(f, delimiter="\t"))


def get_verbal(notation):
    return VERBAL.get(notation)





#
# Read all embeddings once
#


def load_embeddings(filename):
    embeddings = []

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as fin:

        for line in fin:

            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            notation = parts[0]

            embedding = np.array(
                [float(x) for x in parts[1:]],
                dtype=float,
            )

            embeddings.append({
                "notation": notation,
                "embedding": embedding,
                "norm": np.linalg.norm(embedding),
            })

    return embeddings



embeddings = load_embeddings("ilc_embeddings/classes.tsv")



#model = SentenceTransformer(
#    "all-MiniLM-L6-v2",
#    cache_folder="./models"
#)

model = SentenceTransformer(
    "./models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

def first_paragraph(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
    }

    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": "ILCAI/1.0 (your@email.com)"}
    )

    #print("HTTP status:", r.status_code)
    #print("Content type:", r.headers.get("content-type"))
    #print("First 200 chars:", r.text[:200])
    r.raise_for_status()
    data = r.json()

    page = next(iter(data["query"]["pages"].values()))
    #print(r.text)
    return page.get("extract", "")



def cosine_similarity(a, b, bnorm):
    return np.dot(a, b) / (np.linalg.norm(a) * bnorm)




def notation_depth(notation):
    return len(notation)






def keep_candidate(e, reason, similarity, query_embedding, results):
    notation = e["notation"]

    add_result(
        results,
        e,
        similarity,
        reason
    )


def best_at_depth(depth, query_embedding):

    candidates = [
        e
        for e in embeddings
        if (
            notation_depth(e["notation"]) == depth
        )
    ]


    if not candidates:
        return None

    return max(
        candidates,
        key=lambda e: cosine_similarity(
            query_embedding,
            e["embedding"],
            e["norm"],
        ),
    )



def get_verbal_from_sqlite(notation):
    with sqlite3.connect("ilc.sqlite") as conn:
        row = conn.execute(
            "SELECT verbal FROM ilc WHERE notation = ?",
            (notation,)
        ).fetchone()
    return row[0] if row else None



def do_word(title):

    print("=" * 70)
    print("Wikipedia:", title)
    print("=" * 70)

    text = first_paragraph(title)
    query_embedding = model.encode(text)

    results = {}

    # -------------------------------------------------
    # Find best match at every depth
    # -------------------------------------------------

    for depth in range(1, MAX_DEPTH + 1):

        best = best_at_depth(depth, query_embedding)

        if best is None:
            continue

        similarity = cosine_similarity(
            query_embedding,
            best["embedding"],
            best["norm"],
        )

        keep_candidate(
            best,
            f"best match of length {depth}",
            similarity,
            query_embedding,
            results,
        )

    # -------------------------------------------------
    # Final result: three best distinct classes
    # -------------------------------------------------

    for r in results.values():
      r["adjusted_score"] = adjusted_score(r)

    final = sorted(
        results.values(),
        key=lambda x: x["adjusted_score"],
        reverse=True,
    )[:10]

    for i, result in enumerate(final, 1):
        verbal = get_verbal(result["notation"]) or ""

        print(
            f"{i:2d}. "
            f"{result['notation']:15s} "
            f"{verbal:25.25s} "
            f"{result['similarity']:7.4f}  "
            f"{result['adjusted_score']:7.4f}"
        )

def do_word_OLD(title):

    print("=" * 70)
    print("Wikipedia:", title)
    print("=" * 70)

    # Get Wikipedia text
    text = first_paragraph(title)

    # Compute embedding of Wikipedia text
    query_embedding = model.encode(text)

    #
    # PHASE 1:
    # Find the best match at each depth first.
    # No hierarchy descent happens here.
    #

    selected = []

    for depth in range(1, MAX_DEPTH + 1):

        print(f"SEARCHING LENGTH {depth}")
        print("-" * 70)

        best = best_at_depth(depth, query_embedding)

        if best is None:
            print("No candidates.")
            continue

        similarity = cosine_similarity(
            query_embedding,
            best["embedding"],
            best["norm"],
        )

        print(
            f"BEST LENGTH {depth}: "
            f"{best['notation']} "
            f"{similarity:.4f}"
        )

        # Keep the selected candidate, but don't descend yet.
        keep_candidate(
            best,
            f"best match of length {depth}",
            similarity,
            query_embedding,
        )

        selected.append(best)



def adjusted_score(candidate):
    length = len(candidate["notation"])
    return candidate["similarity"] * 0.95 ** (length - 1)



def add_result(results, e, similarity, method):
    notation = e["notation"]

    # Keep the best score if the same class was found by several methods
    if notation not in results or similarity > results[notation]["similarity"]:
        results[notation] = {
            "notation": notation,
            "similarity": similarity,
            "method": method,
        }


do_word("Johann Sebastian Bach")
#do_word("Joe DiMaggio")
#do_word("Cary Grant")
#do_word("Great Barrier Reef")
#do_word("Black Death")
#do_word("Theory of relativity")
#do_word("The Beatles")
#do_word("Mona Lisa")
#do_word("French Revolution")
#do_word("Dungeons & Dragons")
#do_word("Linux")
#do_word("Pizza")
#do_word("Whale")
#do_word("Apple")
#do_word("Mercury")
#do_word("Jaguar")
#do_word("Java")
#do_word("Python")
#do_word("Titan")
#do_word("Amazon")
#do_word("Matrix")
#do_word("Black hole")
#do_word("The Great Gatsby")
#do_word("1984")
#do_word("Animal Farm")
#do_word("The Scream")
#do_word("Mona Lisa")
#do_word("Guernica")
#do_word("Bohemian Rhapsody")
#do_word("The Lord of the Rings")
#do_word("Star Wars")
#do_word("Game theory")
#do_word("Information theory")
#do_word("Cybernetics")
#do_word("Complexity theory")
#do_word("Evolutionary game theory")
#do_word("Cognitive science")
#do_word("Artificial intelligence")
#do_word("Consciousness")
#do_word("Cryptography")
#do_word("Network science")
#do_word("Printing press")
#do_word("Internet")
#do_word("Industrial Revolution")
#do_word("Renaissance")
#do_word("Enlightenment")
#do_word("French Revolution")
#do_word("Silk Road")
#do_word("Manhattan Project")
#do_word("Apollo 11")
#do_word("DNA")
#do_word("Neuron")
#do_word("Antibiotic")
#do_word("Vaccine")
#do_word("Computer")
#do_word("Algorithm")
#do_word("Language")
#do_word("Money")
#do_word("Religion")
