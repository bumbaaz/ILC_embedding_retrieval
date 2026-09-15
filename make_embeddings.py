from sentence_transformers import SentenceTransformer
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)



#model = SentenceTransformer(
#    "all-MiniLM-L6-v2",
#    cache_folder="./models"
#)


model = SentenceTransformer(
    "./models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


input_file = "verbals.tsv"
output_file = "ilc_embeddings.tsv"

counter = 0

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.rstrip("\n")
        if not line:
            continue

        parts = line.split("\t")

        notation = parts[0]

        if len(notation)>6:

            counter = counter + 1
            if counter%100 == 0:
                print(counter)

            verbals = parts[1:5]  # the 4 verbal entries

            #verbals[1] = "Topic: " + verbals[1]

            if len(verbals) != 4:
                print(f"Skipping malformed line: {line}")
                continue

            # Compute embeddings for the four verbal descriptions
            embeddings = model.encode(verbals)

            # Average the four embeddings
            avg_embedding = np.mean(embeddings, axis=0)

            # Write notation + embedding coordinates
            fout.write(
                notation + "\t" +
                "\t".join(str(x) for x in avg_embedding) +
                "\n"
            )

print("done")
