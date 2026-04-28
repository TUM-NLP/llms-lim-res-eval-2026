import json
import re
from sacrebleu import corpus_bleu, corpus_chrf
import sys

def extract(text):
    m = re.search(r"\s*<corrected>\s*(.*?)\s*(?:</corrected>|$)", text)
    return m.group(1) if m else ""

preds, refs = [], []

with open(sys.argv[1]) as f:
    for line in f:
        ex = json.loads(line)
        pred = extract(ex["resps"][0][0])
        ref = ex["doc"]["original_sentence"]

        preds.append(pred)
        refs.append(ref)

print("BLEU:", corpus_bleu(preds, [refs]).score)
print("chrF:", corpus_chrf(preds, [refs]).score)