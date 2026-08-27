import argparse
import json
from collections import Counter

import fasttext
from huggingface_hub import hf_hub_download


def load_model(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return fasttext.load_model(model_path)


def predict(model, text):
    # fasttext chokes on newlines, so we flatten each response to one line.
    flat = " ".join(text.split())
    if not flat:
        return None, 0.0
    labels, scores = model.predict(flat, k=1)
    lang = labels[0].replace("__label__", "")
    return lang, float(scores[0])


def main():
    parser = argparse.ArgumentParser(
        description="Run language identification (GlotLID) on the 'resps' field "
        "of an lm-eval-harness samples jsonl file."
    )
    parser.add_argument("input", help="Path to a samples_*.jsonl file")
    parser.add_argument("output", help="Path to write the results JSON to")
    parser.add_argument(
        "--repo-id",
        default="cis-lmu/glotlid",
        help="Hugging Face repo id of the fastText langid model",
    )
    parser.add_argument(
        "--filename",
        default="model.bin",
        help="Model filename within the Hugging Face repo",
    )
    args = parser.parse_args()

    model = load_model(args.repo_id, args.filename)

    per_doc = []
    lang_counts = Counter()

    with open(args.input) as f:
        for line in f:
            ex = json.loads(line)
            resps = ex.get("resps", [])
            predictions = []
            for resp_group in resps:
                for text in resp_group:
                    lang, score = predict(model, text)
                    predictions.append({"lang": lang, "score": score})
                    if lang is not None:
                        lang_counts[lang] += 1
            per_doc.append({"doc_id": ex.get("doc_id"), "predictions": predictions})

    results = {
        "input": args.input,
        "model": args.repo_id,
        "lang_counts": dict(lang_counts),
        "per_doc": per_doc,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
