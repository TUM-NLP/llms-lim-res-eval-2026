import re


def process_results(doc, results):
    response = results[0]
    wrong_match = re.search(r"<wrong> (.*?) </wrong>", response)
    # </corrected> is the generation stop token, so it won't appear in the response
    corrected_match = re.search(r"<corrected> (.+)", response)
    wrong_pred = wrong_match.group(1).strip() if wrong_match else ""
    corrected_pred = corrected_match.group(1).strip() if corrected_match else ""
    return {
        "exact_match_wrong": int(wrong_pred == doc["incorrect_word"]),
        "exact_match_corrected": int(corrected_pred == doc["correct_word"]),
    }
