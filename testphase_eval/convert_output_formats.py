import argparse
import json
import os
import re
from collections import defaultdict
from enum import StrEnum, auto

import numpy as np

LANGCODES_3_TO_2 = {
    "eng": "en",
    "deu": "de",
    "ukr": "uk",
    "ces": "cs",
    "hsb": "hsb",
    "dsb": "dsb"
}


class Task(StrEnum):
    QA = auto()
    MT = auto()
    SC = auto()
    GC = auto()
    MR = auto()


class LangTrack(StrEnum):
    SORBIAN = "sb"
    UKRAINIAN = "ukr"


def parse_mt_line(line: str, subset: str):
    """Read MT JSON line from the lm-harness output."""
    line_data = json.loads(line)

    src_lang = LANGCODES_3_TO_2[subset.split("-")[0]]
    output_pred_data = {
        'dataset_id': line_data['doc']['dataset_id'],  # f'wmtslavicllm2025_{lang_pair}',
        'sent_id': line_data["doc"]["sent_id"],
        'source': line_data["doc"][src_lang], # not strictly needed but helpful for manual checking
        'pred': line_data['filtered_resps'][0]
    }
    return output_pred_data


def parse_qa_line(line: str):
    """Read question answering JSON line from the lm-harness output."""
    line_data = json.loads(line)

    arguments_run = line_data['arguments']
    tried_answers = []
    for j in range(len(arguments_run)):
        tried_answers.append(arguments_run[f'gen_args_{j}']['arg_1'])

    responses = line_data['filtered_resps']
    responses_val = [float(resp[0]) for resp in responses]

    question_id = line_data['doc']['question_id']
    tried_answers = [int(resp) for resp in tried_answers]

    pred_idx = np.argmax(responses_val)

    if pred_idx is not None:
        prediction = tried_answers[pred_idx]
    else:
        print('No QA answer')
        prediction = None

    output_pred_data = {
        'dataset_id': line_data['doc']['dataset_id'],
        'question_id': question_id,
        'question': line_data['doc']['question'],
        "possible_answers": line_data["doc"]["possible_answers"],
        'pred': prediction,
    }

    return output_pred_data


## Spell-checking and grammar-checking
def parse_checking_output(output_text):
    """Parse the output of the checking tasks to extract both the wrong and corrected words."""
    wrong_match = re.search(r'<wrong>\s*(.*?)\s*</wrong>', output_text, flags=re.DOTALL)
    corrected_match = re.search(r'<corrected>\s*(.*?)\s*</corrected>', output_text, flags=re.DOTALL)

    wrong = wrong_match.group(1).strip() if wrong_match else 'NO_OUTPUT'
    corrected = corrected_match.group(1).strip() if corrected_match else 'NO_OUTPUT'
    if not wrong:
        wrong = 'NO_OUTPUT'
    if not corrected:
        corrected = 'NO_OUTPUT'
    return [wrong, corrected]


def parse_gc_sc_line(line: str):
    """Read spell-checking and grammar-checking JSON line from the lm-harness output."""
    line_data = json.loads(line)
    parsed_output = parse_checking_output(line_data['filtered_resps'][0])

    output_pred_data = {
        'dataset_id': line_data['doc']['dataset_id'],
        'id': line_data['doc']['id'],
        'input_sentence': line_data['doc']['input_sentence'],
        'pred_incorrect': parsed_output[0],
        'pred_corrected': parsed_output[1]
    }
    return output_pred_data


def parse_mr_line(line: str):
    """Read spell-checking and grammar-checking JSON line from the lm-harness output."""
    line_data = json.loads(line)
    output_pred_data = {
        'dataset_id': line_data['doc']['dataset_id'],
        'id': line_data['doc']['id'],
        'question': line_data['doc']['question'],
        'pred': line_data['filtered_resps'][0]
    }
    return output_pred_data


def parse_filename(file_name: str):
    """Parsing lm-harness file names."""
    task_name = file_name.split('_')[1]

    if len(task_name) == 5: # SC, GC, MR, most QA
        task = task_name[-2:]
        language = task_name[:3]
        lang_track = LangTrack.UKRAINIAN if language == LangTrack.UKRAINIAN else LangTrack.SORBIAN if LangTrack.SORBIAN in language else None
        subset = None
    elif '-' in task_name: # MT
        task = Task.MT
        language = task_name
        lang_track = LangTrack.UKRAINIAN if LangTrack.UKRAINIAN in task_name else LangTrack.SORBIAN if LangTrack.SORBIAN in task_name else None
        subset = task_name
    elif "mmlu" in task_name: # ukrmmlu
        lang_track = LangTrack.UKRAINIAN
        task = Task.QA
        language = LangTrack.UKRAINIAN
        subset = "mmlu"
    else:
        raise ValueError(f"Error while parsing task name: {task_name}")

    if task == Task.QA:
        if lang_track == LangTrack.UKRAINIAN and subset is None:
            subset = "zno"
        elif lang_track == LangTrack.SORBIAN:
            split_name = file_name.split('_')[3]
            subset = split_name.split("-")[-1]

    return lang_track, task, subset, language


def write_out_predictions(save_file: str, lang_track: LangTrack, task: Task, subsets: dict[str, list[str]]):
    if len(subsets) == 0:
        print(f"Warning: No subsets parsed for {lang_track} {task}")
        return
    if lang_track == LangTrack.SORBIAN:
        if len(subsets) == 2:  # sb_sc, sb_gc, sb_mr
            subset_order = sorted(list(subsets.keys()), reverse=True)
            write_out_in_order(save_file, subset_order, subsets)
            return
        elif task == Task.QA:
            subset_order = ["hsb_a1", "hsb_a2", "hsb_b1", "hsb_b2", "hsb_c1", "dsb_a1", "dsb_a2", "dsb_b1", "dsb_b2", "dsb_c1"]
            write_out_in_order(save_file, subset_order, subsets)
        elif task == Task.MT:
            subset_order = ["deu-hsb", "hsb-deu", "deu-dsb", "dsb-deu", "hsb-dsb", "dsb-hsb"]
            write_out_in_order(save_file, subset_order, subsets)
    else:
        if len(subsets.keys()) == 1:  # ukr_sc, ukr_gc, ukr_mr
            subset_order = list(subsets.keys())
            write_out_in_order(save_file, subset_order, subsets)
            return
        elif task == Task.QA:
            subset_order = ["ukr_zno", "ukr_mmlu"]
            write_out_in_order(save_file, subset_order, subsets)
        elif task == Task.MT:
            subset_order = ["ces-ukr", "eng-ukr"]
            write_out_in_order(save_file, subset_order, subsets)


def write_out_in_order(save_file: str, subset_order: list[str], subsets: dict[str, list[str]]):
    with open(save_file, 'w', encoding="utf-8") as out_file:
        for subset_name in subset_order:
            for entry in subsets[subset_name]:
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(eval_harness_outputs: list[str], model_name: str, predictions_save_folder: str, data_split: str):
    os.makedirs(predictions_save_folder, exist_ok=True)
    results_dict = {}
    for lang_track in LangTrack:
        results_dict[lang_track] = {}
        for task_type in Task:
            results_dict[lang_track][task_type] = defaultdict(list)

    for folder in eval_harness_outputs:
        read_folder = os.path.join(folder, model_name)

        for filename in os.listdir(read_folder):
            if not filename.endswith(".jsonl"):
                continue
            lang_track, task, subset, language = parse_filename(filename)

            file_entries = []

            with (open(os.path.join(read_folder, filename), 'r', encoding="utf-8") as eval_harness_file):
                for line in eval_harness_file:
                    match task:
                        case Task.QA:
                            output_dict = parse_qa_line(line)
                        case Task.MT:
                            output_dict = parse_mt_line(line, subset)
                        case Task.SC | Task.GC:
                            output_dict = parse_gc_sc_line(line)
                        case Task.MR:
                            output_dict = parse_mr_line(line)
                        case _:
                            raise ValueError(f"Unknown task {task}")
                    file_entries.append(output_dict)
            if task == Task.MT:
                results_dict[lang_track][task][subset] = file_entries
            else:
                results_dict[lang_track][task][f"{language}_{subset}"] = file_entries

    for lang_track in LangTrack:
        for task_type in Task:
            predictions_save_file = os.path.join(predictions_save_folder, f"{lang_track}_{task_type}_{data_split}.jsonl")
            subsets = results_dict[lang_track][task_type]
            write_out_predictions(predictions_save_file, lang_track, task_type, subsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lm-evalharness outputs to expected submission format.")
    parser.add_argument("--eval_harness_outputs", type=str, nargs="+",
                        default=["../baseline_test_output_qa", "../baseline_test_output_thinking_off"])
    parser.add_argument("--model_name", type=str, default="Qwen__Qwen3.5-2B")
    parser.add_argument("--predictions_save_folder", type=str, default="submission_predictions")
    parser.add_argument("--data_split", type=str, default="test")

    args = parser.parse_args()
    main(args.eval_harness_outputs, args.model_name, args.predictions_save_folder, args.data_split)