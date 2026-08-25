import pandas as pd

import re

# Gold data path
main_gold_folder = 'WMT2026_ST/test_data' # TO CHANGE!
TASK_GOLD_DICT = {
    # Sorbian track
    'de-hsb': f'{main_gold_folder}/de-hsb_mt_test.jsonl',
    'hsb-de': f'{main_gold_folder}/hsb-de_mt_test.jsonl',
    'de-dsb': f'{main_gold_folder}/de-dsb_mt_test.jsonl',
    'dsb-de': f'{main_gold_folder}/dsb-de_mt_test.jsonl',
    'hsb-dsb': f'{main_gold_folder}/hsb-dsb_mt_test.jsonl',
    'dsb-hsb': f'{main_gold_folder}/dsb-hsb_mt_test.jsonl',

    'hsb-qa': f'{main_gold_folder}/hsb_qa_test.jsonl',
    'dsb-qa': f'{main_gold_folder}/dsb_qa_test.jsonl',

    'hsb-sc': f'{main_gold_folder}/hsb_sc_test.jsonl',
    'dsb-sc': f'{main_gold_folder}/dsb_sc_test.jsonl',

    'hsb-gc': f'{main_gold_folder}/hsb_gc_test.jsonl',
    'dsb-gc': f'{main_gold_folder}/dsb_gc_test.jsonl',

    'hsb-mr': f'{main_gold_folder}/hsb_mr_test.jsonl',
    'dsb-mr': f'{main_gold_folder}/dsb_mr_test.jsonl',

    # Ukrainian track
    'cs-uk': f'{main_gold_folder}/cs-ukr_mt_test.jsonl',
    'en-uk': f'{main_gold_folder}/en-ukr_mt_test.jsonl',

    'uk-qa': f'{main_gold_folder}/ukr_qa_test.jsonl',
    'uk-mmlu_qa': f'{main_gold_folder}/ukr_mmlu_qa_test.jsonl',

    'uk-sc': f'{main_gold_folder}/ukr_sc_test.jsonl',
    'uk-gc': f'{main_gold_folder}/ukr_gc_test.jsonl',
    'uk-mr': f'{main_gold_folder}/ukr_mr_test.jsonl',
}


# Submission extraction
def extract_file(submission_json):
    '''Extract file name and task in a dictionary: {task: file_name}.'''
    track_file_dict = dict()
    for file_info in submission_json:
        # track_file_dict[file_info['language_pair']] = file_info['file_name'][12:]
        task_name_list = file_info['test_set'].split('-')
        task_name = f'{task_name_list[3]}-{task_name_list[2]}'
        track_file_dict[task_name] = file_info['file_name'][12:]
    return track_file_dict


def extract_subtask(aggregated_df):
    '''Extract each subset for a given task.'''
    subtask_list = list(set(aggregated_df.dataset_id.to_list()))
    print(subtask_list)
    task_submission_dict = dict()
    for subtask in subtask_list:
        # Handling the subtask name from the dataset ID
        subtask_split = subtask[19:].split('_')
        if 'ukr' in subtask_split[1]:
            subtask_split[1] = re.sub('ukr', 'uk', subtask_split[1])
        subtask_name = f'{subtask_split[1]}-{subtask_split[0]}'
        if subtask_split[0] == 'mt':
            subtask_name = f'{subtask_split[1]}'
        if subtask_name == 'mmlu-qa':
            subtask_name = 'uk-mmlu_qa'
            
        task_submission_dict[subtask_name] = aggregated_df[aggregated_df['dataset_id'] == subtask]
    return task_submission_dict


def read_team_submission(team_name, submission_dict):
    '''Read and split the submission from a team.'''
    team_submission_dict = dict()

    for task, file_name in submission_dict[team_name].items():
        print(task)
        if task[0] == 's': # Sorbian
            file_path = f'WMT2026_submissions/Sorbian/{file_name}'
        elif task[0] == 'u': # Ukrainian
            file_path = f'WMT2026_submissions/Ukrainian/{file_name}'
        else:
            print('TRACK NOT RECOGNISED', task)

        aggregated_task_df = pd.read_json(path_or_buf=file_path, lines=True)
        print(aggregated_task_df.shape)

        team_submission_dict[task] = extract_subtask(aggregated_task_df)
        print(task, ':', list(team_submission_dict[task].keys()))

    print(f'{len(team_submission_dict)} submissions')
    assert len(team_submission_dict) % 5 == 0, 'Some tasks are missing in the submission.'
    
    return team_submission_dict


def read_gold_file(gold_file_list):
    '''Read and split the gold files from the OCELoT version.'''
    gold_dict = dict()

    for file_name in gold_file_list:
        file_name_split = file_name.split('_')
        task = f'{file_name_split[0]}-{file_name_split[1]}'
        print(task)
        
        file_path = f'WMT2026_ST/gold_ocelot/{file_name}'

        aggregated_task_df = pd.read_json(path_or_buf=file_path, lines=True)
        print(aggregated_task_df.shape)

        gold_dict[task] = extract_subtask(aggregated_task_df)
        print(task, ':', list(gold_dict[task].keys()))

    print(f'{len(gold_dict)} files')
    assert len(gold_dict) % 5 == 0, 'Some tasks are missing.'
    
    return gold_dict


