# Multitask LLMs with Limited Resources---WMT26 Shared Task

This is a fork of the lm-evaluation-harness to aid with development for the WMT26-LRL shared task. 
This repository shows how the baseline scores were obtained, and may be used by participants to inference their models, particularly during the development phase.

## Setup

Clone the repository:

```bash
git clone --depth 1 https://github.com/TUM-NLP/llms-lim-res-eval-2026.git
```
Remember to set up a fresh Python environment with your favourite package manager. 
The baselines were run with Python 3.12.
Install the package:

```bash
cd llms-lim-res-eval-2026
pip install -e .
```

Then clone the data repository into the root folder:

```bash
git clone https://github.com/TUM-NLP/llms-limited-resources2026
```

## Running the baselines

The following commands were used to run the baselines:

```bash
# generative tasks
python3 -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3.5-2B,enable_thinking=False --apply_chat_template \
    --tasks ukrainian_dev sorbian_dev \
    --device cuda:0 --batch_size auto \
    --output_path baseline_output --log_samples
    
# qa tasks (loglikelihood-based evaluation)
python3 -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3.5-2B \
    --tasks ukrqa ukrmmlu hsbqa dsbqa \
    --device cuda:0 --batch_size auto \
    --output_path baseline_output --log_samples
```

Note that Qwen3.5-2B is a "reasoning" model that will output thinking traces by default.
For the baseline, we decided not to use reasoning mode and explicitly set the parameter `enable_thinking=False` for the generative tasks. 
The parameter `--apply_chat_template` is necessary for this change to apply.
The loglikelihood evaluation for QA tasks performed better without the chat template, hence why we ran these in a separate process.

For reference, here is the baseline output for Ukrainian:

```bash
|  Tasks   |Version|  Filter   |n-shot|       Metric        |   | Value |   |Stderr|
|----------|------:|-----------|-----:|---------------------|---|------:|---|------|
| - ces-ukr|      0|remove_tags|     0|bleu                 |↑  |11.8923|±  |0.2858|
|          |       |remove_tags|     0|chrf_pp              |↑  |34.1566|±  |   N/A|
| - eng-ukr|      0|remove_tags|     0|bleu                 |↑  | 9.8272|±  |0.1868|
|          |       |remove_tags|     0|chrf_pp              |↑  |30.7793|±  |   N/A|
| - ukrsc  |      0|none       |     0|exact_match_corrected|↑  | 0.1725|±  |0.0085|
|          |       |none       |     0|exact_match_wrong    |↑  | 0.1745|±  |0.0085|
| - ukrgc  |      0|none       |     0|exact_match_corrected|↑  | 0.0255|±  |0.0035|
|          |       |none       |     0|exact_match_wrong    |↑  | 0.0425|±  |0.0045|
| - ukrmr  |      0|remove_tags|     0|exact_match          |↑  | 0.1250|±  |0.0690|

| Tasks |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|-------|------:|------|-----:|------|---|-----:|---|-----:|
|ukrqa  |      0|none  |     0|acc   |↑  |0.2985|±  |0.0185|
|ukrmmlu|      0|none  |     0|acc   |↑  |0.4561|±  |0.0296|


```

## Running "reasoning" mode

In order to turn on reasoning mode, it's recommended that you set this explicitly as well.
The evaluation harness framework will attempt to strip out the reasoning trace so that the final answer can be evaluated correctly:
Additionally, adjust the decoding parameters according to the base model's recommendation, and allow the model a larger generation budget:

```bash
# generative tasks
python3 -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3.5-2B,enable_thinking=True,think_end_token="</think>" --apply_chat_template \
    --gen_kwargs max_gen_toks=4096,top_p=0.95 \
    --tasks ukrainian_dev sorbian_dev \
    --device cuda:0 --batch_size auto \
    --output_path baseline_output --log_samples
```
 

## Customising the tasks

Each task has a corresponding YAML file that defines the task, found in `lm_eval/tasks/wmt26-lrl/`.

For example, here is the `deu-hsb.yaml`:

```yaml
task: deu-hsb
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files:  
    test: llms-limited-resources2026/Sorbian/MT/de-hsb_mt_dev.jsonl
test_split: test
doc_to_text: "Translate the following German text to Upper Sorbian. Put it in this format <hsb> Upper Sorbian translation </hsb>.\n<deu> {{de}} </deu>"
doc_to_target: "{{hsb}}"
generation_kwargs:
  until:
    - "<|endoftext|>"
    - "<|im_end|>"
  max_gen_toks: 256
  temperature: 1
  top_p: 1.0
  top_k: 20
  min_p: 0.0
  repetition_penalty: 1.0
filter_list:
  - name: "remove_tags"
    filter:
      - function: "strip_thinking"
      - function: "regex"
        regex_pattern: "<hsb>\\s*([\\s\\S]*?)\\s*(?:</hsb>|$)"
      - function: "take_first"
metric_list:
  - metric: bleu
  - metric: !function lm_eval.tasks.wmt26-lrl.utils.chrf_pp
    aggregation: chrf++
    higher_is_better: true
metadata:
  version: 0.0
```

### Changing the prompt

The `doc_to_text` field in the YAML file prompts the model to put pseudo-html tags around the translation, 
and then the translation is post-processed with `filter_list` to only consider anything inside these tags. 
(We do this because LLMs like to output other text, such as explanations.) 
The output is compared to `doc_to_target` for metrics. {{de}} and {{hsb}} refer to columns in the CSV.

### Reducing the development set size

If running frequent evaluations is taking too long, you can evaluate on a smaller dev set while developing your model:
Add the flag `--limit {n}` to the calling command.

## Further Information

You can find more details in the original [lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness). The final evaluation will be done on outputs you submit, so you have full control over any pre-processing, prompting, and post-processing.
