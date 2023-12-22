# Evaluating Large Language Models at Evaluating Instruction Following

This repository contains the data and code for paper *[Evaluating Large Language Models at Evaluating Instruction Following](https://arxiv.org/abs/2310.07641)*.
In this paper, we introduce a challenging meta-evaluation benchmark, LLMBar, designed to test the ability of an LLM evaluator in discerning instruction-following outputs.
LLMBar consists of 419 instances, where each entry contains an instruction paired with two outputs: one faithfully and correctly follows the instruction and the other deviates from it.
There is also a gold preference label indicating which output is objectively better for each instance.

## Quick Links

- [Requirements](#requirements)
- [Data](#data)
- [Hugging Face Datasets](#hugging-face-datasets)
- [Code Structure](#code-structure)
- [Run LLM Evaluators](#run-llm-evaluators)
- [Bug or Questions?](#bug-or-questions)
- [Citation](#citation)

## Requirements

Please install the packages by `pip install -r requirements.txt`. This codebase has been tested with Python 3.10.4.

## Data

All the data are stored in `Dataset/`.
The Natural set of LLMBar is stored in `Dataset/Natural`.
The four subsets of LLMBar Adversarial set are stored in `Dataset/LLMBar/Adversarial/{Neighbor, GPTInst, GPTOut, Manual}`.

The five evaluation subsets we studied in *4.6 Case Study: A More Challenging Meta-Evaluation Set* are stored in `Dataset/CaseStudy/{Constraint, Negation, Normal, Base_9, Base_10}`.

We also evaluate LLM evaluators on FairEval, LLMEval $^2$, and MT-Bench.
We remove LLMEval $^2$ instances whose instructions are empty or non-English and add the task description before the raw input to get the instruction.
For MT-Bench, we get the gold preferences by majority vote.
We remove all ``TIE'' instances and randomly sample 200 instances for LLMEval $^2$ and MT-Bench respectively.
The processed data are stored in `Dataset/Processed/{FairEval, LLMEval^2, MT-Bench}`.

All the evaluation instances in each folder are stored in `dataset.json`.
Each instance is a JSON object with the format:

```json
{
    "input": "Infer the implied meaning of the following sentence: She is not what she used to be.",
    "output_1": "She is not as she once was.",
    "output_2": "She has changed substantially over time.",
    "label": 2
}
```

`"input"` is the input instruction.
`"output_1"` and `"output_2"` are the two evaluated outputs $O_1$ and $O_2$  respectively.
`label` is either `1` or `2`, indicating which output is objectively better.

## Hugging Face Datasets

Our dataset is now available on [Hugging Face Datasets](https://huggingface.co/datasets/princeton-nlp/LLMBar)! You can access and utilize it using the 🤗 Datasets library.

```python
from datasets import load_dataset
LLMBar = load_dataset("princeton-nlp/LLMBar", "LLMBar")
CaseStudy = load_dataset("princeton-nlp/LLMBar", "CaseStudy")
```

## Code Structure

All the codes are stored in `LLMEvaluator/`.

* `evaluate.py`: run file to reproduce our baselines.
* `evaluators/config`: folder that contains all config files to reproduce baselines.
* `evaluators/prompts`: folder that contains all prompt files.

## Run LLM Evaluators

You can reproduce LLM evaluators from our paper by
```bash
cd LLMEvaluator

python evaluate.py \
    --path {path_to_data_folder} \
    --evaluator {base_llm}/{prompting_strategy} \
    --num_procs {number_of_processes}
    # The default value of num_procs is 10
    # See the following content for more arguments
```

`{base_llm}` is one of `GPT-4`, `ChatGPT`, `LLaMA2`, `PaLM2`, `Falcon`, and `ChatGPT-0301`.

- If you use `GPT-4`, `ChatGPT`, or `ChatGPT-0301`, you will also need to pass the OpenAI API arguments:
    ```bash
    --api_type {your_api_type} \
    --api_version {your_api_version} \
    --api_base {your_api_base} \
    --api_key {your_api_key}
    --organization {your_organization}
    # If you use Azure API, you may need to pass api_type, api_version, api_base, and api_key.
    # Otherwise, you may need to pass api_key and organization.
    ```
    Also, ensure that the arguments in the config files align with those expected by the function.
- If you use `PaLM2`, you will also need to pass the PaLM API key:
    ```bash
    --palm_api_key {your_palm_api_key}
    ```
- If you use `LLaMA2` ([LLaMA-2-70B-Chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)) or `Falcon` ([Falcon-180B-Chat](https://huggingface.co/tiiuae/falcon-180B-chat)), you will also need to pass the Hugging Face authorization token (please make sure your account has the access to the model):
    ```bash
    --hf_use_auth_token {your_use_auth_token}
    ```

An example of the command:

```bash
python evaluate.py \
    --path ../Dataset/LLMBar/Natural \
    --evaluator GPT-4/Vanilla \
    --api_type azure \
    --api_version 2023-05-15 \
    --api_base {your_api_base} \
    --api_key {your_api_key}
```

The current list of `prompting_strategy` (check out our paper for more details) includes:

- `Vanilla_NoRules`: **Vanilla**
- `Vanilla`: **Vanilla\*** (**Vanilla+Rules**)
- `Vanilla_1shot`: **Vanilla\*** (**Vanilla+Rules**) w/ 1-shot in-context learning
- `Vanilla_2shot`: **Vanilla\*** (**Vanilla+Rules**) w/ 2-shot in-context learning
- `CoT`: **CoT\*** (**CoT+Rules**)
- `Metrics`: **Metrics\*** (**Rules+Metrics**)
- `Reference`: **Reference\*** (**Rules+Reference**)
- `Metrics_Reference`: **Metrics+Reference\*** (**Rules+Metrics+Reference**)
- `Swap`: **Swap\*** (**Rules+Swap**)
- `Swap+CoT`: **Swap+CoT\*** (**Rules+Swap+CoT**)
- `Rating_NoRules`: **Vanilla** w/ the rating approach
- `Rating`: **Vanilla\*** (**Vanilla+Rules**) w/ the rating approach
- `Rating_Metrics`: **Metrics\*** (**Rules+Metrics**) w/ the rating approach
- `Rating_Reference`: **Reference\*** (**Rules+Reference**) w/ the rating approach
- `Rating_Metrics_Reference`: **Metrics+Reference\*** (**Rules+Metrics+Reference**) w/ the rating approach

After running the code, the results will be stored in `{path_to_data_folder}/evaluators/{base_llm}/{prompting_strategy}`.
`result.json` is the intermediate results for evaluating the LLM evaluators on all instances.
`statistics.json` is the final statistics of the evaluation, where `"correct_average"` and `equal` represent *average accuracy* (Acc.) and *positional agreement rate* (Agr.) respectively.
We have already put our results (reported in our paper) in the repository.

## Bug or Questions?

If you have any questions related to the code or the paper, feel free to email Zhiyuan Zeng (`zhiyuan1zeng@gmail.com` or `zengzy20@mails.tsinghua.edu.cn`).
If you encounter any problems when using the code, or want to report a bug, you can open an issue.
Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use this repo in your work:

```bibtex
@article{zeng2023llmbar,
  title={Evaluating Large Language Models at Evaluating Instruction Following},
  author={Zeng, Zhiyuan and Yu, Jiatong and Gao, Tianyu and Meng, Yu and Goyal, Tanya and Chen, Danqi},
  journal={arXiv preprint arXiv:2310.07641},
  year={2023}
}
```
