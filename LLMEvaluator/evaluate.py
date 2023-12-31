import re
import os
import tqdm
import time
import json
import torch
import openai
import logging
import argparse
import tiktoken
import functools
import multiprocessing
import google.generativeai as palm
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path) :
    with open(path, "r") as fin :
        obj = json.load(fin)
    return obj
def dump_json(obj, path) :
    with open(path, "w") as fout :
        json.dump(obj, fout, indent = 2)


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    """Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    def string_to_dict(to_convert):
        """Converts a string with equal signs to dictionary. E.g.
        >>> string_to_dict(" name=user university=stanford")
        {'name': 'user', 'university': 'stanford'}
        """
        return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


model_and_tokenizer = {}
def get_modules(args) :
    def process_openai_kwargs(model_name, openai_kwargs : dict) :
        tokenizer = tiktoken.encoding_for_model(model_name)
        logit_bias = {}

        if "tokens_to_avoid" in openai_kwargs :
            for t in openai_kwargs["tokens_to_avoid"] :
                curr_tokens = tokenizer.encode(t)
                if len(curr_tokens) != 1 :
                    continue
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = -100  # avoids certain tokens
            openai_kwargs.pop("tokens_to_avoid")

        if "tokens_to_favor" in openai_kwargs :
            for t in openai_kwargs["tokens_to_favor"]:
                curr_tokens = tokenizer.encode(t)
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = 7  # increase log prob of tokens to match
            openai_kwargs.pop("tokens_to_favor")

        if logit_bias :
            openai_kwargs["logit_bias"] = logit_bias
        
        return openai_kwargs

    configs = load_json(os.path.join("evaluators", "config", args.evaluator + ".json"))
    def get_module(config : dict) :
        with open(os.path.join("evaluators", "prompts", config["prompt"]), "r") as fin :
            prompt = fin.read()
        if "meta-llama/Llama-2" in config["model_name"] or "tiiuae/falcon-180B" in config["model_name"] :
            if config["model_name"] not in model_and_tokenizer :
                if "meta-llama/Llama-2" in config["model_name"] :
                    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map = "auto", torch_dtype = torch.bfloat16, use_auth_token = args.hf_use_auth_token)
                elif "tiiuae/falcon-180B" in config["model_name"] :
                    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map = "auto", torch_dtype = torch.bfloat16, load_in_8bit = True, use_auth_token = args.hf_use_auth_token)
                else :
                    raise NotImplementedError
                model.eval()
                model_and_tokenizer[config["model_name"]] = {
                    "tokenizer" : AutoTokenizer.from_pretrained(config["model_name"], use_auth_token = args.hf_use_auth_token),
                    "model" : model,
                }
            return {
                "prompt" : prompt,
                "model" : model_and_tokenizer[config["model_name"]]["model"],
                "tokenizer" : model_and_tokenizer[config["model_name"]]["tokenizer"],
                "generate_kwargs" : config["generate_kwargs"],
                "parsing" : config["parsing"],
                "model_name" : config["model_name"],
            }
        elif "PaLM" in config["model_name"] :
            return {
                "prompt" : prompt,
                "palm_kwargs" : config["palm_kwargs"],
                "parsing" : config["parsing"],
            }
        return {
            "prompt" : prompt,
            "openai_kwargs" : process_openai_kwargs(config["model_name"], config["openai_kwargs"]),
            "parsing" : config["parsing"],
        }
    modules = list(map(get_module, configs))
    return modules


def openai_completion(
    prompt,
    openai_kwargs : dict,
    sleep_time : int = 5,
    request_timeout : int = 30,
) :
    openai_kwargs = openai_kwargs.copy()
    
    while True :
        try :
            completion_batch = openai.ChatCompletion.create(messages = prompt_to_chatml(prompt), request_timeout = request_timeout, **openai_kwargs)

            choice = completion_batch.choices[0]
            assert choice.message.role == "assistant"

            def cost_calculation(usage) :
                if ("engine" in openai_kwargs) and ("gpt-35-turbo" in openai_kwargs["engine"]) :
                    return usage["prompt_tokens"] / 1000 * 0.0015 + usage["completion_tokens"] / 1000 * 0.0020
                elif ("engine" in openai_kwargs) and ("gpt-4" in openai_kwargs["engine"]) :
                    return usage["prompt_tokens"] / 1000 * 0.03 + usage["completion_tokens"] / 1000 * 0.06
                else :
                    raise NotImplementedError

            return ("" if "content" not in choice.message else choice.message.content), choice.finish_reason, cost_calculation(completion_batch.usage)
        except openai.error.OpenAIError as e :
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e) :
                openai_kwargs["max_tokens"] = int(openai_kwargs["max_tokens"] * 0.8)
                logging.warning(f"Reducing target length to {openai_kwargs['max_tokens']}, Retrying...")
            elif "content management policy" in str(e) :
                return "", "content management policy", 0.0
            elif "0 is less than the minimum of" in str(e) :
                return "", "0 is less than the minimum of", 0.0
            else :
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.


def hf_model_generate(prompt, module) :
    model = module["model"]
    tokenizer = module["tokenizer"]

    channels = prompt_to_chatml(prompt)
    assert channels[0]["role"] == "system"
    if "meta-llama/Llama-2" in module["model_name"] :
        prompt = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n".format(channels[0]["content"])
        for index, channel in enumerate(channels[1 :]) :
            if index % 2 == 0 :
                assert channel["role"] == "user"
                prompt += "{} [/INST]".format(channel["content"])
                if index < len(channels) - 2 :
                    prompt += " "
            else :
                assert channel["role"] == "assistant"
                prompt += "{} </s><s>[INST] ".format(channel["content"])
        input_ids = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False)["input_ids"]
    elif "tiiuae/falcon-180B" in module["model_name"] :
        prompt = "System: {}\n".format(channels[0]["content"])
        for index, channel in enumerate(channels[1 :]) :
            if index % 2 == 0 :
                assert channel["role"] == "user"
                prompt += "User: {}\n".format(channel["content"])
            else :
                assert channel["role"] == "assistant"
                prompt += "Falcon: {}\n".format(channel["content"])
        prompt += "Falcon:"
        input_ids = tokenizer(prompt, return_tensors = "pt")["input_ids"]
    else :
        raise NotImplementedError
    assert channels[-1]["role"] == "user"

    with torch.inference_mode() :
        if torch.cuda.is_available() :
            input_ids = input_ids.cuda()
        generation = model.generate(input_ids = input_ids, **module["generate_kwargs"])
        completion = tokenizer.decode(generation[0][len(input_ids[0]) :], skip_special_tokens = True, clean_up_tokenization_spaces = True)
    return completion.strip(), "", 0.0


def palm_completion(
    prompt,
    palm_kwargs : dict,
    sleep_time : int = 30
) :
    channels = prompt_to_chatml(prompt)
    assert len(channels) == 2 and channels[0]["role"] == "system" and channels[1]["role"] == "user"
    while True :
        try :
            completion = palm.generate_text(
                prompt = "\n\n".join([channels[0]["content"], channels[1]["content"]]),
                **palm_kwargs,
            )
            if completion.result is None :
                completion.result = ""
            return completion.result, "", 0.0
        except :
            logging.warning("Retrying...")
            time.sleep(sleep_time)  # Annoying rate limit on requests.
    

def Complete(prompt, module) :
    if "model" in module :
        return hf_model_generate(prompt, module)
    elif "palm_kwargs" in module :
        return palm_completion(prompt, module["palm_kwargs"])
    elif "openai_kwargs" in module :
        return openai_completion(prompt, module["openai_kwargs"])
    else :
        raise NotImplementedError


def completion_with_parsing(prompt, module) :
    completion, finish_reason, cost = Complete(prompt, module)
    completion = completion.strip()
    parsing : dict = module["parsing"]
    for return_value, exp in parsing.items() :
        if (re.compile(exp)).search(completion) :
            return completion, finish_reason, cost, return_value
    return completion, finish_reason, cost, None


NonSwap = (
    "GPT-4/Vanilla", "GPT-4/Vanilla_NoRules", "GPT-4/Vanilla_1shot", "GPT-4/Vanilla_2shot",
    "GPT-4/CoT",
    "GPT-4/Metrics",
    "GPT-4/Reference",
    "GPT-4/Metrics_Reference",

    "ChatGPT/Vanilla", "ChatGPT/Vanilla_NoRules", "ChatGPT/Vanilla_1shot", "ChatGPT/Vanilla_2shot",
    "ChatGPT/CoT",
    "ChatGPT/Metrics",
    "ChatGPT/Reference",
    "ChatGPT/Metrics_Reference",

    "LLaMA2/Vanilla", "LLaMA2/Vanilla_NoRules", "LLaMA2/Vanilla_1shot", "LLaMA2/Vanilla_2shot",
    "LLaMA2/CoT",
    "LLaMA2/Metrics",
    "LLaMA2/Reference",
    "LLaMA2/Metrics_Reference",

    "PaLM2/Vanilla", "PaLM2/Vanilla_NoRules",
    "PaLM2/CoT",
    "PaLM2/Metrics",
    "PaLM2/Reference",
    "PaLM2/Metrics_Reference",

    "Falcon/Vanilla", "Falcon/Vanilla_NoRules", "Falcon/Vanilla_1shot", "Falcon/Vanilla_2shot",
    "Falcon/CoT",
    "Falcon/Metrics",
    "Falcon/Reference",
    "Falcon/Metrics_Reference",

    "ChatGPT-0301/Vanilla", "ChatGPT-0301/Vanilla_NoRules", "ChatGPT-0301/Vanilla_1shot", "ChatGPT-0301/Vanilla_2shot",
    "ChatGPT-0301/CoT",
    "ChatGPT-0301/Metrics",
    "ChatGPT-0301/Reference",
    "ChatGPT-0301/Metrics_Reference",
)
SwapBased = (
    "GPT-4/Swap", "GPT-4/Swap_CoT",

    "ChatGPT/Swap", "ChatGPT/Swap_CoT",

    "LLaMA2/Swap", "LLaMA2/Swap_CoT",

    "PaLM2/Swap", "PaLM2/Swap_CoT",

    "Falcon/Swap", "Falcon/Swap_CoT",

    "ChatGPT-0301/Swap", "ChatGPT-0301/Swap_CoT",
)
Rating = (
    "ChatGPT/Rating", "ChatGPT/Rating_NoRules",

    "GPT-4/Rating", "GPT-4/Rating_NoRules",
    "GPT-4/Rating_Metrics",
    "GPT-4/Rating_Reference",
    "GPT-4/Rating_Metrics_Reference",
)


def annotate(instance, args, modules) :
    total_cost = 0.0
    def reverse_label(label) :
        if label is None :
            return None
        elif label == "1" :
            return "2"
        elif label == "2" :
            return "1"
        else :
            raise NotImplementedError

    results = []
    if args.evaluator in NonSwap :
        final_kwargs = dict(input = instance["input"])

        for index, module in enumerate(modules[: -1]) :
            prompt = module["prompt"].format(input = instance["input"])
            auxiliary_input, finish_reason, cost = Complete(prompt, module)
            auxiliary_input = auxiliary_input.strip()
            results.append([auxiliary_input, finish_reason])
            total_cost += cost
            final_kwargs["auxiliary_input_{}".format(index)] = auxiliary_input

        module = modules[-1]
        result = {}
        for swap in (False, True) :
            final_kwargs["output_1"] = instance["output_1"] if not swap else instance["output_2"]
            final_kwargs["output_2"] = instance["output_2"] if not swap else instance["output_1"]
            prompt = module["prompt"].format_map(final_kwargs)
            completion, finish_reason, cost, winner = completion_with_parsing(prompt, module)
            result["swap = {}".format(swap)] = {
                "completion" : [completion, finish_reason],
                "winner" : winner if not swap else reverse_label(winner),
            }
            total_cost += cost
        results.append(result)
    elif args.evaluator in SwapBased : # Swap and Synthesize
        assert len(modules) == 2

        final_kwargs = dict(input = instance["input"])
        module = modules[0]
        result = {}
        for swap in (False, True) :
            final_kwargs["output_1"] = instance["output_1"] if not swap else instance["output_2"]
            final_kwargs["output_2"] = instance["output_2"] if not swap else instance["output_1"]
            prompt = module["prompt"].format_map(final_kwargs)
            completion, finish_reason, cost, winner = completion_with_parsing(prompt, module)
            result["swap = {}".format(swap)] = {
                "completion" : [completion, finish_reason],
                "winner" : winner if not swap else reverse_label(winner),
            }
            total_cost += cost
        results.append(result)

        if {result["swap = False"]["winner"], result["swap = True"]["winner"]} == {"1", "2"} :
            winner2swap = {
                result["swap = False"]["winner"]: False,
                result["swap = True"]["winner"] : True,
            }
            actual_explanation_1 = result["swap = {}".format(winner2swap["1"])]["completion"][0]
            actual_explanation_2 = result["swap = {}".format(winner2swap["2"])]["completion"][0]

            module = modules[1]
            result = {}
            for swap in (False, True) :
                final_kwargs["output_1"] = instance["output_1"] if not swap else instance["output_2"]
                final_kwargs["output_2"] = instance["output_2"] if not swap else instance["output_1"]

                def reverse_OutputAB(explanation) :
                    for output in ("Output", "output") :
                        explanation = explanation.replace("{} (a)".format(output), "**********{} (a)**********".format(output))
                        explanation = explanation.replace("{} (b)".format(output), "##########{} (b)##########".format(output))
                        explanation = explanation.replace("**********{} (a)**********".format(output), "{} (b)".format(output))
                        explanation = explanation.replace("##########{} (b)##########".format(output), "{} (a)".format(output))
                    return explanation
                explanation_1 = actual_explanation_1 if swap == winner2swap["1"] else reverse_OutputAB(actual_explanation_1)
                explanation_2 = actual_explanation_2 if swap == winner2swap["2"] else reverse_OutputAB(actual_explanation_2)

                final_kwargs["explanation_1"] = explanation_1 if not swap else explanation_2
                final_kwargs["explanation_2"] = explanation_2 if not swap else explanation_1
                prompt = module["prompt"].format_map(final_kwargs)

                completion, finish_reason, cost, winner = completion_with_parsing(prompt, module)
                result["swap = {}".format(swap)] = {
                    "prompt" : prompt,
                    "completion" : [completion, finish_reason],
                    "winner" : winner if not swap else reverse_label(winner),
                }
                total_cost += cost
            results.append(result)
    elif args.evaluator in Rating : # Rating
        final_kwargs = dict(input = instance["input"])

        for index, module in enumerate(modules[: -1]) :
            prompt = module["prompt"].format(input = instance["input"])
            auxiliary_input, finish_reason, cost = Complete(prompt, module)
            auxiliary_input = auxiliary_input.strip()
            results.append([auxiliary_input, finish_reason])
            total_cost += cost
            final_kwargs["auxiliary_input_{}".format(index)] = auxiliary_input

        module = modules[-1]
        result = {}
        for index in range(1, 2 + 1) :
            final_kwargs["output"] = instance["output_{}".format(index)]
            prompt = module["prompt"].format_map(final_kwargs)
            completion, finish_reason, cost = Complete(prompt, module)
            result["score_{}".format(index)] = [completion, finish_reason]
            total_cost += cost
        try :
            score_1, score_2 = int(result["score_1"][0]), int(result["score_2"][0])
            result["swap = False"] = {
                "winner" : "1" if score_1 > score_2 else "2"
            }
            result["swap = True"] = {
                "winner" : "2" if score_2 > score_1 else "1"
            }
        except :
            result["swap = False"] = {
                "winner" : "1"
            }
            result["swap = True"] = {
                "winner" : "2"
            }
        results.append(result)
    else :
        raise NotImplementedError
    instance["results"] = results
    return instance, total_cost


def set_api(args) :
    if args.api_type is not None :
        openai.api_type = args.api_type
    if args.api_version is not None :
        openai.api_version = args.api_version
    if args.api_base is not None :
        openai.api_base = args.api_base
    if args.api_key is not None :
        openai.api_key = args.api_key
    if args.organization is not None :
        openai.organization = args.organization
    
    if args.palm_api_key is not None :
        palm.configure(api_key = args.palm_api_key)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, required = True)
    parser.add_argument(
        "--evaluator",
        type = str,
        required = True,
        choices = NonSwap + SwapBased + Rating,
    )
    parser.add_argument("--num_procs", type = int, default = 10)
    parser.add_argument("--api_type", type = str, default = None)
    parser.add_argument("--api_version", type = str, default = None)
    parser.add_argument("--api_base", type = str, default = None)
    parser.add_argument("--api_key", type = str, default = None)
    parser.add_argument("--organization", type = str, default = None)
    parser.add_argument("--hf_use_auth_token", type = str, default = None)
    parser.add_argument("--palm_api_key", type = str, default = None)
    args = parser.parse_args()

    set_api(args)

    dataset = load_json(os.path.join(args.path, "dataset.json"))
    modules = get_modules(args)
    
    if model_and_tokenizer :
        dataset = [annotate(instance, args, modules) for instance in tqdm.tqdm(dataset)]
    else :
        with multiprocessing.Pool(args.num_procs) as p :
            _annotate = functools.partial(annotate, args = args, modules = modules)
            dataset = list(
                tqdm.tqdm(
                    p.imap(_annotate, dataset),
                    desc = "dataset",
                    total = len(dataset),
                )
            )
    total_cost = sum(list(map(lambda pair : pair[1], dataset)))
    print("total_cost = {}".format(total_cost))
    dataset = list(map(lambda pair : pair[0], dataset))
    
    os.makedirs(os.path.join(args.path, "evaluators", args.evaluator), exist_ok = True)
    dump_json(dataset, os.path.join(args.path, "evaluators", args.evaluator, "result.json"))
    correct_False, correct_True, correct_both, equal = 0, 0, 0, 0
    kappa_y1_False, kappa_y1_True, kappa_y2 = [], [], []
    for instance in dataset :
        label = str(instance["label"])
        output_False = instance["results"][-1]["swap = False"]["winner"]
        output_True  = instance["results"][-1]["swap = True"]["winner"]
        correct_False += (output_False == label)
        correct_True += (output_True == label)
        correct_both += ((output_False == label) and (output_True == label))
        equal += (output_True == output_False)

        label = instance["label"]
        output_False = int(output_False) if output_False in ("0", "1") else 0
        output_True = int(output_True) if output_True in ("0", "1") else 0
        kappa_y1_False.append(output_False)
        kappa_y1_True.append(output_True)
        kappa_y2.append(label)
    
    statistics = {
        "correct_False" : "{} / {} = {}%".format(correct_False, len(dataset), correct_False / len(dataset) * 100),
        "correct_True" : "{} / {} = {}%".format(correct_True, len(dataset), correct_True / len(dataset) * 100),
        "correct_average" : "{}%".format((correct_False + correct_True) / 2 / len(dataset) * 100),
        "correct_both" : "{} / {} = {}%".format(correct_both, len(dataset), correct_both / len(dataset) * 100),
        "equal" : "{} / {} = {}%".format(equal, len(dataset), equal / len(dataset) * 100),
    }
    statistics["kappa_False"] = cohen_kappa_score(kappa_y1_False, kappa_y2)
    statistics["kappa_True"] = cohen_kappa_score(kappa_y1_True, kappa_y2)
    statistics["kappa_average"] = (statistics["kappa_False"] + statistics["kappa_True"]) / 2
    statistics["kappa_agreement"] = cohen_kappa_score(kappa_y1_False, kappa_y1_True)
    dump_json(
        statistics,
        os.path.join(args.path, "evaluators", args.evaluator, "statistics.json")
    )