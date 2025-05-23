import argparse
import torch
import json
import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import ast
import torch.nn.functional as F


def get_loss(model, tokenizer, prompts: list, answers: list, batch_size=10):
    inputs_ids = []
    labels = []
    max_length = 0
    losses = []
    for prompt, answer in zip(prompts, answers):
        question = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = tokenizer(answer, add_special_tokens=False)
        inputs_ids.append(prompt_ids["input_ids"])
        labels.append([-100] * len(question["input_ids"]) + prompt_ids["input_ids"][len(question["input_ids"]):])
        max_length = max(max_length, len(prompt_ids["input_ids"]))
    for i in range(len(inputs_ids)):
        inputs_ids[i] = inputs_ids[i] + [tokenizer.pad_token_id] * (max_length - len(inputs_ids[i]))
        labels[i] = labels[i] + [-100] * (max_length - len(labels[i]))
    num_batches = (len(inputs_ids) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_inputs = torch.tensor(inputs_ids[start_idx:end_idx])
        batch_labels = torch.tensor(labels[start_idx:end_idx])
        with torch.no_grad():
            output = model(
                input_ids=batch_inputs.to('cuda'),
            )
        logits = output.logits[:, :-1, :].to('cpu')
        batch_labels = torch.tensor(batch_labels[:, 1:]).view(-1)
        batch_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch_labels,
            ignore_index=-100,
            reduction="none"
        )
        batch_loss = batch_loss.view(batch_size, -1)
        batch_loss = (batch_loss.sum(dim=1) / (batch_loss != 0).sum(dim=1)).tolist()
        losses += batch_loss
    return torch.tensor(losses)


def prompt_text(lang1, lang2):
    prompt = "Please translate words from <lang1> to <lang2>.\n\n<example><lang1>: [X]\n<lang2>:"
    return prompt.replace("<lang1>", prompt_list[lang1]).replace("<lang2>", prompt_list[lang2])


def get_fewshot_prompt(lang1, word1, lang2, word2):
    return f"{prompt_list[lang1]}: {word1}\n{prompt_list[lang2]}: {word2}\n\n"


def get_word_capability(model, tokenizer, data_path, save_path, lang1, lang2, num_of_dataset,
                        num_of_shot=0, times=5, restart=False):
    print(data_path)
    dataset = load_dataset("json", data_files=data_path)['train']
    dataset = dataset.select(range(max(num_of_dataset[0], 0), min(num_of_dataset[1], len(dataset))))
    prompt = prompt_text(lang1, lang2)
    if not os.path.exists(save_path) or restart:
        with open(save_path, "w"):
            pass
    num = -1
    try:
        with open(save_path, "r") as f:
            for num, line in enumerate(f):
                if dataset[num][lang2][lang2] != json.loads(line)["word2"]:
                    with open(save_path, "w"):
                        pass
                    num = -1
                    break
    except:
        with open(save_path, "w"):
            pass
    for i, words in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if i <= num:
            continue
        choice = [words[lang1][lang1]] + words[lang2][lang1][:9]
        choice_label = {}
        for label in choice:
            choice_label[label] = 0
        sum_of_fewshot = []
        for _ in range(times):
            sample = []
            fewshot = ""
            for idx in random.sample(range(0, len(dataset)), num_of_shot):
                if idx != i and idx not in sample:
                    sample.append(idx)
                    fewshot += get_fewshot_prompt(lang1, dataset[idx][lang1][lang1], lang2, dataset[idx][lang2][lang2])
            sum_of_fewshot.append(fewshot)
        batch_prompt = []
        batch_prompt_answer = []
        for label in choice:
            for example in sum_of_fewshot:
                batch_prompt.append(prompt.replace("[X]", label).replace("<example>", example))
                batch_prompt_answer.append(
                    prompt.replace("[X]", label).replace("<example>", example) + f" {words[lang2][lang2]}")
        import time
        start = time.time()
        losses = get_loss(model, tokenizer, batch_prompt, batch_prompt_answer)
        losses = losses.view(len(choice), -1).sum(dim=1) / times
        for i, label in enumerate(choice):
            choice_label[label] = losses[i].tolist()
        with open(save_path, "a") as f:
            f.write(json.dumps({"word2": words[lang2][lang2], "word1": words[lang1][lang1], "loss": choice_label},
                               ensure_ascii=False) + '\n')


def main(
        model_paths,
        lang_list,
        data_path,
        save_dir,
        name: str,
        num_of_shot=3,
        times=5,
        num_of_dataset=[0, 100],
        restart=False
):
    if type(num_of_dataset) == int:
        num_of_dataset = [0, num_of_dataset]
    for model_path, model_name, model_step in tqdm.tqdm(model_paths, total=len(model_paths)):
        print(model_path)
        if not os.path.exists(model_path):
            continue
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        except:
            continue
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        for lang1 in lang_list:
            for lang2 in lang_list:
                # if lang1 != "en" and lang2 != "en" and lang1 != "fr" and lang2 != "fr":
                #     continue
                if lang1 == lang2:
                    continue
                save_path = f"{save_dir}/{lang1}-{lang2}"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = save_path + f"/{model_name}"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = save_path + f"/{name}.jsonl"
                get_word_capability(model=model, tokenizer=tokenizer, data_path=data_path,
                                    save_path=save_path, lang1=lang1, lang2=lang2,
                                    num_of_shot=num_of_shot, times=times,
                                    num_of_dataset=num_of_dataset, restart=restart)
        del model
        del tokenizer
        import gc
        import time
        time.sleep(0.5)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="1",
        help=""
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="1",
        help=""
    )
    parser.add_argument(
        "--num_of_shot",
        type=int,
        default=5,
        help=""
    )
    parser.add_argument(
        "--num_of_dataset",
        type=int,
        default=2000,
        help=""
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/hekaiyu/cross-lingual/word/dictionary/dataset.jsonl",
        help=""
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/userdata/hekaiyu/cross_lingual/result/result",
        help=""
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/userdata/hekaiyu/model",
        help=""
    )
    parser.add_argument(
        "--language_list",
        type=str,
        default="['fr', 'ja']",
        help=""
    )
    args = parser.parse_known_args()[0]
    lang_list = ast.literal_eval(args.language_list)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if type(args.num_of_dataset) == str:
        try:
            args.num_of_dataset = ast.literal_eval(args.num_of_dataset)
        except:
            args.num_of_dataset = int(args.num_of_dataset)
    model_paths = [(args.model_path, args.model_name, -1)]
    print(model_paths)
    prompt_list = {
        "en": "English",
        "zh": "Chinese",
        "ru": "Russian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese"
    }
    # lang_list = ["en", "zh", "ru"]
    main(model_paths=model_paths,  # (path, model_name, step)
         data_path=args.data_path,
         name=args.name,
         save_dir=args.save_dir,  # save_path = f"{save_dir}/{lang1}-{lang2}/{model_name}/{model_step}-{id}.jsonl"
         lang_list=lang_list,
         times=5,
         num_of_shot=args.num_of_shot,
         num_of_dataset=args.num_of_dataset,
         restart=False
         )
