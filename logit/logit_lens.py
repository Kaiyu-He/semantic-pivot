import json
import time
import ast
import argparse
import math
import heapq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import colorcet as cc
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset


def get_logit_lens(model, tokenizer, prompt, last):
    def decode(model, tokenizer, output):  # 解码隐藏层输出
        layer_output = model.lm_head(model.model.norm(output))
        final, logits = [], []
        for i in range(len(layer_output)):
            final.append(torch.argmax(layer_output[i]))
            logits.append(torch.max(layer_output[i]).item())

        final = [tokenizer.decode(tokens) + f"/ {tokens}" for tokens in final]
        return final, logits

    output_saving = []

    def forward_hook(module, input, output):
        output_saving.append(output)

    for layer in model.model.layers: # 获取中间层输出
        layer.register_forward_hook(forward_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    # 输入句子的 token 列表
    """sentence_shattered 是输入句子的不同 token 的列表"""
    sentence_shattered = tokenizer.encode(prompt, add_special_tokens=False)  # 编码输入句子为 token 列表
    for i in range(len(sentence_shattered)):
        sentence_shattered[i] = tokenizer.decode(sentence_shattered[i])  # 将每个 token 解码为单词

    # 初始化存储解码单词和 logits 的列表
    """我将不同的解码单词存储在 words_df 中，将它们的 logits 存储在 logits_df 中"""
    words_df, logits_df = [sentence_shattered], [[0] * len(sentence_shattered)]  # 初始化 words_df 和 logits_df
    # 遍历每一层的输出并解码
    for i, element in enumerate(tqdm(output_saving[:])):
        final, logits = decode(model, tokenizer, element[0][0])
        words_df.append(final)  # 将解码后的单词添加到 words_df
        logits_df.append(logits)  # 将对应的 logits 添加到 logits_df

    index = [f"Layer {i}" for i in range(len(output_saving))]  # 创建索引，表示每一层的名称
    index.insert(0, "sentence")  # 在索引的开头插入 "sentence"，表示输入句子
    words_df = pd.DataFrame(words_df, index=index)  # 将 words_df 转换为 DataFrame
    logits_df = pd.DataFrame(logits_df, index=index)  # 将 logits_df 转换为 DataFrame
    words_df = words_df.dropna(axis=1).iloc[:, -last:]
    logits_df = logits_df.dropna(axis=1).iloc[:, -last:]
    return words_df, logits_df


def logit_lens(model, tokenizer, text, last):
    words_df, logits_df = get_logit_lens(model, tokenizer, text, last)
    fig, ax = plt.subplots(figsize=(25, 8))
    ax = sns.heatmap(logits_df.astype(int), ax=ax, annot=words_df,
                     cbar=True, fmt="", cmap=cc.kbgyw[::-1],
                     linewidth=0.0, cbar_kws={'label': 'Logits'})
    ax.set(title="Logit Lens Experiments", xlabel="Tokens", ylabel="Layer index", )
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    plt.show()


def logit_lens_multi_tokens(model, tokenizer, prompt, answer, word):
    output_saving = []

    def forward_hook(module, input, output):
        output_saving.append(output)

    def decode(model, output, labels, word, num_token):
        layer_output = model.lm_head(model.model.norm(output))
        criterion = nn.CrossEntropyLoss()
        num_token = min(num_token, len(word))
        loss = criterion(layer_output[-num_token - 1:-1], word[:num_token])
        loss = loss.item()
        return math.exp(-loss), layer_output

    for layer in model.model.layers:
        layer.register_forward_hook(forward_hook)
    inputs = tokenizer(prompt + answer, return_tensors="pt").to(f"cuda:{cuda_id}")
    prompt = tokenizer(prompt, return_tensors="pt").to(f"cuda:{cuda_id}")
    word = tokenizer(word, return_tensors="pt").to(f"cuda:{cuda_id}")['input_ids'][0]
    num_token = len(inputs["input_ids"][0]) - len(prompt['input_ids'][0])

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    # 遍历每一层的输出并解码
    logits = []
    for i, element in enumerate(tqdm(output_saving[:])):
        loss, layer_out = decode(model, element[0][0], inputs["input_ids"][0], word, num_token)
        logits.append(loss)
    return logits


def get_loss(model, tokenizer, prompts: list, answers: list, batch_size=1):
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
                input_ids=batch_inputs,
            )
        logits = output.logits[:, :-1, :]
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


def draw_line(x, y, title=""):
    plt.figure()
    for label, ans in y.items():
        plt.plot(x, ans, label=f"{label}")
    plt.title(title)
    plt.axhline(y=0, color='gray', linestyle='--', label='y = 0')
    plt.ylabel("loss")
    plt.ylim((0, 1))
    plt.legend()
    plt.show()


def get_fewshot_prompt(lang1, word1, lang2, word2): # 构建few-shot示例
    return f"{language_list[lang1]}: {word1}\n{language_list[lang2]}: {word2}\n\n"


def prompt_text(lang1, lang2): # 构建提示词
    prompt_template = "Please translate words from <lang1> to <lang2>.\n\n<example><lang1>: [X]\n<lang2>:"
    return prompt_template.replace("<lang1>", language_list[lang1]).replace("<lang2>", language_list[lang2])


def get_text(word1="impossible", word2="impossible", lang1="fr", lang2="en", num_of_shot=5):
    import random
    dataset = load_dataset("json", data_files=data_path)['train']

    def get_example(word1):
        sample = []
        fewshot = ""
        for idx in random.sample(range(0, len(dataset)), num_of_shot):
            if dataset[idx][lang1] != word1 and idx not in sample:
                sample.append(idx)
                fewshot += get_fewshot_prompt(lang1, dataset[idx][lang1][lang1], lang2, dataset[idx][lang2][lang2])
        return fewshot

    def get_prompt(lang1, word1, lang2, word2):
        prompt_get = prompt_text(lang1, lang2)
        text = prompt_get.replace("[X]", word1).replace("<example>", get_example(word1))
        return text

    text = get_prompt(lang1, word1, lang2, word2)
    return text, f" {word2}\n\n"


def logit_lens_tokens(model, tokenizer, prompt, answer):
    output_saving = []

    def forward_hook(module, input, output):
        output_saving.append(output)

    # 设置输出embedding所在显卡
    lm_head = model.lm_head.to(f"cuda:{cuda_id}")
    norm = model.model.norm.to(f"cuda:{cuda_id}")

    def decode(output, num_token): # 将中间层输出的output解码
        layer_output = lm_head(norm(output))
        layer_output = layer_output[-num_token:]
        return layer_output

    hooks = []
    for layer in model.model.layers: # 获取中间层输出
        hooks.append(layer.register_forward_hook(forward_hook))
    inputs = tokenizer(prompt + answer, return_tensors="pt").to(f"cuda:{cuda_id}")
    prompt = tokenizer(prompt, return_tensors="pt").to(f"cuda:{cuda_id}")
    num_token = len(inputs["input_ids"][0]) - len(prompt['input_ids'][0])
    labels = inputs["input_ids"][0][-num_token:].tolist()
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    layers_out = []
    for i, element in enumerate(output_saving[:]): # 解码每一层
        layer_out = decode(element[0][0], num_token).tolist()
        layers_out.append(layer_out)

    del inputs
    del outputs
    del prompt
    del output_saving
    for hook in hooks:
        hook.remove()  # 移除钩子
    torch.cuda.empty_cache()
    return layers_out, labels


def decode(tokenizer, layer_output, top_k=10):
    layer_result = []
    for token in layer_output:
        softmax_output = F.softmax(torch.tensor(token), dim=0).tolist()
        top_k_token = heapq.nlargest(top_k, enumerate(softmax_output), key=lambda x: x[1])
        layer_result.append([])
        for token in top_k_token:
            layer_result[-1].append((tokenizer.decode([token[0]]), token[0], token[1]))
    return layer_result


def preprocess(data_path, save_dir, model, tokenizer, lang1='en', lang2='zh', topk=10):
    dataset = load_dataset("json", data_files=data_path)['train']
    if not os.path.exists(f"{save_dir}"):
        os.mkdir(f"{save_dir}")
    if not os.path.exists(f"{save_dir}/{lang1}-{lang2}"):
        os.mkdir(f"{save_dir}/{lang1}-{lang2}")
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        save_path = f"{save_dir}/{lang1}-{lang2}/{idx}.jsonl"
        if os.path.exists(save_path):
            continue
        prompt, answer = get_text(dataset[idx][lang1][lang1], dataset[idx][lang2][lang2], lang1, lang2)
        prompt = prompt.split(f"{language_list[lang1]}: {dataset[idx][lang1][lang1]}")[0]

        layers_out, labels = logit_lens_tokens(model=model, tokenizer=tokenizer, prompt=prompt,
                                               answer=f"{language_list[lang1]}: {dataset[idx][lang1][lang1]}\n{language_list[lang2]}: {dataset[idx][lang2][lang2]}\n\n")
        save_layer_out = {}
        lie = 0
        for i in reversed(range(len(labels))):
            if labels[i] == 27: # 定位输出位置
                lie = i
                break
        for i, layer_output in enumerate(layers_out):
            layer_result = []
            layer_output = layer_output[lie:lie + 2]
            for token in layer_output:
                softmax_output = F.softmax(torch.tensor(token), dim=0).tolist() # 求取概率
                index, value = max(enumerate(softmax_output), key=lambda x: x[1]) # 取最大值
                layer_result.append((tokenizer.decode([index]), index, value))
            save_layer_out[i] = layer_result
        result = {
            "idx": idx,
            "word": dataset[idx][lang1][lang1],
            "labels": labels[lie:lie + 2],
            "layer_out": save_layer_out
        }
        with open(save_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def preprocess2(data_path, save_dir, model, tokenizer, lang1='en', lang2='zh'):
    dataset = load_dataset("json", data_files=data_path)['train']
    print(f"{lang1}-{lang2}")
    if not os.path.exists(f"{save_dir}"):
        os.mkdir(f"{save_dir}")
    if not os.path.exists(f"{save_dir}/{lang1}-{lang2}"):
        os.mkdir(f"{save_dir}/{lang1}-{lang2}")
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        save_path = f"{save_dir}/{lang1}-{lang2}/{idx}.jsonl"
        if os.path.exists(save_path) and os.path.getsize(save_path) >= 50:
            continue
        prompt, answer = get_text(dataset[idx][lang1][lang1], dataset[idx][lang2][lang2], lang1, lang2)
        print(prompt)
        prompt = prompt.split(f"{language_list[lang1]}: {dataset[idx][lang1][lang1]}")[0]
        start = time.time()
        layers_out, labels = logit_lens_tokens(model=model, tokenizer=tokenizer, prompt=prompt,
                                               answer=f"{language_list[lang1]}: {dataset[idx][lang1][lang1]}\n{language_list[lang2]}: {dataset[idx][lang2][lang2]}\n\n")
        save_layer_out = {}
        lie = 0
        for i in reversed(range(len(labels))):
            if labels[i] == 27:
                lie = i
                break
        labels = labels[lie:]
        for i, layer_output in enumerate(layers_out):
            save_layer_out[i] = layer_output[lie: lie + 2]
        result = {
            "idx": idx,
            "word": dataset[idx][lang1][lang1],
            "labels": labels,
            "layer_out": save_layer_out
        }
        start = time.time()
        with open(save_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_devices = torch.cuda.device_count()
    cuda_id = num_devices - 1
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--language_list",
        type=str,
        default="['en', 'zh']",
        help=""
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="。/word/dictionary/dataset.jsonl",
        help=""
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./result/logit_lens/logit",
        help=""
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/OLMo-7B-0424-hf/step400000",
        help=""
    )
    args = parser.parse_known_args()[0]
    lang_list = ast.literal_eval(args.language_list)
    data_path = args.data_path
    save_dir = args.save_dir
    print(data_path)
    print(save_dir)
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    language_list = {
        "en": "English",
        "zh": "Chinese",
        "ru": "Russian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese"
    }
    for lang1 in lang_list:
        for lang2 in lang_list:
            if lang1 == lang2:
                continue
            # preprocess2(data_path, save_dir, model, tokenizer, lang1, lang2)
            preprocess(data_path, save_dir, model, tokenizer, lang1, lang2)
