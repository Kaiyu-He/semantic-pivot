import copy
import gzip
import json
import os.path
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent
import threading
from threading import Lock

import tqdm
from datasets import load_dataset


def deepseek_query(word, answer, lang1, lang2):
    from openai import OpenAI
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    query = f"""### Instruction:
你是一个单词翻译问题的出题人。
请根据提供的单词，设计九个干扰项，且保证最终翻译问题的答案唯一。

- 干扰项对应的要求如下：
1、干扰项应不为给定单词在任何语境下的翻译。
2、干扰项互不重复。
3、确保干扰项使用的语言正确。
4、干扰项因保证为单个词语。

- 请按照如下步骤进行回答：
1. 在 "answer" 中，列举出单词的所有翻译结果。
2. 在 "options" 中，列举出干扰项,。

请务必确保干扰项不为给定单词的翻译。

EXAMPLE1 INPUT: 
English: hand
Chinese answer: 手
Chinese options: 

EXAMPLE1 JSON OUTPUT:
{{
    "answer": "手, 方面, 作用",
    "options": "['手指', '腿', '手心', '手掌', '手臂', '肩膀', '指关节', '腕', '脚趾']"
}}

EXAMPLE2 INPUT: 
French: acceptation
Chinese answer: 接受
Chinese options: 

EXAMPLE2 JSON OUTPUT:
{{
    "answer": "接受, 接纳, 同意, 认可"
    "options": "['接收', '认清', '批准', '拒绝', '否认', '接替', '认知', '忽视', '误解']"
}}

EXAMPLE3 INPUT: 
English: thought
French answer: pensée
French options: 

EXAMPLE3 JSON OUTPUT:
{{
    "answer": "pensée"
    "options": "['idée', 'réflexion', 'concept', 'esprit', 'considération', 'avis', 'notion']"
}}

### Input:
{language[lang1]}: {word}
{language[lang2]} answer: {answer}
{language[lang2]} options:

### Response:
"""
    messages1 = [
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages1,
        response_format={
            'type': 'json_object'
        },
        temperature=1.2
    )
    text = json.loads(response.choices[0].message.content)
    text = text["options"]
    import ast
    interference = ast.literal_eval(text)
    return interference


def check_deepseek(word, selects, answer, lang1, lang2):
    from openai import OpenAI
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    selects.append(answer)
    random.shuffle(selects)
    query = f"""### Instruction:
You are an expert in word translation. Please select the correct translation of the given word from the ten options provided.

EXAMPLE INPUT: 
English: hand
Options: ['手指', '腿', '手心', '手掌', '手臂', '肩膀', '指关节', '腕', '脚趾']
Chinese: 

EXAMPLE JSON OUTPUT:
{{
    'Chinese': '手'
}}

### Input:
{language[lang1]}: {word}
Options: {selects}
{language[lang2]}: 

### Response:
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": query},
        ],
        response_format={
            'type': 'json_object'
        },
        stream=False,

    )
    text = response.choices[0].message.content
    try:
        text = json.loads(text)[f"{language[lang2]}"]
        if text == answer:
            return True
    except:
        return False
    return False


def deepseek(query):
    from openai import OpenAI
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    query = f"""{query}"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": query},
        ],
        stream=False,
    )
    text = response.choices[0].message.content
    text = text.split("```")[1][6:]
    return json.loads(text)


def create_multi_dataset(word_path, save_path, already_path, language_list, restart=False):
    dataset = []
    with open(word_path) as f:
        for line in f:
            dataset.append(json.loads(line))
    # dataset = load_dataset("json", data_files=word_path)['train']
    if restart:
        processed = {}
        with open(save_path, "w"):
            pass
    else:
        processed = {}
        with open(save_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    word = json.loads(line)
                    processed[word['en']] = i
                except:
                    processed[word['en']['en']] = i
    futures = []
    file_lock = threading.Lock()
    pbar = tqdm.tqdm(total=len(dataset))

    def process(data, idx):
        option = {}
        for lang in language_list:
            option[lang] = {}
        save = True
        for lang1 in language_list:
            for lang2 in language_list:
                if save is False:
                    continue
                if lang1 == lang2:
                    option[lang1][lang2] = data[lang1]
                    continue
                vis = True
                for _ in range(3):
                    start = time.time() # 构建干扰项
                    option[lang1][lang2] = deepseek_query(data[lang1], data[lang2], lang1, lang2)
                    print("1: ", time.time() - start)
                    vis = True
                    for _ in range(1): # 过滤有歧义的干扰项
                        vis = (vis and check_deepseek(data[lang1], copy.deepcopy(option[lang1][lang2]), data[lang2],
                                                      lang1, lang2))
                    # print(time.time()-start)
                    if vis:
                        break
                if vis is False:
                    save = False
        if save is False:
            option = {}
            for lang in language_list:
                option[lang] = data[lang]
        print("finish:", idx, save)
        with file_lock:
            with open(save_path, "a") as file:
                file.write(json.dumps(option, ensure_ascii=False) + "\n")
        print("idx: ", idx)
        pbar.update(1)
        return option

    with ThreadPoolExecutor(max_workers=20) as executor:
        for i, word in enumerate(dataset):
            if word['en'] in processed:
                pbar.update(1)
                continue
            future = executor.submit(process, word, i)
        #     save_result(save_path, result)


def get_Azure_translate(word, lang_list=None):
    if lang_list is None:
        lang_list = ["zh", ]
    import requests, uuid, json

    # Add your key and endpoint
    key = "BxMOx5QdFtSz5BBRRTo4mMDNChRIDw2Y5thlk00WZiYKpgXR6EY5JQQJ99AKAC3pKaRXJ3w3AAAbACOGkAGg"
    endpoint = "https://api.cognitive.microsofttranslator.com"

    # location, also known as region.
    # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
    location = "eastasia"

    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': lang_list
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': f'{word}'
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    result = {}
    result["en"] = word
    for trans in response[0]['translations']:
        result[trans['to'][:2]] = trans['text']

    return result


def get_word_dataset(word_path, language_list):
    data_path = "/netdisk/hekaiyu/cross_lingual/dataset/word_19k.jsonl"
    dataset = load_dataset("json", data_files=data_path)['train'] # 获取字典单词
    try:
        words = load_dataset("json", data_files=word_path)['train']
        processed_idx = len(words)
    except:
        processed_idx = 0

    def processed(word):
        result = get_Azure_translate(word['en'], language_list) # 构建平行语料
        for lang in language_list:
            if lang in word:
                result[lang] = word[lang]
        return result

    futures = []
    results = {}
    with ThreadPoolExecutor(max_workers=40) as executor:
        for i, word in enumerate(dataset):
            if i < processed_idx:
                continue
            future = executor.submit(processed, word)
            futures.append((i, future))
        for i, future in tqdm.tqdm(futures, total=len(futures)):
            result = future.result()
            results[i] = result
            while processed_idx in results:
                with open(word_path, "a") as file:
                    file.write(json.dumps(results[processed_idx], ensure_ascii=False) + "\n")
                processed_idx += 1


def filter(word_path, data_path, save_path):
    result = {}
    with open(data_path, "r") as f:
        for line in f:
            try:
                line = json.loads(line)
                result[line['en']['en']] = line
            except:
                continue
    dataset = load_dataset("json", data_files=word_path)['train']
    with open(save_path, "w"):
        pass
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        with open(save_path, "a") as f:
            try:
                f.write(json.dumps(result[data['en']], ensure_ascii=False) + '\n')
            except:
                continue
    dataset = load_dataset("json", data_files=save_path)['train']
    dataset = dataset.shuffle(seed=98).select(range(0, 2000))
    dataset.to_json("/netdisk/hekaiyu/cross_lingual/dataset/dataset.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    word_path = "/netdisk/hekaiyu/cross_lingual/dataset/word_19k_2.jsonl"
    data_path = "/netdisk/hekaiyu/cross_lingual/dataset/datasets.jsonl"
    save_path = "/netdisk/hekaiyu/cross_lingual/dataset/datasets_option.jsonl"
    already_path = ""
    language = {
        "en": "English",
        "zh": "Chinese",
        "ru": "Russian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese"
    }
    deepseek_api_key = "sk-df3548002777404da2b96456cb349046"
    # print(deepseek_query("academy","学院", "en", "zh"))
    # print(deepseek_query("académie", "学院", "fr", "zh"))
    #
    language_list = ["en", "fr", "zh", "ja"]
    # get_word_dataset(word_path,language_list)
    create_multi_dataset(word_path, data_path, already_path, language_list)
    # data_path = "/netdisk/hekaiyu/cross_lingual/dataset/dataset.jsonl"
    # filter(word_path, data_path, save_path)
    # filter()
