import json
import os.path
import ast
import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import requests


def search_keyword(string_list):
    payload = {
        'index': 'v4_dolma-v1_7_llama',
        'query_type': 'count',
        'query': string_list,
    }
    times = 0
    while 1:
        try:
            result = requests.post('https://api.infini-gram.io/', json=payload).json()
            if "error" in result:
                times += 1
                if times >= 5:
                    return -1
                print(f"{string_list} :尝试 {times} 次")
            else:
                return result['count']
        except:
            continue

def word_count(
        data_path,
        save_path,
        lang1='en',
        lang2='zh',
):

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            count = json.load(f)
    except:
        count = {}
    with open(data_path, "r") as f:
        total_num = 0
        for _ in f:
            total_num += 1
    dataset = load_dataset("json", data_files=data_path)['train']
    for data in tqdm.tqdm(dataset, total=len(dataset)):
        word1 = data[lang1][lang1]
        word2 = data[lang2][lang2]
        if word1 not in count:
            count[word1] = search_keyword(word1)
        if word2 not in count:
            count[word2] = search_keyword(word2)
        if f"{word1} AND {word2}" not in count:
            count[f"{word1} AND {word2}"] = search_keyword(f"{word1} AND {word2}")
        with open(save_path, "w") as f:
            json.dump(count, f, indent=4, ensure_ascii=False)
    return

def word_count_thread(data_path, save_path, lang1='en', lang2='zh', thread=20):
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            count = json.load(f)
    except:
        count = {}

    dataset = load_dataset("json", data_files=data_path)['train']

    def process(data, count, lang1, lang2):
        word1 = data[lang1][lang1]
        word2 = data[lang2][lang2]
        if word1 not in count:
            count[word1] = search_keyword(word1)
        if word2 not in count:
            count[word2] = search_keyword(word2)
        if f"{word1} AND {word2}" not in count:
            count[f"{word1} AND {word2}"] = search_keyword(f"{word1} AND {word2}")
        return count
    with ThreadPoolExecutor(max_workers=thread) as executor:
        futures = []
        for data in dataset:
            futures.append(executor.submit(process, data, count.copy(), lang1, lang2))
        finished = 0
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            count.update(result)
            finished += 1
            if finished % 50 == 0:
                with open(save_path, "w") as f:
                    json.dump(count, f, indent=4, ensure_ascii=False)

    with open(save_path, "w") as f:
        json.dump(count, f, indent=4, ensure_ascii=False)

    return count
def search_word_count_token(doc1, doc2, doc_none, num_thread=40, n_gram=1):
    tokens_set1 = count_in_doc(doc1, n_gram)
    tokens_set2 = count_in_doc(doc2, n_gram)
    tokens = count_in_doc(doc_none, n_gram)
    tokens = dict(sorted(tokens.items(), key=lambda item: item[1], reverse=True))
    tokens_set1 = dict(sorted(tokens_set1.items(), key=lambda item: item[1], reverse=True))
    tokens_set2 = dict(sorted(tokens_set2.items(), key=lambda item: item[1], reverse=True))
    return tokens, tokens_set1, tokens_set2


def get_topk_list(tokens: dict, top_k=10, reverse=True):
    top = 0
    find_tokens = []
    tokens = dict(sorted(tokens.items(), key=lambda item: item[1], reverse=reverse))
    for key, count in tokens.items():
        find_tokens.append([key, round(count, 4)])
        top += 1
        if top == top_k:
            break
    return find_tokens


def find(dataset, idx, lang1='en', lang2='zh', doc_num=2000, num_thread=40, top_k=100):
    word1 = dataset[idx][lang1]
    word2 = dataset[idx][lang2]
    word_message = {}
    with open("/home/hekaiyu/cross-lingual/word/embedding/result/words_token_OLMo-7B-0424-hf.jsonl", "r") as f:
        for line in tqdm.tqdm(f):
            line = json.loads(line)
            word_message[line["word"]] = line
    tokens1 = word_message[word1]['token']
    tokens2 = word_message[word2]['token']
    doc1, doc2, doc_none = get_doc(word1, word2, tokens1, tokens2, doc_num, num_thread=num_thread)
    result = search_word_count_token(doc1, doc2, doc_none, num_thread)
    return find_medium(doc1, doc2, doc_none, result, top_k=top_k)


def read_find(result, n_gram=1, top_k=50):
    tokens1 = result["tokens_set1"]
    tokens2 = result["tokens_set2"]
    tokens_background = result["token_set_none"]
    for key, count in tokens1.items():
        tokens1[key] -= tokens_background[key] if key in tokens_background else 0
    for key, count in tokens2.items():
        tokens2[key] -= tokens_background[key] if key in tokens_background else 0
    token_set1 = get_topk_list(tokens1, top_k=-1)
    token_set2 = get_topk_list(tokens2, top_k=-1)
    result = {}
    vis = {}
    for rank1, token1 in enumerate(token_set1):
        vis[f"{token1[0]}"] = token1[1]
    for rank2, token2 in enumerate(token_set2):
        if f"{token2[0]}" in vis:
            result[f"{token2[0]}"] = -min(vis[token2[0]], token2[1])
    medium_token = get_topk_list(result, top_k=top_k, reverse=False)
    return medium_token

if __name__ == "__main__":
    lang_list = [
        "en",  # English
        "zh",  # Chinese
        # "ru",  # Russian
        # "es",  # Spanish
        "fr",  # French
        # "de"
    ]
    for lang1 in lang_list:
        for lang2 in lang_list:
            if lang1 == lang2:
                continue
            data_path = "/mnt/userdata/cross_lingual/dataset/datasets_all.jsonl"
            save_path = f"/mnt/userdata/cross_lingual/word/word_count/{lang1}-{lang2}.json"
            # word_count(data_path=data_path,save_path=save_path,lang1=lang1, lang2=lang2)
            word_count_thread(data_path=data_path, save_path=save_path, lang1=lang1, lang2=lang2)

