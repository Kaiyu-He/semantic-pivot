import json
import os.path
import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import requests

def count_in_doc(doc_list, num_thread, n_gram):
    def count(docs):
        tokens_set = {}
        for doc in docs:
            doc_token = {}
            for i in range(len(doc)):
                if i + n_gram > len(doc):
                    continue
                gram = doc[i: i + n_gram]
                key = f"{gram}"
                if key in doc_token:
                    continue
                if key in tokens_set:
                    tokens_set[key] += 1
                else:
                    tokens_set[key] = 1
                doc_token[key] = 1
        return tokens_set

    sum_doc = {}
    size = len(doc_list) // num_thread
    split_doc = [doc_list[i * size: min(len(doc_list), (i + 1) * size)] for i in range(num_thread)]
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = [executor.submit(count, docs) for docs in split_doc]
        for future in as_completed(futures):
            doc_result = future.result()
            for key, value in doc_result.items():
                if key not in sum_doc:
                    sum_doc[key] = value
                else:
                    sum_doc[key] += value
    return sum_doc


def search_word_count_token(doc, num_thread=40, n_gram=1):
    tokens_set = count_in_doc(doc, num_thread, n_gram)
    tokens_set = dict(sorted(tokens_set.items(), key=lambda item: item[1], reverse=True))
    return tokens_set


def get_doc(word, tokens, max_num=200, num_thread=20):
    doc = []

    def search_word_doc(string):
        payload = {
            'index': 'v4_dolma-v1_7_llama',
            'query_type': 'search_docs',
            'query': string,
            'maxnum': 10
        }
        times = 0
        while 1:
            try:
                result = requests.post('https://api.infini-gram.io/', json=payload).json()
                if "error" in result or 'documents' not in result:
                    times += 1
                else:
                    return result
            except:
                continue

    def whether_word_in_doc(tokens, doc_token):
        lens = len(tokens)
        for i in range(len(doc_token) - lens + 1):
            for j in range(lens):
                try:
                    if doc_token[i + j] == tokens[j]:
                        return 1
                except:
                    continue
        return 0

    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llama2_tokenizer = AutoTokenizer.from_pretrained("/netdisk/hekaiyu/model/llama2")

    def process_documents(word, tokens, num_of_doc):
        doc_container = []
        for _ in range(num_of_doc // 5):
            result = search_word_doc(word)
            for document in result['documents']:
                doc = llama2_tokenizer.decode(document["token_ids"])
                doc = tokenizer(doc, add_special_tokens=False)['input_ids']
                if not len(word) or whether_word_in_doc(tokens, doc):
                    doc_container.append(doc)
                    if len(doc_container) >= num_of_doc:
                        doc_container = doc_container[:num_of_doc]
                        break
            if len(doc_container) >= num_of_doc:
                doc_container = doc_container[:num_of_doc]
                break

        return doc_container

    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = [executor.submit(process_documents, word, tokens, max_num // num_thread) for _ in range(num_thread)]
        for future in as_completed(futures):
            doc_result = future.result()
            if doc_result:
                doc += doc_result
    return doc


def main(tokenizer, lang='en', data_path="./dataset.json", save_dir="./word_set.json", doc_num=2000, num_thread=50, n_gram=1):
    save_path = f"{save_dir}/n_gram_{n_gram}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + f"/{lang}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dataset = load_dataset("json", data_files=data_path)['train']
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        path = f"{save_path}/{i}.json"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    word = json.load(f)
                if word['word'] == data[lang][lang]:
                    continue
                else:
                    with open(path, "w"):
                        pass
            except:
                with open(path, "w"):
                    pass
        word = data[lang][lang]
        tokens = tokenizer(f" {word}")['input_ids']
        with open(path, "a") as f:
            doc = get_doc(word, tokens, doc_num, num_thread=num_thread)
            token_set = search_word_count_token(doc, num_thread, n_gram)
            result = {
                "word": word,
                "token": tokens,
                "tokens_set": token_set,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # doc = get_doc("", [], 2000, num_thread=40)
    # token_set = search_word_count_token(doc, 50, 1)
    # result = {
    #     "word": "",
    #     "token": [],
    #     "tokens_set": token_set,
    # }
    # with open("/netdisk/hekaiyu/cross_lingual/word/word_set/n_gram_1/none.json", "a") as f:
    #     f.write(json.dumps(result, ensure_ascii=False) + "\n")

    lang_list = [
        "en",  # English
        "zh",  # Chinese
        # "ru",  # Russian
        # "es",  # Spanish
        "fr",  # French
        # "de"
    ]
    data_path = "/netdisk/hekaiyu/cross_lingual/dataset/datasets_all.jsonl"
    save_dir = "/netdisk/hekaiyu/cross_lingual/word/word_set"
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for lang in lang_list:
        main(lang=lang, data_path=data_path, save_dir=save_dir, tokenizer=tokenizer, n_gram=1)
