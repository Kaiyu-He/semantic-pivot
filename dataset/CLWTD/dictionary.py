import os
import tqdm
import json

path = "./download/DICT/DICT/"


def get_dictionary(path):
    path_list = []
    for f1 in os.listdir(path):
        for f2 in os.listdir(path + f1 + "/"):
            path_list.append(f"{path}{f1}/{f2}")
    path_list.sort()
    dictionary = {}
    for path in path_list:
        with open(path, 'r', encoding='gbk', errors='ignore') as file:
            word = ""
            for i, line in enumerate(file):
                if not i % 2:
                    word = line
                    while len(word) and (word[-1] == ' ' or word[-1] == '\n'):
                        word = word[:-1]
                    continue
                else:
                    describe = line
                if not len(word):
                    continue
                dictionary[word] = {
                    "describe": describe,
                    "word_type": describe.split('/ ')[-1].split(' ')[0]
                }
                if dictionary[word]['word_type'] not in ['n', 'v', 'adj', 'adv']:
                    dictionary[word]['word_type'] = None
                # print(f"{word}:: {line}")
    # print(path_list)
    return dictionary


def filter_word(data_path, dictionary, total_num=20000):
    word = []

    def check(describe, doc):
        return doc['describe'].find(describe) >= 0 and doc['word_type'] == 'n'

    with open(data_path, "r") as f:
        for line in tqdm.tqdm(f, total=total_num):
            data = json.loads(line)
            en = data["en"]
            zh = data["zh"]
            if len(en) <= 2 or en == zh:
                continue
            if en in dictionary and check(zh, dictionary[en]):
                word.append(data)
    return word


def main():
    dict = get_dictionary("./download/DICT/DICT/")
    words = filter_word(
        data_path="/home/hekaiyu/cross-lingual/word/dictionary/word_dataset_deepseek.jsonl",
        dictionary=dict
    )
    path = "/home/hekaiyu/cross-lingual/word/dictionary/word_dataset_filter_n.jsonl"
    with open(path, "w"):
        pass
    with open(path, "a") as f:
        for word in words:
            f.write(json.dumps(word, ensure_ascii=False) + '\n')
    print(len(words))


def get_dictionary_word():
    path = "./download/DICT/DICT/"
    path_list = []
    for f1 in os.listdir(path):
        for f2 in os.listdir(path + f1 + "/"):
            path_list.append(f"{path}{f1}/{f2}")
    path_list.sort()
    dictionary = {}
    words = []
    def f(word):
        for c in word:
            if not 'a' <= c <= 'z':
                return 0
        return 1
    for path in path_list:
        with open(path, 'r', encoding='gbk', errors='ignore') as file:
            word = ""
            for i, line in enumerate(file):
                if not i % 2:
                    word = line
                    while len(word) and (word[-1] == ' ' or word[-1] == '\n'):
                        word = word[:-1]
                    continue
                else:
                    describe = line
                if not len(word):
                    continue
                if word in dictionary and dictionary[word]["word_type"] == 'n':
                    continue
                dictionary[word] = {
                    "describe": "",
                    "word_type": describe.split('/ ')[-1].split(' ')[0]
                }
                if dictionary[word]['word_type'] in ['n', 'v', 'adj']:
                    dictionary[word]["describe"] = describe
                    if word not in words and f(word):
                        words.append(word)
                # print(f"{word}:: {line}")
    # print(path_list)
    return dictionary, words


if __name__ == "__main__":
    dictionary, words = get_dictionary_word()
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="/netdisk/hekaiyu/cross_lingual/dataset/dataset.jsonl")['train']
    count = {}
    for data in dataset:
        word = data['en']['en']
        if dictionary[word]['word_type'] in count:
            count[dictionary[word]['word_type']] += 1
        else:
            count[dictionary[word]['word_type']] = 1
        if dictionary[word]['word_type'] not in ['n', 'v', 'adj']:
            print(dictionary[word])
            print(word)
    print(count)
