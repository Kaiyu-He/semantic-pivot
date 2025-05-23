import json
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
import torch
from pathlib import Path
import time
import requests
import os
import tqdm
from datasets import load_dataset

import numpy as np
from cached_path import cached_path
from transformers import AutoTokenizer

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset


# Update these paths to what you want:


def creat_indice(start_ckpt, end_ckpt):
    import numpy as np
    path = "/netdisk/hekaiyu/dataset/Olmo/1B/multilingual/global_indices.npy"
    data_list = []
    for i in tqdm.tqdm(range(start_ckpt * 1000 * 2048)):
        data_list.append(0)
    for i in tqdm.tqdm(range((end_ckpt - start_ckpt + 1) * 1000 * 2048)):
        data_list.append(i)
    data_array = np.array(data_list, dtype=np.uint32)
    data_array_mmap = np.memmap(
        Path(path),  # 文件路径
        dtype=np.uint32,  # 数据类型
        mode="w+",  # 读写模式，如果文件存在则覆盖
        shape=(len(data_array),)  # 数组形状
    )

    data_array_mmap[:] = data_array

    data_array_mmap.flush()
    del data_array_mmap

    select = np.memmap(path, mode="r", dtype=np.uint32)
    print(select[0])
    print(select[-1])
    print(len(select))

def get_doc(chunk):
    token_ids = []
    for index in chunk:
        num = 0
        while 1:
            num += 1
            try:
                data = dataset[int(index)]
                token_ids.append(data['input_ids'].tolist())
                break
            except Exception as e:
                print(f"{index}: {e}")
    return token_ids


def write(data_list, save_path):
    token_ids = np.array(data_list, dtype=np.uint16)
    fp = np.memmap(save_path, dtype='uint16', mode='w+', shape=token_ids.shape)
    fp[:] = token_ids[:]
    print(f"saved: {save_path}")
    del fp


def creat_dataset_1(start_ckpt, save_dir):

    def get_indices(global_indices, start_ckpt) -> list[list[int]]:
        start = start_ckpt * 1000 * 2048
        end = start + 1000 * 2048
        indices = []
        for idx in global_indices[start:end]:
            indices.append(idx)
        return indices

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    data_order_file_path = cached_path("/netdisk/hekaiyu/cross_lingual/train/1B/1/global_indices.npy")
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    indices = get_indices(global_indices, start_ckpt)
    chunk_size = 32
    save_step = 200
    num = 1
    while os.path.exists(save_dir + f"/ckpt{start_ckpt}-{num:04d}.npy"):
        num += 1
    print(save_dir + f"/ckpt{start_ckpt}-{num:04d}.npy")
    data = []
    with multiprocessing.Pool(processes=200) as pool:
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        chunks = chunks[int((num - 1) * save_step * batch_size / chunk_size):]
        pbar = tqdm.tqdm(total=len(chunks))
        for result in pool.imap(get_doc, [chunk for chunk in chunks]):
            for doc in result:
                data.append(doc)
            pbar.update(1)
            if len(data) % (2048 * save_step) == 0:
                save_path = save_dir + f"/ckpt{start_ckpt}-{num:04d}.npy"
                print(save_path)
                write(data, save_path)
                data = []
                num += 1


def get_language(token_ids, lid_model, tokenizer):
    text = tokenizer.decode(token_ids)
    text = text.replace("\n", "   ")
    language = lid_model.predict(text)
    return language[0][0][9:]


def download_ckpt(ckpt):
    def download_with_progress(url, save_path):
        try:
            downloaded_size = os.path.getsize(save_path)
        except FileNotFoundError:
            downloaded_size = 0

        # 设置请求头，支持断点续传
        headers = {"Range": f"bytes={downloaded_size}-"}
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('content-length', 0)) + downloaded_size

        # 以追加模式打开文件
        with open(save_path, 'ab') as file, tqdm.tqdm(
                desc=save_path.split('/')[-2],
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                initial=downloaded_size,  # 设置初始值
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        print(f"文件已下载到：{save_path}")

    files = ['config.yaml']
    for file in files:
        url = f"https://olmo-checkpoints.org/ai2-llm/olmo-small/mkunaie6/step{ckpt}000-unsharded/{file}"
        save_path = f"/netdisk/hekaiyu/dataset/Olmo/1B/ckpt{ckpt}-unsharded/{file}"
        if not os.path.exists(f"/netdisk/hekaiyu/dataset/Olmo/1B/ckpt{ckpt}-unsharded"):
            os.mkdir(f"/netdisk/hekaiyu/dataset/Olmo/1B/ckpt{ckpt}-unsharded")
        download_with_progress(url, save_path)

def download_dataset():
    for ckpt in range(638, 676):
        creat_dataset_1(ckpt, f"/netdisk/hekaiyu/dataset/Olmo/1B/data/ckpt{ckpt}")


def creat_multilingual_dataset(start_ckpt=605, concat_num=15, ckpt_num=1):
    save_dir = "/netdisk/hekaiyu/dataset/Olmo/1B/multi-15"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    multidata = []
    num_of_multilingual = {
        'en': 0,
        "fr": 0,
        "zh": 0,
        "ja": 0,
    }
    doc_num = 0
    for j in range(ckpt_num):
        for ckpt in range(start_ckpt*j, start_ckpt*j+15):
            data = np.memmap(f"/netdisk/hekaiyu/dataset/Olmo/1B/multickpt/ckpt{ckpt}-multi.npy", dtype='uint16', mode='r')
            for i in tqdm.tqdm(range(int(len(data)/2048))):
                multidata.append(data[i * 2048 : (i+1) * 2048].tolist())
                doc_num += 1
            with open(f"/netdisk/hekaiyu/dataset/Olmo/1B/multickpt/ckpt{ckpt}-multi-num.npy", "r")as f:
                num = json.load(f)
                for lang, value in num_of_multilingual.items():
                    num_of_multilingual[lang] += num[lang]

                num_of_multilingual['num'] = doc_num
                print(num_of_multilingual)
                save_dir = f"/netdisk/hekaiyu/dataset/Olmo/1B/multi{j}-{concat_num}"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                with open(f"{save_dir}/multi-{end_ckpt-start_ckpt}-num.json", "w")as f:
                    f.write(json.dumps(num_of_multilingual, indent=4))
                raise
        englishdata = []
        print("get english data")
        for part in range(5):
            data = np.memmap(f"/netdisk/hekaiyu/dataset/Olmo/1B/multilingual/ckpt605-en-000{part}.npy", dtype='uint16', mode='r')
            for i in tqdm.tqdm(range(int(len(data) / 2048))):
                englishdata.append(data[i * 2048: (i + 1) * 2048].tolist())
        random.shuffle(englishdata)
        englishdata = englishdata[: 2048 * 1000 - len(multidata)]
        for doc in tqdm.tqdm(englishdata, total=len(englishdata)):
            multidata.append(doc)
        random.shuffle(multidata)
        print(len(multidata))
        for i in range(5):
            write(multidata[i*2048*200: (i+1)*2048*200], f"{save_dir}/multi-{end_ckpt-start_ckpt}-000{i}.npy")
if __name__ == "__main__":

    # download_ckpt(605)

    # creat_indice(605, 610)
    # wget -c -P /netdisk/hekaiyu/cross_lingual/train/1B/1 https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy

    train_config_path = "/home/hekaiyu/cross-lingual/OLMo-0.5.1/configs/official/OLMo-1B.yaml"
    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    print(dataset[208868312])
    batch_size = cfg.global_train_batch_size
    download_dataset()
    # creat_multilingual_dataset()