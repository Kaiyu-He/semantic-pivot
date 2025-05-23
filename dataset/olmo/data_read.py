import json
from pathlib import Path

import numpy as np
import tqdm

data = np.memmap(f"/netdisk/hekaiyu/cross_lingual/train/1B/1/global_indices.npy", dtype='uint32', mode='r')
print(len(data)/2048)
print(data[:2048])
print(data[2048:4096])
print(data[4096:6144])
data = np.memmap(f"/netdisk/hekaiyu/dataset/Olmo/1B/multilingual/multi-0000.npy", dtype='uint16', mode='r')
print(len(data)/2048)
print(data[:2048])
print(data[2048:4096])
for i in tqdm.tqdm(range(int(len(data)/2048))):
    if data[i*2048] == 0:
        print(i)
print(data[4096:6144])
for ckpt in range(605, 610):
    data = np.memmap(f"/netdisk/hekaiyu/dataset/Olmo/1B/multickpt/ckpt{ckpt}-multi.npy", dtype='uint16', mode='r')
    print(len(data)/2048)
    print(data[:2048])
    print(data[2048:4096])