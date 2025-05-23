import os

import wandb

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

from huggingface_hub import login


def download(model_name, save_path, step, tokens):
    snapshot_download(repo_id=f"allenai/{model_name}",
                      revision=f"step{step}-tokens{tokens}B",
                      cache_dir=f'{save_path}',  # 缓存目录
                      local_dir=f'{save_path}',  # 目标目录
                      local_dir_use_symlinks=False,  # 使用软连接
                      resume_download=True, )  # 断点续传


def download_bloom(save_path):
    step_list1 = [1000, 10000, 50000, 100000, 150000, 200000, 250000, 300000]
    step_list2 = [1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000]
    model_size_list = ["7b1", "3b", "1b7", "1b1", "560m"]
    for model_size in model_size_list:
        if model_size == "1b1" or model_size == "560m":
            step_list = step_list2
        else:
            step_list = step_list1
        for step in step_list:
            try:
                snapshot_download(repo_id=f"bigscience/bloom-{model_size}-intermediate",
                                  revision=f"global_step{step}",
                                  cache_dir=f'{save_path}/bloom-{model_size}-intermediate/{step}',  # 缓存目录
                                  local_dir=f'{save_path}/bloom-{model_size}-intermediate/{step}',  # 目标目录
                                  local_dir_use_symlinks=False,  # 使用软连接
                                  resume_download=True, )  # 断点续传
            except:
                continue
        snapshot_download(repo_id=f"bigscience/bloom-{model_size}",
                          cache_dir=f'{save_path}/bloom-{model_size}-intermediate/last',  # 缓存目录
                          local_dir=f'{save_path}/bloom-{model_size}-intermediate/last',  # 目标目录
                          local_dir_use_symlinks=False,  # 使用软连接
                          resume_download=True, )  # 断点续传
        print(f"bigscience/bloom-{model_size}")

def download_olmo_7B(save_path):
    model_name = "OLMo-7B-0424-hf"
    step_list = [400, 380, 360, 340, 320, 300, 280, 260, 240, 220,
                 200, 180, 160, 140, 120, 100,  80,  60,  40,  20]
    token_size = [1677, ]
    for i in step_list:
        step = int(i * 1000)
        tokens = int(step * 2719 / 648650)
        try:
            print(f"step{step}-tokens{tokens}B")
            snapshot_download(repo_id=f"allenai/{model_name}",
                              revision=f"step{step}-tokens{tokens}B",
                              cache_dir=f'{save_path}/{model_name}/step{step}',  # 缓存目录
                              local_dir=f'{save_path}/{model_name}/step{step}',  # 目标目录
                              local_dir_use_symlinks=False,  # 使用软连接
                              resume_download=True)  # 断点续传
        except:
            try:
                tokens = int(step * 2719 / 648650) +1
                print(f"step{step}-tokens{tokens}B")
                snapshot_download(repo_id=f"allenai/{model_name}",
                                  revision=f"step{step}-tokens{tokens}B",
                                  cache_dir=f'{save_path}/{model_name}/step{step}',  # 缓存目录
                                  local_dir=f'{save_path}/{model_name}/step{step}',  # 目标目录
                                  local_dir_use_symlinks=False,  # 使用软连接
                                  resume_download=True)  # 断点续传
            except:
                try:
                    tokens = int(step * 2719 / 648650) - 1
                    print(f"step{step}-tokens{tokens}B")
                    snapshot_download(repo_id=f"allenai/{model_name}",
                                      revision=f"step{step}-tokens{tokens}B",
                                      cache_dir=f'{save_path}/{model_name}/step{step}',  # 缓存目录
                                      local_dir=f'{save_path}/{model_name}/step{step}',  # 目标目录
                                      local_dir_use_symlinks=False,  # 使用软连接
                                      resume_download=True)  # 断点续传
                except Exception as e:
                    print(f"Failed to download for step={step}, tokens={tokens:.2f}B: {e}")
def download_olmo_1B(save_path):
    model_name = "OLMo-1B-hf"
    get = []
    step_list = [605, 606,
                 738, 710, 680, 650, 630, 580, 555, 530, 505, 480,
                 455, 430, 405, 380, 355, 330, 110,  80,  50,  20]
    for i in step_list:
        step = int(i * 1000)
        tokens = int(step * 2536 / 605000)
        try:
            print(f"step{step}-tokens{tokens}B")
            snapshot_download(repo_id=f"allenai/{model_name}",
                              revision=f"step{step}-tokens{tokens}B",
                              cache_dir=f'{save_path}/{model_name}/step{step}',  # 缓存目录
                              local_dir=f'{save_path}/{model_name}/step{step}',  # 目标目录
                              local_dir_use_symlinks=False,  # 使用软连接
                              resume_download=True)  # 断点续传
        except:
            try:
                tokens = int(step * 2719 / 648650) + 1
                print(f"step{step}-tokens{tokens}B")
                snapshot_download(repo_id=f"allenai/{model_name}",
                                  revision=f"step{step}-tokens{tokens}B",
                                  cache_dir=f'{save_path}/{model_name}/step{step}',  # 缓存目录
                                  local_dir=f'{save_path}/{model_name}/step{step}',  # 目标目录
                                  local_dir_use_symlinks=False,  # 使用软连接
                                  resume_download=True)  # 断点续传
            except:
                try:
                    tokens = int(step * 2719 / 648650) - 1
                    print(f"step{step}-tokens{tokens}B")
                    snapshot_download(repo_id=f"allenai/{model_name}",
                                      revision=f"step{step}-tokens{tokens}B",
                                      cache_dir=f'{save_path}/{model_name}/step{step}',  # 缓存目录
                                      local_dir=f'{save_path}/{model_name}/step{step}',  # 目标目录
                                      local_dir_use_symlinks=False,  # 使用软连接
                                      resume_download=True)  # 断点续传
                except Exception as e:
                    print(f"Failed to download for step={step}, tokens={tokens:.2f}B: {e}")


# os.environ["WANDB_PROJECT"] = "download"
# os.environ["WANDB_API_KEY"] = "666bff72d04cfb09fb241f5152a186617627651b"
if __name__ == "__main__":
    # login("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")
    save_path = "/netdisk/hekaiyu/model"
    # download_bloom(save_path)
    # download_olmo(save_path)
    model_size = "1b7"
    step = 250000
    while 1:
        try:
            # print("start")
            snapshot_download(repo_id=f"bigscience/bloom-{model_size}-intermediate",
                              revision=f"global_step{step}",
                              cache_dir=f'{save_path}/bloom-{model_size}-intermediate/{step}',  # 缓存目录
                              local_dir=f'{save_path}/bloom-{model_size}-intermediate/{step}',  # 目标目录
                              local_dir_use_symlinks=False,  # 使用软连接
                              resume_download=True, )  # 断点续传
            # snapshot_download(repo_id=f"allenai/OLMo-1B-hf",
            #                   revision=f"step605000-tokens2536B",
            #                   cache_dir=f'{save_path}/OLMo-1B-hf/step605000',  # 缓存目录
            #                   local_dir=f'{save_path}/OLMo-1B-hf/step605000',  # 目标目录
            #                   local_dir_use_symlinks=False,  # 使用软连接
            #                   resume_download=True)
            # print("end")
            # snapshot_download(repo_id=f"Qwen/Qwen2.5-1.5B",
            #                   cache_dir=f'{save_path}/Qwen2.5-1.5B',  # 缓存目录
            #                   local_dir=f'{save_path}/Qwen2.5-1.5B',  # 目标目录
            #                   local_dir_use_symlinks=False,  # 使用软连接
            #                   resume_download=True)
            # snapshot_download(repo_id=f"Qwen/Qwen2.5-0.5B",
            #                   cache_dir=f'{save_path}/Qwen2.5-0.5B',  # 缓存目录
            #                   local_dir=f'{save_path}/Qwen2.5-0.5B',  # 目标目录
            #                   local_dir_use_symlinks=False,  # 使用软连接
            #                   resume_download=True)
            # download_olmo_7B(save_path)
            break
        except Exception as e:
            print(f"{e}")
