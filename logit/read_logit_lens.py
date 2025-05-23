import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
import math
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
import torch.nn as nn
import torch
import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def decode(tokenizer, layer_output, top_k=5):
    layer_result = []
    for token in layer_output:
        softmax_output = F.softmax(torch.tensor(token), dim=0).tolist()
        top_k_token = heapq.nlargest(top_k, enumerate(softmax_output), key=lambda x: x[1])
        layer_result.append([])
        for token in top_k_token:
            layer_result[-1].append((tokenizer.decode([token[0]]), token[0], token[1]))
    return layer_result


def get_P(layer_output, token_idx):
    layer_result = []
    for token in layer_output:
        softmax_output = F.softmax(torch.tensor(token), dim=0).tolist()
        layer_result.append(softmax_output[token_idx])
    return layer_result


def read(idx, top_k=5):
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_path = "/netdisk/hekaiyu/cross_lingual/dataset/datasets_all.jsonl"
    dataset = load_dataset("json", data_files=data_path)['train']
    print(f"English: {dataset[idx]['en']['en']} // Chinese: {dataset[idx]['zh']['zh']}")
    save_dir = "/netdisk/hekaiyu/cross_lingual/logit_lens/result/en-zh"
    save_path = f"{save_dir}/{idx}.jsonl"
    with open(save_path, "r") as f:
        result = json.load(f)
    message = []
    for layer_output in result["layer_out"]:
        message.append(decode(tokenizer, layer_output, top_k))
    return message, result["labels"], result["question"]


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def all_token(message):
    tokens = []
    with open("/home/hekaiyu/cross-lingual/word/decode_tokens.json", "r") as f:
        decode_token = json.load(f)
    for i in range(len(message)):
        for j, cell in enumerate(message[i]):
            for single in cell:
                token = single[1]
                if single[2] <= 0.01:
                    continue
                try:
                    print(f"{i}-{j}//{token}//{decode_token[f'{token}']['str']}\n{decode_token[f'{token}']['doc'][0]}")
                except:
                    print(f"{i}-{j}//{token}")


def draw_logit_lens_plt(message, rank=1):
    data = {
        'X': [],  # 行标签
        'Y': [],  # 列标签
        'Probability': [[] for _ in range(len(message))],
        'text': [[] for _ in range(len(message))]
    }
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open("/home/hekaiyu/cross-lingual/word/decode_tokens.json", "r") as f:
        decode_token = json.load(f)
    for token in labels:
        data['Y'].append(f"{token}//{tokenizer.decode([token])}")
    for i in range(len(message)):
        data['X'].append(f"{len(message) - i - 1}")
        for j, cell in enumerate(message[i]):
            data['Probability'][len(message) - i - 1].append(cell[rank - 1][2])
            try:
                token = cell[rank - 1][1]
                text = decode_token[f"{token}"]['doc'][0][1].split("<|token|>")
                text1 = decode_token[f"{token}"]['doc'][0][0][len(text[0]) - 2:-(len(text[1]) - 2)]
                text2 = decode_token[f"{token}"]['doc'][0][1][len(text[0]) - 2:-(len(text[1]) - 2)].replace("<|token|>",
                                                                                                            "<token>")
            except:
                text1 = ""
                text2 = ""
            data['text'][len(message) - i - 1].append(
                f"{cell[rank - 1][0]}//{text1}//{round(cell[rank - 1][2], 3)}//{token}")

    df = pd.DataFrame(data['Probability'], columns=data['Y'], index=data['X'])
    plt.figure(figsize=(40, 20))

    colors = ["#FFFFFF", "#ADD8E6"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue_white", colors)

    heatmap = plt.imshow(df, cmap=cmap, aspect='auto')

    plt.colorbar(heatmap, label='Probability')
    plt.xticks(np.arange(len(data['Y'])), data['Y'], rotation=45, ha='right')
    plt.yticks(np.arange(len(data['X'])), data['X'])

    for i in range(len(data['X'])):
        for j in range(len(data['Y'])):
            text_color = 'black' if df.iloc[i, j] < 50 else 'white'
            plt.text(
                j, i, data['text'][i][j],
                ha='center', va='center',
                color=text_color, fontsize=8
            )

    # 添加标题和标签
    plt.title("Logit Lens Heatmap")
    plt.xlabel("Input Tokens")
    plt.ylabel("Layers")
    plt.title(f"rank:{rank}", fontsize=10, pad=20)
    # 显示热图
    # plt.tight_layout()
    # plt.savefig(f"/home/hekaiyu/cross-lingual/word/logit_lens/image/{top_k}.png", bbox_inches='tight', dpi=300)  # 保存为文件

    plt.show()
    return data


def table(message, rank=1):
    data = {
        'X': [],  # 行标签
        'Y': [],  # 列标签
        'Probability': [[] for _ in range(len(message))],
        'text': [[] for _ in range(len(message))]
    }
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open("/home/hekaiyu/cross-lingual/word/decode_tokens.json", "r") as f:
        decode_token = json.load(f)
    for token in labels:
        data['Y'].append(f"{token}//{tokenizer.decode([token])}")
    for i in range(len(message)):
        data['X'].append(f"{len(message) - i - 1}")
        for j, cell in enumerate(message[i]):
            data['Probability'][len(message) - i - 1].append(cell[rank - 1][2])
            try:
                token = cell[rank - 1][1]
                text = decode_token[f"{token}"]['doc'][0][1].split("<|token|>")
                text1 = decode_token[f"{token}"]['doc'][0][0][len(text[0]) - 2:-(len(text[1]) - 2)]
                text2 = decode_token[f"{token}"]['doc'][0][1][len(text[0]) - 2:-(len(text[1]) - 2)].replace("<|token|>",
                                                                                                            "<token>")
            except:
                text1 = ""
                text2 = ""
            data['text'][len(message) - i - 1].append(
                f"{cell[rank - 1][0]}//{text1}//{round(cell[rank - 1][2], 3)}//{token}")
    X = data['X']
    Y = data['Y']
    text_matrix = data['text']
    Y = [str(y).replace("\n", "\\n") for y in Y]
    # 表头（第一行是 Y 的值）
    header_row = "|   | " + " | ".join(map(str, Y)) + " |"
    separator_row = "| " + " | ".join(["---"] * (len(Y) + 1)) + " |"

    # 数据行（每行以 X 的值开头，后接 text 矩阵的行）
    data_rows = []
    for i in range(len(X)):
        row = [str(X[i])] + [cell.replace("\n", "\\n") for cell in text_matrix[i]]
        data_row = "| " + " | ".join(row) + " |"
        data_rows.append(data_row)

    # 合并表头、分隔符和数据行
    markdown_table = "\n".join([header_row, separator_row] + data_rows)
    with open(f"/home/hekaiyu/cross-lingual/word/logit_lens/image/{rank}.md", "w") as f:
        f.write(markdown_table)
    return markdown_table


def token_P_plt(idx, tokens):
    def P(layer_output, tokens):
        layer_result = []
        for token in layer_output:
            sum_P = 0
            softmax_output = F.softmax(torch.tensor(token), dim=0).tolist()
            for i in tokens:
                sum_P += softmax_output[i]
            layer_result.append(sum_P)
        return layer_result

    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_path = "/netdisk/hekaiyu/cross_lingual/dataset/datasets_all.jsonl"
    dataset = load_dataset("json", data_files=data_path)['train']
    print(f"English: {dataset[idx]['en']['en']} // Chinese: {dataset[idx]['zh']['zh']}")
    save_dir = "/netdisk/hekaiyu/cross_lingual/logit_lens/result_only_answer/en-zh"
    save_path = f"{save_dir}/{idx}.jsonl"
    with open(save_path, "r") as f:
        result = json.load(f)
    print(result['labels'])
    message = []
    for layer_output in result["layer_out"]:
        message.append(P(layer_output, tokens))
    x = [i for i in range(len(message))]
    for i in range(len(message[0])):
        y = []
        for j in range(len(message)):
            y.append(message[j][i])
        plt.plot(x, y, linestyle="-", label=f"lie {i}")
    plt.xlabel("layer")
    plt.ylabel("P")
    plt.legend()
    plt.show()


def check(lang1='en', lang2='zh'):
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_path = "/netdisk/hekaiyu/cross_lingual/dataset/datasets_all.jsonl"
    dataset = load_dataset("json", data_files=data_path)['train']

    # count_path = "/home/hekaiyu/cross-lingual/word/Central/Central_count/count.jsonl"
    # count = load_dataset("json", data_files=count_path)['train']
    def process(idx):
        word1 = f" {dataset[idx][lang1][lang1]}"
        word2 = f" {dataset[idx][lang2][lang2]}"
        token1 = tokenizer(word1)['input_ids']
        token2 = tokenizer(word2)['input_ids']
        save_dir = f"/netdisk/hekaiyu/cross_lingual/logit_lens/result/{lang1}-{lang2}"
        save_path = f"{save_dir}/{idx}.jsonl"
        with open(save_path, "r") as f:
            result = json.load(f)
        p = {
            "vis": 0,
            'p1': 0,
            'p2': 0,
            'Central': 0,
            'layer_out': []
        }
        max_p = 0
        for layer_output in result["layer_out"]:
            output = decode(tokenizer, layer_output, 1)
            lie = 0
            for i in reversed(range(len(result['labels']))):
                if result['labels'][i] == 27:
                    output = output[i:]
                    lie = i
                    break
            if result['labels'][lie + 1] == 209:
                output = output[1]
                lie += 1
            else:
                output = output[0]
            p['layer_out'].append(output)
            if max_p < output[0][2] and output[0][1] not in token2:
                p['Central'] = output[0][1]
                max_p = output[0][2]
            output = output[0][1]

            for token in token1:
                P = get_P(layer_output, token)[lie]
                p['p1'] = max(p['p1'], P)
            for token in token2:
                P = get_P(layer_output, token)[lie]
                p['p2'] = max(p['p2'], P)
            if output in token1:
                p['vis'] = 1
        return p

    num = 0
    with open(f"/home/hekaiyu/cross-lingual/word/logit_lens/result/{lang1}-{lang2}.jsonl", "r") as f:
        for line in f:
            num += 1
    for i in tqdm.tqdm(range(len(dataset)), total=len(dataset)):
        if i < num:
            continue
        vis = process(i)
        result = {
            "word1": dataset[i]['en']['en'],
            "word2": dataset[i]['zh']['zh'],
            "logit": vis
        }
        with open(f"/home/hekaiyu/cross-lingual/word/logit_lens/result/{lang1}-{lang2}.jsonl", "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def draw_point2(x, y, point_type, messages):
    import plotly.graph_objs as go
    from scipy.stats import linregress
    import plotly.utils
    x = np.array(x)
    y = np.array(y)

    # 计算拟合直线
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    fit_line = slope * x + intercept
    clr = []
    sym = []
    for t in point_type:
        if t == 0:
            clr.append('rgba(255, 182, 193, .9)')  # 类型 1 的点为浅粉色
            sym.append('circle')  # 类型 1 的点为圆形
        elif t == 1:
            clr.append('rgba(255, 0, 0, .9)')  # 类型 3 的点为橙色
            sym.append('triangle-up')  # 类型 3 的点为三角形
        else:
            clr.append('rgba(0, 0, 255, .9)')  # 类型 2 的点为蓝色
            sym.append('square')  # 类型 2 的点为方形

    # 创建散点图
    trace_scatter = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=5,
            color=clr,
            symbol=sym,
            line=dict(
                width=0.5,
            )
        ),
        name='Data Points',
        hovertemplate=
        '<b>Word1:</b> %{customdata[0]}<br>' +
        '<b>Word2:</b> %{customdata[1]}<br>' +
        '<b>Word1_count:</b> %{customdata[8]}<br>' +
        '<b>Word2_count:</b> %{customdata[9]}<br>' +
        '<b>p1:</b> %{customdata[2]}<br>' +
        '<b>Central:</b> %{customdata[3]}<br>' +
        '<b>Central1:</b> %{customdata[4]}<br>' +
        '<b>Central2:</b> %{customdata[5]}<br>' +
        '<b>ratio1:</b> %{customdata[6]}<br>' +
        '<b>ratio2:</b> %{customdata[7]}<br>' +
        '<extra></extra>',
        customdata=np.stack(([msg['word1'] for msg in messages],
                             [msg['word2'] for msg in messages],
                             [msg['p1'] for msg in messages],
                             [msg['Central'] for msg in messages],
                             [msg['Central1'] for msg in messages],
                             [msg['Central2'] for msg in messages],
                             [msg['ratio1'] for msg in messages],
                             [msg['ratio2'] for msg in messages],
                             [msg['word1_count'] for msg in messages],
                             [msg['word2_count'] for msg in messages],
                             ), axis=-1)
    )

    # 创建拟合直线
    trace_line = go.Scatter(
        x=x,
        y=fit_line,
        mode='lines',
        line=dict(
            color='rgba(255, 0, 0, .9)',
            width=2
        ),
        name='Fit Line'
    )

    # 创建图表布局
    layout = go.Layout(
        title='',
        xaxis=dict(title=''),
        yaxis=dict(title=''),
        hovermode='closest',
    )

    # 创建图表
    fig = go.Figure(data=[trace_scatter, trace_line], layout=layout)

    # 将图表转换为JSON格式
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # 计算相关性
    correlation = r_value
    fig.show()
    print("相关性系数: ", correlation)
    print('直线方程: ', f'y = {slope:.2f}x + {intercept:.2f}')
def get_max(layer_out):
    softmax_output = F.softmax(torch.tensor(layer_out), dim=0).tolist()
    index, value = max(enumerate(softmax_output), key=lambda x: x[1])
    return index, value


def thought_type(args):
    idx, dataset, lang1, lang2, read_dir, tokenizer = args

    word1 = f" {dataset[idx][lang1][lang1]}"
    word2 = f" {dataset[idx][lang2][lang2]}"
    token1 = tokenizer(word1)['input_ids']
    token2 = tokenizer(word2)['input_ids']
    save_dir = f"{read_dir}/{lang1}-{lang2}"
    save_path = f"{save_dir}/{idx}.jsonl"
    with open(save_path, "r", encoding='utf-8') as f:
        logit = json.load(f)

    for layer, layer_out in logit["layer_out"].items():
        if logit['labels'][1] == 209:
            layer_out = layer_out[1]
        else:
            layer_out = layer_out[0]

        index, value = get_max(layer_out)

        if value >= 0.1 and index not in token1 and index not in token2:
            return idx, 1
    return idx, 0


def merge_message(data_path, score_dir, count_path, word_set_path, total_save_dir, lang1, lang2):
    print(f"{lang1}-{lang2}")
    path = f"{score_dir}/{lang1}-{lang2}/OLMo-7B-0424-hf"
    loss = []
    loss.append([])
    result_path = path + f"/step400000-shot-5.jsonl"
    dataset = load_dataset("json", data_files=data_path)['train']
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # with open(f"{count_path}/{lang1}-{lang2}.json", "r") as f:
    #     count = json.load(f)

    def ratio(idx):
        word1 = f" {dataset[idx][lang1][lang1]}"
        word2 = f" {dataset[idx][lang2][lang2]}"
        token1 = tokenizer(word1)['input_ids']
        token2 = tokenizer(word2)['input_ids']
        count = []
        with open(f"{word_set_path}/n_gram_1/{lang1}/{idx}.json", "r") as f:
            token_set = json.load(f)
            count.append(token_set['tokens_set'][f"[{token1[0]}]"])
        with open(f"{word_set_path}/n_gram_1/{lang2}/{idx}.json", "r") as f:
            token_set = json.load(f)
            if f"[{token2[0]}]" in token_set['tokens_set']:
                count.append(token_set['tokens_set'][f"[{token2[0]}]"])
            else:
                count.append(0)
        return count

    def process(choice_loss, ans):
        sum = 0
        score = 0
        for i, (label, label_loss) in enumerate(choice_loss.items()):
            label_loss = math.exp(-label_loss)
            if ans == label:
                score = label_loss
            else:
                sum += label_loss / 9
        return score - sum, sum

    with open(f"{total_save_dir}/{lang1}-{lang2}.jsonl", "w"):
        pass

    with open(result_path, "r", encoding="utf-8") as file:
        for i, line in tqdm.tqdm(enumerate(file), total=len(dataset)):
            line = json.loads(line)
            p_processed, p_others = process(line["loss"], line["word1"])
            if line['word1'] != dataset[i][lang1][lang1]:
                raise
            message = {
                "word1": line["word1"],
                "word2": line["word2"],
                # "word1_count": count[line["word1"]],
                # "word2_count": count[line["word2"]],
                # "word1_AND_word2_count": count[f"{line['word1']} AND {line['word2']}"],
                "p_processed": p_processed,
                "p_others": p_others,
                # "ratio": ratio(i)
                # "thought_type": thought_type(i)
            }
            with open(f"{total_save_dir}/{lang1}-{lang2}.jsonl", "a") as f:
                f.write(json.dumps(message, ensure_ascii=False) + '\n')


def probability_distribution(Central_dir, logit_dir, logit_distribution_path, lang1, lang2):
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def token_P_plt_output(idx, tokens):
        def P(layer_output, tokens):
            sum_P = 0
            softmax_output = F.softmax(torch.tensor(layer_output), dim=0).tolist()
            for i in tokens:
                sum_P += softmax_output[i]
            return sum_P

        save_path = f"{logit_dir}/{lang1}-{lang2}/{idx}.jsonl"
        with open(save_path, "r") as f:
            result = json.load(f)
        P_tokens = []
        lie = 0
        for i in reversed(range(len(result['labels']))):
            if result['labels'][i] == 27:
                lie = i
                break
        if result['labels'][lie + 1] == 209:
            lie += 1
        for i, layer_output in result["layer_out"].items():
            P_tokens.append(P(layer_output[lie], tokens))
        return P_tokens
    def process(i, line):
        Central_list = []
        word1 = f" {line['word1']}"
        word2 = f" {line['word2']}"
        token1 = tokenizer(word1)['input_ids']
        token2 = tokenizer(word2)['input_ids']
        vis = 0
        for token in token1:
            if token in token2:
                vis = 1
        if vis:
            return None, None, None
        for Central_token in line['Central-processed']:
            token = int(Central_token.split('[')[-1].split(']')[0])
            if token in token1 or token in token2:
                continue
            Central_list.append(token)
        y1 = token_P_plt_output(i, token1)
        y2 = token_P_plt_output(i, Central_list)
        y3 = token_P_plt_output(i, token2)
        return y1, y2, y3
    save_path = f"{logit_distribution_path}/{lang1}-{lang2}.jsonl"
    try:
        save = load_dataset("json", data_files=save_path)['train']
        processed = save[-1]['idx']
    except:
        processed = 0
    with open(f"{Central_dir}/{lang1}-{lang2}.jsonl", "r") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            if i < processed:
                continue
            line = json.loads(line)
            y_source, y_Central, y_target = process(i, line)
            if y_source == None:
                continue
            with open(save_path, "a") as f:
                f.write(json.dumps({
                    "idx": i,
                    "source": y_source,
                    "central": y_Central,
                    "target": y_target
                }, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--language_list",
        type=str,
        default="['en', 'fr', 'zh', 'ja']",
        help=""
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/hekaiyu/cross-lingual/word/dictionary/dataset.jsonl",
        help=""
    )
    parser.add_argument(
        "--thought_dir",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/logit_lens/thought_type",
        help=""
    )
    parser.add_argument(
        "--logit_dir",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/logit_lens/logit",
        help=""
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/result/result",
        help=""
    )
    parser.add_argument(
        "--count_path",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/word/word_count",
        help=""
    )
    parser.add_argument(
        "--word_set_path",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/word/word_set",
        help=""
    )
    parser.add_argument(
        "--total_save_dir",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/merge_result",
        help=""
    )
    parser.add_argument(
        "--Central_dir",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/word/Central/find",
        help=""
    )
    parser.add_argument(
        "--logit_distribution_dir",
        type=str,
        default="/netdisk/hekaiyu/cross_lingual/logit_lens/distribution",
        help=""
    )
    args = parser.parse_known_args()[0]
    import ast
    lang_list = ast.literal_eval(args.language_list)
    data_path = args.data_path
    logit_dir = args.logit_dir
    for lang1 in lang_list:
        for lang2 in lang_list:
            if lang1 == lang2:
                continue
            # merge_message(data_path, args.score_dir, args.count_path, args.word_set_path,
            #               args.total_save_dir, lang1, lang2)
            probability_distribution(args.Central_dir, args.logit_dir, args.logit_distribution_dir, lang1, lang2)