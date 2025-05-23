import json
import os.path

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from scipy.stats import pearsonr
from transformers import AutoTokenizer


def line(x, y, colors, lang1, lang2):
    plt.figure(figsize=(8, 6))  # 调整图像尺寸和分辨率
    c = []
    colors_select = ['#2ca02c', '#1f77b4', '#FF0000']  # 绿 蓝 红
    labels = ['none', 'pivot', 'direct']  # 根据你的需求修改标签

    # 创建颜色列表
    for label in colors:
        c.append(colors_select[label])

    # 计算拟合直线
    coefficients = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coefficients)
    y_fit = poly(x)

    # 计算皮尔逊相关系数
    corr_coef, p_value = pearsonr(x, y)

    # 创建散点图，并为每个点指定标签
    scatter = plt.scatter(x, y, c=c, s=3, label=[labels[label] for label in colors])

    # 绘制拟合直线
    plt.plot(x, y_fit)

    # 添加标题、标签和注释
    plt.title(f'{lang1}-{lang2}')
    plt.xlabel('co-occurrence count')
    plt.ylabel('probability')
    plt.annotate(f'r = {corr_coef:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12)

    # 创建图例
    # 每种颜色对应一个标签，确保每行包含一种颜色
    handles = []
    for i, color in enumerate(colors_select):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=labels[i])
        handles.append(handle)

    # 放置图例
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 0.8), loc='upper left', ncol=1, fontsize=12)
    plt.subplots_adjust(right=0.8)  # 为图例留出右侧空间

    # 保存图像
    save_dir = "/netdisk/hekaiyu/cross_lingual/image/co-occurrence"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 保存图像时扩展边界
    plt.savefig(f"{save_dir}/{lang1}-{lang2}.png", bbox_inches='tight', pad_inches=0.5)
    plt.show()


def get_xy_merge_result(lang1, lang2):
    model_path = "/netdisk/hekaiyu/model/OLMo-7B-0424-hf/step400000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"{lang1}-{lang2}")
    x, y, labels = [], [], []
    logit_top_dir = f"/netdisk/hekaiyu/cross_lingual/logit_lens/top1/{lang1}-{lang2}"
    # count_path = f"/netdisk/hekaiyu/cross_lingual/word/word_count/{lang1}-{lang2}.json"
    count_path = f"/netdisk/hekaiyu/cross_lingual/word/wimbd/word_count/{lang1}-{lang2}.json"
    with open(count_path, "r") as f:
        count_sum = json.load(f)

    def get_thought_type(idx):
        with open(logit_top_dir + f"/{idx}.json", "r") as f:
            logit = json.load(f)
        head_layer = 31
        rear_layer = 31
        # print(logit['token1'])
        # print(logit['token2'])
        # print(logit['word1'])
        # print(logit['word2'])

        for token1 in logit['token1']:
            for token2 in logit['token2']:
                if token1 == 209:
                    continue
                if token1 == token2:
                    return 0

        top1 = [0, 0]
        top_layer = 0
        for layer in range(32):
            if logit['top'][f"{layer}"][1] >= top1[1]:
                top1 = logit['top'][f"{layer}"]
                top_layer = layer
        # if top1[1] <= 0.1:
        #     return 0
        if logit['top']["31"][0] == top1[0]:
            return 2
        if top1[0] not in logit['token1']:
            # print(top1)
            # print(f"-{tokenizer.decode([top1[0]])}--")
            return 1
        return 2
        # while logit['top'][f"{rear_layer}"][0] in logit['token2']:
        #     rear_layer -= 1
        # while head_layer and logit['top'][f"{head_layer}"][0] not in logit['token1']:
        #     head_layer -= 1
        # print(head_layer, rear_layer)
        # for layer in range(head_layer + 1, rear_layer + 1):
        #     if logit['top'][f"{layer}"][1] >= 0.3:
        #         print(logit['top'][f"{layer}"])
        #         print(f"-{tokenizer.decode([logit['top'][f'{layer}'][0]])}--")
        #         return 1
        # return 2

    with open(f"/netdisk/hekaiyu/cross_lingual/word/wimbd/num/{lang1}.json", "r") as count_file:
        word1_count = json.load(count_file)
    with open(f"/netdisk/hekaiyu/cross_lingual/word/wimbd/num/{lang2}.json", "r") as count_file:
        word2_count = json.load(count_file)
    with open(f"/netdisk/hekaiyu/cross_lingual/merge_result/{lang1}-{lang2}.jsonl", "r") as f:

        for i, line in tqdm.tqdm(enumerate(f), total=2000):
            line = json.loads(line)
            if line['word1'] == line['word2']:
                continue
            # num = count_sum[f"{line['word1']} AND {line['word2']}"]
            # num = count_sum[f"{line['word1']}-{line['word2']}"]
            # num = word1_count[line['word1']]
            num = word2_count[line['word2']]
            if num == 0:
                count = 0
            else:
                count = math.log(num)
            x.append(count)
            y.append(line["p_processed"])
            labels.append(get_thought_type(i))
    return x, y, labels


def concate(dir_path="/netdisk/hekaiyu/cross_lingual/image/co-occurrence"):
    from PIL import Image, ImageDraw, ImageFont

    language_list = ['en', 'fr', 'zh', 'ja']
    size = len(language_list)
    path = f"{dir_path}/{language_list[0]}-{language_list[1]}.png"
    width, height = Image.open(path).size
    texts = language_list

    text_height = 40
    text_width = 40

    big_image = Image.new('RGB', (size * width + text_width + 10, size * height + text_height), color=(255, 255, 255))

    draw = ImageDraw.Draw(big_image)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    for row in range(len(language_list)):
        for col in range(len(language_list)):
            if row != col:
                try:
                    path = f"{dir_path}/{language_list[row]}-{language_list[col]}.png"
                    x = col * width + text_width
                    y = row * height
                    big_image.paste(Image.open(path), (x, y))
                except:
                    continue

    for col in range(len(language_list)):
        text = texts[col]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width_current = bbox[2] - bbox[0]
        text_height_current = bbox[3] - bbox[1]
        text_position = (
            text_width + col * width + (width - text_width_current) // 2,
            size * height + (text_height - text_height_current) // 2
        )
        draw.text(text_position, text, fill="black", font=font)

    for row in range(len(language_list)):
        text = texts[row]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width_current = bbox[2] - bbox[0]
        text_height_current = bbox[3] - bbox[1]
        text_position = (
            (text_width - text_width_current) // 2,
            text_height + row * height + (height - text_height_current) // 2
        )
        draw.text(text_position, text, fill="black", font=font)
    big_image.save(f"{dir_path}/concat.png")
    big_image.show()


def auc(x, labels, lang1, lang2):
    concat = []
    label = []
    for i, j in zip(x, labels):
        if j == 0:
            continue
        elif j == 1:
            concat.append((i, 0))
            label.append(0)
        else:
            concat.append((i, 1))
            label.append(1)
    sorted_concat = sorted(concat, key=lambda item: item[0], reverse=True)
    fn = sum(label)
    tn = len(label) - fn
    fp = 0
    tp = 0
    x_points = [0]
    y_points = [0]
    for item in sorted_concat:
        if item[1] == 1:
            tp += 1
            fn -= 1
        else:
            tn -= 1
            fp += 1
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        x_points.append(fpr)
        y_points.append(tpr)
    auc_value = 0
    for i in range(1, len(x_points)):
        auc_value += (x_points[i] - x_points[i - 1]) * (y_points[i] + y_points[i - 1]) / 2
    plt.figure()
    plt.plot(x_points, y_points, color='darkorange', lw=2, label=f'ROC (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC {lang1}-{lang2}')
    plt.legend(loc="lower right")
    save_dir = "/netdisk/hekaiyu/cross_lingual/image/roc"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # plt.savefig(f"{save_dir}/{lang1}-{lang2}.png", bbox_inches='tight')
    # plt.show()
    print(f"{lang1}-{lang2}: {auc_value:.4f}")
    return auc_value
def Central_contribution(lang1, lang2):
    co_occurrence = []
    p = []
    word1_count = []
    word2_count = []
    with open(f"/netdisk/hekaiyu/cross_lingual/merge_result/{lang1}-{lang2}.jsonl", "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            if line['word1'] == line['word2']:
                continue
            # if line['word1_AND_word2_count'] == 0:
            #     count = 0
            # else:
            #     count = math.log(line['word1_AND_word2_count'])
            count = math.log(line['word1_count'] * line['ratio'][1])
            word1_count.append(line['word1_count'])
            word2_count.append(line['word2_count'])
            co_occurrence.append(count)
            p.append(line["p_processed"])
    count = []
    path = f"/netdisk/hekaiyu/cross_lingual/word/Central/find/{lang1}-{lang2}.jsonl"
    if not os.path.exists(path):
        path = f"/netdisk/hekaiyu/cross_lingual/word/Central/find/{lang2}-{lang1}.jsonl"
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            if line['word1'] == line['word2']:
                continue
            count.append(0)
            for central in line['Central-processed']:
                if central.split('-')[1].split(':')[0] in line['Central']:
                    continue
                try:
                    set1 = int(central.split('(')[1].split(')')[0].split(',')[1])
                    set2 = int(central.split('(')[1].split(')')[0].split(',')[2])

                    count[-1] += math.log(min(set1 * word1_count[i], set2 * word2_count[i]))
                    # count[-1] = 0
                except:
                    continue
    lens = min(len(p), len(count), len(co_occurrence))
    co_occurrence = co_occurrence[:lens]
    count = count[:lens]
    p = p[:lens]
    return co_occurrence, count, p


def fit_line(lang1, lang2):
    x1, x2, p = Central_contribution(lang1, lang2)
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # 定义目标函数
    def target_func(X, k1, k2, c):
        x1, x2 = X
        return k1 * x1 + k2 * x2 + c

    X = np.column_stack((x1, x2))
    y = np.array(p)

    # 拟合模型
    params, _ = curve_fit(target_func, (X[:, 0], X[:, 1]), y)
    k1, k2, c = params

    # 预测值
    y_pred = target_func((X[:, 0], X[:, 1]), k1, k2, c)

    # 计算组合变量和相关系数
    x_combined = k1 * X[:, 0] + k2 * X[:, 1]
    corr_coef, _ = pearsonr(x_combined, y)
    print(f"Pearson correlation coefficient: {corr_coef:.4f}")

    # 可视化
    plt.scatter(x_combined, y, s=3, color='blue', label='Data')
    plt.plot(x_combined, y_pred, color='red', label=f'y = {k1:.2f}x1 + {k2:.2f}x2 + {c:.2f}')
    plt.xlabel('co_occurrence + k * central')
    plt.ylabel('p')
    plt.title('')
    plt.legend()
    save_dir = "/netdisk/hekaiyu/cross_lingual/image/relation"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(f"{save_dir}/{lang1}-{lang2}.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # concate()
    # concate("/netdisk/hekaiyu/cross_lingual/image/roc")
    lang_list = ['en', 'fr', 'zh', 'ja']
    auc_score = {}
    avg = []
    for lang1 in lang_list:
        for lang2 in lang_list:
            if lang1 not in auc_score:
                auc_score[lang1] = {}
            if lang1 == lang2:
                auc_score[lang1][lang2] = None
                continue
            # fit_line(lang1, lang2)
            x, y, labels = get_xy_merge_result(lang1, lang2)
            # line(x, y, labels, lang1, lang2)
            auc_score[lang1][lang2] = auc(x, labels, lang1, lang2)
            avg.append(auc_score[lang1][lang2])
    df = pd.DataFrame(auc_score)
    print(df)
    print(sum(avg)/len(avg))
    # concate()
    # concate("/netdisk/hekaiyu/cross_lingual/image/roc")
    #
