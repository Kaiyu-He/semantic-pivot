import argparse
import torch
import json
import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import ast
import sacrebleu


def get_translation(model, tokenizer, prompt: str):
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    output = model.generate(**prompt_ids,
                            max_new_tokens=100)[0]
    output = tokenizer.decode(output[len(prompt_ids[0]):])
    return output.split('\n')[0]


def get_translation_batch(model, tokenizer, prompts: list):
    tokenizer.padding_side = "left"
    prompt_ids = tokenizer(prompts, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt').to('cuda')
    outputs = model.generate(**prompt_ids, max_new_tokens=150)
    translations = []
    for output in outputs:
        translations.append(tokenizer.decode(output[len(prompt_ids[0]):]).split('\n')[0].split('<|endoftext|>')[0])
    return translations


def prompt_text(lang1, lang2):
    prompt = "Translate the sentence from <lang1> to <lang2>.\n\n<example><lang1>: [X]\n<lang2>: "
    return prompt.replace("<lang1>", prompt_list[lang1]).replace("<lang2>", prompt_list[lang2])


def get_fewshot_prompt(lang1, sentence1, lang2, sentence2):
    return f"{prompt_list[lang1]}: {sentence1}\n{prompt_list[lang2]}: {sentence2}\n\n"


def flores(model, tokenizer, data_path, lang1, lang2, num_of_dataset):
    print(f"{lang1}-{lang2}")
    language = {
        "en": "sentence_eng_Latn",
        "fr": "sentence_fra_Latn",
        "zh": "sentence_zho_Hans",
        "ja": "sentence_jpn_Jpan"
    }
    dataset = load_dataset(data_path, "all", trust_remote_code=True)['dev']
    dataset = dataset.select(range(max(num_of_dataset[0], 0), min(num_of_dataset[1], len(dataset))))
    score = []
    batch = []
    batch_answer = []
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        prompt = prompt_text(lang1, lang2)
        prompt = prompt.replace('[X]', data[language[lang1]])
        prompt = prompt.replace('<example>',
                                get_fewshot_prompt(lang1, dataset[i - 1][language[lang1]],
                                                   lang2, dataset[i - 1][language[lang2]]))
        batch.append(prompt)
        batch_answer.append(data[language[lang2]])
        if len(batch) >= 8:
            translations = get_translation_batch(model, tokenizer, batch)
            for translation, answer in zip(translations, batch_answer):
                chrf = sacrebleu.sentence_chrf(translation, [answer])
                score.append(chrf.score)
            batch = []
            batch_answer = []
    translations = get_translation_batch(model, tokenizer, batch)
    for translation, answer in zip(translations, batch_answer):
        chrf = sacrebleu.sentence_chrf(translation, [answer])
        score.append(chrf.score)

    return sum(score) / len(score)


def main(
        model_path,
        lang_list,
        data_path,
        save_dir,
        name,
        num_of_shot=3,
        num_of_dataset=[0, 100],
        restart=False
):
    if type(num_of_dataset) == int:
        num_of_dataset = [0, num_of_dataset]
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    try:
        with open(f"{save_dir}/{name}.json", "r") as f:
            result = json.load(f)
    except:
        result = {}
    scores = []
    for lang1 in lang_list:
        for lang2 in lang_list:
            # if not (f"{lang1}-{lang2}" == "en-fr" or f"{lang1}-{lang2}" == "ja-zh" or f"{lang1}-{lang2}" == "ja-fr"):
            #     continue
            if lang1 == lang2:
                continue
            try:
                if result[lang1][lang2] > 0:
                    scores.append(result[lang1][lang2])
                    continue
            except:
                score = flores(model=model, tokenizer=tokenizer, data_path=data_path,
                               lang1=lang1, lang2=lang2,
                               num_of_shot=num_of_shot,
                               num_of_dataset=num_of_dataset)
                if lang1 not in result:
                    result[lang1] = {}
                result[lang1][lang2] = score
                scores.append(score)
                with open(f"{save_dir}/{name}.json", "w") as f:
                    f.write(json.dumps(result, indent=4))
    del model
    del tokenizer
    import gc
    import time
    time.sleep(0.5)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="OLMo-7B-0424-hf",
        help=""
    )
    parser.add_argument(
        "--num_of_shot",
        type=int,
        default=5,
        help=""
    )
    parser.add_argument(
        "--num_of_dataset",
        type=int,
        default=2000,
        help=""
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/flores",
        help=""
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./cross_lingual/result/flores",
        help=""
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/OLMo-7B-0424-hf/step400000",
        help=""
    )
    parser.add_argument(
        "--language_list",
        type=str,
        default="['en', 'zh', 'fr', 'ja']",
        help=""
    )
    args = parser.parse_known_args()[0]
    lang_list = ast.literal_eval(args.language_list)
    model_path = args.model_path
    prompt_list = {
        "en": "English",
        "zh": "Chinese",
        "ru": "Russian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese"
    }

    models = []
    # models.append((f"./model/OLMo-7B-0424-hf/step{step}000", f"{step}"))
    save_dir = args.save_dir
    models.append((args.model_path, args.name))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for model_path, name in models:
        print(model_path)
        print(name)
        main(model_path=model_path,
             data_path=args.data_path,
             name=name,
             save_dir=save_dir,
             lang_list=lang_list,
             num_of_shot=args.num_of_shot,
             num_of_dataset=args.num_of_dataset,
             restart=False
             )
