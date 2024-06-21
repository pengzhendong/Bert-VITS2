import sys

from g2p_mix import G2pMix
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import config

LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

models = dict()
g2per = G2pMix()


def find_bpes(words, bpes):
    res = []
    begin = 0
    for target in words:
        word = ''
        for end in range(begin, len(bpes)):
            word += bpes[end]
            if word == target or word == "[UNK]":
                res.append((target, bpes[begin:end + 1]))
                begin = end + 1
                break
    return res


def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        embs = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        g2p_toks = [item["word"].lower() for item in g2per.g2p(text)]
        g2p_toks = ["[CLS]"] + g2p_toks + ["[SEP]"]
        tokens = [token.replace("##", "") for token in inputs.tokens()]
        slices = [len(item[1]) for item in find_bpes(g2p_toks, tokens)]

        embs = torch.split(embs, slices, dim=0)
        embs = torch.cat([torch.sum(emb, dim=0, keepdim=True) for emb in embs], dim=0)
        embs = [(embs[i] / num).repeat(num, 1) for i, num in enumerate(word2ph)]
        return torch.cat(embs, dim=0).T


if __name__ == "__main__":
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
