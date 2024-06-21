import os
import re

from g2p_mix import G2pMix

from text.symbols import punctuation

try:
    from tn.chinese.normalizer import Normalizer

    normalizer = Normalizer().normalize
except ImportError:
    import cn2an

    print("tn.chinese.normalizer not found, use cn2an normalizer")
    normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

g2per = G2pMix()


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    else:
        tone = 3
    return phn.lower(), tone


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def g2p(text):
    phones, tones, word2ph, languages = _g2p(text)
    assert sum(word2ph) == len(phones)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    languages = ["ZH"] + languages + ["ZH"]
    return phones, tones, word2ph, languages


def _g2p(text):
    languages_list = []
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in g2per.g2p(text, sandhi=True):
        if seg["word"] == seg["phones"]:
            punct = seg["word"]
            assert punct in punctuation
            word2ph.append(1)
            phones_list.append(punct)
            tones_list.append(0)
            languages_list.append("ZH")
        elif seg["phones"][-1][0].islower():
            c, v = seg["phones"]
            raw_pinyin = c + v
            v_without_tone = v[:-1]
            tone = v[-1]

            pinyin = c + v_without_tone
            assert tone in "12345"
            if c:
                # 多音节
                v_rep_map = {
                    "uei": "ui",
                    "iou": "iu",
                    "uen": "un",
                }
                if v_without_tone in v_rep_map.keys():
                    pinyin = c + v_rep_map[v_without_tone]
            else:
                # 单音节
                pinyin_rep_map = {
                    "ing": "ying",
                    "i": "yi",
                    "in": "yin",
                    "u": "wu",
                }
                if pinyin in pinyin_rep_map.keys():
                    pinyin = pinyin_rep_map[pinyin]
                else:
                    single_rep_map = {
                        "v": "yu",
                        "e": "e",
                        "i": "y",
                        "u": "w",
                    }
                    if pinyin[0] in single_rep_map.keys():
                        pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
            assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
            phone = pinyin_to_symbol_map[pinyin].split(" ")
            word2ph.append(len(phone))
            phones_list += phone
            tones_list += [int(tone)] * len(phone)
            languages_list += ["ZH"] * len(phone)
        if seg["phones"][-1][0].isupper():
            word2ph.append(len(seg["phones"]))
            for phn in seg["phones"]:
                phn, tone = refine_ph(phn)
                phones_list.append(phn)
                tones_list.append(tone)
                languages_list.append("EN")

    return phones_list, tones_list, word2ph, languages_list


def text_normalize(text):
    text = normalizer(text)
    text = replace_punctuation(text)
    return text


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
