"""
版本管理、兼容推理及模型加载实现。
版本说明：
    1. 版本号与github的release版本号对应，使用哪个release版本训练的模型即对应其版本号
    2. 请在模型的config.json中显示声明版本号，添加一个字段"version" : "你的版本号"
特殊版本说明：
    1.1.1-fix： 1.1.1版本训练的模型，但是在推理时使用dev的日语修复
    2.3：当前版本
"""

import time

import torch
from vits2.utils import commons
from text import cleaned_text_to_sequence, get_bert

from typing import Union
from text.cleaner import clean_text
from vits2.utils import task

from vits2.models import SynthesizerTrn
from text.symbols import symbols


def get_net_g(model_path: str, device: str, hps):
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = task.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g


def get_text(text, device):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph, language = clean_text(text)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language)

    phone = commons.intersperse(phone, 0)
    tone = commons.intersperse(tone, 0)
    language = commons.intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, device)
    del word2ph
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, phone, tone, language


def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    hps,
    net_g,
    device,
    skip_start=False,
    skip_end=False,
):
    begin = time.time()
    bert, phones, tones, lang_ids = get_text(text, device)
    print(f"g2p and bert cost: {int((time.time() - begin) * 1000)}ms")
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        is_first = True
        for o in net_g.infer_streaming(
            x_tst,
            x_tst_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        ):
            if is_first:
                is_first = False
                print(f"first package cost: {int((time.time() - begin) * 1000)}ms")
            yield o

        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
