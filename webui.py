# flake8: noqa: E402
import os
import logging
import soxr
import re_matching
from tools.sentence import split_by_language

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, get_net_g
import gradio as gr
import webbrowser
import numpy as np
from config import config
import librosa

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def tts_streaming_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    speed_scale,
    sample_rate,
    skip_start=False,
    skip_end=False,
):
    with torch.no_grad():
        chunks = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=1 / speed_scale,
            sid=speaker,
            hps=hps,
            net_g=net_g,
            device=device,
            skip_start=skip_start,
            skip_end=skip_end,
        )
    if sample_rate != 44100:
        rs = soxr.ResampleStream(44100, sample_rate, 1, dtype=np.int16)
    for chunk in chunks:
        chunk = (chunk * 32767).astype(np.int16)
        if sample_rate != 44100:
            chunk = rs.resample_chunk(chunk)
        yield chunk


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    speed_scale,
    sample_rate,
):
    sample_rate = int(sample_rate)
    chunks = tts_streaming_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, speed_scale, sample_rate)
    audios = []
    for chunk in chunks:
        audios.append(chunk)
    audios = np.concatenate(audios)
    return ("Success", (sample_rate, audios))


if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    net_g = get_net_g(
        model_path=config.webui_config.model, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="输入文本内容",
                    placeholder="",
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="音色"
                )
                sample_rate = gr.Dropdown(
                    choices=["8000", "16000", "22050", "44100"], value="44100", label="音频采样率"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.7, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0, maximum=2, value=0.6, step=0.1, label="Noise"
                )
                noise_scale_w = gr.Slider(
                    minimum=0, maximum=2, value=0.7, step=0.1, label="Noise_W"
                )
                speed_scale = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="语速"
                )
                btn = gr.Button("生成音频！", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="状态信息")
                audio_output = gr.Audio(label="输出音频")
        btn.click(
            tts_fn,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                speed_scale,
                sample_rate,
            ],
            outputs=[text_output, audio_output],
        )

    text = "您好，你准备好了的话我们就开始吧。"
    tts_fn(text, speakers[0], 0.7, 0.6, 0.7, 1.0, 44100)
    print("推理页面已开启!")
    print(f"http://127.0.0.1:{config.webui_config.port}")
    app.launch(share=config.webui_config.share, server_name="0.0.0.0", server_port=config.webui_config.port)
