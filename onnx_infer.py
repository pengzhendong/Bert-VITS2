import logging
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from infer import get_text


formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path


class OnnxInferenceSession:
    def __init__(self, path, providers=["CPUExecutionProvider"]):
        self.enc = ort.InferenceSession(path["enc"], providers=providers)
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=providers)
        self.dp = ort.InferenceSession(path["dp"], providers=providers)
        self.sdp = ort.InferenceSession(path["sdp"], providers=providers)
        self.flow = ort.InferenceSession(path["flow"], providers=providers)
        self.dec = ort.InferenceSession(path["dec"], providers=providers)

    def __call__(
        self,
        seq,
        tone,
        language,
        bert,
        sid,
        seed=7723,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:
            seq = np.expand_dims(seq, 0)
        if tone.ndim == 1:
            tone = np.expand_dims(tone, 0)
        if language.ndim == 1:
            language = np.expand_dims(language, 0)
        assert seq.ndim == 2
        assert tone.ndim == 2
        assert language.ndim == 2

        begin = time.time()
        g = self.emb_g.run(None, {"sid": sid.astype(np.int64)})[0]
        logging.info("vits2 spk_emb cost: %sms", int((time.time() - begin) * 1000))
        g = np.expand_dims(g, -1)

        begin = time.time()
        enc_rtn = self.enc.run(
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert": bert.astype(np.float32),
                "g": g.astype(np.float32),
            },
        )
        logging.info("vits2 txt_enc cost: %sms", int((time.time() - begin) * 1000))

        np.random.seed(seed)
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale

        begin = time.time()
        dp_logw = self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[0]
        logging.info("vits2 dp cost: %sms", int((time.time() - begin) * 1000))

        begin = time.time()
        sdp_logw = self.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0]
        logging.info("vits2 sdp cost: %sms", int((time.time() - begin) * 1000))

        logw = sdp_ratio * sdp_logw + (1 - sdp_ratio) * dp_logw
        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )

        begin = time.time()
        z = self.flow.run(
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g,
            },
        )[0]
        logging.info("vits2 flow cost: %sms", int((time.time() - begin) * 1000))

        # begin = time.time()
        # o = self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]
        # logging.info("vits2 dec cost: %sms", int((time.time() - begin) * 1000))
        # yield o[0, 0]

        chunk_size = 40
        padding = 19
        upsample_rates = 512
        z_len = z.shape[-1]
        for start in range(0, z_len, chunk_size):
            end = start + chunk_size
            if start < padding:
                # 历史数据长度小于 padding，pad 上所有历史数据
                l_pad = start * upsample_rates
                start = 0
            else:
                l_pad = padding * upsample_rates
                start -= padding
            if end > z_len:
                r_pad = 0
                end = z_len
            elif end + padding > z_len:
                # 未来数据长度小于 padding，pad 上所有未来数据
                r_pad = (z_len - end) * upsample_rates
                end = z_len
            else:
                r_pad = padding * upsample_rates
                end += padding

            begin = time.time()
            o = self.dec.run(
                None, {"z_in": z[:, :, start:end].astype(np.float32), "g": g}
            )[0]
            if start == 0:
                logging.info("vits2 dec cost: %sms", int((time.time() - begin) * 1000))

            o = o[:, :, l_pad:-r_pad] if r_pad != 0 else o[:, :, l_pad:]
            o = o[0, 0]
            # clip heading sil
            if start == 0:
                # process add blank
                index = w_ceil[0, 0][0].item() + w_ceil[0, 0][1].item()
                index = int(index * upsample_rates * 0.5)
                o = o[index:]
            yield o


def tts(session, text, sid=0):
    begin = time.time()
    bert, phone, tone, language = get_text(text, "cuda")
    logging.info("bert cost: %sms", int((time.time() - begin) * 1000))
    bert = bert.numpy().T
    sid = np.array([sid])
    audios = []

    for chunk in session(phone.numpy(), tone.numpy(), language.numpy(), bert, sid):
        audios.append(chunk)
    audios = np.concatenate(audios)
    return audios


def main():
    session = OnnxInferenceSession(
        {
            "enc": "onnx/BertVits2.3PT/BertVits2.3PT_enc_p.onnx",
            "emb_g": "onnx/BertVits2.3PT/BertVits2.3PT_emb.onnx",
            "dp": "onnx/BertVits2.3PT/BertVits2.3PT_dp.onnx",
            "sdp": "onnx/BertVits2.3PT/BertVits2.3PT_sdp.onnx",
            "flow": "onnx/BertVits2.3PT/BertVits2.3PT_flow.onnx",
            "dec": "onnx/BertVits2.3PT/BertVits2.3PT_dec.onnx",
        },
        # providers=["CPUExecutionProvider"],
        # EXHAUSTIVE, HEURISTIC, DEFAULT
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})],
    )

    audio = tts(session, "一二三四") # 预热
    audio = tts(session, "也不知道这群牛娃进入社会后能不能change the world", 1)
    sf.write("test.wav", audio, 44100)


if __name__ == "__main__":
    main()
