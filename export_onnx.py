import json
import os

from onnx_modules import SynthesizerTrn, symbols
from text.symbols import symbols
from vits2.utils.task import get_hparams_from_file, load_checkpoint


def export_onnx(export_path, model_path, config_path):
    hps = get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    _ = net_g.eval()
    _ = load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    net_g.cpu()
    net_g.export_onnx(export_path)

    spklist = []
    for key in hps.data.spk2id.keys():
        spklist.append(key)

    MoeVSConf = {
        "Folder": export_path,
        "Name": export_path,
        "Type": "BertVits",
        "Symbol": symbols,
        "Rate": hps.data.sampling_rate,
        "Characters": spklist,
        "BertPath": "chinese-roberta-wwm-ext-large",
        "BertSize": 1024,
    }

    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)


if __name__ == "__main__":
    export_path = "BertVits2.3PT"
    model_path = "Data/models/G_0.pth"
    config_path = "Data/config.json"
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    export_onnx(export_path, model_path, config_path)
