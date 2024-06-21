from vits2.utils.task import get_hparams_from_file, load_checkpoint
import json


def export_onnx(export_path, model_path, config_path, novq, dev, Extra):
    hps = get_hparams_from_file(config_path)
    version = hps.version[0:3]
    enable_emo = False
    BertPaths = ["chinese-roberta-wwm-ext-large"]
    from .V230 import SynthesizerTrn, symbols

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

    LangDict = {"ZH": [0, 0], "JP": [1, 6], "EN": [2, 8]}
    BertSize = 1024

    MoeVSConf = {
        "Folder": f"{export_path}",
        "Name": f"{export_path}",
        "Type": "BertVits",
        "Symbol": symbols,
        "Cleaner": "",
        "Rate": hps.data.sampling_rate,
        "CharaMix": True,
        "Characters": spklist,
        "LanguageMap": LangDict,
        "Dict": "BasicDict",
        "BertPath": BertPaths,
        "Clap": ("clap-htsat-fused" if enable_emo else False),
        "BertSize": BertSize,
    }

    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)
