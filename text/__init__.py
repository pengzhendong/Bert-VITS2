from text.symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, languages):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tones = [tone + language_tone_start_map[language] for tone, language in zip(tones, languages)]
    lang_ids = [language_id_map[language] for language in languages]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, device, style_text=None, style_weight=0.7):
    from .chinese_bert import get_bert_feature as zh_bert

    return zh_bert(norm_text, word2ph, device, style_text, style_weight)
