from text import chinese, cleaned_text_to_sequence


def clean_text(text):
    norm_text = chinese.text_normalize(text)
    phones, tones, word2ph, languages = chinese.g2p(norm_text)
    return norm_text, phones, tones, word2ph, languages


def clean_text_bert(text):
    norm_text, phones, tones, word2ph, languages = clean_text(text)
    bert = chinese.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text):
    norm_text, phones, tones, word2ph, languages = clean_text(text)
    return cleaned_text_to_sequence(phones, tones, languages)


if __name__ == "__main__":
    pass
