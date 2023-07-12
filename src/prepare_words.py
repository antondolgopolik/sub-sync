import sys
import pymongo
import re
import spacy
import pysrt
from pysrt import SubRipFile
from pysrt import SubRipItem


def drop_formatting(s):
    s = re.sub('<\\s*b\\s*>', '', s)
    s = re.sub('<\\s*/\\s*b\\s*>', '', s)
    s = re.sub('<\\s*i\\s*>', '', s)
    s = re.sub('<\\s*/\\s*i\\s*>', '', s)
    s = re.sub('<\\s*u\\s*>', '', s)
    s = re.sub('<\\s*/\\s*u\\s*>', '', s)
    s = re.sub('<\\s*font\\s*color\\s*=\\s*"[^"]*"\\s*>', '', s)
    s = re.sub('<\\s*/\\s*font\\s*>', '', s)
    return s


def is_word(token):
    return words_coll.find_one({'lang': lang, 'word': token}) is not None


def is_allowed_word(token):
    return is_word(token) and token not in nlp.Defaults.stop_words


def process_sub_item(sub_item: SubRipItem, word_stat):
    txt = drop_formatting(sub_item.text)
    doc = nlp(txt)
    lemmatized_tokens = [token.lemma_ for token in doc]
    words = filter(is_allowed_word, lemmatized_tokens)
    for word in words:
        if word in word_stat:
            word_stat[word] += 1
        else:
            word_stat[word] = 1


def process_sub_items(subs: SubRipFile):
    word_stat = {}
    for sub_item in subs:
        process_sub_item(sub_item, word_stat)
    sorted_word_stat = sorted(word_stat.items(), key=lambda item: item[1], reverse=True)
    return sorted_word_stat[:1000]


def extract_words(path_to_subs):
    subs = pysrt.open(path_to_subs)
    words = process_sub_items(subs)
    for word in words:
        print(f'{word[0]} {word[1]}')


lang = sys.argv[1]
lang_to_spacy_model = {'en': 'en_core_web_lg', 'de': 'de_core_news_lg', 'ru': 'ru_core_news_lg'}

mongodb_client = pymongo.MongoClient('mongodb://localhost:27017/')
word_db = mongodb_client['word_db']
words_coll = word_db['words']

nlp = spacy.load(lang_to_spacy_model[lang])

extract_words(sys.argv[2])
