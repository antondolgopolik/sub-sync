import pymongo


def load_words():
    with open('wordlist-german.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words


mongodb_client = pymongo.MongoClient('mongodb://localhost:27017/')
word_db = mongodb_client['word_db']
words_coll = word_db['words']

print(words_coll.find_one({'lang': 'en', 'word': 'world'}) is not None)

