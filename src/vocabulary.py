import re
import codecs
import pickle

from default import *


class Vocabulary:
    unk_token = "#UNK#"
    eom_token = "#EOM#"
    go_token = "#GO#"

    def __init__(self):
        self.word2id = {}
        self.id2word = []

    def normalize(self, token):
        if token in [Vocabulary.unk_token, "<s>", "</s>", Vocabulary.eom_token, Vocabulary.go_token]:
            return token
        elif token == "-LCB-":
            token = "{"
        elif token == "-LRB-":
            token = "("
        elif token == "-LSB-":
            token = "["
        elif token == "-RCB-":
            token = "}"
        elif token == "-RRB-":
            token = ")"
        elif token == "-RSB-":
            token = "]"
        else:
            token = re.sub("[0-9]", "0", token)

        return token

    def load_from_file(self, path):
        self.word2id = {}
        self.id2word = []

        f = codecs.open(path, "r",encoding='utf-8', errors='ignore')
        for line in f:
            token = self.normalize(line.strip())
            self.id2word.append(token)
            self.word2id[token] = len(self.id2word) - 1

        f.close()

        if Vocabulary.unk_token not in self.word2id:
            raise Exception("the dictionary doesn't have #UNK#")

    def size(self):
        return len(self.id2word)

    def get_id(self, token):
        tok = self.normalize(token)
        if tok not in self.word2id:
            tok = Vocabulary.unk_token
        return self.word2id[tok]

    @staticmethod
    def load_word_embeddings(path):
        data = pickle.load(open(path, "rb"))
        type = data["type"]
        wembs = data["word embeddings"]

        voca = Vocabulary()
        embs = [wembs[Vocabulary.unk_token], wembs["<s>"], wembs["</s>"], wembs[Vocabulary.eom_token]]

        for token, vect in wembs.items():
            if token not in [Vocabulary.unk_token, "<s>", "</s>", Vocabulary.eom_token, Vocabulary.go_token]:
                embs.append(vect)
                voca.id2word.append(token)
                voca.word2id[token] = len(voca.id2word) - 1

        embs = np.array(embs)

        return voca, embs
