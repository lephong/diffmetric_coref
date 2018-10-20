import argparse
import pickle
import vocabulary

import numpy as np
from vocabulary import Vocabulary

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--words_path", type=str,
                    help="word file",
                    default="../data/conll-2012/small_rawtext_hdf5/voca.txt")
parser.add_argument("--word_emb_path", type=str,
                    help="word embedding file",
                    default="../data/conll-2012/small_rawtext_hdf5/w2v_300.txt")
parser.add_argument("--output_prefix", type=str,
                    help="output file",
                    default="../data/conll-2012/small_rawtext_hdf5/words")

args = parser.parse_args()

if __name__ == "__main__":
    for arg,value in vars(args).items():
        print(arg, value)

    voca = Vocabulary()
    print("load vocabulary")
    voca.load_from_file(args.words_path)

    print("load full word embeddings")
    wembs = {}
    dim = -1
    f = open(args.word_emb_path, "r")
    for line in f:
        line = line.strip()
        comps = line.split(" ")
        if len(comps) == 2:
            continue

        token = comps[0]
        if token in voca.word2id:
            wembs[comps[0]] = np.array([float(x) for x in comps[1:]])
            if dim == -1:
                dim = wembs[comps[0]].size
            elif not dim == wembs[comps[0]].size:
                raise Exception("dimension not matched")

    for w in [Vocabulary.eom_token, Vocabulary.go_token, Vocabulary.unk_token, "<s>", "</s>"]:
        if w not in wembs:
            wembs[w] = 0.1 * np.random.randn(dim)

    print("output")
    fword = open(args.output_prefix + ".words", "w")
    embs = []
    for w, e in wembs.items():
        fword.write(w + "\n")
        embs.append(np.reshape(e, [1, -1]))

    fword.close()
    embs = np.concatenate(embs, axis=0)
    np.savetxt(args.output_prefix + ".embs", embs)