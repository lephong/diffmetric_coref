import sys
from print_coref import read_raw_docs

def read_full_voca(fname):
    """
    read the full vocabulary (word2vec, glove, ...)
    :param fname:
    :return:
    """
    voca = {}
    f = open(fname, "r")
    for line in f:
        line = line.strip()
        voca[line] = False

    f.close()
    return voca

def must_split(doc):
    """
    sentences must be split at mentions' borders
    :param doc:
    :return:
    """
    must_split_positions = [None] * len(doc["sentences"])

    for m_i in range(len(doc["ment_properties"])):
        ment_prop = doc["ment_properties"][m_i]
        sent = doc["sentences"][ment_prop[0]]
        if must_split_positions[ment_prop[0]] is None:
            must_split_positions[ment_prop[0]] = [False] * (len(sent) + 1)

        must_split_positions[ment_prop[0]][ment_prop[1]] = True
        must_split_positions[ment_prop[0]][ment_prop[2]] = True

    return must_split_positions

def process_document(full_voca, doc):
    """
    process a document, combine tokens into phrases
    :param full_voca:
    :param doc:
    :return:
    """
    new_positions = [None] * len(doc["sentences"])
    new_sentences = [None] * len(doc["sentences"])
    new_ment_properties = [None] * len(doc["ment_properties"])

    # combine tokens into phrase, keep track positions of original tokens
    # note: not allow any merge at the borders of mentions
    must_split_positions = must_split(doc)

    for sen_i in range(len(doc["sentences"])):
        sent = doc["sentences"][sen_i]

        new_positions_sent = [0] * len(sent)
        new_sent = []

        for w_i in range(len(sent)):
            token = sent[w_i]
            if token == "-LCB-":
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
            sent[w_i] = token

        w_i = 0
        while w_i < len(sent):
            cur_item = sent[w_i]
            best_item = None
            found = -1
            if cur_item in full_voca:
                found = 0
                best_item = cur_item
            for d in range(1, 6):
                if w_i + d < len(sent):
                    if must_split_positions[sen_i] is not None and must_split_positions[sen_i][w_i + d]:
                        break

                    cur_item += "_" + sent[w_i + d]
                    if cur_item in full_voca:
                        found = d
                        best_item = cur_item
                else:
                    break

            if w_i > 0:
                new_positions_sent[w_i] = new_positions_sent[w_i - 1] + 1
            if found > 0:
                for d in range(found + 1):
                    new_positions_sent[w_i + d] = new_positions_sent[w_i]

            if found > -1:
                full_voca[best_item] = True

            new_sent.append(best_item if best_item is not None else sent[w_i])

            if found > -1:
                w_i += found + 1
            else:
                w_i += 1

        new_positions[sen_i] = new_positions_sent
        new_sentences[sen_i] = new_sent

    # re-index mentions (head, start, end)
    for m_i in range(len(doc["ment_properties"])):
        sent_id, start_id, end_id, head_id, type_id, number, gender = doc["ment_properties"][m_i]
        new_start_id = new_positions[sent_id][start_id]
        new_end_id = new_positions[sent_id][end_id - 1] + 1
        new_head_id = new_positions[sent_id][head_id]
        new_ment_properties[m_i] = (sent_id, new_start_id, new_end_id, new_head_id, type_id, number, gender)

    new_doc = {"sentences": new_sentences,
               "ment_properties": new_ment_properties}

    return new_doc

if __name__ == "__main__":
    raw_doc_path = sys.argv[1]
    full_voca_path = sys.argv[2]
    output_path = sys.argv[3]

    print("load raw docs")
    raw_docs = read_raw_docs(raw_doc_path)

    print("load full vocabulary")
    full_voca = read_full_voca(full_voca_path)

    print("process")
    for doc in raw_docs:
        new_doc = process_document(full_voca, doc)

        # for sen_i in range(len(doc["sentences"])):
        #     sentence = ""
        #     for tok in doc["sentences"][sen_i]:
        #         sentence += " " + tok
        #
        #     new_sentence = ""
        #     for tok in new_doc["sentences"][sen_i]:
        #         new_sentence += " " + tok
        #
        #     if sentence != new_sentence:
        #         print(new_sentence)
        #
        # for m_i in range(len(doc["ment_properties"])):
        #     sent_id, start_id, end_id, head_id, type_id, number, gender = doc["ment_properties"][m_i]
        #     sent = doc["sentences"][sent_id]
        #     print(sent[start_id:end_id], sent[head_id], type_id, number, gender)
        #
        #     sent_id, start_id, end_id, head_id, type_id, number, gender = new_doc["ment_properties"][m_i]
        #     sent = new_doc["sentences"][sent_id]
        #     print(sent[start_id:end_id], sent[head_id], type_id, number, gender)
        #
        #     print("\n")

    print("output")
    f = open(output_path, 'w')
    f.write("#UNK#\n")
    f.write("<s>\n")
    f.write("</s>\n")

    for k,v in full_voca.items():
        if v:
            f.write(k + "\n")

    f.close()
