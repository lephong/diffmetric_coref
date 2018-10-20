import codecs
import sys
from corpus import GENDER, NUMBER

COLORS = ['blue', 'brown', 'cyan', 'darkgray', 'green', 'lime', 'magenta', 'olive',
          'orange', 'pink', 'purple', 'red', 'teal', 'violet', 'yellow'] * 10


def read_raw_docs(fname):
    # reading sentences and position of mentions
    raw_docs = []

    f = codecs.open(fname, "r", encoding='utf-8', errors='ignore')
    while(True):
        sentences = []
        ment_words = []
        ment_properties = []
        prev_is_ment = False

        done = False

        while True:
            line = f.readline()
            if line == "":
                done = True
                break

            line = line.strip()
            if line == "":
                break

            line = line.replace("$", "\$").replace("#", "\#").replace("&", "\&").replace("%", "\%").replace("_", "\_").replace("^", "\^")
            comps = line.split()

            if len(comps) == 8 and comps[0].isdigit() and comps[1].isdigit() and \
                    comps[2].isdigit() and comps[3].isdigit():
                sent_id = int(comps[0])
                start_id = int(comps[1]) 
                end_id = int(comps[2]) 
                head_id = int(comps[3])
                type_id = comps[4]
                number = NUMBER[comps[6]]
                gender = GENDER[comps[7]]

                ment_words.append(sentences[sent_id][start_id:end_id])
                ment_properties.append((sent_id, start_id, end_id, head_id, type_id, number, gender))

                prev_is_ment = True
            else:
                if prev_is_ment:
                    raise Exception("we are reading mentions")
                sentences.append(comps)

        if done:
            break

        raw_docs.append({"sentences": sentences,
                         "mentions": ment_words,
                         "ment_properties": ment_properties,
                         "clusters": [],
                         "antecedents": None},)

    f.close()
    return raw_docs

def read_gold_clusters(fname, raw_docs):
    f = open(fname, "r")
    doc_id = 0

    for line in f:
        line = line.strip()
        clusters = line.split('|')

        for clus in clusters:
            items = clus.split(" ")
            if len(items) > 1:
                raw_docs[doc_id]["clusters"].append([int(i) for i in items])

        doc_id += 1

    f.close()
    return raw_docs


def not_singleton(ante):
    not_sing = [False] * len(ante)
    for i in range(len(ante)):
        if ante[i] != i:
            not_sing[i] = True
            not_sing[ante[i]] = True

    return not_sing

def read_system_antecedents(fname1, fname2, docs):
    f1 = open(fname1, "r")
    f2 = open(fname2, "r")
    doc_id = 0

    for line1 in f1:
        line1 = line1.strip()
        ante_id1 = [int(i) for i in line1.split(" ")]

        line2 = f2.readline().strip()
        ante_id2 = [int(i) for i in line2.split(" ")]

        docs[doc_id]["antecedents"] = (ante_id1, ante_id2)
        doc_id += 1

    f1.close()
    f2.close()
    return docs


def annotate_old(docs):
    for doc in docs:
        # system antecedents
        marked = {}
        for i in range(len(doc["antecedents"][0])):
            ante_id1 = doc["antecedents"][0][i]
            ante_id2 = doc["antecedents"][1][i]

            if ante_id1 == i and ante_id2 == i:
                continue

            if ante_id1 != i and ante_id1 not in marked:
                marked[ante_id1] = 1
                ment_prop = doc["ment_properties"][ante_id1]
                sent = doc["sentences"][ment_prop[0]]
                sent[ment_prop[1]] = "$_{" + str(ante_id1) + "}$[" + sent[ment_prop[1]]
                sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "]"
 
            if ante_id2 != i and ante_id2 not in marked:
                marked[ante_id2] = 1
                ment_prop = doc["ment_properties"][ante_id2]
                sent = doc["sentences"][ment_prop[0]]
                sent[ment_prop[1]] = "$_{" + str(ante_id2) + "}$[" + sent[ment_prop[1]]
                sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "]"
                
            marked[i] = 1
            ment_prop = doc["ment_properties"][i]
            sent = doc["sentences"][ment_prop[0]]
            sent[ment_prop[1]] = "$_{" + str(i) +  "}$[" + sent[ment_prop[1]]
            sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "]$^{" + str(ante_id1) + "}_{" + str(ante_id2) + "}$"
   
        # color clusters
        for i in range(len(doc["clusters"])):
            color = COLORS[i]
            for id in doc["clusters"][i]:
                ment_prop = doc["ment_properties"][id]
                sent = doc["sentences"][ment_prop[0]]
                # print(sent)
                # print(ment_prop)
                sent[ment_prop[1]] = "{\color{" + color + "} " + sent[ment_prop[1]]
                sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "}"    

    return docs


def print_cluster(doc, clus, i, ante_id):
    text = []
    for j in clus:
        ment_prop = doc["ment_properties"][j]
        sent = doc["sentences"][ment_prop[0]]
        surface = ""
        for t in range(ment_prop[1], ment_prop[2]):
            surface += sent[t] + " "

        surface += " [" + sent[ment_prop[3]] + "]"

        if j == i:
            surface += "*"

        text.append(surface)
        if j == i:
            break

    surface = ""
    ment_prop = doc["ment_properties"][ante_id]
    sent = doc["sentences"][ment_prop[0]]
    surface = ""
    for t in range(ment_prop[1], ment_prop[2]):
        surface += sent[t] + " "
    surface += " [" + sent[ment_prop[3]] + "]"

    if i == ante_id:
        surface += "*"

    print(text, surface)


def annotate(docs):
    self_count_1 = 0
    count1 = 0
    count2 = 0
    total = 0

    for doc in docs:
        # system antecedents
        marked = {}
        not_sing_1 = not_singleton(doc["antecedents"][0])

        for i in range(len(doc["antecedents"][0])):
            ante_id1 = doc["antecedents"][0][i]
            ante_id2 = doc["antecedents"][1][i]

            ante_id1_wrong = "*"
            ante_id2_wrong = "*"
            i_wrong = "*"
            in_cluster = False
            cluster = None

            for clus in doc["clusters"]:
                if i in clus:
                    cluster = clus
                    if i != clus[0]:
                        ante_id1_wrong = "" if ante_id1 in clus and ante_id1 != i else "*"
                        ante_id2_wrong = "" if ante_id2 in clus and ante_id2 != i else "*"

                        if doc["ment_properties"][i][4] != "PRONOMINAL" and not_sing_1[i]:
                            if ante_id1_wrong != "":
                                print_cluster(doc, clus, i, ante_id1)
                            if ante_id1 == i and doc["ment_properties"][ante_id1][4] != "PRONOMINAL":
                                self_count_1 += 1
                            if ante_id1 in clus and ante_id1 != i and doc["ment_properties"][ante_id1][4] != "PRONOMINAL":
                                count1 += 1
                            if ante_id2 in clus and ante_id2 != i and doc["ment_properties"][ante_id2][4] != "PRONOMINAL":
                                count2 += 1
                            total += 1
                    else:
                        ante_id1_wrong = "" if ante_id1 in clus else "*"
                        ante_id2_wrong = "" if ante_id2 in clus else "*"

                    i_wrong = ""
                    in_cluster = True
                    break

            if not in_cluster:
                if ante_id1 == i and ante_id2 == i:
                    continue

                ante_id1_wrong = "" if ante_id1 == i else "*"
                ante_id2_wrong = "" if ante_id2 == i else "*"

            if cluster is None:
                continue

        #     if ante_id1 != i and ante_id1 not in marked:
        #         marked[ante_id1] = 1
        #         ment_prop = doc["ment_properties"][ante_id1]
        #         sent = doc["sentences"][ment_prop[0]]
        #         sent[ment_prop[1]] = "$_{" + str(ante_id1) + "}$[" + sent[ment_prop[1]]
        #         sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "]"
        #
        #     if ante_id2 != i and ante_id2 not in marked:
        #         marked[ante_id2] = 1
        #         ment_prop = doc["ment_properties"][ante_id2]
        #         sent = doc["sentences"][ment_prop[0]]
        #         sent[ment_prop[1]] = "$_{" + str(ante_id2) + "}$[" + sent[ment_prop[1]]
        #         sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "]"
        #
        #     marked[i] = 1
        #     ment_prop = doc["ment_properties"][i]
        #     sent = doc["sentences"][ment_prop[0]]
        #     sent[ment_prop[1]] = "$_{" + str(i) + i_wrong + "}$[" + sent[ment_prop[1]]
        #     sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "]$^{" + (str(ante_id1) if ante_id1 != i else "") + ante_id1_wrong + \
        #                              "}_{" + (str(ante_id2) if ante_id2 != i else "") + ante_id2_wrong + "}$"
        #
        # # color clusters
        # for i in range(len(doc["clusters"])):
        #     color = COLORS[i]
        #     for id in doc["clusters"][i]:
        #         ment_prop = doc["ment_properties"][id]
        #         sent = doc["sentences"][ment_prop[0]]
        #         # print(sent)
        #         # print(ment_prop)
        #         sent[ment_prop[1]] = "{\color{" + color + "} " + sent[ment_prop[1]]
        #         sent[ment_prop[2] - 1] = sent[ment_prop[2] - 1] + "}"

    total_clusters = 0
    for doc in docs:
        total_clusters += len(doc["clusters"])

    print(self_count_1, count1, count2, total, total_clusters)
    return docs


def print_to_file(docs, fname):
    f = open(fname, "w")
    f.write("\\documentclass{article}\n \\usepackage[utf8]{inputenc}\n \\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}\n\n \\begin{document}\n\n\n")

    for doc in docs:
        f.write("\section{doc}\n")        
        for sent in doc["sentences"]:
            sent = " ".join(sent)
            f.write(sent + " ")

        f.write("\n\n")

    f.write("\n\n\n\\end{document}")
    f.close()


if __name__ == "__main__":
    raw_doc_path = sys.argv[1]
    gold_cluster_path = sys.argv[2]
    sys1_ante_path = sys.argv[3]
    sys2_ante_path = sys.argv[4]
    output_path = sys.argv[5]

    raw_docs = read_raw_docs(raw_doc_path)
    docs = read_gold_clusters(gold_cluster_path, raw_docs)
    docs = read_system_antecedents(sys1_ante_path, sys2_ante_path, docs) 
    docs = annotate(docs)
    print_to_file(docs, output_path)
    



