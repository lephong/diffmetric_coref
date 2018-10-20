import sys

short_voca = {}
long_voca = {}

f = open(sys.argv[1])
for line in f:
    short_voca[line.strip()] = 0
f.close()

f = open(sys.argv[2])
for line in f:
    comps = line.strip().split()
    if comps[0] in short_voca:
        print(line.strip())
    elif comps[0] in {'<UNK>', '<S>', '</S>', '(', ')', '[', ']'}:
        print(line.strip())
f.close()