import sys
import random

if len(sys.argv) != 7:
    raise Exception("fa fal fp fc maxlen output_dir")

# read file
fa = open(sys.argv[1], "r")
fal = open(sys.argv[2], "r")
fp = open(sys.argv[3], "r")
fc = open(sys.argv[4], "r")
lines = []

print("reading files")
while True:
    la = fa.readline()
    if la == "":
        break

    lal = ""
    while True:
        l = fal.readline()
        lal += l
        if l == "\n":
            break

    if len(lal) < 100:
        raise Exception("sthing wrong with lex file")

    lp = fp.readline()
    lc = fc.readline()
    lines.append((la, lal, lp, lc))
    if len(lines) % 10 == 0:
        print(len(lines), "\r")

print("total", len(lines), "documents")
fa.close()
fal.close()
fp.close()
fc.close()

random.shuffle(lines)

# write to files
max_len = int(sys.argv[5])
id = 0
cur_line = -1
output_dir = sys.argv[6] + "/"

print("writing files")
while True:
    fa = open(output_dir + "na." + str(id) + ".txt", "w")
    fal = open(output_dir + "na_lex." + str(id) + ".txt", "w")
    fp = open(output_dir + "pw." + str(id) + ".txt", "w")
    fc = open(output_dir + "OPCs" + "." + str(id) + ".txt", "w")

    done = False

    for i in range(max_len):
        cur_line += 1
        if cur_line >= len(lines):
            done = True
            break

        fa.write(lines[cur_line][0])
        fal.write(lines[cur_line][1])
        fp.write(lines[cur_line][2])
        fc.write(lines[cur_line][3])
        if cur_line % 10 == 0:
            print(cur_line, "\r")

    fa.close()
    fal.close()
    fp.close()
    fc.close()
    id += 1

    if done:
        break

