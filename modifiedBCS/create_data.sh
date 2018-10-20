java -jar -Xmx30g ./target/scala-2.11/moarcoref-assembly-1.jar ++./base.conf -execDir execdir -numberGenderData gender.data -animacyPath animate.unigrams.txt -inanimacyPath inanimate.unigrams.txt -trainPath flat_train_2012 -devPath flat_dev_2012 -testPath flat_test_2012  -mode SMALLER -conjType NONE -pairwiseFeats FINAL+MOARANAPH+MOARPW

cd data

rm -r split/* 
mkdir split/train

rm original_files/*
mv ../SMALL* original_files/

python3 shuffle_and_split.py original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphTrainFeats.txt original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphTrainLex.txt original_files/SMALL-FINAL+MOARANAPH+MOARPW-pwTrainFeats.txt original_files/SMALLTrainOPCs.txt 500 split/


bash convert_to_hd5.sh split/

mv split/OPCs.* split/*.h5 split/*lex* split/train

python2 ./text_feats_to_hdf5_replacezero.py -n 10 original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphDevFeats.txt split/dev_small ana

python2 ./text_feats_to_hdf5_replacezero.py -n 10 original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphTestFeats.txt split/test_small ana

python2 ./text_feats_to_hdf5_replacezero.py -n 10 original_files/SMALL-FINAL+MOARANAPH+MOARPW-pwDevFeats.txt split/dev_small pw

python2 ./text_feats_to_hdf5_replacezero.py -n 10 original_files/SMALL-FINAL+MOARANAPH+MOARPW-pwTestFeats.txt split/test_small pw

cp original_files/SMALLDevOPCs.txt split/DevOPCs.txt

mkdir small_rawtext_hdf5

mv split/dev_small-* split/test_small-* split/DevOPCs.txt split/train/ small_rawtext_hdf5/

cp original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphMapping.txt small_rawtext_hdf5/anaphMapping.txt
cp original_files/SMALL-FINAL+MOARANAPH+MOARPW-pwMapping.txt small_rawtext_hdf5/pwMapping.txt

cp original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphDevLex.txt small_rawtext_hdf5/dev_small-na-lex.txt
cp original_files/SMALL-FINAL+MOARANAPH+MOARPW-anaphTestLex.txt small_rawtext_hdf5/test_small-na-lex.txt

mv small_rawtext_hdf5/ ../../data/conll-2012/

cd ../../data/conll-2012/
cat anaphMapping.txt | grep -v 'UNK_FEAT' | awk '{print $3, $1}' | sort | awk '{print $0,(NR-1)}' > anaphReMapping.txt 
cat pwMapping.txt | grep -v 'UNK_FEAT' | awk '{print $3, $1}' | sort | awk '{print $0,(NR-1)}' > pwReMapping.txt
