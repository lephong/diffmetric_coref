#!/bin/bash

cd $PROJECT_HOME
DATA_DIR=$PROJECT_HOME/data/conll-2012/small_rawtext_hdf5/

TRAIN_DATA_DIR=$DATA_DIR/train/

DEV_NA_PREFIX=$DATA_DIR/dev_small-na-
DEV_NA_LEX=$DATA_DIR/dev_small-na-lex.txt
DEV_PW_PREFIX=$DATA_DIR/dev_small-pw-
DEV_ORACLE_CLUSTER=$DATA_DIR/DevOPCs.txt

MENT_MAP_FILE=$DATA_DIR/anaphReMapping.txt
PW_MAP_FILE=$DATA_DIR/pwReMapping.txt

EXPERIMENT_DIR=$PROJECT_HOME/experiment_anadet_1/

VOCA=$PROJECT_HOME/data/lm/small_voca.txt

OPTIMIZER=AdaGrad

PW_FEAT_DIM=700
A_FEAT_DIM=200
MAX_LENTH=1000
N_EPOCHS=100
N_PORTIONS=6
L1_LEARNING_RATE=0.1
L2_LEARNING_RATE=0.1
GEN_LEARNING_RATE=0.1
KEEP_PROB=1.
GPU=

mkdir $EXPERIMENT_DIR

for ((epoch=0;epoch<$N_EPOCHS;epoch++))
do
    echo "=========================== epoch " $epoch "============================="
    for ((portion=0;portion<$N_PORTIONS;portion++))
    do
        echo "************** portion " $portion "********************"
        if [ $epoch -eq 0 ] && [ $portion -eq 0 ] ; then
            python3 -u src/anaphoricity_detection.py $GPU \
                --mode train --optimizer $OPTIMIZER \
                --n_epochs 1 \
                --eval_every_k_epochs -1 \
                --max_length $MAX_LENTH \
                --train_na_prefix $TRAIN_DATA_DIR/na.$portion.txt-na- \
                --train_na_lex $TRAIN_DATA_DIR/na_lex.$portion.txt \
                --train_pw_prefix $TRAIN_DATA_DIR/pw.$portion.txt-pw- \
                --train_oracle_cluster $TRAIN_DATA_DIR/OPCs.$portion.txt \
                --voca $VOCA \
                --ment_map_file $MENT_MAP_FILE \
                --pw_map_file $PW_MAP_FILE \
                --experiment_dir $EXPERIMENT_DIR \
                --pw_feat_dim $PW_FEAT_DIM \
                --a_feat_dim $A_FEAT_DIM \
                --layer_1_learning_rate $L1_LEARNING_RATE \
                --layer_2_learning_rate $L2_LEARNING_RATE \
                --gen_learning_rate $GEN_LEARNING_RATE

        else
            if [ $portion -eq 0 ] ; then
                cp $EXPERIMENT_DIR/coref.model-$((epoch - 1)) $EXPERIMENT_DIR/coref.model-$epoch
            fi

            python3 -u src/anaphoricity_detection.py $GPU \
                --mode train_cont --optimizer $OPTIMIZER \
                --n_epochs 1 \
                --eval_every_k_epochs -1 \
                --model_path $EXPERIMENT_DIR/coref.model-$epoch \
                --max_length $MAX_LENTH \
                --train_na_prefix $TRAIN_DATA_DIR/na.$portion.txt-na- \
                --train_na_lex $TRAIN_DATA_DIR/na_lex.$portion.txt \
                --train_pw_prefix $TRAIN_DATA_DIR/pw.$portion.txt-pw- \
                --train_oracle_cluster $TRAIN_DATA_DIR/OPCs.$portion.txt \
                --voca $VOCA \
                --ment_map_file $MENT_MAP_FILE \
                --pw_map_file $PW_MAP_FILE \
                --experiment_dir $EXPERIMENT_DIR \
                --pw_feat_dim $PW_FEAT_DIM \
                --a_feat_dim $A_FEAT_DIM \
                --layer_1_learning_rate $L1_LEARNING_RATE \
                --layer_2_learning_rate $L2_LEARNING_RATE \
                --gen_learning_rate $GEN_LEARNING_RATE

            mv $EXPERIMENT_DIR/coref.model-$epoch-0 $EXPERIMENT_DIR/coref.model-$epoch
        fi
    done

    echo "*************** eval ***************"
    python3 -u src/anaphoricity_detection.py $GPU \
        --mode eval \
        --voca $VOCA \
        --ment_map_file $MENT_MAP_FILE \
        --pw_map_file $PW_MAP_FILE \
        --experiment_dir $EXPERIMENT_DIR \
        --a_feat_dim $A_FEAT_DIM \
        --pw_feat_dim $PW_FEAT_DIM \
        --model_path $EXPERIMENT_DIR/coref.model-$epoch \
        --eval_output $EXPERIMENT_DIR/load_and_pred.bps$epoch \
        --dev_na_prefix $DEV_NA_PREFIX \
        --dev_na_lex $DEV_NA_LEX \
        --dev_pw_prefix $DEV_PW_PREFIX \
        --dev_oracle_cluster $DEV_ORACLE_CLUSTER
done
