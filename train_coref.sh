#!/bin/bash

cd $PROJECT_HOME
DATA_DIR=$PROJECT_HOME/data/conll-2012/minimal_lex_hd5/

TRAIN_DATA_DIR=$DATA_DIR/train/

#DEV_NA_PREFIX=$DATA_DIR/dev_small-na-
#DEV_NA_LEX=$DATA_DIR/dev_small-na-lex.txt
#DEV_PW_PREFIX=$DATA_DIR/dev_small-pw-
#DEV_ORACLE_CLUSTER=$DATA_DIR/DevOPCs.txt

DEV_NA_PREFIX=$DATA_DIR/train/na.0.txt-na-
DEV_NA_LEX=$DATA_DIR/train/na_lex.0.txt
DEV_PW_PREFIX=$DATA_DIR/train/pw.0.txt-pw-
DEV_ORACLE_CLUSTER=$DATA_DIR/train/OPCs.0.txt


EXPERIMENT_DIR=$PROJECT_HOME/experiment_unsup/

TRAIN_SCORE_TYPE=cluster
TEST_SCORE_TYPE=cluster
PW_FEAT_DIM=100
A_FEAT_DIM=100
CLUS_EMB_DIM=100
WORD_EMB_DIM=100
MAX_LENTH=200
N_EPOCHS=100
N_PORTIONS=1
L1_LEARNING_RATE=0.1
L2_LEARNING_RATE=0.1
CLUSTER_LEARNING_RATE=0.1
GEN_LEARNING_RATE=0.1
KEEP_PROB=1.
HARD_ATT=
ORACLE_CLS=
GPU=
UNSUP=--unsup

mkdir $EXPERIMENT_DIR

for ((epoch=0;epoch<$N_EPOCHS;epoch++))
do
    echo "=========================== epoch " $epoch "============================="
    for ((portion=0;portion<$N_PORTIONS;portion++))
    do
        echo "************** portion " $portion "********************"
        if [ $epoch -eq 0 ] && [ $portion -eq 0 ] ; then
            python3 -u src/coref.py $GPU \
                --mode train --score_type $TRAIN_SCORE_TYPE $UNSUP \
                --n_epochs 1 \
                --eval_every_k_epochs -1 \
                --max_length $MAX_LENTH \
                --train_na_prefix $TRAIN_DATA_DIR/na.$portion.txt-na- \
                --train_na_lex $TRAIN_DATA_DIR/na_lex.$portion.txt \
                --train_pw_prefix $TRAIN_DATA_DIR/pw.$portion.txt-pw- \
                --train_oracle_cluster $TRAIN_DATA_DIR/OPCs.$portion.txt \
                --experiment_dir $EXPERIMENT_DIR \
                --pw_feat_dim $PW_FEAT_DIM \
                --a_feat_dim $A_FEAT_DIM \
                --cluster_emb_dim $CLUS_EMB_DIM \
                --word_emb_dim $WORD_EMB_DIM \
                --layer_1_learning_rate $L1_LEARNING_RATE \
                --layer_2_learning_rate $L2_LEARNING_RATE \
                --cluster_learning_rate $CLUSTER_LEARNING_RATE \
                --gen_learning_rate $GEN_LEARNING_RATE \
                --keep_prob $KEEP_PROB $HARD_ATT  $ORACLE_CLS

        else
            if [ $portion -eq 0 ] ; then
                cp $EXPERIMENT_DIR/coref.model-$((epoch - 1)) $EXPERIMENT_DIR/coref.model-$epoch
                cp $EXPERIMENT_DIR/coref.model-$((epoch - 1)).meta $EXPERIMENT_DIR/coref.model-$epoch.meta
            fi

            python3 -u src/coref.py $GPU \
                --mode train  --score_type $TRAIN_SCORE_TYPE $UNSUP \
                --n_epochs 1 \
		        --eval_every_k_epochs -1 \
                --model_path $EXPERIMENT_DIR/coref.model-$epoch \
                --max_length $MAX_LENTH \
                --train_na_prefix $TRAIN_DATA_DIR/na.$portion.txt-na- \
                --train_na_lex $TRAIN_DATA_DIR/na_lex.$portion.txt \
                --train_pw_prefix $TRAIN_DATA_DIR/pw.$portion.txt-pw- \
                --train_oracle_cluster $TRAIN_DATA_DIR/OPCs.$portion.txt \
                --experiment_dir $EXPERIMENT_DIR \
                --pw_feat_dim $PW_FEAT_DIM \
                --a_feat_dim $A_FEAT_DIM \
                --cluster_emb_dim $CLUS_EMB_DIM \
                --word_emb_dim $WORD_EMB_DIM \
                --layer_1_learning_rate $L1_LEARNING_RATE \
                --layer_2_learning_rate $L2_LEARNING_RATE \
                --cluster_learning_rate $CLUSTER_LEARNING_RATE \
                --gen_learning_rate $GEN_LEARNING_RATE \
                --keep_prob $KEEP_PROB $HARD_ATT  $ORACLE_CLS

            mv $EXPERIMENT_DIR/coref.model-$epoch-0 $EXPERIMENT_DIR/coref.model-$epoch            
            mv $EXPERIMENT_DIR/coref.model-$epoch-0.meta $EXPERIMENT_DIR/coref.model-$epoch.meta            
        fi
    done

    echo "*************** eval ***************"
    python3 -u src/coref.py $GPU \
        --mode eval $HARD_ATT $ORACLE_CLS --score_type $TEST_SCORE_TYPE \
        --experiment_dir $EXPERIMENT_DIR \
        --a_feat_dim $A_FEAT_DIM \
        --pw_feat_dim $PW_FEAT_DIM \
        --cluster_emb_dim $CLUS_EMB_DIM \
        --word_emb_dim $WORD_EMB_DIM \
        --model_path $EXPERIMENT_DIR/coref.model-$epoch \
        --eval_output $EXPERIMENT_DIR/load_and_pred.bps$epoch \
        --dev_na_prefix $DEV_NA_PREFIX \
        --dev_na_lex $DEV_NA_LEX \
        --dev_pw_prefix $DEV_PW_PREFIX \
        --dev_oracle_cluster $DEV_ORACLE_CLUSTER 

done
