#!/bin/bash



DATAHOME=/media/hessam/SSD_AI/NLP_NEU/data/news_data

EXEHOME=/media/hessam/SSD_AI/NLP_NEU/code/NeuSum/neusum_pt


SAVEPATH=${DATAHOME}/models/neusum


python3.7 train.py -save_path ${SAVEPATH} \
                -online_process_data \
                -max_doc_len 120 \
                -train_oracle ${DATAHOME}/train/train.rouge_bigram_F1.oracle.100.regGain \
                -train_src ${DATAHOME}/train/train.txt.src.100 \
                -train_src_rouge ${DATAHOME}/train/train.rouge_bigram_F1.oracle.100.regGain \
                -src_vocab ${DATAHOME}/train/vocab \
                -train_tgt ${DATAHOME}/train/train.txt.tgt.100 \
                -tgt_vocab ${DATAHOME}/train/vocab \
                -layers 1 -word_vec_size 50 -sent_enc_size 256 -doc_enc_size 256 -dec_rnn_size 256 \
                -sent_brnn -doc_brnn \
                -dec_init simple \
                -att_vec_size 256 \
                -norm_lambda 20 \
                -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0\
                -batch_size 64 -beam_size 1 \
                -epochs 100 \
                -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                -curriculum 0 -extra_shuffle \
                -start_eval_batch 1000 -eval_per_batch 1000 \
                -log_interval 100
                -pre_word_vecs_enc ${DATAHOME}/glove/glove.6B.50d.txt \
                -freeze_word_vecs_enc \
                -dev_input_src ${DATAHOME}/dev/val.txt.src.shuffle.4k \
                -dev_ref ${DATAHOME}/dev/val.txt.tgt.shuffle.4k \
                -max_decode_step 3 -force_max_len
