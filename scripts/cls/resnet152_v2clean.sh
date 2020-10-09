#!/bin/bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
#--------------------prepare data-------------------#
SRC0_DATA_DIR="data"
SRC0_LABEL_PATH="${SRC0_DATA_DIR}/data_list/label_81313.txt"

TAG="resnet152_v2clean_448_GPU"${1}
BACKBONE="resnet152"
MODEL_NAME="cls_model"
CHECKPOINTS_NAME="google_landmark_2020_"$TAG
CHECKPOINTS_DIR="checkpoints/cls/${BACKBONE}"

PRETRAINED="pretrained_models/7x7resnet152-imagenet.pth"
RESUME_MODEL="${CHECKPOINTS_DIR}/${CHECKPOINTS_NAME}_latest.pth"

CONFIG_FILE="configs/cls/base_multitask_classifier_448.conf"
MAX_ITERS=200000
TEST_INTERVAL=20000
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=24
BASE_LR=0.01
LOSS_TYPE="ce_loss"

SHUFFLE_TRANS_SEQ="random_contrast random_hue random_saturation random_brightness random_perm random_blur"
TRANS_SEQ="random_flip resize random_rotate random_resize random_crop random_pad"

LOG_DIR="./log/cls/${BACKBONE}"
LOG_FILE="${LOG_DIR}/${CHECKPOINTS_NAME}.log"



if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

if [[ ${1} == 4 ]]; then
    GPU_LIST="0 1 2 3"
else
    GPU_LIST="0 1 2 3 4 5 6 7"
fi

PYTHON="python -u -m torch.distributed.launch --nproc_per_node=${1}"
${PYTHON} main.py --config_file ${CONFIG_FILE} --gpu ${GPU_LIST} --workers 4 --num_data_sources 1 \
                  --src0_data_dir ${SRC0_DATA_DIR} --src0_label_path ${SRC0_LABEL_PATH} --src0_train_batch_size ${TRAIN_BATCH_SIZE} --src0_val_batch_size ${VAL_BATCH_SIZE} --src0_num_classes 81313 --src0_min_count 5 --src0_linear_type arc0.3_30  --src0_train_loader default \
                  --backbone ${BACKBONE} --model_name ${MODEL_NAME} --loss_type ${LOSS_TYPE} --fc_dim 512 --fc_bn y --fc_relu n \
                  --max_iters ${MAX_ITERS} --base_lr ${BASE_LR} --bb_lr_scale 0.1 --is_warm y --warm_iters 5000 --warm_freeze n --test_interval ${TEST_INTERVAL} --dist y --gather n --display_iter 100 \
                  --shuffle_trans_seq ${SHUFFLE_TRANS_SEQ} --trans_seq ${TRANS_SEQ} --include_val n --val_ratio 0.01 \
                  --pretrained ${PRETRAINED} --resume_continue n --resume_strict n --resume ${RESUME_MODEL} \
                  --checkpoints_dir ${CHECKPOINTS_DIR} --checkpoints_name ${CHECKPOINTS_NAME} --save_iters 2500 2>&1 | tee ${LOG_FILE}