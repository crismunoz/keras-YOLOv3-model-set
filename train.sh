#!/bin/bash

# GLOBAL VARIABLES
SIF_DOCKER_NAME='/home/cristian/docker/relation-extraction_horovod.sif'
HOME='/home/cristian'
PROJECT_HOME='${HOME}/Documents/Github/keras-YOLOv3-model-set'

# MODEL VARIABLES
MODEL_TYPE=yolo4_efficientnet
ANCHOR_PATH=configs/yolo4_anchors.txt
MODEL_IMAGE=512x512

# DATA VARIABLES
ANNOTATION_FILE=../data/train_tf.record
VAL_ANNOTATION_FILE=../data/valid_tf.record
CLASS_PATH=../data/doc_classes.txt

PROGRAM="singularity"
USE_HOROVOD=""
GPUS_NUM=1

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 #,1,2,3
$PROGRAM run -B $HOME --nv ${SIF_DOCKER_NAME} bash -c "cd ${PROJECT_HOME}; HOME=${PROJECT_HOME} python train.py --model_type=$MODEL_TYPE \
--anchors_path=$ANCHOR_PATH \
--annotation_file=$ANNOTATION_FILE \
--val_annotation_file=$VAL_ANNOTATION_FILE \
--gpu_num $GPUS_NUM \
--data_shuffle \
--transfer_epoch 20 \
--total_epoch 250 \
--learning_rate 0.001 \
--model_image_size=$MODEL_IMAGE \
--classes_path=$CLASS_PATH \
--batch_size 1 $USE_HOROVOD"