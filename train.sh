
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR=/path/to/datasets_and_save_model

IMG_SIZE=224 # 224, 384
MODE=swintiny # swintiny, cvt13, t2t, resnet50, vit
CONFIG=swin_tiny_patch4_window7 # swin_tiny_patch4_window7, cvt_13, t2tvit_14, resnet_50, vit_base_16
LAMBDA_DRLOC=0.5 # swin: 0.5, t2t: 0.1, cvt: 0.1
DRLOC_MODE=l1 # l1, ce, cbr

DATASET=imagenet-100 # imagenet-100, imagenet, cifar-10, cifar-100, svhn, places365, flowers102, clipart, infograph, painting, quickdraw, real, sketch
NUM_CLASSES=100

DISK_DATA=${DATA_DIR}/datasets/${DATASET}
TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-drloc${LAMBDA_DRLOC}-bs128-g8
SAVE_DIR=${DATA_DIR}/swintransformer-expr/${TARGET_FOLDER}

python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 12345 main.py \
    --cfg ./configs/${CONFIG}_${IMG_SIZE}.yaml \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --data-path ${DISK_DATA} \
    --batch-size 128 \
    --output ${SAVE_DIR} \
    --lambda_drloc ${LAMBDA_DRLOC} \
    --drloc_mode ${DRLOC_MODE} \
    --use_drloc \
    --use_abs
