
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set pretrained model path and dataset 
PRETRAINED_MODEL=/path/to/pretrained_model
DISK_DATA=/path/to/dataset_root
# set dataset
DATASET=imagenet-100 
NUM_CLASSES=100 

# set model
MODEL=swintiny # swintiny, cvt13, t2t
CONFIG=swin_tiny_patch4_window7 # swin_tiny_patch4_window7, cvt_13, t2tvit_14

# set dense relative localization loss
LAMBDA_RELD=0.5 # swin: 0.5, t2t: 0.1, cvt: 0.1
RELM_MODE=l1 

python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 12345 main.py \
    --cfg ./configs/${CONFIG}_224.yaml \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --data-path ${DISK_DATA} \
    --pretrain_model ${PRETRAINED_MODEL} \
    --batch-size 128 \
    --output ./eval \
    --lambda_drloc ${LAMBDA_RELD} \
    --drloc_mode ${RELM_MODE} \
    --use_drloc \
    --use_abs \
    --eval
    
