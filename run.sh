#!/bin/bash
root=$(dirname $0)
experiment='eva02_tiny'
backbone_name='tiny'
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env $root/main.py \
    --batch_size 4 \
    --lr 5e-5 \
    --lr_backbone 5e-6\
    --epochs 300 \
    --backbone_name $backbone_name \
    --coco_path $root/../VOC2012\
    --pre_trained $root/../pretrained/eva02_Ti_pt_in21k_ft_in1k_p14.pt\
    --eval_size 512 \
    --init_pe_size 800 1333 \
    --output_dir $root/../outputs/$experiment
    # --resume $root/../outputs/$experiment/checkpoint.pth
