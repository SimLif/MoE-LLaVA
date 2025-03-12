#!/bin/bash

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="/mnt/data/haoqiang/workspace/data/medmoe-vqa/3vqa"
IMAGE_FOLDER="/mnt/data/haoqiang/workspace/data/medmoe-vqa/images"
cd ~/workspace/05-moe-llava
export WANDB_PROJECT=moe-llava-med
export NCCL_P2P_DISABLE=1
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
deepspeed --include=localhost:0,1 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/data/haoqiang/workspace/models/moe-llava-qwen-stage2 \
    --skip_moe_init False \
    --load_k_experts True \
    --k_experts_path /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/moe-llava-qwen-stage2-k-10epoch \
    --version qwen \
    --data_path ${JSON_FOLDER}/train_all_converted.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /mnt/data/haoqiang/workspace/models/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/moe-llava-qwen-1.8b-4e-s2-k-9epoch\
    --num_train_epochs 9 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "test-3vqa" \
    --cache_dir "./cache_dir"
