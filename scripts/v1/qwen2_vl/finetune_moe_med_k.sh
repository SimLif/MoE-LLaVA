#!/bin/bash

moe_mode="sparse"
num_experts=8
top_k_experts=2
use_residual=False
router_aux_loss_coef=1
# JSON_FOLDER="/mnt/data/haoqiang/workspace/data/medmoe-vqa/3vqa"
# JSON_FOLDER="/mnt/data/haoqiang/workspace/data/biomed-visual-instructions"
JSON_FOLDER="/mnt/data/haoqiang/workspace/data/med-k-nlp"
IMAGE_FOLDER="/mnt/data/haoqiang/workspace/data/medmoe-vqa/images"
# IMAGE_FOLDER="/mnt/data/haoqiang/workspace/data/pubmedvision/images"
cd ~/workspace/05-moe-llava
export WANDB_PROJECT=moe-qwen2vl-med-k
export NCCL_P2P_DISABLE=1
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --include=localhost:1,5 --master_port=29500 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.gate_proj mlp.up_proj mlp.down_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct \
    --skip_moe_init True \
    --image_min_pixels $((16 * 28 * 28)) \
    --image_max_pixels $((576 * 28 * 28)) \
    --mone_enable True \
    --mone_expert_type "embedding_expert" \
    --mone_gate_type "token_gating" \
    --mone_r 2 \
    --mone_num_heads 8 \
    --version med-moe \
    --data_path ${JSON_FOLDER}/zh-huatuo-knowledge-graph-qa-800k.json \
                ${JSON_FOLDER}/zh-disc-med-sft-cmekg-50k.json \
                ${JSON_FOLDER}/en-medical-ai-cleaned-alpaca-257k.json \
                ${JSON_FOLDER}/en-pubmedqa-212k.json \
                ${JSON_FOLDER}/en-medical-meadow-medical-flashcards-34k.json \
                ${JSON_FOLDER}/en-medical-meadow-medqa-10k.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e${top_k_experts}-med-k-ns-ee-1363k\
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "3vqa-nano" \
    --cache_dir "./cache_dir"


# --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e${top_k_experts}-ada-nano-test\
# --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e8x${top_k_experts}-med-nano-ee-smi-5e\
    # --report_to wandb \