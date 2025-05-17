#!/bin/bash

moe_mode="sparse"
num_experts=192
top_k_experts=64
gpu_id=2
epochs=5
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="/mnt/data/haoqiang/workspace/data/medmoe-vqa/3vqa"
# JSON_FOLDER="/mnt/data/haoqiang/workspace/data/mmed"
# JSON_FOLDER="/mnt/data/haoqiang/workspace/data/biomed-visual-instructions"
IMAGE_FOLDER="/mnt/data/haoqiang/workspace/data/medmoe-vqa/images"
# IMAGE_FOLDER="/mnt/data/haoqiang/workspace/data/pubmedvision/images"
cd ~/workspace/05-moe-llava
export WANDB_PROJECT=moe-qwen2vl-med
export NCCL_P2P_DISABLE=1
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --include=localhost:${gpu_id},$((${gpu_id}+1)) --master_port=$((${gpu_id}+29500)) moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.gate_proj mlp.up_proj mlp.down_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct \
    --image_min_pixels $((16 * 28 * 28)) \
    --image_max_pixels $((576 * 28 * 28)) \
    --skip_moe_init False \
    --load_k_experts False \
    --k_experts_path /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-12e4-ada-nano-ds-tok-share-1epoch \
    --from_pretrained False \
    --from_pretrained_path /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-96e32-mmed-nano-ds-tok-share-1epoch/pytorch_model.bin \
    --warm_up_experts False \
    --use_shared_experts True \
    --use_combined_gate False \
    --combined_gate_type cmr \
    --freeze_shared False \
    --unfreeze_shared_epoch 1 \
    --mone_enable True \
    --mone_expert_type "small_expert" \
    --mone_gate_type "token_gating" \
    --mone_r $((8960/${top_k_experts})) \
    --mone_num_heads 1 \
    --mone_use_expert_gate True \
    --mone_load_original True \
    --version med-moe \
    --data_path ${JSON_FOLDER}/train_all_converted.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e${top_k_experts}-nano-s-tok-share-${epochs}epoch \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
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
    --run_name "s-tok-share" \
    --cache_dir "./cache_dir"


# --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e${top_k_experts}-ada-nano-test\
# --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e8x${top_k_experts}-med-nano-ee-smi-5e\
# --report_to wandb \
# --shared_lr 2e-5 \
# --data_path ${JSON_FOLDER}/train_all_converted.json \

# ${JSON_FOLDER}/adamllm-image-caption-and-synthetic-vqa.json \
#                 ${JSON_FOLDER}/pubmedvision-instruction-vqa.json \
#                 ${JSON_FOLDER}/nlp-tune-40k.json \
#                 ${JSON_FOLDER}/en-medical-meadow-medical-flashcards-34k.json \
#                 ${JSON_FOLDER}/zh-huatuo-knowledge-graph-qa-30k.json \
# --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e${top_k_experts}-med-ada-5epoch-test \
# --max_steps 1000 