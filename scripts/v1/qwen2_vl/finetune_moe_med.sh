#!/bin/bash
#SBATCH --account=c4gcot 
#SBATCH --partition=preempt
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --time=6:00:00
#SBATCH --job-name=moe-llava
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

# Load modules if needed (e.g., module load cuda)
# Activate venv if needed (e.g., source activate my_env)
module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"


moe_mode="sparse"

# num_experts=6
num_experts=12
# num_experts=24
# num_experts=48
# num_experts=96
# num_experts=192

# top_k_experts=2
top_k_experts=$((${num_experts}/3))
# run_name="fgmoe-mrg-f2-${num_experts}k${top_k_experts}"
run_name="fgmoe-food-e1-ada-${num_experts}k${top_k_experts}"
gpu_id=0
expert_type="dense_mask_expert"
batch_size=4
epochs=1
use_residual=False
router_aux_loss_coef=0.01
# JSON_FOLDER="/project/c4gcot/datasets/medmoe-vqa/3vqa"
# JSON_FOLDER="/project/c4gcot/datasets/mimic-cxr-dataset"
JSON_FOLDER="/project/c4gcot/datasets/food-vqa-benchmark/"
# JSON_FOLDER="/home/hguoau/workspace/data/mmed"
# JSON_FOLDER="/home/hguoau/workspace/data/biomed-visual-instructions"
# IMAGE_FOLDER="/project/c4gcot/datasets/medmoe-vqa/images"
# IMAGE_FOLDER="/project/c4gcot/datasets/mimic-cxr-dataset/images"
IMAGE_FOLDER="/project/c4gcot/datasets/food-vqa-benchmark/images"
# IMAGE_FOLDER="/home/hguoau/workspace/data/pubmedvision/images"
cd ~/workspace/05-moe-llava
export WANDB_PROJECT=moe-qwen2vl-med
# export NCCL_P2P_DISABLE=1
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run deepspeed --include=localhost:${gpu_id},$((${gpu_id}+1)) --master_port=$((${gpu_id}+29500+${top_k_experts})) moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.gate_proj mlp.up_proj mlp.down_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /project/c4gcot/models/Qwen2-VL-2B-Instruct \
    --image_min_pixels $((16 * 28 * 28)) \
    --image_max_pixels $((576 * 28 * 28)) \
    --skip_moe_init False \
    --load_k_experts False \
    --k_experts_path /home/hguoau/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-12e4-ada-nano-ds-tok-share-1epoch \
    --from_pretrained True \
    --from_pretrained_path /project/c4gcot/models/FGMoE/qwen2-vl-2b-instruct-12e4-ada-nano-ds-tok-share-1epoch/pytorch_model.bin \
    --warm_up_experts False \
    --use_shared_experts True \
    --use_combined_gate False \
    --combined_gate_type cmr \
    --freeze_shared False \
    --unfreeze_shared_epoch 1 \
    --mone_enable True \
    --mone_expert_type ${expert_type} \
    --mone_gate_type "token_gating" \
    --mone_r $((8960/${top_k_experts})) \
    --mone_num_heads 1 \
    --mone_use_expert_gate True \
    --mone_load_original False \
    --version med-moe \
    --data_path ${JSON_FOLDER}/train_all_converted.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /project/c4gcot/models/Qwen2-VL-2B-Instruct \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /scratch/c4gcot/hqguo/checkpoints/${run_name} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $((16/${batch_size})) \
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
    --run_name ${run_name} \
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
    # --output_dir ./checkpoints/qwen2-vl-2b-instruct-${num_experts}e${top_k_experts}-${epochs}epoch \