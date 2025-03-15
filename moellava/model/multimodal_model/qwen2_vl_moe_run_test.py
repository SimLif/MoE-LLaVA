import torch
import argparse
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor
from dataclasses import dataclass

from transformers import logging
logging.set_verbosity_info()

# 导入自定义的MoE模型
from moellava.model.multimodal_model.qwen2_vl_moe import MoEQwen2VLForConditionalGeneration, MoEQwen2VLConfig
from qwen_vl_utils import process_vision_info

import deepspeed
deepspeed.init_distributed(dist_backend='nccl')

@dataclass
class ModelArguments:
    """MoE模型参数配置"""
    moe_enable: bool = True
    moe_mode: str = "sparse"  # 可选：sparse, dense, first_half, second_half, custom
    moe_layers_idx: list = None  # 若为None，则根据moe_mode自动设置
    num_experts: list = None  # 每层专家数量
    ep_size: int = 1  # 专家并行度
    top_k_experts: int = 2  # 每个token选择的专家数
    capacity_factor: float = 1.0  # 容量因子
    eval_capacity_factor: float = 1.0  # 评估时的容量因子
    min_capacity: int = 4  # 最小容量
    use_residual: bool = False  # 是否使用残差连接
    router_aux_loss_coef: float = 0.01  # 路由器辅助损失系数
    train_modules: list = None  # 要训练的模块，其他将被冻结
    # LoRA配置（若需要）
    lora_enable: bool = False
    only_lora_ffn: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"


def test_moe_initialization():
    """测试MoE模型初始化"""
    print("===== 测试MoE模型初始化 =====")
    
    # 加载原始Qwen2-VL模型配置
    config = MoEQwen2VLConfig.from_pretrained("/mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct")
    
    # 设置MoE参数
    model_args = ModelArguments(
        moe_enable=True,
        moe_mode="sparse",  # 稀疏MoE，每隔一层使用MoE
        moe_layers_idx=[0, 2, 4, 6],  # 每隔一层使用MoE
        num_experts=[4, 4, 4, 4],  # 每层4个专家
        train_modules=["mlp"]  # 只训练MLP层
    )
    
    # 初始化MoE模型
    model = MoEQwen2VLForConditionalGeneration(config)
    model.initialize_moe_modules(model_args)
    
    print(f"模型配置: {model.config.moe}")
    print(f"MoE层索引: {model.config.moe['moe_layers_idx']}")
    print(f"专家数量: {model.config.moe['num_experts']}")
    print(f"模型初始化成功!")
    
    return model, config


def test_model_forward(model, processor):
    """测试模型前向传播"""
    print("\n===== 测试模型前向传播 =====")
    
    # 加载测试图像
    image_url = 'https://fastly.picsum.photos/id/114/300/300.jpg?hmac=HflrZa5lTOtH8oy__ekxg31MHzkKMKLDt0P-kK2JRtM'
    image = Image.open(BytesIO(requests.get(image_url).content))
    
    # 使用正确的消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # 直接传递PIL图像对象
                },
                {"type": "text", "text": "Describe this image in detail:"},
            ],
        }
    ]
    
    # 按照参考代码处理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 处理视觉信息 - 需要导入process_vision_info函数
    # 如果没有此函数，可以从messages中提取图像
    # image_inputs = [image]
    # video_inputs = []
    
    # 如果有process_vision_info函数
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 完整处理输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    print(f"输入文本: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"输入图像: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        print(f"图像网格: {inputs['image_grid_thw']}")
    
    # 检查图像token是否正确插入
    if hasattr(model.config, 'image_token_id'):
        image_token_count = (inputs['input_ids'] == model.config.image_token_id).sum().item()
        print(f"图像token数量: {image_token_count}")
    
    # 移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"输出logits形状: {outputs.logits.shape}")
    
    # 检查是否有MoE损失
    if hasattr(outputs, "moe_loss"):
        print(f"MoE损失: {outputs.moe_loss}")
    
    return inputs


def test_model_generation(model, processor, inputs):
    """测试模型生成能力"""
    print("\n===== 测试模型生成能力 =====")
    
    # 移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 获取输入长度以便后续截取生成部分
    input_length = inputs["input_ids"].shape[1]
    
    # 确保传递所有必要的参数
    generation_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    # 添加视觉相关参数（如果存在）
    if "pixel_values" in inputs:
        generation_inputs["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        generation_inputs["image_grid_thw"] = inputs["image_grid_thw"]
    if "video_grid_thw" in inputs:
        generation_inputs["video_grid_thw"] = inputs["video_grid_thw"]
    if "pixel_values_videos" in inputs:
        generation_inputs["pixel_values_videos"] = inputs["pixel_values_videos"]
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **generation_inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 只解码新生成的部分（剔除输入部分）
    generated_ids_trimmed = generated_ids[:, input_length:]
    
    # 解码生成的文本
    generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    print(f"生成的文本: {generated_text}")


def test_expert_routing(model):
    """测试专家路由机制"""
    print("\n===== 测试专家路由机制 =====")
    
    # 检查MoE层
    moe_layers_idx = model.config.moe['moe_layers_idx']
    print(f"MoE层索引: {moe_layers_idx}")
    
    # 输出每个MoE层的专家数量和配置
    for layer_idx, num_experts in zip(moe_layers_idx, model.config.moe['num_experts']):
        layer = model.model.layers[layer_idx]
        moe_layer = layer.mlp
        
        if hasattr(moe_layer, 'deepspeed_moe'):
            print(f"层 {layer_idx}: {num_experts}个专家")
            print(f"  - Top-k专家: {model.config.moe['top_k_experts']}")
            print(f"  - 容量因子: {model.config.moe['capacity_factor']}")
            print(f"  - 使用残差: {model.config.moe['use_residual']}")
        else:
            print(f"警告: 层 {layer_idx} 不是MoE层")

# 打印模型参数
def print_model_parameters(model):
    print("模型参数：")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

def main():
    print("开始测试MoE Qwen2-VL模型")
    
    # 步骤1: 测试模型初始化
    model, config = test_moe_initialization()
    
    # 步骤2: 加载处理器
    processor = AutoProcessor.from_pretrained("/mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct")
    
    # 步骤3: 测试前向传播
    inputs = test_model_forward(model, processor)
    
    # 步骤4: 测试生成能力
    test_model_generation(model, processor, inputs)
    
    # 步骤5: 测试专家路由
    test_expert_routing(model)

    # 步骤6: 打印模型参数
    print_model_parameters(model)
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    main()