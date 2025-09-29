from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer

from moellava.train.args import ModelArguments, DataArguments, TrainingArguments
from moellava.model import EvalMoEQwen2VLForConditionalGeneration, MoEQwen2VLForConditionalGeneration, MoEQwen2VLConfig


class DictToObject:
    """
    将字典及其嵌套字典转换为可以通过点操作符访问属性的对象。
    """
    def __init__(self, data_dict):
        if not isinstance(data_dict, dict):
            # 如果您期望输入总是字典，可以取消注释下一行
            # raise ValueError("Input must be a dictionary.")
            # 或者，如果它可能已经是所需的对象类型，则直接返回或处理
            if hasattr(data_dict, '__dict__'): # 简单检查它是否像一个对象
                 self.__dict__.update(data_dict.__dict__)
            return


        for key, value in data_dict.items():
            # 确保键是有效的属性名（例如，不含空格，不以数字开头等）
            # 为简单起见，此处未做严格的键名检查
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value)) # 递归转换嵌套字典
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    # 如果需要类似字典的 get 方法
    def get(self, key, default=None):
        return getattr(self, key, default)

    # 如果需要通过 [] 访问 (可选)
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e)) from e



def convert_dense_mask_to_small_expert(
    dense_mask_model_path: str,
    base_model_path: str, # e.g., qwen2-vl-2b-instruct (non-MoE base)
    output_path: str,
    device: str = "cpu"
):
    print(f"\n[Step 1] Loading dense_mask_expert model from: {dense_mask_model_path}")

    try:
        # 使用 EvalMoEQwen2VLForConditionalGeneration 是因为它在 __init__ 中有重新构建 MoE 层的逻辑，
        # 这有助于我们确保原始模型是按其配置正确加载的。
        original_config = MoEQwen2VLConfig.from_pretrained(dense_mask_model_path)
        dense_mask_model = EvalMoEQwen2VLForConditionalGeneration.from_pretrained(
            dense_mask_model_path,
            config=original_config,
        )
        dense_mask_model.eval()
    except Exception as e:
        print(f"Error loading dense_mask_expert model: {e}")
        print("Please ensure that EvalMoEQwen2VLForConditionalGeneration and MoEQwen2VLConfig are correctly defined")
        print("and can load your dense_mask_expert model structure and weights.")
        return
    print("Dense_mask_expert model loaded successfully.")

    print("\n[Step 2] Preparing configuration for the new small_expert model...")
    parser = transformers.HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]
    
    print(f'Type of model_args: {type(original_config)}')
    tmp_mone = original_config.mone
    tmp_moe = original_config.moe
    original_config.mone = DictToObject(original_config.mone)
    original_config.moe = DictToObject(original_config.moe)

    model_args.mone_enable = original_config.mone.mone_enable
    model_args.mone_expert_type = 'small_expert'
    model_args.mone_gate_type = original_config.mone.mone_gate_type
    model_args.mone_r = original_config.mone.mone_r
    model_args.mone_dropout = original_config.mone.mone_dropout
    model_args.mone_num_heads = original_config.mone.mone_num_heads
    model_args.mone_use_query_bn = original_config.mone.mone_use_query_bn
    model_args.mone_act_fn = original_config.mone.mone_act_fn
    model_args.mone_use_expert_gate = original_config.mone.mone_use_expert_gate
    model_args.mone_load_original = False
    model_args.mone_forward_mode = original_config.mone.mone_forward_mode

    model_args.moe_enable = original_config.moe.moe_enable
    model_args.moe_mode = original_config.moe.moe_mode
    model_args.moe_layers_idx = original_config.moe.moe_layers_idx

    model_args.num_experts = original_config.moe.num_experts
    model_args.top_k_experts = original_config.moe.top_k_experts
    model_args.use_shared_experts = original_config.moe.use_shared_experts

    model_args.train_modules = original_config.moe.train_modules
    model_args.ep_size = original_config.moe.ep_size
    model_args.capacity_factor = original_config.moe.capacity_factor
    model_args.eval_capacity_factor = original_config.moe.eval_capacity_factor
    model_args.min_capacity = original_config.moe.min_capacity
    model_args.use_residual = original_config.moe.use_residual
    model_args.router_aux_loss_coef = original_config.moe.router_aux_loss_coef
    model_args.use_combined_gate = original_config.moe.get('use_combined_gate', False)
    model_args.combined_gate_type = original_config.moe.get('combined_gate_type', 'ds')
    model_args.combined_gate_drop = original_config.moe.get('combined_gate_drop', 0.002)
    model_args.kd_align = original_config.moe.get('kd_align', False)
    structure = original_config.moe.get('structure', 'new')

    print(f"\n[Step 3] Initializing small_expert model structure from base: {base_model_path}")
    try:
        # Use MoEQwen2VLForConditionalGeneration for creating the new model
        small_expert_model = MoEQwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
        )
        # This call will replace standard MLP layers with MoE layers (with SmallExperts)
        # The experts' weights will be randomly initialized at this point.
        small_expert_model.initialize_moe_modules(model_args=model_args)
        small_expert_model.eval()
    except Exception as e:
        print(f"Error initializing small_expert model structure: {e}")
        print("Please ensure MoEQwen2VLForConditionalGeneration.from_pretrained and .initialize_moe_modules work correctly.")
        return
    print("Small_expert model structure initialized.")

    # --- Step 4: Copy weights from dense_mask_model to small_expert_model ---
    print("\n[Step 4] Copying weights...")
    source_state_dict = dense_mask_model.state_dict()
    target_state_dict = small_expert_model.state_dict() # This is a reference

    # print(f"  Name, Shape for source model:")
    # for name, param in source_state_dict.items():
    #     print(f"    {name}: {param.shape}")
    # print(f"  Name, Shape for target model:")
    # for name, param in target_state_dict.items():
    #     print(f"    {name}: {param.shape}")
    # return

    # Get hidden_size and expert_intermediate_dim from config for reshaping
    hidden_size = original_config.hidden_size # Should be consistent
    # model_args.mone_r is the expert_intermediate_dim for SmallExpert
    expert_intermediate_dim = model_args.mone_r
    if expert_intermediate_dim is None:
        print("ERROR: model_args.mone_r (expert_intermediate_dim) is not defined. Cannot proceed with expert weight copying.")
        return

    with torch.no_grad():
        # Part A: Copy non-MoE-expert weights and MoE router gate weights
        print("  Copying general weights (embeddings, lm_head, vision tower, non-expert MoE gates, etc.)...")
        for name, source_param in source_state_dict.items():
            is_dense_mask_internal_expert_param = False
            # Check if the parameter belongs to the flattened experts of DenseMaskMoE
            for layer_idx_cfg in original_config.moe.get('moe_layers_idx', []):
                # These are the flattened weights of the DenseMaskMoE experts.
                # Based on your provided keys:
                # model.layers.{idx}.mlp.moe.expert_gate.weight
                # model.layers.{idx}.mlp.moe.expert_down.weight
                # model.layers.{idx}.mlp.moe.expert_up.weight
                if f"model.layers.{layer_idx_cfg}.mlp" in name:
                    is_dense_mask_internal_expert_param = True
                    break
            
            if is_dense_mask_internal_expert_param:
                # print(f"    Skipping dense_mask internal expert weight: {name} (will be handled specifically)")
                continue

            if name in target_state_dict:
                if target_state_dict[name].shape == source_param.shape:
                    target_state_dict[name].data.copy_(source_param.data)
                    # print(f"    Copied: {name}")
                else:
                    print(f"    Shape mismatch for '{name}': source {source_param.shape}, target {target_state_dict[name].shape}. Skipping.")
            else:
                print(f"    Weight '{name}' from source not found in target model. Skipping.")

        # Part B: Copy MoE expert weights from DenseMaskMoE to SmallExperts
        print("  Copying MoE expert weights...")
        if not model_args.moe_layers_idx:
            print("    No MoE layers configured to copy expert weights for.")

        for layer_idx_pos, layer_idx in enumerate(model_args.moe_layers_idx):
            num_experts_in_this_layer = model_args.num_experts[layer_idx_pos]
            print(f"    Processing MoE Layer {layer_idx} with {num_experts_in_this_layer} experts, intermediate_dim {expert_intermediate_dim}, hidden_size {hidden_size}")

            if structure == 'new':
                moe_key = 'moe'
                shared_key = 'shared'
            else:
                moe_key = 'moe_layer'
                shared_key = 'original_mlp'
            for ele in ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']:
                source_shared = source_state_dict.get(f"model.layers.{layer_idx}.mlp.{shared_key}.{ele}")
                target_shared_key = f"model.layers.{layer_idx}.mlp.shared.{ele}"
                target_state_dict[target_shared_key].data.copy_(source_shared.data)
            source_gate = source_state_dict.get(f"model.layers.{layer_idx}.mlp.{moe_key}.gate.wg.weight")
            target_gate_key = f"model.layers.{layer_idx}.mlp.moe.deepspeed_moe.gate.wg.weight"
            target_state_dict[target_gate_key].data.copy_(source_gate.data)
            
            # Source keys for flattened expert weights in DenseMaskMoE
            # Based on your provided state_dict keys for dense_mask_expert
            src_expert_gate_key = f"model.layers.{layer_idx}.mlp.{moe_key}.expert_gate.weight"
            src_expert_down_key = f"model.layers.{layer_idx}.mlp.{moe_key}.expert_down.weight" # Corresponds to SmallExpert's up_proj
            src_expert_up_key   = f"model.layers.{layer_idx}.mlp.{moe_key}.expert_up.weight"   # Corresponds to SmallExpert's down_proj

            source_expert_gate_flat = source_state_dict.get(src_expert_gate_key)
            source_expert_down_flat = source_state_dict.get(src_expert_down_key)
            source_expert_up_flat   = source_state_dict.get(src_expert_up_key)

            # Check if essential source weights exist
            missing_weights = False
            if model_args.mone_use_expert_gate and source_expert_gate_flat is None:
                print(f"      WARNING: mone_use_expert_gate is True, but source weights {src_expert_gate_key} not found for DenseMaskMoE layer {layer_idx}.")
            if source_expert_down_flat is None:
                print(f"      ERROR: Source weights {src_expert_down_key} (for up_proj) not found for DenseMaskMoE layer {layer_idx}.")
                missing_weights = True
            if source_expert_up_flat is None:
                print(f"      ERROR: Source weights {src_expert_up_key} (for down_proj) not found for DenseMaskMoE layer {layer_idx}.")
                missing_weights = True
            
            if missing_weights:
                print(f"      Skipping expert weight copy for layer {layer_idx} due to missing source weights.")
                continue

            # Get the container for SmallExperts in the target model
            try:
                # Determine the path to the MoE block (which contains deepspeed_experts)
                # based on whether shared experts are used in the target model.
                if model_args.use_shared_experts:
                    # Assuming CombinedLayer stores the MoE block as 'moe_block'
                    # You might need to adjust 'moe_block' if your CombinedLayer uses a different name
                    moe_block_in_target = small_expert_model.model.layers[layer_idx].mlp.moe
                else:
                    moe_block_in_target = small_expert_model.model.layers[layer_idx].mlp
                
                target_experts_list = moe_block_in_target.deepspeed_moe.experts.deepspeed_experts
            except AttributeError as e:
                print(f"      ERROR: Could not access target_experts_list for layer {layer_idx}. Path incorrect or model structure changed. Error: {e}")
                print(f"      Attempted path involved: model.layers[{layer_idx}].mlp{'.moe_block' if model_args.use_shared_experts else ''}.deepspeed_moe.experts.deepspeed_experts")
                continue
            
            if len(target_experts_list) != num_experts_in_this_layer:
                print(f"      ERROR: Mismatch in number of experts for layer {layer_idx}. Source config: {num_experts_in_this_layer}, Target structure: {len(target_experts_list)}")
                continue

            for expert_idx in range(num_experts_in_this_layer):
                target_small_expert_module = target_experts_list[expert_idx] # This is a SmallExpert instance

                # 1. Copy to SmallExpert's gate_proj.weight [target shape: (expert_intermediate_dim, hidden_size)]
                if model_args.mone_use_expert_gate and source_expert_gate_flat is not None:
                    # source_expert_gate_flat[expert_idx] shape: (expert_intermediate_dim * hidden_size)
                    source_flat_slice = source_expert_gate_flat[expert_idx]
                    try:
                        reshaped_weight = source_flat_slice.view(hidden_size, expert_intermediate_dim).t()
                        if target_small_expert_module.gate_proj.weight.shape == reshaped_weight.shape:
                            target_small_expert_module.gate_proj.weight.data.copy_(reshaped_weight)
                        else:
                            print(f"        Shape mismatch for layer {layer_idx} expert {expert_idx} gate_proj: "
                                  f"Target {target_small_expert_module.gate_proj.weight.shape}, Reshaped Source {reshaped_weight.shape}")
                    except RuntimeError as e:
                        print(f"        RuntimeError reshaping gate_proj for layer {layer_idx} expert {expert_idx}: {e}")
                        print(f"        Source slice shape: {source_flat_slice.shape}, Target view: ({expert_intermediate_dim}, {hidden_size})")


                # 2. Copy to SmallExpert's up_proj.weight [target shape: (expert_intermediate_dim, hidden_size)]
                # source_expert_down_flat[expert_idx] shape: (expert_intermediate_dim * hidden_size)
                source_flat_slice = source_expert_down_flat[expert_idx]
                try:
                    reshaped_weight = source_flat_slice.view(hidden_size, expert_intermediate_dim).t()
                    if target_small_expert_module.up_proj.weight.shape == reshaped_weight.shape:
                        target_small_expert_module.up_proj.weight.data.copy_(reshaped_weight)
                    else:
                        print(f"        Shape mismatch for layer {layer_idx} expert {expert_idx} up_proj: "
                              f"Target {target_small_expert_module.up_proj.weight.shape}, Reshaped Source {reshaped_weight.shape}")
                except RuntimeError as e:
                    print(f"        RuntimeError reshaping up_proj for layer {layer_idx} expert {expert_idx}: {e}")
                    print(f"        Source slice shape: {source_flat_slice.shape}, Target view: ({expert_intermediate_dim}, {hidden_size})")


                # 3. Copy to SmallExpert's down_proj.weight [target shape: (hidden_size, expert_intermediate_dim)]
                # source_expert_up_flat[expert_idx] shape: (expert_intermediate_dim * hidden_size)
                # This was originally (expert_intermediate_dim, hidden_size) then flattened.
                # So, reshape to (expert_intermediate_dim, hidden_size) then transpose.
                source_flat_slice = source_expert_up_flat[expert_idx]
                try:
                    reshaped_weight_intermediate = source_flat_slice.view(expert_intermediate_dim, hidden_size)
                    final_reshaped_weight = reshaped_weight_intermediate.t() # Transpose
                    if target_small_expert_module.down_proj.weight.shape == final_reshaped_weight.shape:
                        target_small_expert_module.down_proj.weight.data.copy_(final_reshaped_weight)
                    else:
                        print(f"        Shape mismatch for layer {layer_idx} expert {expert_idx} down_proj: "
                              f"Target {target_small_expert_module.down_proj.weight.shape}, Reshaped Source {final_reshaped_weight.shape}")
                except RuntimeError as e:
                    print(f"        RuntimeError reshaping down_proj for layer {layer_idx} expert {expert_idx}: {e}")
                    print(f"        Source slice shape: {source_flat_slice.shape}, Intermediate view: ({expert_intermediate_dim}, {hidden_size})")

            print(f"      Attempted to copy weights for {num_experts_in_this_layer} experts in layer {layer_idx}.")
    
    print("Weight copying complete.")

    # --- Step 5: Save the converted model, processor, and tokenizer ---
    print(f"\n[Step 5] Saving converted model and tokenizer/processor to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        small_expert_model.save_pretrained(str(output_dir)) # save_pretrained expects a string
        print(f"  Converted model saved to {output_dir}")

        # Save processor and tokenizer from the original model's path
        # as they are generally compatible.
        print(f"  Loading processor/tokenizer from {dense_mask_model_path} to save alongside converted model...")
        processor = AutoProcessor.from_pretrained(
            dense_mask_model_path,
            max_pixels=(576 * 28 * 28), # Retain original settings if possible
            min_pixels=(16 * 28 * 28)
        )
        tokenizer = AutoTokenizer.from_pretrained(dense_mask_model_path)

        processor.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"  Processor and Tokenizer saved to {output_dir}")

        tmp_moe['structure'] = 'new'
        tmp_mone['mone_expert_type'] = 'small_expert'
        original_config.moe = tmp_moe
        original_config.mone = tmp_mone
        small_expert_model.config = original_config
        small_expert_model.config.save_pretrained(str(output_dir))
        print(f"  Config saved to {output_dir}")

    except Exception as e:
        print(f"Error during saving: {e}")
        return

    print("\nConversion process finished successfully!")


if __name__ == '__main__':
    # --- Configuration for the conversion ---
    # !!! IMPORTANT: Replace these paths with your actual model paths !!!

    # model_name = 'qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch'
    # 'qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch',
    # model_name_list = [
    #     'qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-5epoch',
    #     'qwen2-vl-2b-instruct-24e8-med-nano-ds-tok-share-5epoch',
    #     'qwen2-vl-2b-instruct-48e16-med-nano-ds-tok-share-5epoch',
    #     'qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch',
    #     'qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch',
    # ]

    # model_name_list = [
    #     'qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-ada-pre-5epoch',
    #     'qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-mmed-5epoch',
    #     'qwen2-vl-2b-instruct-96e32-ada-med-5epoch',
    #     'qwen2-vl-2b-instruct-96e32-mmed-med-5epoch'
    # ]

    model_name_list = [
        'qwen2-vl-2b-instruct-12e4-ada-nano-ds-tok-share-1epoch',
        'qwen2-vl-2b-instruct-12e4-mmed-nano-ds-tok-share-re-1epoch'
    ]

    for model_name in model_name_list:

        DENSE_MASK_EXPERT_MODEL_PATH = f'/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/{model_name}'
        BASE_PRETRAINED_MODEL_PATH = '/mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct' # e.g., the original Qwen2-VL-2B-Instruct
        if 'ds' in model_name: 
            OUTPUT_CONVERTED_MODEL_PATH = DENSE_MASK_EXPERT_MODEL_PATH.replace('-ds-', '-ds2s-') # Example output path
        else:
            OUTPUT_CONVERTED_MODEL_PATH = DENSE_MASK_EXPERT_MODEL_PATH + '-ds2s'
        import os
        if os.path.exists(OUTPUT_CONVERTED_MODEL_PATH):
            print(f"Output path {OUTPUT_CONVERTED_MODEL_PATH} already exists. Skipping conversion.")
            continue

        # Before running, ensure all custom classes like MoEQwen2VLForConditionalGeneration, SmallExpert,
        # MoEQwen2VLConfig, etc., are correctly defined or imported above this `if __name__ == '__main__':` block.
        # You also need to have `ModelArguments`, `DataArguments`, `TrainingArguments` defined if your
        # `MoEQwen2VLForConditionalGeneration.initialize_moe_modules` or other parts rely on them directly.
        # For this script, I've made `model_args` a simple temporary class.

        print("="*50)
        print("MoE Model Conversion Script: dense_mask_expert -> small_expert")
        print("="*50)
        print(f"Source (dense_mask_expert) model: {DENSE_MASK_EXPERT_MODEL_PATH}")
        print(f"Base model for structure: {BASE_PRETRAINED_MODEL_PATH}")
        print(f"Output (small_expert) model: {OUTPUT_CONVERTED_MODEL_PATH}")
        print("="*50)
        
        # Check if paths exist (optional but good practice)
        if not Path(DENSE_MASK_EXPERT_MODEL_PATH).exists():
            print(f"ERROR: Source model path does not exist: {DENSE_MASK_EXPERT_MODEL_PATH}")
        elif not Path(BASE_PRETRAINED_MODEL_PATH).exists():
            print(f"ERROR: Base model path does not exist: {BASE_PRETRAINED_MODEL_PATH}")
        else:
            # Execute the conversion
            convert_dense_mask_to_small_expert(
                dense_mask_model_path=DENSE_MASK_EXPERT_MODEL_PATH,
                base_model_path=BASE_PRETRAINED_MODEL_PATH,
                output_path=OUTPUT_CONVERTED_MODEL_PATH
            )
        # print("\nINFO: Conversion call is commented out. Please ensure:")
        # print("1. All necessary class definitions (MoEQwen2VLForConditionalGeneration, SmallExpert, etc.) are provided or imported correctly.")
        # print("2. The paths for DENSE_MASK_EXPERT_MODEL_PATH, BASE_PRETRAINED_MODEL_PATH, and OUTPUT_CONVERTED_MODEL_PATH are correct.")
        # print("3. Your environment has all required packages (torch, transformers, etc.).")
        # print("4. The `initialize_moe_modules` method in `MoEQwen2VLForConditionalGeneration` correctly uses `model_args.mone_expert_type = 'small_expert'` to build SmallExperts.")
        # print("5. The structure `model.layers[idx].mlp.deepspeed_moe.experts.deepspeed_experts` is correct for accessing the list of expert modules in your target model.")
        # print("Uncomment the 'convert_dense_mask_to_small_expert(...)' call above to run the script.")



# import torch

# # 假设 dense_mask_model2 和 dense_mask_model3 已经加载
# # model_path = "你的模型路径"
# # dense_mask_model2 = EvalMoEQwen2VLForConditionalGeneration.from_pretrained(model_path)
# # dense_mask_model3 = EvalMoEQwen2VLForConditionalGeneration.from_pretrained(model_path)

# def compare_models_plus(model1, model2):
#     state_dict1 = model1.state_dict()
#     state_dict2 = model2.state_dict()

#     # if len(state_dict1) != len(state_dict2):
#     #     print("模型参数数量不一致！")
#     #     return False

#     for key in state_dict1:
#         is_dense_mask_internal_expert_param = False
#         for layer_idx_cfg in original_config.moe.get('moe_layers_idx', []):
#                 # These are the flattened weights of the DenseMaskMoE experts.
#                 # Based on your provided keys:
#                 # model.layers.{idx}.mlp.moe.expert_gate.weight
#                 # model.layers.{idx}.mlp.moe.expert_down.weight
#                 # model.layers.{idx}.mlp.moe.expert_up.weight
#                 if f"model.layers.{layer_idx_cfg}.mlp" in key:
#                     print(f"Skipping expert parameter {key} for comparison.")
#                     is_dense_mask_internal_expert_param = True
#                     break
#         if is_dense_mask_internal_expert_param:
#             continue # Skip expert parameters for comparison.
#         if key not in state_dict2:
#             print(f"参数 {key} 在 model2 中不存在！")
#             return False
#         if not torch.equal(state_dict1[key], state_dict2[key]):
#             print(f"参数 {key} 的值不一致！")
#             # 你可以进一步打印出不一致的张量以供调试
#             # print("Model 1 tensor:", state_dict1[key])
#             # print("Model 2 tensor:", state_dict2[key])
#             return False
#         else:
#             print(f"参数 {key} 相同！")
#     return True

# def compare_models_plus(model1, model2):
#     state_dict1 = model1.state_dict()
#     state_dict2 = model2.state_dict()

#     # if len(state_dict1) != len(state_dict2):
#     #     print("模型参数数量不一致！")
#     #     return False

#     for key in state_dict1:
#         if 'moe_layer.gate' in key:
#             key1 = key
#             key2 = key.replace('moe_layer', 'moe.deepspeed_moe')
#         else:
#             key1 = key
#             key2 = key
#             continue
#         # is_dense_mask_internal_expert_param = False
#         # for layer_idx_cfg in original_config.moe.get('moe_layers_idx', []):
#         #         # These are the flattened weights of the DenseMaskMoE experts.
#         #         # Based on your provided keys:
#         #         # model.layers.{idx}.mlp.moe.expert_gate.weight
#         #         # model.layers.{idx}.mlp.moe.expert_down.weight
#         #         # model.layers.{idx}.mlp.moe.expert_up.weight
#         #         if f"model.layers.{layer_idx_cfg}.mlp" in key:
#         #             print(f"Skipping expert parameter {key} for comparison.")
#         #             is_dense_mask_internal_expert_param = True
#         #             break
#         # if is_dense_mask_internal_expert_param:
#         #     continue # Skip expert parameters for comparison.
#         # if key not in state_dict2:
#         #     print(f"参数 {key} 在 model2 中不存在！")
#         #     return False
#         if not torch.equal(state_dict1[key1], state_dict2[key2]):
#             print(f"参数 {key} 的值不一致！")
#             # 你可以进一步打印出不一致的张量以供调试
#             # print("Model 1 tensor:", state_dict1[key])
#             # print("Model 2 tensor:", state_dict2[key])
#             return False
#         else:
#             print(f"参数 {key} 相同！")
#     return True


# def compare_models_expert(model1, model2):
#     state_dict1 = model1.state_dict()
#     state_dict2 = model2.state_dict()

#     # if len(state_dict1) != len(state_dict2):
#     #     print("模型参数数量不一致！")
#     #     return False

#     for key in state_dict1:
#         if 'moe_layer.expert_gate' in key:
#             layer = int(key.split('.')[2])
#             value1 = state_dict1[key]
#             for i in range(6):
#                 expert_i = value1[i]
#                 vaule1 =  expert_i.view(1536, 4480).t()
#                 key2 = f'model.layers.{layer}.mlp.moe.deepspeed_moe.experts.deepspeed_experts.{i}.gate_proj.weight'
#                 value2 = state_dict2[key2]
#                 if not torch.equal(vaule1, value2):
#                     print(f"参数 {key2} 的值不一致！")
#                     # 你可以进一步打印出不一致的张量以供调试
#                     # print("Model 1 tensor:", state_dict1[key])
#                     # print("Model 2 tensor:", state_dict2[key])
#                     return False
#                 else:
#                     print(f"参数 {key2} 相同！")
#         elif 'moe_layer.expert_down' in key:
#             layer = int(key.split('.')[2])
#             value1 = state_dict1[key]
#             for i in range(6):
#                 expert_i = value1[i]
#                 vaule1 =  expert_i.view(1536, 4480).t()
#                 key2 = f'model.layers.{layer}.mlp.moe.deepspeed_moe.experts.deepspeed_experts.{i}.up_proj.weight'
#                 value2 = state_dict2[key2]
#                 if not torch.equal(vaule1, value2):
#                     print(f"参数 {key2} 的值不一致！")
#                     # 你可以进一步打印出不一致的张量以供调试
#                     # print("Model 1 tensor:", state_dict1[key])
#                     # print("Model 2 tensor:", state_dict2[key])
#                     return False
#                 else:
#                     print(f"参数 {key2} 相同！")
#         elif 'moe_layer.expert_up' in key:
#             layer = int(key.split('.')[2])
#             value1 = state_dict1[key]
#             for i in range(6):
#                 expert_i = value1[i]
#                 vaule1 =  expert_i.view(4480, 1536).t()
#                 key2 = f'model.layers.{layer}.mlp.moe.deepspeed_moe.experts.deepspeed_experts.{i}.down_proj.weight'
#                 value2 = state_dict2[key2]
#                 if not torch.equal(vaule1, value2):
#                     print(f"参数 {key2} 的值不一致！")
#                     # 你可以进一步打印出不一致的张量以供调试
#                     # print("Model 1 tensor:", state_dict1[key])
#                     # print("Model 2 tensor:", state_dict2[key])
#                     return False
#                 else:
#                     print(f"参数 {key2} 相同！")
#     return True



