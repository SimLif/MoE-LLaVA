def replace_specific_moe_layer(model_a, model_b, layer_indices):
    """
    只替换指定层的MoE专家
    
    Args:
        model_a: 包含MoE层的模型
        model_b: 包含常规MLP层的模型
        layer_indices: 要替换的层索引列表，如[2, 4, 6]
    """
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    
    for i in layer_indices:
        # 构建参数名
        expert0_w1_name = f'transformer.h.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.0.w1.weight'
        expert0_w2_name = f'transformer.h.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.0.w2.weight'
        expert0_c_proj_name = f'transformer.h.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.0.c_proj.weight'
        
        mlp_w1_name = f'transformer.h.{i}.mlp.w1.weight'
        mlp_w2_name = f'transformer.h.{i}.mlp.w2.weight'
        mlp_c_proj_name = f'transformer.h.{i}.mlp.c_proj.weight'
        
        # 进行替换
        if all(name in state_dict_a for name in [expert0_w1_name, expert0_w2_name, expert0_c_proj_name]) and \
           all(name in state_dict_b for name in [mlp_w1_name, mlp_w2_name, mlp_c_proj_name]):
            
            state_dict_a[expert0_w1_name] = state_dict_b[mlp_w1_name].clone()
            state_dict_a[expert0_w2_name] = state_dict_b[mlp_w2_name].clone()
            state_dict_a[expert0_c_proj_name] = state_dict_b[mlp_c_proj_name].clone()
            print(f"已替换层 {i} 的MoE专家0")
        else:
            print(f"层 {i} 的某些参数不存在，无法替换")
    
    model_a.load_state_dict(state_dict_a)
    return model_a