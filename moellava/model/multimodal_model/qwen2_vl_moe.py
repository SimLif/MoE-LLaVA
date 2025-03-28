import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import DynamicCache, Cache
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, \
    _prepare_4d_causal_attention_mask_for_sdpa
from transformers import Qwen2VLConfig, Qwen2VLModel, Qwen2VLForConditionalGeneration

from transformers.modeling_outputs import CausalLMOutputWithPast

from deepspeed.moe.layer import MoE
from deepspeed.moe.sharded_moe import TopKGate
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_vl.modeling_qwen2_vl import logger
from transformers.utils import ModelOutput

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# 定义小专家类
class SmallExpert(nn.Module):
    def __init__(self, hidden_size, r, dropout_rate=0.0):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, r, bias=False)
        self.act_fn = nn.SiLU()  # 与 Qwen2 MLP 保持一致
        self.dropout = nn.Dropout(dropout_rate)  # 可选的 dropout
        self.up_proj = nn.Linear(r, hidden_size, bias=False)
        
    def forward(self, x):
        x = self.down_proj(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x
    

# 定义组合层，结合原始MLP和MoE
class CombinedLayer(nn.Module):
    def __init__(self, original_mlp, moe_layer):
        super().__init__()
        self.original_mlp = original_mlp
        self.moe_layer = moe_layer
        
    def forward(self, x):
        mlp_out = self.original_mlp(x)
        moe_out = self.moe_layer(x)
        # 处理MoE返回的(output, loss)元组
        if isinstance(moe_out, tuple) and len(moe_out) >= 2:
            return mlp_out + moe_out[0], moe_out[1]
        else:
            return mlp_out + moe_out, None


try:
    from deepspeed.utils import logger
    from deepspeed.utils.timer import SynchronizedWallClockTimer
except ImportError:
    # 创建简单的替代品以允许在没有DeepSpeed的情况下使用
    class DummyLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    logger = DummyLogger()
    
    class SynchronizedWallClockTimer:
        def __init__(self): pass
        def __call__(self, name): return self
        def start(self): pass
        def stop(self): pass
        def elapsed(self, reset=True): return 0.0

def _one_hot_to_float(indices, num_classes):
    """将整数索引转换为one-hot浮点表示"""
    device = indices.device
    indices_shape = list(indices.shape)
    reshaped_indices = indices.reshape(-1)
    
    one_hot = torch.zeros(reshaped_indices.shape[0], num_classes,
                         device=device, dtype=torch.float)
    one_hot.scatter_(1, reshaped_indices.unsqueeze(1), 1)
    one_hot = one_hot.reshape(indices_shape + [num_classes])
    return one_hot

class TopKGateAdapter(nn.Module):
    """TopKGate到SimilarityGate接口的轻量级适配器"""
    
    def __init__(self, model_dim, num_experts, k=1, **kwargs):
        super().__init__()
        
        # 过滤相关参数
        gate_kwargs = {
            'model_dim': model_dim, 
            'num_experts': num_experts,
            'k': k
        }
        
        # 添加TopKGate支持的其他参数
        for param in ['capacity_factor', 'eval_capacity_factor', 'min_capacity',
                      'noisy_gate_policy', 'drop_tokens', 'use_rts', 'ep_group', 
                      'top2_2nd_expert_sampling']:
            if param in kwargs:
                gate_kwargs[param] = kwargs[param]
        
        # 创建原始TopKGate
        self.gate = TopKGate(**gate_kwargs)
        self.num_experts = num_experts
        self.k = k
    
    def _set_ep_group(self, ep_group):
        self.gate._set_ep_group(ep_group)
    
    def forward(self, x, used_token=None):
        """适配TopKGate输出到SimilarityGate格式"""
        # 获取原始输出
        # print(f'type of x: {type(x)}')
        # print(f'x shape: {x.shape}')
        # d_model = x[0].shape[-1]
        # reshaped_x = x[0].reshape(-1, d_model)
        d_model = x.shape[-1]
        reshaped_x = x.reshape(-1, d_model)
        outputs = self.gate(reshaped_x, used_token)
        l_aux, combine_weights, dispatch_mask, exp_counts = outputs
        
        # 从dispatch_mask提取专家索引
        # dispatch_mask形状为 [batch_tokens, num_experts, capacity]
        batch_tokens = dispatch_mask.size(0)
        
        # 计算每个专家的总权重
        expert_weights = dispatch_mask.sum(dim=-1)  # [batch_tokens, num_experts]
        
        # 使用topk获取每个token的前k个专家
        k_to_use = min(self.k, expert_weights.size(1))
        _, top_indices = torch.topk(
            expert_weights, 
            k=k_to_use,
            dim=1, 
            largest=True
        )  # [batch_tokens, k_to_use]
        
        # 创建最终的专家索引张量，确保大小为 [batch_tokens, k]
        expert_indices = torch.zeros(
            (batch_tokens, self.k), 
            dtype=torch.long, 
            device=dispatch_mask.device
        )
        
        # 填充真实的专家索引
        if k_to_use > 0:  # 确保有专家被选择
            expert_indices[:, :k_to_use] = top_indices
        
        return l_aux, combine_weights, dispatch_mask, exp_counts, expert_indices


class SimilarityGate(nn.Module):
    """
    基于产品键的专家选择门控网络
    使用两个子键集合，为输入分配到高维专家空间
    """
    
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        k: int = 16,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        drop_tokens: bool = True,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        num_heads: int = 8,
        use_query_bn: bool = True,
    ) -> None:
        super().__init__()
        
        # 确保num_experts是完全平方数
        sqrt_experts = int(math.sqrt(num_experts))
        assert sqrt_experts * sqrt_experts == num_experts, f"专家数量必须是完全平方数，而不是{num_experts}"
        
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.sqrt_experts = sqrt_experts
        self.k = k
        self.num_heads = num_heads
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
        self.ep_group = ep_group
        
        # 查询投影网络 - 产生用于专家选择的查询向量
        query_dim = model_dim // num_heads * 2
        self.query_proj = nn.Linear(model_dim, query_dim * num_heads, bias=False)
        
        # 可选的查询批量归一化以提高稳定性
        self.use_query_bn = use_query_bn
        if use_query_bn:
            self.query_bn = nn.BatchNorm1d(query_dim * num_heads)
        
        # 初始化产品子键 - 这些是用于快速检索的两组子键
        key_dim = query_dim // 2
        self.register_parameter(
            "sub_keys1", 
            nn.Parameter(torch.randn(sqrt_experts, model_dim // num_heads) / math.sqrt(model_dim // num_heads))
        )
        self.register_parameter(
            "sub_keys2", 
            nn.Parameter(torch.randn(sqrt_experts, model_dim // num_heads) / math.sqrt(model_dim // num_heads))
        )
        
        # 计时器用于性能分析
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
    
    def _set_ep_group(self, ep_group):
        """设置专家并行组"""
        assert self.ep_group is None, '尝试覆盖已存在的ep_group'
        self.ep_group = ep_group
    
    def forward(self, input: torch.Tensor, used_token: Optional[torch.Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """专家选择的前向传播
        
        参数:
            input: 形状为[batch_size, seq_len, hidden_dim]的输入张量
            used_token: 可选的掩码，指示要处理的有效token
            
        返回:
            l_aux: 负载均衡损失
            combine_weights: 路由权重
            dispatch_mask: 调度掩码
            exp_counts: 每个专家被选择的次数
        """
        if self.wall_clock_breakdown:
            self.timers("gate_timer").start()
        
        batch_size, seq_len, _ = input.shape
        batch_tokens = batch_size * seq_len
        
        # 应用token掩码（如果提供）
        if used_token is not None:
            input = input * used_token.unsqueeze(-1)
            
        # 扁平化批次和序列维度
        flattened_input = input.reshape(-1, self.model_dim)
        
        # 投影输入到查询空间
        query = self.query_proj(flattened_input)
        
        # 应用批量归一化（如果启用）
        if self.use_query_bn and self.training:
            query = self.query_bn(query)
        
        # 重塑查询以进行多头处理
        d_key = query.size(1)
        query = query.view(batch_tokens, self.num_heads, d_key // self.num_heads)
        
        # 分割查询用于产品键检索
        d_half = d_key // (2 * self.num_heads)
        query1, query2 = query[..., :d_half], query[..., d_half:]
        
        # 使用子键计算相似度分数
        scores1 = torch.einsum('bhd,ed->bhe', query1, self.sub_keys1)
        scores2 = torch.einsum('bhd,ed->bhe', query2, self.sub_keys2)

        # 使用sqrt(k)作为每个维度的选择数量，但确保不超过可用的子键数量
        subkey_k = min(int(math.ceil(math.sqrt(self.k))), self.sqrt_experts)
        
        # 获取每个子键集的top-k索引
        top_scores1, top_indices1 = torch.topk(scores1, k=subkey_k, dim=-1)
        top_scores2, top_indices2 = torch.topk(scores2, k=subkey_k, dim=-1)
        
        # 优化版本：使用批量化张量操作代替嵌套循环
        # 计算笛卡尔积的分数 - 使用广播计算
        combined_scores = top_scores1.unsqueeze(-1) + top_scores2.unsqueeze(-2)  # [batch_tokens, num_heads, subkey_k, subkey_k]
        # 重塑并找到每个(token,head)的top-k
        flat_scores = combined_scores.reshape(batch_tokens, self.num_heads, -1)  # [batch_tokens, num_heads, subkey_k*subkey_k]
        final_k = min(self.k, subkey_k * subkey_k)
        top_k_scores, top_k_indices = torch.topk(flat_scores, k=final_k, dim=-1)  # [batch_tokens, num_heads, final_k]
        
        # 计算原始subkey_k*subkey_k网格中的行列索引
        i1_indices = top_k_indices // subkey_k  # 获取行索引
        i2_indices = top_k_indices % subkey_k   # 获取列索引
        
        # 使用这些索引获取实际的子键索引
        selected_i1 = torch.gather(top_indices1, dim=2, index=i1_indices)  # [batch_tokens, num_heads, final_k]
        selected_i2 = torch.gather(top_indices2, dim=2, index=i2_indices)  # [batch_tokens, num_heads, final_k]
        
        # 计算最终的专家索引
        indices = selected_i1 * self.sqrt_experts + selected_i2  # [batch_tokens, num_heads, final_k]
        scores = top_k_scores  # 直接使用topk返回的分数
        
        # 扁平化结果
        flat_indices = indices.reshape(batch_tokens, -1)  # [batch_tokens, num_heads*final_k]
        flat_scores = scores.reshape(batch_tokens, -1)    # [batch_tokens, num_heads*final_k]

        num_selected = self.num_heads * final_k
        
        # 应用softmax获取路由概率
        gates = F.softmax(flat_scores, dim=-1)
        
        # 优化: 向量化计算专家计数
        if used_token is not None:
            # 创建扁平化的有效token掩码
            flat_mask = used_token.reshape(-1)
            # 只选择有效token的专家索引
            valid_indices = flat_indices[flat_mask.bool()]
        else:
            valid_indices = flat_indices.reshape(-1)
        
        # 使用bincount快速计算专家计数
        exp_counts = torch.zeros(self.num_experts, device=input.device)
        for i in range(num_selected):
            if used_token is not None:
                counts = torch.bincount(valid_indices[:, i], minlength=self.num_experts)
            else:
                counts = torch.bincount(flat_indices[:, i], minlength=self.num_experts)
            exp_counts += counts
        
        # 计算负载均衡损失 - 更高效的实现
        valid_tokens = batch_tokens if used_token is None else used_token.sum().item()
        
        # 预计算常量
        me = torch.zeros(self.num_experts, dtype=gates.dtype, device=input.device)
        ce = torch.zeros(self.num_experts, dtype=gates.dtype, device=input.device)
        
        # 使用scatter_add_进行向量化计算
        token_indices = torch.arange(batch_tokens, device=input.device)
        token_indices = token_indices.repeat_interleave(num_selected)
        expert_indices = flat_indices.reshape(-1)
        
        if used_token is not None:
            # 只处理有效token
            valid_mask = used_token.reshape(-1).repeat_interleave(num_selected).bool()
            token_indices = token_indices[valid_mask]
            expert_indices = expert_indices[valid_mask]
            gate_values = gates.reshape(-1)[valid_mask]
        else:
            gate_values = gates.reshape(-1)
        
        # 向量化更新me
        me.scatter_add_(0, expert_indices, gate_values / valid_tokens)
        
        # 更新ce - 每个专家分配的计数
        ce.scatter_add_(0, expert_indices, 
                    torch.ones_like(expert_indices, dtype=gate_values.dtype) / (valid_tokens * num_selected))
        
        # 计算负载均衡损失
        l_aux = torch.mean(me * ce) * self.num_experts * self.num_experts / num_selected
        
        # 创建掩码用于容量处理
        mask = torch.ones_like(gates, dtype=torch.bool)
        
        # 处理专家容量约束 - 这部分很难有效向量化，因为有顺序依赖
        if self.drop_tokens:
            capacity_factor = self.capacity_factor if self.training else self.eval_capacity_factor
            capacity = max(self.min_capacity, 
                        int(capacity_factor * valid_tokens * final_k / self.num_experts))
            
            # 这部分仍需要循环，但可以减少item()调用
            expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=input.device)
            
            # 创建一个专家索引的副本，避免多次调用item()
            expert_indices_cpu = flat_indices.cpu()
            
            for i in range(batch_tokens):
                if used_token is None or used_token.reshape(-1)[i]:
                    for j in range(num_selected):
                        expert_idx = expert_indices_cpu[i, j].item()
                        if expert_counts[expert_idx] < capacity:
                            expert_counts[expert_idx] += 1
                        else:
                            mask[i, j] = False
        else:
            capacity = torch.max(exp_counts).long()
            if self.ep_group is not None:
                dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self.ep_group)
        
        # 重新归一化被掩码的门控值
        gates_masked = gates * mask
        gate_sums = torch.sum(gates_masked, dim=-1, keepdim=True)
        gate_sums = torch.clamp(gate_sums, min=torch.finfo(gates.dtype).eps)
        gates = gates_masked / gate_sums
        
        # 计算位置 - 这部分也很难向量化
        locations = torch.zeros_like(flat_indices)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=input.device)
        
        # 同样使用CPU索引减少GPU-CPU同步
        mask_cpu = mask.cpu()
        
        for i in range(batch_tokens):
            if used_token is None or used_token.reshape(-1)[i]:
                for j in range(num_selected):
                    if mask_cpu[i, j]:
                        expert_idx = expert_indices_cpu[i, j].item()
                        locations[i, j] = expert_counts[expert_idx]
                        expert_counts[expert_idx] += 1
        
        # 创建one-hot位置表示
        locations_sc = _one_hot_to_float(locations * mask, capacity)
        combine_weights = torch.einsum("se,sec->sec", gates, locations_sc)
        dispatch_mask = combine_weights.bool()
        
        return l_aux, combine_weights, dispatch_mask, exp_counts, flat_indices


class EmbeddedExpertsMoE(nn.Module):
    """
    基于嵌入式存储的大规模MoE实现
    使用产品键专家选择和嵌入表参数存储
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 1024,  # 默认值较小，可以设置更大的值
        expert_dim: int = 1,      # 单神经元专家的内部维度
        k: int = 16,              # 每个token选择的专家数
        gate_type: str = "topk",  # 专家选择门控类型
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        use_residual: bool = False,  # 是否添加残差连接
        num_heads: int = 8,          # 专家选择的头数
        use_query_bn: bool = True,   # 是否在查询中使用批量归一化
        act_fn: str = "silu",        # 激活函数类型
        bias: bool = False,          # 是否使用偏置
        dropout: float = 0.0,        # Dropout率
        init_scale: float = 1.0,     # 初始化缩放
        expert_parallel: bool = False, # 是否并行化专家
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.k = k
        self.use_residual = use_residual
        self.expert_parallel = expert_parallel
        self.ep_group = ep_group
        
        # 确保num_experts是完全平方数
        sqrt_experts = int(math.sqrt(num_experts))
        if sqrt_experts * sqrt_experts != num_experts:
            # 找到最接近的完全平方数
            new_sqrt = int(sqrt_experts) + (1 if sqrt_experts != int(sqrt_experts) else 0)
            num_experts = new_sqrt ** 2
            logger.warning(f"调整专家数量为最接近的完全平方数: {num_experts}")
            self.num_experts = num_experts
        
        # 专家选择门控网络
        if gate_type == "token_gating":
            self.gate = TopKGateAdapter(
                model_dim=hidden_size,
                num_experts=num_experts,
                k=k,
                capacity_factor=capacity_factor,
                eval_capacity_factor=eval_capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=True,
                ep_group=ep_group,
                top2_2nd_expert_sampling=True,
            )
        elif gate_type == "similarity_gating":
            self.gate = SimilarityGate(
                model_dim=hidden_size,
                num_experts=num_experts,
                k=k,
                capacity_factor=capacity_factor,
                eval_capacity_factor=eval_capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=True,
                ep_group=ep_group,
                num_heads=num_heads,
                use_query_bn=use_query_bn,
            )
        else:
            raise ValueError(f"不支持的门控类型: {gate_type}")
        
        # 嵌入式专家参数
        self.expert_down = nn.Embedding(num_experts, hidden_size * expert_dim).to(dtype=dtype)
        self.expert_up = nn.Embedding(num_experts, expert_dim * hidden_size).to(dtype=dtype)
        
        # 设置激活函数
        if act_fn == "relu":
            self.activation = F.relu
        elif act_fn == "gelu":
            self.activation = F.gelu
        elif act_fn == "silu" or act_fn == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"不支持的激活函数: {act_fn}")
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 可选的残差系数
        if use_residual:
            self.coefficient = nn.Linear(hidden_size, 2)
        
        # 初始化参数
        with torch.no_grad():
            # 使用高斯初始化下投影权重
            std = math.sqrt(2.0 / (hidden_size + expert_dim)) * init_scale
            nn.init.normal_(self.expert_down.weight, mean=0.0, std=std)
            
            # 使用高斯初始化上投影权重
            std = math.sqrt(1.0 / hidden_size) * init_scale
            nn.init.normal_(self.expert_up.weight, mean=0.0, std=std)
    
    def forward(self, hidden_states: torch.Tensor, used_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MoE前向传播
        
        参数:
            hidden_states: 形状为[batch_size, seq_len, hidden_size]的输入
            used_token: 可选的掩码指示有效token
            
        返回:
            output: 模型输出
            l_aux: 负载均衡损失
            exp_counts: 专家计数
        """
        # 保存原始形状和设备
        original_shape = hidden_states.shape
        batch_size, seq_len, hidden_size = original_shape
        device = hidden_states.device
        input_dtype = hidden_states.dtype

        # 获取专家路由信息
        l_aux, combine_weights, dispatch_mask, exp_counts, expert_indices = self.gate(hidden_states, used_token)
        combine_weights = combine_weights.to(dtype=input_dtype)
        dispatch_mask = dispatch_mask.to(dtype=input_dtype)        

        # 扁平化输入
        hidden_states = hidden_states.reshape(-1, hidden_size)  # [batch*seq, hidden]
        batch_tokens = hidden_states.shape[0]
        
        # 提取路由相关尺寸
        num_selected_experts = expert_indices.shape[1]  # num_heads * k
        
        # 检索和处理专家参数
        # 对每个token，我们只获取该token选择的专家的参数
        
        # 计算专家输出
        outputs = torch.zeros((batch_tokens, hidden_size), device=device, dtype=hidden_states.dtype)
        
        # 使用选定的专家处理每个token
        # flat_mask = dispatch_mask.reshape(-1).bool()
        # flat_indices = expert_indices.reshape(-1)[flat_mask]
        # flat_locs = torch.nonzero(flat_mask).squeeze(1)
        # active_positions = torch.nonzero(dispatch_mask)
        # token_indices = active_positions[:, 0]
        # expert_indices_from_mask = active_positions[:, 1]
        # capacity_indices = active_positions[:, 2]
        
        # 使用选定的专家处理每个token
        active_positions = torch.nonzero(dispatch_mask)
        if len(active_positions) > 0:  # 确保有选定的专家
            # 提取活跃位置的索引
            token_indices = active_positions[:, 0]
            expert_indices_from_mask = active_positions[:, 1]
            capacity_indices = active_positions[:, 2]
            
            # 获取对应token的隐藏状态
            flat_hidden = hidden_states[token_indices]
            
            # 获取专家参数 - 使用从dispatch_mask获取的专家索引
            expert_down_w = self.expert_down(expert_indices_from_mask)  # [active_tokens, hidden*expert_dim]
            expert_up_w = self.expert_up(expert_indices_from_mask)  # [active_tokens, expert_dim*hidden]
            
            # 重塑为矩阵形式
            expert_down_w = expert_down_w.view(-1, hidden_size, self.expert_dim)
            expert_up_w = expert_up_w.view(-1, self.expert_dim, hidden_size)
            
            # 计算中间激活
            intermediate = torch.bmm(flat_hidden.unsqueeze(1), expert_down_w).squeeze(1)
            intermediate = self.activation(intermediate)
            intermediate = self.dropout(intermediate)
            
            # 计算输出
            expert_outputs = torch.bmm(intermediate.unsqueeze(1), expert_up_w).squeeze(1)
            
            # 获取组合权重 - 使用活跃位置直接索引combine_weights
            flat_weights = combine_weights[token_indices, expert_indices_from_mask, capacity_indices].unsqueeze(1)
            
            # 准备用于scatter_add的索引
            token_indices_expanded = token_indices.unsqueeze(1).expand(-1, hidden_size)
            weighted_outputs = flat_weights * expert_outputs
            
            # 使用scatter_add累加输出
            outputs.scatter_add_(0, token_indices_expanded, weighted_outputs)
        
        # 重塑回原始形状
        outputs = outputs.reshape(original_shape)
        
        # 如果使用残差连接
        if self.use_residual:
            # 通过标准MLP计算残差
            coef = self.coefficient(hidden_states.reshape(original_shape))
            coef = F.softmax(coef, dim=-1)
            outputs = outputs * coef[..., 0:1] + hidden_states.reshape(original_shape) * coef[..., 1:]
        
        return outputs, l_aux, exp_counts


class EmbeddedMoELayer(nn.Module):
    """
    嵌入式MoE层，可作为标准Transformer层的替代品
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 1024,
        expert_dim: int = 1,
        k: int = 16,
        gate_type: str = "topk",
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        use_residual: bool = True,
        num_heads: int = 8,
        use_query_bn: bool = True,
        act_fn: str = "silu",
        bias: bool = False,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        norm_type: str = "layernorm",
        use_pre_norm: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.use_pre_norm = use_pre_norm
        
        # 归一化层
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm_type == "rmsnorm":
            # 简单的RMSNorm实现
            from functools import partial
            def rms_norm(x, eps=1e-5):
                return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            self.norm = partial(rms_norm)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        # MoE层
        self.moe = EmbeddedExpertsMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            expert_dim=expert_dim,
            k=k,
            gate_type=gate_type,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            num_heads=num_heads,
            use_query_bn=use_query_bn,
            act_fn=act_fn,
            bias=bias,
            dropout=dropout,
            init_scale=init_scale,
            dtype=dtype,
        )
    
    def forward(self, hidden_states: torch.Tensor, used_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """层前向传播
        
        参数:
            hidden_states: 形状为[batch_size, seq_len, hidden_size]的输入
            used_token: 可选的掩码指示有效token
            
        返回:
            output: 层输出
            l_aux: 负载均衡损失
            exp_counts: 专家计数
        """
        if self.use_pre_norm:
            # Pre-LN风格：先应用归一化，再处理
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states, l_aux, exp_counts = self.moe(hidden_states, used_token)
            output = residual + hidden_states
        else:
            # Post-LN风格：先处理，再应用归一化
            residual = hidden_states
            hidden_states, l_aux, exp_counts = self.moe(hidden_states, used_token)
            output = self.norm(residual + hidden_states)
        
        return output, l_aux, exp_counts


class MoEQwen2VLConfig(Qwen2VLConfig):
    model_type = "moe_qwen2_vl"

    def __init__(self,
                 moe_enable=True,
                 moe_mode='sparse',
                 moe_layers_idx=None,
                 ep_size=1,
                 top_k_experts=2,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        self.moe = dict(
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=[]
        )
        self.lora = {}
        self.mone = {}

        super(MoEQwen2VLConfig, self).__init__(**kwargs)


@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


def MoEQwen2DecoderLayer_forward(self):
    def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings for rotary attention.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Handle MoE layer
        mlp_output = self.mlp(hidden_states)

        moe_losses = []
        if isinstance(mlp_output, tuple) and len(mlp_output) >= 2:
            moe_losses.append(mlp_output[1])
            hidden_states = mlp_output[0]
        else:
            hidden_states = mlp_output
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs

    return forward


def MoEQwen2VLModel_forward(self):
    def forward(
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_moe_loss = [] if output_moe_loss else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_moe_loss:
                all_moe_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_moe_loss] if
                v is not None)
        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_loss_list=all_moe_loss,
        )

    return forward


class MoEQwen2VLModel(Qwen2VLModel):
    config_class = MoEQwen2VLConfig
    
    def __init__(self, config):
        super().__init__(config)
        self._attn_implementation = config._attn_implementation
        
    # We need to inherit the _update_causal_mask method to ensure proper functionality
    # This is referenced in the forward method
    _update_causal_mask = Qwen2VLModel._update_causal_mask


class MoEQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    config_class = MoEQwen2VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MoEQwen2VLModel(config)
        
        # Initialize or reuse components
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict




        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    print(f"input_ids: {input_ids}")
                    print(f"image_grid_thw: {image_grid_thw}")
                    print(f"image_embeds: {image_embeds}") 
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Calculate position IDs and rope deltas if needed
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # Use the previously calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        moe_loss, moe_losses = None, []
        if hasattr(outputs, "moe_loss_list") and outputs.moe_loss_list and len(outputs.moe_loss_list) > 0:
            moe_loss_list = outputs.moe_loss_list
            for moe_loss_item in moe_loss_list:
                if moe_loss_item is not None:
                    moe_losses.append(moe_loss_item)
            if moe_losses:
                moe_loss = self.router_aux_loss_coef * sum(moe_losses)
                if labels is not None:
                    # print(f"Loss: {loss}, MoE Loss: {sum(moe_losses)}, Total: {loss + moe_loss}")
                    loss += moe_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            moe_loss_list=outputs.moe_loss_list if hasattr(outputs, "moe_loss_list") else None,
        )

    def initialize_moe_modules(self, model_args):
        if getattr(model_args, 'lora_enable', False):
            self.config.lora['lora_enable'] = model_args.lora_enable
            self.config.lora['only_lora_ffn'] = model_args.only_lora_ffn
            self.config.lora['lora_r'] = model_args.lora_r
            self.config.lora['lora_alpha'] = model_args.lora_alpha
            self.config.lora['lora_dropout'] = model_args.lora_dropout
            self.config.lora['lora_bias'] = model_args.lora_bias
            self.config.lora['target_modules'] = model_args.train_modules
        
        if getattr(model_args, 'mone_enable', False):
            self.config.mone['mone_expert_type'] = model_args.mone_expert_type
            self.config.mone['mone_gate_type'] = model_args.mone_gate_type
            self.config.mone['mone_r'] = model_args.mone_r
            self.config.mone['mone_dropout'] = model_args.mone_dropout
            self.config.mone['mone_num_heads'] = model_args.mone_num_heads
            self.config.mone['mone_use_query_bn'] = model_args.mone_use_query_bn
            self.config.mone['mone_act_fn'] = model_args.mone_act_fn

        self.config.moe['moe_enable'] = model_args.moe_enable
        self.config.moe['train_modules'] = model_args.train_modules
        self.config.moe['moe_mode'] = model_args.moe_mode
        self.config.moe['moe_layers_idx'] = model_args.moe_layers_idx
        self.config.moe['ep_size'] = model_args.ep_size
        self.config.moe['top_k_experts'] = model_args.top_k_experts
        self.config.moe['capacity_factor'] = model_args.capacity_factor
        self.config.moe['eval_capacity_factor'] = model_args.eval_capacity_factor
        self.config.moe['min_capacity'] = model_args.min_capacity
        self.config.moe['use_residual'] = model_args.use_residual
        self.config.moe['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef
        
        # Freeze all parameters except those specified in train_modules
        if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.moe['train_modules']):
                    continue
                else:
                    p.requires_grad = False

        num_layers = self.config.num_hidden_layers

        # Determine which layers will be converted to MoE
        moe_layers_idx = model_args.moe_layers_idx
        if model_args.moe_layers_idx is not None:
            model_args.moe_mode = 'custom'
            assert len(model_args.moe_layers_idx) <= num_layers
            assert max(model_args.moe_layers_idx) < num_layers
            assert min(model_args.moe_layers_idx) >= 0
        else:
            if model_args.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif model_args.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif model_args.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif model_args.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense"], but found {model_args.moe_mode}')

        self.config.moe['moe_layers_idx'] = moe_layers_idx
        self.config.moe['num_experts'] = model_args.num_experts
        
        # Handle single num_experts value
        if len(model_args.num_experts) == 1:
            self.config.moe['num_experts'] = model_args.num_experts * len(moe_layers_idx)
        assert len(self.config.moe['num_experts']) == len(moe_layers_idx)

        # Convert specified layers to MoE
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            if not getattr(model_args, 'mone_enable', False):
                pretrained_state_dict = self.model.layers[layer_num].mlp.state_dict()
                self.model.layers[layer_num].mlp = MoE(
                    self.config.hidden_size,
                    expert=self.model.layers[layer_num].mlp,
                    num_experts=num_experts,
                    ep_size=model_args.ep_size,
                    k=model_args.top_k_experts,
                    capacity_factor=model_args.capacity_factor,
                    eval_capacity_factor=model_args.eval_capacity_factor,
                    min_capacity=model_args.min_capacity,
                    use_residual=model_args.use_residual,
                )
                # Verify weights are properly copied
                for e in self.model.layers[layer_num].mlp.deepspeed_moe.experts.deepspeed_experts:
                    loaded_state_dict = e.state_dict()
                    assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()])
                    assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()])
            else:
                rank0_print(f"Using MoE with Mixture of Nano Experts")
                # 保存原始MLP
                original_mlp = self.model.layers[layer_num].mlp
                mone_r = self.config.mone['mone_r']
                mone_dropout = self.config.mone['mone_dropout']
                
                if model_args.mone_expert_type == 'small_expert' and model_args.mone_gate_type == 'token_gating':
                    # 创建小专家实例
                    small_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                    # 创建MoE层
                    moe_layer = MoE(
                        self.config.hidden_size,
                        expert=small_expert,  # 使用小专家
                        num_experts=num_experts,
                        ep_size=model_args.ep_size,
                        k=model_args.top_k_experts,
                        capacity_factor=model_args.capacity_factor,
                        eval_capacity_factor=model_args.eval_capacity_factor,
                        min_capacity=model_args.min_capacity,
                        use_residual=model_args.use_residual,
                    )
                elif model_args.mone_expert_type == 'embedding_expert':
                    moe_layer = EmbeddedMoELayer(
                        hidden_size=self.config.hidden_size,
                        num_experts=num_experts,  # 可设置为1M专家
                        expert_dim=mone_r,           # 单神经元专家
                        k=model_args.top_k_experts,                   # 每个token选择的专家数
                        gate_type=model_args.mone_gate_type,
                        capacity_factor=model_args.capacity_factor,
                        eval_capacity_factor=model_args.eval_capacity_factor,
                        min_capacity=model_args.min_capacity,
                        use_residual=model_args.use_residual,
                        num_heads=self.config.mone['mone_num_heads'],            # 多头专家选择, 8
                        use_query_bn=self.config.mone['mone_use_query_bn'],      # 使用批量归一化提高稳定性, True
                        act_fn=self.config.mone['mone_act_fn'],          # 可选: "relu", "gelu", "silu", silu
                        dropout=self.config.mone['mone_dropout'],            # 专家dropout率
                        dtype=next(original_mlp.parameters()).dtype,
                    )
                else:
                    raise NotImplementedError(f"Unsupported expert type: {model_args.mone_expert_type}")
                
                # 替换原始MLP为组合层
                self.model.layers[layer_num].mlp = CombinedLayer(original_mlp, moe_layer)

                for name, param in self.model.named_parameters():
                    if 'deepspeed_moe' in name:
                        param.requires_grad = True
                    # else:
                    #     param.requires_grad = False

        # # 冻结普通MLP层，只训练MoE层
        # for name, param in self.model.named_parameters():
        #     # 如果是普通MLP层参数（不是MoE层）
        #     if 'mlp' in name and 'deepspeed_moe' not in name:
        #         param.requires_grad = False
        #     # 可选：冻结其他非MLP层参数
        #     elif 'mlp' not in name:
        #         param.requires_grad = False  # 如果只想训练MoE部分
        
        
        rank0_print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        # Replace forward methods to handle MoE outputs
        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        rank0_print(f'replace Qwen2DecoderLayer.forward to MoEQwen2DecoderLayer.forward')
        
        self.model.forward = MoEQwen2VLModel_forward(self.model)
        rank0_print(f'replace Qwen2VLModel.forward to MoEQwen2VLModel.forward')

    get_rope_index = Qwen2VLForConditionalGeneration.get_rope_index
    prepare_inputs_for_generation = Qwen2VLForConditionalGeneration.prepare_inputs_for_generation
    _get_image_nums_and_video_nums = Qwen2VLForConditionalGeneration._get_image_nums_and_video_nums
    _expand_inputs_for_generation = Qwen2VLForConditionalGeneration._expand_inputs_for_generation


class EvalMoEQwen2VLForConditionalGeneration(MoEQwen2VLForConditionalGeneration):
    config_class = MoEQwen2VLConfig

    def __init__(self, config):
        super(EvalMoEQwen2VLForConditionalGeneration, self).__init__(config)
        if getattr(self.config, 'lora', False) and self.config.lora.get('lora_enable', False):
            from peft import LoraConfig, get_peft_model
            pre_lora_config = self.config.lora
            lora_config = LoraConfig(
                r=pre_lora_config['lora_r'],
                lora_alpha=pre_lora_config['lora_alpha'],
                target_modules=pre_lora_config['target_modules'],
                lora_dropout=pre_lora_config['lora_dropout'],
                bias=pre_lora_config['lora_bias'],
                task_type="CAUSAL_LM",
            )
            print("Adding LoRA adapters...")
            get_peft_model(self, lora_config)
        
        if getattr(self.config, 'mone', False) and self.config.mone.get('mone_enable', False):
            mone_expert_type = self.config.mone['mone_expert_type']
            mone_gate_type = self.config.mone['mone_gate_type']
            mone_r = self.config.mone['mone_r']
            mone_dropout = self.config.mone['mone_dropout'] 

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']

        # Reinitialize MoE layers for evaluation
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            if getattr(self.config, 'mone', False) and self.config.mone.get('mone_enable', False):
                original_mlp = self.model.layers[layer_num].mlp
                mone_r = self.config.mone['mone_r']
                mone_dropout = self.config.mone['mone_dropout']
                
                if mone_expert_type == 'small_expert' and mone_gate_type == 'token_gating':
                    # 创建小专家实例
                    small_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                    # 创建MoE层
                    moe_layer = MoE(
                        self.config.hidden_size,
                        expert=small_expert,  # 使用小专家
                        num_experts=num_experts,
                        ep_size=self.config.moe['ep_size'],
                        k=self.config.moe['top_k_experts'],
                        capacity_factor=self.config.moe['capacity_factor'],
                        eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                        min_capacity=self.config.moe['min_capacity'],
                        use_residual=self.config.moe['use_residual'],
                    )
                elif mone_expert_type == 'embedding_expert':
                    moe_layer = EmbeddedMoELayer(
                        hidden_size=self.config.hidden_size,
                        num_experts=num_experts,  # 可设置为1M专家
                        expert_dim=mone_r,           # 单神经元专家
                        k=self.config.moe['top_k_experts'],                   # 每个token选择的专家数
                        gate_type=self.config.moe['mone_gate_type'],
                        capacity_factor=self.config.moe['capacity_factor'],
                        eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                        min_capacity=self.config.moe['min_capacity'],
                        use_residual=self.config.moe['use_residual'],
                        num_heads=self.config.mone['mone_num_heads'],            # 多头专家选择, 8
                        use_query_bn=self.config.mone['mone_use_query_bn'],      # 使用批量归一化提高稳定性, True
                        act_fn=self.config.mone['mone_act_fn'],          # 可选: "relu", "gelu", "silu", silu
                        dropout=self.config.mone['mone_dropout'],            # 专家dropout率
                        dtype=self.config.torch_dtype
                    )
                else:
                    raise NotImplementedError(f"Unsupported expert type: {mone_expert_type}")
                
                # 替换原始MLP为组合层
                self.model.layers[layer_num].mlp = CombinedLayer(original_mlp, moe_layer)
            else:
                self.model.layers[layer_num].mlp = MoE(
                    self.config.hidden_size,
                    expert=self.model.layers[layer_num].mlp,
                    num_experts=num_experts,
                    ep_size=self.config.moe['ep_size'],
                    k=self.config.moe['top_k_experts'],
                    capacity_factor=self.config.moe['capacity_factor'],
                    eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                    min_capacity=self.config.moe['min_capacity'],
                    use_residual=self.config.moe['use_residual'],
                )

        rank0_print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        # Replace forward methods for evaluation
        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        rank0_print(f'replace Qwen2DecoderLayer.forward to MoEQwen2DecoderLayer.forward')
        
        self.model.forward = MoEQwen2VLModel_forward(self.model)
        rank0_print(f'replace Qwen2VLModel.forward to MoEQwen2VLModel.forward')
    
    get_rope_index = Qwen2VLForConditionalGeneration.get_rope_index
    prepare_inputs_for_generation = Qwen2VLForConditionalGeneration.prepare_inputs_for_generation
    _get_image_nums_and_video_nums = Qwen2VLForConditionalGeneration._get_image_nums_and_video_nums
    _expand_inputs_for_generation = Qwen2VLForConditionalGeneration._expand_inputs_for_generation


# Register the new model with AutoConfig and AutoModel systems
AutoConfig.register("moe_qwen2_vl", MoEQwen2VLConfig)
AutoModelForCausalLM.register(MoEQwen2VLConfig, MoEQwen2VLForConditionalGeneration)
AutoModelForCausalLM.register(MoEQwen2VLConfig, EvalMoEQwen2VLForConditionalGeneration)