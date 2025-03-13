import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForConditionalGeneration, DynamicCache, Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, \
    _prepare_4d_causal_attention_mask_for_sdpa
from moellava.model.multimodal_model.qwen2_vl import Qwen2VLConfig, Qwen2VLModel, Qwen2VLForConditionalGeneration

from transformers.modeling_outputs import CausalLMOutputWithPast

from deepspeed.moe.layer import MoE
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


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
            **kwargs
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
            output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds
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
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
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

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

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


class MoEQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    config_class = MoEQwen2VLConfig

    def __init__(self, config):
        super(Qwen2VLForConditionalGeneration, self).__init__(config)
        self.model = MoEQwen2VLModel(config)
        
        # Initialize or reuse components
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
        self.router_aux_loss_coef = 0.01  # Default value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if pixel_values is not None and images is not None:
            raise ValueError("You cannot specify both pixel_values and images at the same time")

        if images is not None:
            pixel_values = images

        if pixel_values is not None:
            if not hasattr(self, "visual"):
                vision_outputs = None
            else:
                vision_outputs = self.visual(pixel_values)
        else:
            vision_outputs = None

        if vision_outputs is not None:
            if isinstance(vision_outputs, dict):
                vision_hidden_states = vision_outputs.get("image_features", None) or vision_outputs.get(
                    "last_hidden_state", None
                )
            else:
                vision_hidden_states = vision_outputs

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            mode = getattr(self.config, "multimodal_projector_type", "")
            if "pretrain" in mode.lower():
                if hasattr(self, "projector_embedding") and self.projector_embedding is not None:
                    vision_hidden_states = self.projector_embedding(vision_hidden_states)

            image_sizes = None
            if hasattr(vision_outputs, "image_sizes"):
                image_sizes = vision_outputs.image_sizes

            inputs_embeds, attention_mask, position_ids = self.multimodal_projector(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_features=vision_hidden_states,
                image_sizes=image_sizes,
            )

        outputs = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
                    print(f"Loss: {loss}, MoE Loss: {sum(moe_losses)}, Total: {loss + moe_loss}")
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
                
        rank0_print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        # Replace forward methods to handle MoE outputs
        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        rank0_print(f'replace Qwen2DecoderLayer.forward to MoEQwen2DecoderLayer.forward')
        
        self.model.forward = MoEQwen2VLModel_forward(self.model)
        rank0_print(f'replace Qwen2VLModel.forward to MoEQwen2VLModel.forward')


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

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']

        # Reinitialize MoE layers for evaluation
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
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


# Register the new model with AutoConfig and AutoModel systems
AutoConfig.register("moe_qwen2_vl", MoEQwen2VLConfig)
AutoModelForConditionalGeneration.register(MoEQwen2VLConfig, MoEQwen2VLForConditionalGeneration)
AutoModelForConditionalGeneration.register(MoEQwen2VLConfig, EvalMoEQwen2VLForConditionalGeneration)