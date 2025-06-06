# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import time
import random
import sys
from dataclasses import dataclass, field
import json
import logging
import pathlib
from glob import glob
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import numpy as np
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

from moellava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, MAX_IMAGE_LENGTH, \
    MAX_VIDEO_LENGTH
from torch.utils.data import Dataset
from moellava.train.llava_trainer import LLaVATrainer, QwenMoETrainer

from moellava import conversation as conversation_lib
from moellava.model import *
from moellava.mm_utils import tokenizer_image_token
from moellava.model.language_model.llava_qwen_moe import EvalMoELLaVAQWenForCausalLM
from moellava.train.utils import initialize_moe_with_pretrained_weights, make_supervised_data_module_qwen2_vl
from moellava.train.args import ModelArguments, DataArguments, TrainingArguments
from moellava.train.callbacks import EpochBasedUnfreezeCallback

from PIL import Image
from moellava.utils import order_pick_k

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, add_keywords=None):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'image_tower',
                           'video_tower', 'vision_resampler'] + add_keywords if add_keywords is not None else []
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) or sentence['value'].startswith(DEFAULT_VIDEO_TOKEN):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * data_args.num_frames
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            # print('before replace_token:', [sentence['value']])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # print('after replace_token:', [sentence['value']])
            # ======================================================================================================

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []
    # sys.exit()

    # import ipdb
    # ipdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(11111111, conversations)
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # print(2222222222222, input_ids.shape)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # print('after tokenizer_image_token', input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # print(tokenizer)
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print('total_len', total_len)
        rounds = conversation.split(conv.sep2)
        # print('len(rounds)', len(rounds))
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            # import ipdb
            # ipdb.set_trace()
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []
    # sys.exit()

    # import ipdb
    # ipdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(11111111, conversations)
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # print(2222222222222, input_ids.shape)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # print('after tokenizer_image_token input_ids targets', input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # print(tokenizer)
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    # print('sep', sep)
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print('total_len', total_len)
        rounds = conversation.split(conv.sep2)
        # print('len(rounds)', len(rounds))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # print('i rou, parts', i, rou, parts)
            if len(parts) != 2:
                break
            parts[0] += sep
            # print('after add sep, parts', parts)

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1  # for eos_token
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            # print('round_len, instruction_len, target[cur_len : cur_len + instruction_len]',
            #       round_len, instruction_len, target[cur_len : cur_len + instruction_len], target[cur_len : cur_len + round_len])
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            # import ipdb
            # ipdb.set_trace()
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print(input_ids, target)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_openchat(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []
    # sys.exit()

    # import ipdb
    # ipdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(11111111, conversations)
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # print(2222222222222, input_ids.shape)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # print('after tokenizer_image_token input_ids targets', input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # print(tokenizer)
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    # print('sep\n', sep)
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print('total_len', total_len)
        rounds = conversation.split(conv.sep2)
        # print('len(rounds)', len(rounds))
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # print('i rou, parts\n', i, rou, parts)
            if len(parts) != 2:
                break
            parts[0] += sep
            # print('after add sep, parts\n', parts)

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # print('instruction_len, target[cur_len : cur_len + instruction_len]\n',
            #       instruction_len, target[cur_len : cur_len + instruction_len])
            # print('round_len, target[cur_len : cur_len + round_len]\n',
            #       round_len, target[cur_len : cur_len + round_len])
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # print(cur_len, total_len)
        if cur_len < tokenizer.model_max_length:
            # import ipdb
            # ipdb.set_trace()
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print(input_ids, target)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    # print('sources', sources)
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # print('conversations', conversations)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    # print('after tokenizer_image_token', input_ids)
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    # print('target:', target)
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("phi") or \
            conversation_lib.default_conversation.version.startswith("qwen"):  # for phi and qwen
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("stablelm"):  # stablelm same as phi
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("openchat") or \
        conversation_lib.default_conversation.version.startswith("mistral"):  # for openchat
        return preprocess_openchat(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("minicpm"):  # minicpm same as openchat
        return preprocess_openchat(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def analyze_data_lengths(dataset, max_length=None, sample_size=5, seed=42):
    """
    分析数据集中序列长度的统计信息
    
    参数:
        dataset: 数据集对象
        max_length: 最大长度阈值，用于计算截断比例
        sample_size: 要采样的数据条数，如果为None则分析全部数据
        seed: 随机种子，确保采样可重复
    """
    random.seed(seed)
    
    # 确定要分析的索引
    total_size = len(dataset)
    if sample_size is not None and sample_size < total_size:
        # 随机采样
        indices = random.sample(range(total_size), sample_size)
        print(f"从{total_size}条数据中随机采样{sample_size}条进行统计...")
    else:
        indices = range(total_size)
        print(f"正在统计全部{total_size}条数据的长度...")
    
    input_lengths = []
    label_lengths = []
    
    for i in tqdm(indices):
        item = dataset[i]
        input_lengths.append(len(item['input_ids']))
        label_lengths.append(len(item['labels']))
    
    input_lengths = np.array(input_lengths)
    label_lengths = np.array(label_lengths)
    
    stats = {
        "输入序列 (input_ids)": {
            "最短长度": int(np.min(input_lengths)),
            "最长长度": int(np.max(input_lengths)),
            "平均长度": float(np.mean(input_lengths)),
            "75%分位值": int(np.percentile(input_lengths, 75)),
            "80%分位值": int(np.percentile(input_lengths, 80)),
            "样本长度列表": input_lengths.tolist(),  # 添加原始长度列表供参考
        },
        "标签序列 (labels)": {
            "最短长度": int(np.min(label_lengths)),
            "最长长度": int(np.max(label_lengths)),
            "平均长度": float(np.mean(label_lengths)),
            "75%分位值": int(np.percentile(label_lengths, 75)),
            "80%分位值": int(np.percentile(label_lengths, 80)),
            "样本长度列表": label_lengths.tolist(),  # 添加原始长度列表供参考
        }
    }
    
    # 计算被截断的数据比例
    if max_length is not None:
        truncated_inputs = np.sum(input_lengths >= max_length)
        truncated_labels = np.sum(label_lengths >= max_length)
        
        stats["截断统计"] = {
            "输入序列截断数量": int(truncated_inputs),
            "输入序列截断比例": f"{truncated_inputs / len(input_lengths) * 100:.2f}%",
            "标签序列截断数量": int(truncated_labels),
            "标签序列截断比例": f"{truncated_labels / len(label_lengths) * 100:.2f}%"
        }
    
    return stats


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        # ================================================
        list_data_dict = []
        for data in data_path:
            data = json.load(open(data, "r"))
            for i in data:
                i['id'] = len(list_data_dict)
                list_data_dict.append(i)
        # ================================================

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)
        # return 10

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # ===========================================================================
            cur_len = cur_len if ('image' in sample or 'video' in sample) else -cur_len
            # ===========================================================================
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            # ======================================================================================================
            if 'image' in sources[0] and 'video' not in sources[0]:
                # rank0_print('image')
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor
                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                # print(f"total {len(self.list_data_dict[i]['image'])} now {len(image_file)}")
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                # print(image[0])
                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                # print(image[0].shape)
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                data_dict = preprocess(sources, self.tokenizer, has_image=True)

            elif 'image' not in sources[0] and 'video' in sources[0]:
                # rank0_print('video')
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image
                # image = [torch.randn(3, 8, 224, 224) for i in video]  # fake image
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print('after preprocess_multimodal', sources[0])
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
                # print('after preprocess', data_dict['input_ids'])

            elif 'image' in sources[0] and 'video' in sources[0]:
                # rank0_print('image & video')
                # video must before image
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor

                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor

                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]

                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                video = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image

                image = video + image  # video must before image

                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(sources, self.tokenizer, has_image=False)

            # ==========================================================================================================

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])
            # image exist in the data
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                if hasattr(self.data_args.image_processor, 'crop_size'):
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
                else:
                    size = self.data_args.image_processor.size
                    data_dict['image'] = [torch.zeros(3, size['height'], size['width'])]
            return data_dict
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # print('before Collator', input_ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # print('after Collator', batch)
        # print(input_ids, labels, input_ids.ne(self.tokenizer.pad_token_id))
        # ======================================================================================================
        # origin image, if batch_size=6: [[image], [image], [video], [image, image], [video, video], [video, image]]
        '''
            will be converted to a sequence of list, if batch size=6:
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(8, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
        '''
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]

            # adapt to multi-video or multi-image or multi-image & video
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

        # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
            batch['images'] = images
        #     if all(x is not None and x.shape == images[0].shape for x in images):  # if all images or all videos
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        else:
            raise ValueError(f'pretrain, {instances}')
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.image_tower is not None or model_args.video_tower is not None:
        if not model_args.moe_enable:
            if 'mpt' in model_args.model_name_or_path.lower():
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                config.attn_config['attn_impl'] = training_args.mpt_attn_impl
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen2-vl' in model_args.model_name_or_path.lower():
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                model = LlavaQWenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' in model_args.model_name_or_path.lower():
                model = LlavaQwen1_5ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'openchat' in model_args.model_name_or_path.lower() or 'mistral' in model_args.model_name_or_path.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'phi' in model_args.model_name_or_path.lower():
                model = LlavaPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'minicpm' in model_args.model_name_or_path.lower():
                model = LlavaMiniCPMForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'stablelm' in model_args.model_name_or_path.lower():
                model = LlavaStablelmForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
        else:
            if 'qwen2-vl' in model_args.model_name_or_path.lower():
                model = MoEQwen2VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                if model_args.skip_moe_init:
                    model = EvalMoELLaVAQWenForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        **bnb_model_from_pretrained_args
                    )
                else:
                    model = MoELLaVAQWenForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        **bnb_model_from_pretrained_args
                    )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' in model_args.model_name_or_path.lower():
                model = MoELLaVAQwen1_5ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'phi' in model_args.model_name_or_path.lower():
                model = MoELLaVAPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'minicpm' in model_args.model_name_or_path.lower():
                model = MoELLaVAMiniCPMForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'openchat' in model_args.model_name_or_path.lower() or 'mistral' in model_args.model_name_or_path.lower():
                model = MoELLaVAMistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'stablelm' in model_args.model_name_or_path.lower():
                model = MoELLaVAStablelmForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = MoELLaVALlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # attn_implementation="flash_attention_2",
            # torch_dtype=torch.bfloat16,
            **bnb_model_from_pretrained_args
        )
    rank0_print('LLM init. firstly\n', model)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        print("Enable gradient checkpointing")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            print("Enable input require grads")
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # ==============================================================================================
    training_args.moe_enable = model_args.moe_enable
    training_args.only_lora_ffn = model_args.only_lora_ffn
    model_args.lora_enable = training_args.lora_enable
    if model_args.moe_enable:
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            if 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                target_modules = [
                    'mlp.w1', 'mlp.w2', 'mlp.c_proj'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            elif 'phi' in model_args.model_name_or_path.lower():
                target_modules = [
                    'fc1', 'fc2'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            else:
                target_modules = [
                    'up_proj', 'down_proj', 'gate_proj'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            # modules_to_save = ['wg']  # weight gating for MoE
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                # modules_to_save=modules_to_save,
                task_type="CAUSAL_LM",
            )
            model_args.lora_r = training_args.lora_r
            model_args.lora_alpha = training_args.lora_alpha
            model_args.lora_dropout = training_args.lora_dropout
            model_args.lora_bias = training_args.lora_bias
            # model_args.modules_to_save = modules_to_save
            model_args.target_modules = target_modules
            model_args.train_modules = target_modules
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        model.initialize_moe_modules(model_args=model_args)
    else:
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
    # ==============================================================================================

    if model_args.ffn_only:
        for n, p in model.named_parameters():
            if any(name in n for name in ["mlp.w1", "mlp.w2", "mlp.c_proj"]):
                p.requires_grad = True
            else:
                p.requires_grad = False
            trainable_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        rank0_print(f"Number of trainable parameters: {len(trainable_params)}")
        rank0_print("List of trainable parameters:")
        for param_name in trainable_params:
            rank0_print(f"  {param_name}")
        
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total number of trainable parameters: {total_trainable_params / 1e6:.2f}M")

    if model_args.load_k_experts:
        if 'llava-qwen' in model_args.model_name_or_path.lower():
            model_k = LlavaQWenForCausalLM.from_pretrained(
                model_args.k_experts_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            model = initialize_moe_with_pretrained_weights(model, model_k, model.config.moe['moe_layers_idx'], 'moe-llava-qwen')
        elif 'qwen2-vl' in model_args.model_name_or_path.lower():
            model_k = EvalMoEQwen2VLForConditionalGeneration.from_pretrained(
                model_args.k_experts_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            model_type = 'moe-qwen2-vl'
            if model_args.mone_expert_type == 'dense_mask_expert':
                model_type += '-ds'
            if 'kd' in model_args.k_experts_path:
                model_type += '-kd'
                if 'g' in model_args.k_experts_path:
                    model_type += '-g'
            if 'ada' in model_args.k_experts_path:
                model_type += '-ada'
            if 'share' in model_args.k_experts_path:
                model_type += '-share'
            model = initialize_moe_with_pretrained_weights(model, model_k, model.config.moe['moe_layers_idx'], model_type)
        del model_k
    
    if model_args.from_pretrained:
        state_dict = torch.load(model_args.from_pretrained_path, map_location="cpu")
        config = json.load(open(model_args.from_pretrained_path.replace('pytorch_model.bin', 'config.json'), 'r'))
        if config['moe'].get('structure', "new") == "old":
            for k in list(state_dict.keys()):
                if 'moe_layer' in k:
                    state_dict[k.replace('moe_layer', 'moe')] = state_dict.pop(k)
                elif 'original_mlp' in k:
                    state_dict[k.replace('original_mlp', 'shared')] = state_dict.pop(k)
        incompatible = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)
        rank0_print(f'------------------------------- load from {model_args.from_pretrained_path} -------------------------------')
    # return

    if model_args.kd_align:
        for n, p in model.named_parameters():
            if any(name in n for name in ["moe"]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    if model_args.warm_up_experts:
        for n, p in model.named_parameters():
            if any(name in n for name in ["wg", "expert"]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    if not model_args.moe_enable:
        for n, p in model.named_parameters():
            # if ('mlp' in n) and ('visual' not in n):
            p.requires_grad = True
            # else:
                # p.requires_grad = False 

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # import ipdb
        # ipdb.set_trace()
        if 'qwen2-vl' in model_args.model_name_or_path.lower():
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                # The default setting is padding_side="left"
                # When training using the right-side padding is more efficient.
                padding_side="right",
                use_fast=False
            )
            processor.image_processor.min_pixels = data_args.image_min_pixels
            processor.image_processor.max_pixels = data_args.image_max_pixels
            tokenizer = processor.tokenizer
            processor.tokenizer.model_max_length = training_args.model_max_length
            model.config.tokenizer_model_max_length = training_args.model_max_length
            model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
            from moellava.model.language_model.qwen.tokenization_qwen import QWenTokenizer
            tokenizer = QWenTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>', 'eos_token': '<|endoftext|>'})
        elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
        elif 'phi' in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
        elif 'stablelm' in model_args.model_name_or_path.lower():
            from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
            tokenizer = Arcade100kTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.unk_token = '<|reg0|>'  # FIXME: DO SUPPORT ADD SPECIAL TOKENS
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
    # import ipdb
    # ipdb.set_trace()
    # print(tokenizer)
    # print(tokenizer)
    if 'qwen2-vl' in model_args.model_name_or_path.lower():
        pass
    elif model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        # =============================================================================================================
        model.config.pad_token_id = tokenizer.pad_token_id
        # =============================================================================================================
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # print(conversation_lib.default_conversation)
    # =============================================================================================================
    if 'qwen2-vl' in model_args.model_name_or_path.lower():
        pass
    elif model_args.image_tower is not None or model_args.video_tower is not None:
        # print(model_args)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        if model_args.image_tower is not None:
            image_tower = model.get_image_tower()
            image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.num_frames = video_tower.config.num_frames
    # =============================================================================================================

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        # model.config.tokenizer_model_max_length = tokenizer.model_max_length  # number of video tokens may greater than 2048

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    rank0_print('Vision encoder and proj init.\n', model)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    if training_args.freeze_shared:
        for n, p in model.named_parameters():
            if "shared" in n:
                p.requires_grad = False
        rank0_print('--------------------- Freeze shared parameters ---------------------')
        
    for name, param in model.named_parameters():
        # param.requires_grad = True
        if param.requires_grad:
            rank0_print(f'{name}: {param.shape}')
    rank0_print(model)
    # sys.exit()

    if 'qwen2-vl' in model_args.model_name_or_path.lower():
        data_module = make_supervised_data_module_qwen2_vl(
            model_id=model_args.model_name_or_path,
            processor=processor,
            data_args=data_args
        )
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    # # 分析训练集长度分布
    # max_length = training_args.model_max_length  # 假设这是你的最大长度参数
    # train_stats = analyze_data_lengths(data_module['train_dataset'], max_length, sample_size=int(5*1e4))

    # # 打印统计结果
    # print("\n数据长度统计结果:")
    # for key, value in train_stats.items():
    #     print(f"\n{key}:")
    #     for stat_name, stat_value in value.items():
    #         print(f"  {stat_name}: {stat_value}")
    
    # return

    if 'qwen2-vl' in model_args.model_name_or_path.lower():
        trainer = QwenMoETrainer(
            processing_class=processor,
            model=model,
            args=training_args,
            **data_module
        )
        if training_args.freeze_shared:
            trainer.add_callback(
                EpochBasedUnfreezeCallback(unfreeze_shared_epoch=model_args.unfreeze_shared_epoch)
            )
    else:
        trainer = LLaVATrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)

    # Statistics of time and memory usage. Part 1
    # =============================================================================================================
    STATS_JSON_FILE_PATH = training_args.stat_json_path
    if training_args.enable_stat:
        training_duration_seconds = None
        peak_memory_gb = None
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            stats_dir = os.path.dirname(STATS_JSON_FILE_PATH)
            assert os.path.isdir(stats_dir), f"Error: Directory for statistics '{stats_dir}' does not exist."

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
    # =============================================================================================================
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Statistics of time and memory usage. Part 2
    # =============================================================================================================
    if training_args.enable_stat:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            training_duration_seconds = end_time - start_time

            if torch.cuda.is_available():
                peak_memory_bytes = torch.cuda.max_memory_allocated()
                peak_memory_gb = peak_memory_bytes / (1024**3)
            else:
                peak_memory_gb = 0.0

            if hasattr(model_args, 'moe_enable') and model_args.moe_enable:
                shared_expert_str = "S" if getattr(model_args, 'use_shared_experts', False) else ""
                num_experts_val = getattr(model_args, 'num_experts', 'N')[0]
                top_k_experts_val = getattr(model_args, 'top_k_experts', 'M')
                mone_expert_type_val = getattr(model_args, 'mone_expert_type', 'N/A_expert_type')
                mone_gate_type_val = getattr(model_args, 'mone_gate_type', 'N/A_gate_type')
                stats_key = f'{mone_expert_type_val}-{mone_gate_type_val}-{shared_expert_str}{num_experts_val}k{top_k_experts_val}'
            else:
                # 为非MoE模型或MoE参数不完全的配置提供一个通用key
                model_name_for_key = os.path.basename(model_args.model_name_or_path) if model_args.model_name_or_path else "unknown_model"
                stats_key = f'{model_name_for_key}'
                if hasattr(training_args, 'lora_enable') and training_args.lora_enable:
                    lora_r_val = getattr(training_args, 'lora_r', 'N/A')
                    stats_key += f'-lora_r{lora_r_val}'
                if hasattr(training_args, 'bits') and training_args.bits in [4, 8]:
                    stats_key += f'-bits{training_args.bits}'
                # 如果没有MoE且没有LoRA/bits等特定标识，加一个后缀区分
                if not (stats_key.count('-') > 0): # 简单检查是否添加了后缀
                    stats_key += '-standard'

            current_run_stats = {
                "training_time_seconds": round(training_duration_seconds, 2),
                "peak_memory_usage_gb": round(peak_memory_gb, 2),
                "max_steps_for_stat": training_args.max_steps, # 记录用于统计的max_steps
                "model_path": training_args.output_dir, # 记录模型路径以供参考
                "fp16": training_args.fp16,
                "bf16": training_args.bf16,
                "bits_quantization": training_args.bits if training_args.bits in [4,8] else "None"
            }

            # 加载已有的统计数据，或初始化为空字典
            all_experiment_stats = {}
            if os.path.exists(STATS_JSON_FILE_PATH):
                try:
                    with open(STATS_JSON_FILE_PATH, 'r') as f:
                        all_experiment_stats = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {STATS_JSON_FILE_PATH} was corrupted or not valid JSON. Initializing new stats.")
                    all_experiment_stats = {} # 如果文件损坏，则重新开始
                except Exception as e:
                    print(f"Error loading {STATS_JSON_FILE_PATH}: {e}. Initializing new stats.")
                    all_experiment_stats = {}

            # 更新字典
            all_experiment_stats[stats_key] = current_run_stats

            # 3. 保存到指定的JSON文件
            try:
                # 再次确保目录存在，以防在初始检查后发生变化或初始创建失败但未中止
                stats_dir = os.path.dirname(STATS_JSON_FILE_PATH)
                if not os.path.isdir(stats_dir):
                    os.makedirs(stats_dir, exist_ok=True) # 尝试最后一次创建

                with open(STATS_JSON_FILE_PATH, 'w') as f:
                    json.dump(all_experiment_stats, f, indent=4)
                print(f"Training statistics successfully saved to {STATS_JSON_FILE_PATH}")
                print(f"Stats for key '{stats_key}': {current_run_stats}")
            except IOError as e:
                print(f"Error: Could not write statistics to {STATS_JSON_FILE_PATH}. IOError: {e}")
            except Exception as e: #捕获其他可能的错误
                print(f"An unexpected error occurred while saving statistics: {e}")
    # =============================================================================================================

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable and not model_args.moe_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        # if model_args.moe_enable:
        ckpt = model.state_dict()
        # import ipdb
        # ipdb.set_trace()
        ckpt = {(k[11:] if k.startswith('base_model.') else k): v for k, v in ckpt.items()}
        if any(k.startswith('model.model.') for k in ckpt):
            ckpt = {(k[6:] if k.startswith('model.') else k): v for k, v in ckpt.items()}
        torch.save(ckpt, os.path.join(training_args.output_dir, 'pytorch_model.bin'))
        model.config.save_pretrained(training_args.output_dir)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            [os.remove(i) for i in glob(os.path.join(training_args.output_dir, 'adapter_*'))]
    # print(model.state_dict().keys())

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
