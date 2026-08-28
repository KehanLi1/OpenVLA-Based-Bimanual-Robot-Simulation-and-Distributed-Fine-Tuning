"""训练/微调脚本的工具函数。

动作掩码原理说明:
    labels 序列格式: [-100, -100, ..., -100, A₁, A₂, ..., A_K, </s>, -100(pad)...]
                      ← 全部 IGNORE_INDEX →  ← 动作 bin tokens →  stop

    其中 K = ACTION_DIM × NUM_ACTIONS_CHUNK。
    get_current_action_mask: 选出前 ACTION_DIM 个动作 token（当前时间步的动作）
    get_next_actions_mask:   选出第 ACTION_DIM+1 到第 K 个动作 token（后续 chunk 步的动作）
    二者合并即覆盖全部 K 个动作 token。

    实现方式: 对 (token_ids != IGNORE_INDEX) 做 cumsum，根据累计计数区分当前步与后续步。
"""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    """选出"当前动作"对应的前 ACTION_DIM 个动作 bin token 位置。

    Args:
        token_ids: [B, T] — 通常是 labels[:, 1:]（next-token 对齐后）

    Returns:
        bool mask [B, T]，True 位置为当前步的动作 token
    """
    # 非 IGNORE_INDEX 的位置标记为 True（即动作区域 + stop token）
    newline_positions = token_ids != IGNORE_INDEX

    # cumsum: 从左到右累计非 IGNORE 位的出现次数
    # 当 cumsum 在 [1, ACTION_DIM] 范围内时，就是当前步的动作区域
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # 额外过滤: 只保留 token id > ACTION_TOKEN_BEGIN_IDX 的位置（排除 stop token 等）
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    """选出"后续 chunk 步"对应的动作 bin token 位置（cumsum > ACTION_DIM 的部分）。

    Args:
        token_ids: [B, T] — 通常是 labels[:, 1:]

    Returns:
        bool mask [B, T]，True 位置为后续 chunk 步的动作 token
    """
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)

    # cumsum > ACTION_DIM 的位置就是第 ACTION_DIM+1 个动作 token 之后的区域
    mask = cumsum > ACTION_DIM

    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
