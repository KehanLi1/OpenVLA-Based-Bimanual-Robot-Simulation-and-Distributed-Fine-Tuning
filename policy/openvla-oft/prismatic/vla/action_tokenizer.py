"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    """将连续机器人动作离散化为 token 字符串，映射到 LLM 词表末尾的 bin token。

    编码流程 (每个维度独立):
        连续值 → clip 到 [min_action, max_action] → np.digitize 找到 bin 索引
        → 映射到词表末尾 token id (vocab_size - bin_index) → tokenizer.decode 为字符串

    映射约定: 使用 LlamaTokenizer 词表中最不常用的末尾 token，
    即 token id 范围为 [vocab_size - n_bins, vocab_size - 1]。
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # 在 [min_action, max_action] 上均匀划分 n_bins 个边界
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        # bin 中心值，用于反向解码时将 bin 索引还原为连续值
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 动作 token 的起始 id: vocab_size - (n_bins + 1)
        # 所有 > action_token_begin_idx 的 token id 都是动作 bin token
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """将连续动作编码为 token 字符串。

        单步 action shape [ACTION_DIM] → 返回 str (ACTION_DIM 个 bin token 字符拼接)
        多步 action shape [N, ACTION_DIM] → 返回 List[str] (每步一个字符串)

        token id 计算: vocab_size - digitize_index，即数值越大的动作映射到越靠前的 token。
        """
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """将动作 token id 反向解码为连续动作值。

        反向流程: token_id → bin_index = vocab_size - token_id → clip → bin_centers[index]
        注意 digitize 返回的索引范围是 [1, n_bins]，需要 -1 并 clip 到 [0, n_bins-2]。
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
