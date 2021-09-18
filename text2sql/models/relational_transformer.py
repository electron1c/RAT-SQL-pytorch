#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import traceback
import logging
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# Adapted from
# https://github.com/tensorflow/tensor2tensor/blob/0b156ac533ab53f65f44966381f6e147c7371eee/tensor2tensor/layers/common_attention.py
def relative_attention_logits(query, key, relation):
    # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
    # In this version, relation vectors are shared across heads.
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    # q_t is [batch, num queries, heads, depth]
    q_t = query.permute(0, 2, 1, 3)

    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose(-2, -1)

    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_tr_t_matmul = torch.matmul(q_t, r_t)

    # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

    # [batch, heads, num queries, num kvs]
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

    # Sharing relation vectors across batch and heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [num queries, num kvs, depth].
    #
    # Then take
    # key reshaped
    #   [num queries, batch * heads, depth]
    # relation.transpose(-2, -1)
    #   [num queries, depth, num kvs]
    # and multiply them together.
    #
    # Without sharing relation vectors across heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, heads, num queries, num kvs, depth].
    #
    # Then take
    # key.unsqueeze(3)
    #   [batch, heads, num queries, 1, depth]
    # relation.transpose(-2, -1)
    #   [batch, heads, num queries, depth, num kvs]
    # and multiply them together:
    #   [batch, heads, num queries, 1, depth]
    # * [batch, heads, num queries, depth, num kvs]
    # = [batch, heads, num queries, 1, num kvs]
    # and squeeze
    # [batch, heads, num queries, num kvs]


def relative_attention_values(weight, value, relation):
    # In this version, relation vectors are shared across heads.
    # weight: [batch, heads, num queries, num kvs].
    # value: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # wv_matmul is [batch, heads, num queries, depth]
    wv_matmul = torch.matmul(weight, value)

    # w_t is [batch, num queries, heads, num kvs]
    w_t = weight.permute(0, 2, 1, 3)

    #   [batch, num queries, heads, num kvs]
    # * [batch, num queries, num kvs, depth]
    # = [batch, num queries, heads, depth]
    w_tr_matmul = torch.matmul(w_t, relation)

    # w_tr_matmul_t is [batch, heads, num queries, depth]
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

    return wv_matmul + w_tr_matmul_t


class RelationalAttentionLayer(nn.Module):
    def __init__(self, cfg):
        super(RelationalAttentionLayer, self).__init__()
        d_model = cfg['hidden_size']
        n_head = cfg['num_attention_heads']
        assert d_model % n_head == 0
        d_model_q = cfg.get('query_hidden_size_per_head',
                            d_model // n_head) * n_head
        d_model_v = cfg.get('value_hidden_size_per_head',
                            d_model // n_head) * n_head
        self.n_head = n_head
        self.d_key = d_model_q // n_head
        self.q = nn.Linear(d_model, d_model_q)
        self.k = nn.Linear(d_model, d_model_q)
        self.v = nn.Linear(d_model, d_model_v)
        self.o = nn.Linear(d_model_v, d_model)
        self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

    def forward(self,
                queries,
                keys,
                values,
                relation_k,
                relation_v,
                attn_bias=None,
                past_cache=None):
        """relational attention forward.
        seq_len in `shape` means num queries/keys/values of attention

        Args:
            queries (TYPE): shape = [batch, seq_len, num_heads * hidden]
            keys (TYPE): shape = queries.shape
            values (TYPE): shape = queries.shape
            relation_k (TYPE): shape = [batch, seq_len, seq_len, hidden]
            relation_v (TYPE): shape = relation_k.shape
            attn_bias (TYPE): used as sequence mask. Default is None
            past_cache (TYPE): Default is None

        Returns: TODO

        Raises: NULL
        """
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        # bsz, q_len, q_dim = queries.shape
        # bsz, k_len, k_dim = keys.shape
        # bsz, v_len, v_dim = values.shape
        # assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        cache = (k, v)
        if past_cache is not None:
            cached_k, cached_v = past_cache
            k = torch.cat([cached_k, k], 1)
            v = torch.cat([cached_v, v], 1)

        def _transpose(inputs):
            """reshape and transpose
            Args: inputs: shape = [batch, seq_len, heads * hidden]
            Returns: shape = [batch, heads, seq_len, hidden]
            """
            hidden_size = inputs.shape[-1] // self.n_head
            outputs = inputs.reshape([inputs.shape[0], inputs.shape[1], self.n_head, hidden_size])
            return outputs.permute([0, 2, 1, 3])

        q, k, v = [_transpose(x) for x in (q, k, v)]

        q = q * (self.d_key ** -0.5)
        scores = relative_attention_logits(q, k, relation_k)
        if attn_bias is not None:
            scores += attn_bias
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = relative_attention_values(scores, v, relation_v)
        # input: [batch, heads, seq_len, hidden]
        # output: [batch, seq_len, heads * hidden]
        out = out.permute([0, 2, 1, 3])
        out = out.reshape([out.shape[0], out.shape[1], out.shape[2] * out.shape[3]])
        out = self.o(out)
        return out, cache


class RelationalPointerNet(nn.Module):
    """Pointer Netword with Relations"""

    def __init__(self, hidden_size, num_relations, init_range=0.02):
        """init of class

        Args:
            cfg (TYPE): NULL

        """
        super(RelationalPointerNet, self).__init__()
        self.hidden_size = hidden_size

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

        self.relation_emb = None
        if num_relations > 0:
            self.relation_emb = nn.Embedding(num_relations, hidden_size)
        self.scores = None

    def forward(self, queries, keys, relations, attn_bias=None):
        """relational attention forward.
        seq_len in `shape` means num queries/keys/values of attention

        Args:
            queries (TYPE): shape = [batch, seq_len, num_heads * hidden]
            keys (TYPE): shape = queries.shape
            relations (TYPE): shape = [batch, seq_len, seq_len, hidden]
            attn_bias (TYPE): used as sequence mask. Default is None

        Returns: TODO

        Raises: NULL
        """
        assert len(queries.shape) == len(keys.shape) == 3

        q = self.q(queries)
        k = self.k(keys)
        r = None
        if relations is not None:
            r = self.relation_emb(relations)

        def _transpose(inputs):
            """reshape and transpose
            Args: inputs: shape = [batch, seq_len, heads * hidden]
            Returns: shape = [batch, heads, seq_len, hidden]
            """
            # 1 代表 head 数量，此处恒为 1。
            outputs = inputs.reshape([inputs.shape[0], inputs.shape[1], 1, self.hidden_size])
            return outputs.permute([0, 2, 1, 3])

        q = _transpose(q)
        k = _transpose(k)
        # q = q.scale(self.hidden_size**-0.5)
        scores = relative_attention_logits(q, k, r)
        if attn_bias is not None:
            scores += attn_bias

        self.scores = F.softmax(scores, dim=-1)
        return self.scores.squeeze()


class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, cfg):
        super(PositionwiseFeedForwardLayer, self).__init__()
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        self.act = nn.ReLU()
        self.i = nn.Linear(d_model, d_ffn)
        self.o = nn.Linear(d_ffn, d_model)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs):
        hidden = self.act(self.i(inputs))
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out


class RelationalTransformerBlock(nn.Module):
    """A transformer block with relations"""

    def __init__(self, cfg):
        super(RelationalTransformerBlock, self).__init__()
        d_model = cfg['hidden_size']
        n_heads = cfg['num_attention_heads']
        self.attn = RelationalAttentionLayer(cfg)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForwardLayer(cfg)
        self.ln2 = nn.LayerNorm(d_model)
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)

        # 假设 k/v 的
        rel_hidden = d_model // n_heads
        self.relation_k_emb = nn.Embedding(cfg['num_relations'], rel_hidden)
        self.relation_v_emb = nn.Embedding(cfg['num_relations'], rel_hidden)

    def forward(self, inputs, relations, attn_bias=None, past_cache=None):
        relation_k = self.relation_k_emb(relations)
        relation_v = self.relation_k_emb(relations)

        attn_out, cache = self.attn(
            inputs,
            inputs,
            inputs,
            relation_k,
            relation_v,
            attn_bias,
            past_cache=past_cache)  # self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, cache


class RelationalTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(RelationalTransformerEncoder, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = nn.ModuleList([
            RelationalTransformerBlock(cfg)
            for _ in range(n_layers)
        ])

    def forward(self, inputs, relations, attn_bias=None, past_cache=None):
        """relational transformer encoder, forward stage of
        n layers and m heads transformer blocks with relations

        Args:
            inputs (TYPE): shape= [batch, seq_len, hidden]
            relations (TYPE): shape = [batch, seq_len, seq_len]
            attn_bias (TYPE): mask for inputs sequence. Default is None
            past_cache (TYPE): Default is None

        Returns: (last_hidden_state, all_hidden_state_list, (cache_list_k, cache_list_v))

        Raises: NULL
        """
        if past_cache is not None:
            assert isinstance(past_cache, tuple), 'unknown type of `past_cache`,' + \
                                                  ' expect tuple or list. got %s' % repr(type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

        for b, p in zip(self.block, past_cache):
            inputs, cache = b(inputs,
                              relations,
                              attn_bias=attn_bias,
                              past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(inputs)

        return inputs, hidden_list, (cache_list_k, cache_list_v)


if __name__ == "__main__":
    """run some simple test cases"""
    cfg = {
        "num_hidden_layers": 12,
        "num_attention_heads": 2,
        "num_relations": 99,
        "hidden_size": 4,
        "hidden_act": "relu",
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
    }

    model = RelationalTransformerEncoder(cfg)
    print(model)
    inputs = torch.tensor(
        list(range(24)), dtype=torch.float32).reshape([2, 3, 4])
    relations = torch.tensor(
        list(range(18)), dtype=torch.int64).reshape([2, 3, 3])
    hidden, _, _ = model(inputs, relations)
    print(hidden)
    print(hidden.size())
