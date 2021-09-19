import numpy as np
import torch
import torch.nn.functional as F
import math


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
            for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])), \
            f'Attention mask shape {attn_mask.shape} should be broadcastable with attention shape {attn.shape}'

        attn.data.masked_fill_(attn_mask, -float('inf'))


class Attention(torch.nn.Module):
    def __init__(self, pointer):
        super().__init__()
        self.pointer = pointer
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query_size
        # values shape: batch x num values x value_size

        # attn_logits shape: batch x num values
        attn_logits = self.pointer(query, values, attn_mask)
        # attn_logits shape: batch x num values
        attn = self.softmax(attn_logits)
        # output shape: batch x 1 x value_size
        output = torch.bmm(attn.unsqueeze(1), values)
        output = output.squeeze(1)
        return output, attn


class ScaledDotProductPointer(torch.nn.Module):
    def __init__(self, query_size, key_size):
        super().__init__()
        self.query_proj = torch.nn.Linear(query_size, key_size)
        self.temp = np.power(key_size, 0.5)

    def forward(self, query, keys, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # proj_query shape: batch x key_size x 1
        proj_query = self.query_proj(query).unsqueeze(2)

        # attn_logits shape: batch x num keys
        attn_logits = torch.bmm(keys, proj_query).squeeze(2) / self.temp
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


class ScaledDotProductAttention(Attention):
    def __init__(self, query_size, value_size):
        super().__init__(ScaledDotProductPointer(query_size, value_size))


class BahdanauPointer(torch.nn.Module):
    def __init__(self, query_size, key_size, proj_size):
        super().__init__()
        self.compute_scores = torch.nn.Sequential(
            torch.nn.Linear(query_size + key_size, proj_size),
            torch.nn.Tanh(),
            torch.nn.Linear(proj_size, 1))

    def forward(self, query: torch.Tensor, keys: torch.Tensor, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # query_expanded shape: batch x num keys x query_size
        query_expanded = query.unsqueeze(1).expand(-1, keys.shape[1], -1)

        # scores shape: batch x num keys x 1
        attn_logits = self.compute_scores(
            # shape: batch x num keys x query_size + key_size
            torch.cat((query_expanded, keys),
                      dim=2))
        # scores shape: batch x num keys
        attn_logits = attn_logits.squeeze(2)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


class BahdanauAttention(Attention):
    def __init__(self, query_size, value_size, proj_size):
        super().__init__(BahdanauPointer(query_size, value_size, proj_size))


# Adapted from The Annotated Transformers
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, query_size, value_size, dropout=0.1):
        super().__init__()
        assert query_size % h == 0
        assert value_size % h == 0

        # We assume d_v always equals d_k
        self.d_k = value_size // h
        self.h = h

        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(query_size, value_size),
            torch.nn.Linear(value_size, value_size),
            torch.nn.Linear(value_size, value_size),
            torch.nn.Linear(value_size, value_size),
        ])

        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, values, attn_mask=None):
        "Implements Figure 2"
        if attn_mask is not None:
            # Same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, keys, values = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, values, values))]

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = transformer.sparse_attention(
        x, self.attn = attention(
            query, keys, values, mask=attn_mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = x.squeeze(1)
        return self.linears[-1](x), self.attn


if __name__ == "__main__":
    """run some simple test cases"""
    sdpp = ScaledDotProductPointer(query_size=8, key_size=16)
    sdpa = ScaledDotProductAttention(query_size=8, value_size=16)
    bp = BahdanauPointer(query_size=8, key_size=16, proj_size=12)
    ba = BahdanauAttention(query_size=8, value_size=16, proj_size=12)
    mha = MultiHeadedAttention(h=2, query_size=8, value_size=16)

    q = torch.tensor(list(range(1, 9)), dtype=torch.float32).reshape(1, 8)
    v = torch.tensor(list(range(1, 17)), dtype=torch.float32).reshape(1, 1, 16)

    print(sdpp(q, v))
    print(sdpa(q, v))
    print(bp(q, v))
    print(ba(q, v))
    print(mha(q, v))
