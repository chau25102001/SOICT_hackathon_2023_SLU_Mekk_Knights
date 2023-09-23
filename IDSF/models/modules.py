import numpy as np
import torch
import torch.nn as nn


# TODO: add multi head attention
class Attention(nn.Module):

    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, attention_mask):
        '''
        :param query: embedding of each token in the sentence
        :param context: embedding of the intent
        :param attention_mask:
        :return:
        '''
        _, output_len, _ = query.size()
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        if attention_mask is not None:
            attention_mask = torch.unsqueeze(attention_mask, 2)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        attention_weights = self.softmax(attention_scores)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        output = self.linear_out(combined)
        output = self.tanh(output)

        return output, attention_weights


class ETFLinear(nn.Module):
    def __init__(self, feat_in, feat_out, device='cuda'):
        super().__init__()
        P = self._generate_random_orthogonal_matrix(feat_in, feat_out)
        I = torch.eye(feat_out)
        one = torch.ones(feat_out, feat_out, dtype=torch.float32)
        M = np.sqrt(feat_out / (feat_out - 1)) * \
            torch.matmul(P, I - ((1 / feat_out) * one))
        self.M = M.to(device)

        self.InstanceNorm = nn.InstanceNorm1d(feat_in, affine=False, device=device)
        self.BatchNorm = nn.BatchNorm1d(feat_in, affine=False, device=device)

    def _generate_random_orthogonal_matrix(self, feat_in, feat_out):
        a = np.random.random(size=(feat_in, feat_out))
        P, _ = np.linalg.qr(a)  # This function returns an orthonormal matrix (q) and an upper-triangle matrix r(q)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(feat_out), atol=1e-07), \
            torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(feat_out)))
        return P

    def forward(self, x):
        if x.shape[0] != 1:
            return self.BatchNorm(x) @ self.M
        else:
            return self.InstanceNorm(x) @ self.M


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out=0.0, use_etf=False):
        super().__init__()
        self.dropout = nn.Dropout(drop_out)
        if use_etf:
            self.linear = ETFLinear(input_dim, num_classes)
        else:
            self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self,
                 input_dim,
                 num_intent_labels,
                 num_slot_labels,
                 use_attn=False,
                 attention_embedding_size=200,
                 drop_out=0.0, **kwargs):
        super().__init__()
        self.use_attn = use_attn
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.attention_embedding_size = attention_embedding_size
        output_dim = self.attention_embedding_size
        self.attention = Attention(attention_embedding_size)
        self.linear_slot = nn.Linear(input_dim, self.attention_embedding_size, bias=False)
        self.linear_intent_context = nn.Linear(self.num_intent_labels, self.attention_embedding_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(output_dim, num_slot_labels)

    def forward(self, x, intent_context, attention_mask):
        x = self.linear_slot(x)
        if self.use_attn:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            output, weights = self.attention(x, intent_context, attention_mask)
            x = output
        x = self.dropout(x)
        return self.linear(x)


class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, drop=0.1, num_classes=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
