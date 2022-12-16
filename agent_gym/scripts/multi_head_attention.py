import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# __all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        # print("q,k,v")
        # print(query.shape, key.shape, value.shape)
        # print(query[:5,:5,:5].detach().numpy())
        # print(key[:5, :5, :5].detach().numpy())
        # print(value[:5, :5, :5].detach().numpy())
        dk = query.size()[-1]
        # print("dk    ", dk)
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # print("score")
        # print(scores.shape)
        # print(scores[:5, :5, :5].detach().numpy())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        # print("attention")
        # print(attention.shape)
        # print(attention[:5, :5, :5].detach().numpy())
        # print("value")
        # print(value.shape)
        output = attention.matmul(value)
        # print("output")
        # print(output.shape)
        # print(output[:5, :5, :5].detach().numpy())
        return output


class ScaledDotProductAttentionwithEdge(nn.Module):

    def forward(self, query, key, value, mask=None):
        # print("q,k,v")
        # print(query.shape, key.shape, value.shape)
        dk = query.size()[-1]
        # print("dk    ", dk)
        query = torch.unsqueeze(query, dim=-2)
        # print("q',k,v")
        # print(query.shape, key.shape, value.shape)

        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # print("score")
        # print(scores.shape)
        scores = torch.squeeze(scores, dim=-2)
        # print("score'")
        # print(scores.shape)
        # print(scores[:5, :5, :5].detach().numpy())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        # print("attention")
        # print(attention.shape)
        # print(attention[:5, :5, :5].detach().numpy())
        # print("value")
        # print(value.shape)
        output = attention.matmul(value)
        # print("output")
        # print(output.shape)
        # print(output[:5, :5, :5].detach().numpy())
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 output_dim,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                '`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.output_dim = output_dim
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, output_dim, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(y[:, :, :5].detach().numpy())

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, output_dim={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.output_dim, self.bias, self.activation,
        )


class MultiHeadSelfCrossAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 output_dim,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadSelfCrossAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                '`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.output_dim = output_dim
        self.head_num = head_num // 2
        self.activation = activation
        self.bias = bias
        self.linear_qs = nn.Linear(in_features, in_features // 2, bias)
        self.linear_ks = nn.Linear(in_features, in_features // 2, bias)
        self.linear_vs = nn.Linear(in_features, in_features // 2, bias)

        self.linear_qc = nn.Linear(in_features, in_features // 2, bias)
        self.linear_kc = nn.Linear(in_features, in_features // 2, bias)
        self.linear_vc = nn.Linear(in_features, in_features // 2, bias)

        self.linear_o = nn.Linear(in_features, output_dim, bias)

    def forward(self, f1, f2, masks=None, maskc=None):
        qs, ks, vs = self.linear_qs(f1), self.linear_ks(f1), self.linear_vs(f1)
        qc, kc, vc = self.linear_qc(f1), self.linear_kc(f2), self.linear_vc(f2)

        if self.activation is not None:
            qs = self.activation(qs)
            ks = self.activation(ks)
            vs = self.activation(vs)
            qc = self.activation(qc)
            kc = self.activation(kc)
            vc = self.activation(vc)

        qs = self._reshape_to_batches(qs)
        ks = self._reshape_to_batches(ks)
        vs = self._reshape_to_batches(vs)
        qc = self._reshape_to_batches(qc)
        kc = self._reshape_to_batches(kc)
        vc = self._reshape_to_batches(vc)
        if masks is not None:
            masks = masks.repeat(self.head_num, 1, 1)
            maskc = maskc.repeat(self.head_num, 1, 1)

        # print(masks.shape, maskc.shape)
        ys = ScaledDotProductAttention()(qs, ks, vs, masks)
        # print(ys.shape)
        ys = self._reshape_from_batches(ys)

        yc = ScaledDotProductAttention()(qc, kc, vc, maskc)
        # print(yc.shape)
        yc = self._reshape_from_batches(yc)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(y[:, :, :5].detach().numpy())

        y = torch.cat([ys, yc], dim=-1)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, output_dim={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.output_dim, self.bias, self.activation,
        )


class MultiHeadAttentionWithEdge(MultiHeadAttention):

    def __init__(self,
                 in_features,
                 head_num,
                 output_dim,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionWithEdge, self).__init__(
            in_features=in_features,
            head_num=head_num,
            output_dim=output_dim,
            bias=bias,
            activation=activation)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches_edge(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttentionwithEdge()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(y[:, :, :5].detach().numpy())

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches_edge(self, x):
        batch_size, seq_len, _, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, seq_len, self.head_num, sub_dim) \
            .permute(0, 3, 1, 2, 4) \
            .reshape(batch_size * self.head_num, seq_len, seq_len, sub_dim)


class MultiHeadSelfCrossAttentionWithEdge(MultiHeadSelfCrossAttention):

    def __init__(self,
                 in_features,
                 head_num,
                 output_dim,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadSelfCrossAttentionWithEdge, self).__init__(
            in_features=in_features,
            head_num=head_num,
            output_dim=output_dim,
            bias=bias,
            activation=activation)

        self.linear_qs = nn.Linear(in_features, in_features // 2, bias)
        self.linear_ks = nn.Linear(in_features, in_features // 2, bias)
        self.linear_vs = nn.Linear(in_features, in_features // 2, bias)

        self.linear_qc = nn.Linear(in_features, in_features // 2, bias)
        self.linear_kc = nn.Linear(in_features, in_features // 2, bias)
        self.linear_vc = nn.Linear(in_features, in_features // 2, bias)

    def forward(self, f1, f2, edges, edgec, masks=None, maskc=None):
        qs, ks, vs = self.linear_qs(f1), self.linear_ks(edges), self.linear_vs(f1)
        qc, kc, vc = self.linear_qc(f1), self.linear_kc(edgec), self.linear_vc(f2)

        if self.activation is not None:
            qs = self.activation(qs)
            ks = self.activation(ks)
            vs = self.activation(vs)
            qc = self.activation(qc)
            kc = self.activation(kc)
            vc = self.activation(vc)

        qs = self._reshape_to_batches(qs)
        ks = self._reshape_to_batches_edge(ks)
        vs = self._reshape_to_batches(vs)
        qc = self._reshape_to_batches(qc)
        kc = self._reshape_to_batches_edge(kc)
        vc = self._reshape_to_batches(vc)
        if masks is not None:
            masks = masks.repeat(self.head_num, 1, 1)
            maskc = maskc.repeat(self.head_num, 1, 1)

        # print(masks.shape, maskc.shape)
        ys = ScaledDotProductAttentionwithEdge()(qs, ks, vs, masks)
        # print(ys.shape)
        ys = self._reshape_from_batches(ys)

        yc = ScaledDotProductAttentionwithEdge()(qc, kc, vc, maskc)
        # print(yc.shape)
        yc = self._reshape_from_batches(yc)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(y[:, :, :5].detach().numpy())

        y = torch.cat([ys, yc], dim=-1)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches_edge(self, x):
        batch_size, seq_len, _, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, seq_len, self.head_num, sub_dim) \
            .permute(0, 3, 1, 2, 4) \
            .reshape(batch_size * self.head_num, seq_len, seq_len, sub_dim)


class MultiHeadSelfCrossAttentionWithNodeAndEdge(MultiHeadSelfCrossAttention):

    def __init__(self,
                 in_features,
                 head_num,
                 output_dim,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadSelfCrossAttentionWithNodeAndEdge, self).__init__(
            in_features=in_features,
            head_num=head_num,
            output_dim=output_dim,
            bias=bias,
            activation=activation)

        self.linear_qs = nn.Linear(in_features, in_features // 2, bias)
        self.linear_ks = nn.Linear(in_features * 2, in_features // 2, bias)
        self.linear_vs = nn.Linear(in_features, in_features // 2, bias)

        self.linear_qc = nn.Linear(in_features, in_features // 2, bias)
        self.linear_kc = nn.Linear(in_features * 2, in_features // 2, bias)
        self.linear_vc = nn.Linear(in_features, in_features // 2, bias)

    def forward(self, f1, f2, edges, edgec, masks=None, maskc=None):

        nodes = torch.unsqueeze(f1, dim=-3).repeat(1, f1.shape[-2], 1, 1)
        nodec = torch.unsqueeze(f2, dim=-3).repeat(1, f2.shape[-2], 1, 1)

        assert nodes.shape == edges.shape

        edges_and_nodes = torch.cat([edges, nodes], dim=-1)
        edgec_and_nodec = torch.cat([edgec, nodec], dim=-1)

        qs, ks, vs = self.linear_qs(f1), self.linear_ks(edges_and_nodes), self.linear_vs(f1)
        qc, kc, vc = self.linear_qc(f1), self.linear_kc(edgec_and_nodec), self.linear_vc(f2)

        if self.activation is not None:
            qs = self.activation(qs)
            ks = self.activation(ks)
            vs = self.activation(vs)
            qc = self.activation(qc)
            kc = self.activation(kc)
            vc = self.activation(vc)

        qs = self._reshape_to_batches(qs)
        ks = self._reshape_to_batches_edge(ks)
        vs = self._reshape_to_batches(vs)
        qc = self._reshape_to_batches(qc)
        kc = self._reshape_to_batches_edge(kc)
        vc = self._reshape_to_batches(vc)
        if masks is not None:
            masks = masks.repeat(self.head_num, 1, 1)
            maskc = maskc.repeat(self.head_num, 1, 1)

        # print(masks.shape, maskc.shape)
        ys = ScaledDotProductAttentionwithEdge()(qs, ks, vs, masks)
        # print(ys.shape)
        ys = self._reshape_from_batches(ys)

        yc = ScaledDotProductAttentionwithEdge()(qc, kc, vc, maskc)
        # print(yc.shape)
        yc = self._reshape_from_batches(yc)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(y[:, :, :5].detach().numpy())

        y = torch.cat([ys, yc], dim=-1)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches_edge(self, x):
        batch_size, seq_len, _, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, seq_len, self.head_num, sub_dim) \
            .permute(0, 3, 1, 2, 4) \
            .reshape(batch_size * self.head_num, seq_len, seq_len, sub_dim)
