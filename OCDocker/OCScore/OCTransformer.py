import torch
from torch import nn


import torch
from torch import nn

class RegressionTransformer(nn.Module):
  def __init__(self, d_model, output_dim, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1, init_type: str = 'zeros', init_params: dict = {}, random_seed: int = 42, device=torch.device('cuda')):
    super().__init__()
    self.encoder = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
    self.linear = nn.Linear(d_model, 1)  # Assuming one target variable for regression
    self.output_dim = output_dim

  def forward(self, src, src_mask = None):
    if src_mask is None:
      
    src = self.encoder(src, src_mask)
    # No decoder needed for regression, use encoded output directly
    output = self.linear(torch.mean(src, dim=self.output_dim))
    return output


class Encoder(nn.Module):
  def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
    super().__init__()
    self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, src, src_mask):
    for layer in self.layers:
      src = layer(src, src_mask)
    src = self.norm(src)
    return src


class EncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout):
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, nhead, dropout)
    self.feed_forward = PositionwiseFeedforward(d_model, dim_feedforward, dropout)
    self.dropout = nn.Dropout(p=dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

  def forward(self, src, src_mask):
    src = self.self_attn(src, src, src, src_mask)
    src = self.dropout(src)
    src = src + self.norm1(src)
    src = self.feed_forward(src)
    src = self.dropout(src)
    src = src + self.norm2(src)
    return src

class MultiheadAttention(nn.Module):
  def __init__(self, d_model, nhead, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.w_q = nn.Linear(d_model, d_model * nhead, bias=False)
    self.w_k = nn.Linear(d_model, d_model * nhead, bias=False)
    self.w_v = nn.Linear(d_model, d_model * nhead, bias=False)
    self.dropout = nn.Dropout(p=dropout)
    self.linear = nn.Linear(d_model * nhead, d_model, bias=False)
    self.scale = 1 / (d_model ** 0.5)

  def forward(self, q, k, v, mask=None):
    # (batch, seq_len, d_model) -> (batch, nhead, seq_len, d_model // nhead)
    q = self.w_q(q).view(q.shape[0], q.shape[1], self.nhead, -1)
    k = self.w_k(k).view(k.shape[0], k.shape[1], self.nhead, -1)
    v = self.w_v(v).view(v.shape[0], v.shape[1], self.nhead, -1)

    # (batch, nhead, seq_len, seq_len)
    attn = torch.bmm(q, k.transpose(1, 2)) * self.scale

    if mask is not None:
      attn = attn.masked_fill(mask == 0, -float('inf'))
    attn = self.dropout(torch.softmax(attn, dim=-1))

    # (batch, nhead, seq_len, d_model // nhead) * (batch, nhead, seq_len, d_model // nhead) -> (batch, seq_len, d_model)
    output = torch.bmm(attn, v).transpose(1, 2).contiguous()
    output = self.linear(output.view(output.shape[0], -1, self.d_model))
    return output


class PositionwiseFeedforward(nn.Module):
  def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
    super().__init__()
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.relu = nn.ReLU(inplace=True)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    x = self.relu(self.linear1(x))
    x = self.dropout(x)
    x = self.linear2(x)
    return x
