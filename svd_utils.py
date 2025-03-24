# svd_utils.py (예: 새 파일로 분리 가능)
import os
import copy
import torch
import torch.nn as nn


def truncated_svd(M: torch.Tensor, rank: int, device: torch.device):
    """
    M.shape = (in_dim, out_dim)인 2D 텐서에 대해 rank만큼 truncated SVD 수행
    -> return: U_r, S_r, V_r (torch.Tensor)
    """
    M = M.cpu()
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    # 상위 rank개만 취함
    U_r = U[:, :rank].to(device)           # (in_dim, r)
    S_r = torch.diag(S[:rank]).to(device)  # (r, r)
    V_r = Vh[:rank, :].to(device)          # (r, out_dim)
    return U_r, S_r, V_r


class SVDLinear(nn.Module):
    """
    weight = (U * S * V) 형태로 분해된 '가상' Linear 레이어
      - weight.shape = (in_dim, out_dim)을
        U.shape = (in_dim, r), S.shape = (r,r), V.shape = (r, out_dim)
      - forward 시 x @ U @ S @ V + bias 수행
    """
    def __init__(self, in_features, out_features, U, S, V, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.U = nn.Parameter(U, requires_grad=False)
        self.S = nn.Parameter(S, requires_grad=False)
        self.V = nn.Parameter(V, requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch, in_dim)
        # 1) x @ U   => (batch, r)
        # 2) 그 결과 @ S => (batch, r)
        # 3) 그 결과 @ V => (batch, out_dim)
        z = x @ self.U  # (batch, r)
        z = z @ self.S  # (batch, r)
        z = z @ self.V  # (batch, out_dim)
        if self.bias is not None:
            z += self.bias
        return z
