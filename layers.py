import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import math
import numpy as np
import gc
import psutil
    


class DecomposeLinearSVD(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank, weight, bias):
        super(DecomposeLinearSVD, self).__init__(
            in_features=in_features, out_features=out_features
        )
        self.U, self.S, self.Vh = torch.linalg.svd(weight, full_matrices=False)

        # 추가: U/S/Vh를 buffer로 등록하고 싶다면 아래처럼 가능합니다.
        # (필요 없다면, 원 코드처럼 None으로 만들면 됩니다.)
        ## CHANGED: buffer 등록 예시
        self.register_buffer("U_buf", self.U.clone().detach(), persistent=True)
        self.register_buffer("S_buf", self.S.clone().detach(), persistent=True)
        self.register_buffer("Vh_buf", self.Vh.clone().detach(), persistent=True)

        if not (isinstance(rank, float) or isinstance(rank, int)):
            variance = float(rank.split(":")[-1])
            S_sum = torch.cumsum(self.S.float(), 0)
            self.rank = torch.searchsorted(S_sum, S_sum[-1] * variance).item()
            self.target_budget = self.rank / (
                in_features * out_features / (in_features + out_features)
            )
        else:
            self.rank = rank

        self.weight = weight
        self.weight1 = nn.Parameter(
            torch.zeros(self.rank, in_features, requires_grad=True, device=weight.device, dtype=weight.dtype)
        )
        self.weight2 = nn.Parameter(
            torch.zeros(out_features, self.rank, requires_grad=True, device=weight.device, dtype=weight.dtype)
        )

        # 초기화
        self.weight1.data = torch.transpose(
            torch.transpose(self.Vh[: self.rank, :], 1, 0)
            @ torch.diag((self.S[: self.rank])),
            1,
            0,
        ).to(weight.device).to(weight.dtype)

        # 아래 두 줄은 bias 관련이지만 주석 처리되어 있음
        w1_bias = torch.transpose(
            torch.transpose(self.Vh[self.rank:, :], 1, 0)
            @ torch.diag((self.S[self.rank:])),
            1,
            0,
        ).mean(axis=0).reshape((1,self.weight1.data.shape[1]))

        self.weight2.data = self.U[:, : self.rank].to(weight.device).to(weight.dtype)
        w2_bias = self.U[:, self.rank:].mean(axis = 1).reshape((self.weight2.data.shape[0],1))
        
        print(self.weight1.data.shape, self.weight2.data.shape)

    def forward(self, input):
        return F.linear(
            F.linear(input, self.weight1, None),
            self.weight2,
            None,
        )

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = DecomposeLinearSVD(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            linear_module.weight,
            linear_module.bias,
        )
        # 여기서 기존 weight/bias는 nn.Linear로부터 넘어왔으므로 필요시 제거
        new_linear.U = None
        new_linear.S = None
        new_linear.Vh = None
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        return new_linear


class DecomposeLinearEigen(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank, weight, bias):
        super(DecomposeLinearEigen, self).__init__(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.mf16 = False
        self.init = False
        self.weight = weight
        self.rank = rank

        ## CHANGED: V, b1, Y_sub를 buffer로 등록하기 위해 미리 placeholder로 만들어 둠
        # persistent=False 는 꼭 영구저장이 필요없으면 False로 하셔도 됩니다 (속성)
        self.register_buffer("V_buf", torch.zeros(out_features, rank, dtype=weight.dtype), persistent=True)
        self.register_buffer("b1_buf", torch.zeros(1, out_features, dtype=weight.dtype), persistent=True)
        self.register_buffer("Y_sub_buf", torch.zeros(1, out_features, dtype=weight.dtype), persistent=False)

        self.weight1 = nn.Parameter(
            torch.zeros(
                rank,
                in_features,
                requires_grad=True,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.weight2 = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
                requires_grad=True,
                device=weight.device,
                dtype=weight.dtype,
            )
        )

    def make_float16(self):
        # half로 전환
        self.Y_sub_buf = self.Y_sub_buf.half()
        self.V_buf = self.V_buf.half()
        self.weight.data = self.weight.data.to(torch.float16)
        self.weight2.data = self.weight2.data.to(torch.float16)
        self.weight1.data = self.weight1.data.to(torch.float16)
        self.bias.data = self.bias.data.to(torch.float16)
        self.b1_buf = self.b1_buf.to(torch.float16)
        self.mf16 = True
        gc.collect()

    def init_lowrank(self, input):
        # 아직 self.V_buf이 텅 비어 있을 것이므로 여기서 실제 값 계산
        if torch.count_nonzero(self.V_buf) == 0:
            # 실제 연산
            Y = (
                F.linear(input, self.weight, None)
                .reshape(-1, self.out_features)
                .float()
                .cpu()
            )  # (BS, out)
            Y_mean = torch.mean(Y, dim=0).unsqueeze(0)
            self.Y_sub_buf = (Y - Y_mean)  # buffer에 복사할 예정

            cov = torch.cov(torch.transpose(self.Y_sub_buf, 1, 0))  # (out, out)
            _, V = torch.linalg.eigh(cov.float())  # (out, out)
            self.target_budget = self.rank / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
            V = V[:, -self.rank:].to(self.weight.dtype)  # (out, rank)

            # b1_buf
            b1 = (Y_mean - Y_mean @ V @ V.transpose(1,0))
            self.b1_buf = b1.to(self.weight.device).to(self.weight.dtype)

            # CPU -> GPU 복사
            self.register_buffer("V_buf", V.to(self.weight.device))
            self.register_buffer("b1_buf", self.b1_buf.to(self.weight.device))
            self.register_buffer("Y_sub_buf", self.Y_sub_buf.to(self.weight.device))

        # 최종 weight1, weight2 초기화
        self.weight2.data = self.V_buf
        self.weight1.data = (
            torch.transpose(self.V_buf, 1, 0).to(self.weight.device) @ self.weight
        )

        # bias도 수정
        # 남는 공간 V_prune가 있다면 처리
        V_prune = None
        # 예: V_prune = V[:, :-self.rank] 이렇게 할 수도 있지만, 코드에 따라 변경
        # 여기서는 self.rank만큼 썼으므로 남는 차원은 0. 실제로는 full covariance 시에만 가능
        # 편의상, 여기서는 self.b1_buf를 그냥 bias에 더해줍니다.
        self.bias.data = self.b1_buf.squeeze(0) + self.bias.data
        # 필요 시, 추가적인 편차 적용

        self.init = True

    def get_importance(self, input):
        input_norm = torch.norm(input.reshape(-1, input.shape[-1]), p=2, dim=0)[None, :]
        imp1 = input_norm * self.weight1.abs()
        imp1 = imp1.sum(1)
        input = F.linear(input, self.weight1, None)
        input_norm = torch.norm(input.reshape(-1, input.shape[-1]), p=2, dim=0)[None, :]
        imp2 = input_norm * self.weight2.abs()
        imp2 = imp2.sum(1)
        self.scores = imp1.tolist()

    def forward(self, input):
        if not self.init:
            self.init_lowrank(input)

        out = F.linear(
            F.linear(input, self.weight1, None),
            self.weight2,
            self.bias
        )
        if not self.mf16:
            self.make_float16()
        return out

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = DecomposeLinearEigen(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            linear_module.weight,
            linear_module.bias,
        )
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        return new_linear


class DecomposeLinearSVDPrune(DecomposeLinearSVD):
    def __init__(self, in_features, out_features, rank, budget, weight, bias):
        super(DecomposeLinearSVDPrune, self).__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            weight=weight,
            bias=bias,
        )
        self.zeta = nn.Parameter(torch.ones(1, rank, requires_grad=True, device=weight.device))
        self.mask = nn.Parameter(
            torch.ones(1, rank, requires_grad=False, device=weight.device)
        )
        self.pruned = False
        self.target_budget = budget
        if "auto" in budget:
            variance = float(self.target_budget.split(":")[-1])
            self.S = torch.cumsum(self.S.float(), 0)
            self.active_ranks = torch.searchsorted(self.S, self.S[-1] * variance).item()
            self.target_budget = self.active_ranks / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        else:
            self.active_ranks = int(
                (
                    self.in_features
                    * self.out_features
                    / (self.in_features + self.out_features)
                )
                * float(budget)
            )
            self.target_budget = float(self.target_budget)

    def forward(self, input):
        if self.pruned:
            return F.linear(
                F.linear(input, self.weight1, None),
                self.weight2,
                None,
            )
        return F.linear(
            F.linear(input, self.weight1, None) * self.get_mask(),
            self.weight2,
            None,
        )

    def hard_prune(self, calculate=True):
        if calculate:
            sorted_val, _ = torch.sort(self.zeta.abs(), 1, descending=True)
            threshold = sorted_val[0][self.active_ranks]
            self.mask.data = (
                (self.zeta.abs() >= threshold).to(self.zeta.device).to(self.zeta.dtype)
            )
            self.mask.requires_grad = False
        self.target_budget = (
            self.mask.sum()
            / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        ).item()
        self.pruned = True
        self.mask_indexes = torch.nonzero(self.mask)[:, 1]
        self.weight1 = torch.nn.Parameter(self.weight1.data[self.mask_indexes, :])
        self.weight2 = torch.nn.Parameter(self.weight2.data[:, self.mask_indexes])

    def get_mask(self):
        if self.pruned:
            return self.mask
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, rank, budget):
        new_linear = DecomposeLinearSVDPrune(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            budget,
            linear_module.weight,
            linear_module.bias,
        )
        new_linear.U = None
        new_linear.S = None
        new_linear.Vh = None
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


class DecomposeLinearEigenPrune(DecomposeLinearEigen):
    def __init__(self, in_features, out_features, rank, budget, weight, bias):
        super(DecomposeLinearEigenPrune, self).__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            weight=weight,
            bias=bias,
        )
        self.zeta = nn.Parameter(torch.ones(1, rank, requires_grad=True, device=weight.device))
        self.mask = nn.Parameter(
            torch.ones(1, rank, requires_grad=False, device=weight.device)
        )
        self.pruned = False
        self.target_budget = budget

    def init_lowrank(self, input):
        Y = F.linear(input, self.weight, None).reshape(-1, self.weight.shape[0])  # BS, out
        cov = torch.cov(torch.transpose(Y, 1, 0))  # out, out
        E, V = torch.linalg.eig(cov)  # out, out
        if "auto" in self.target_budget:
            variance = float(self.target_budget.split(":")[-1])
            E = torch.cumsum(E.float(), 0)
            self.active_ranks = torch.searchsorted(E, E[-1] * variance).item()
            self.target_budget = self.active_ranks / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        else:
            self.active_ranks = int(
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
                * float(self.target_budget)
            )
        V = V[:, : self.rank].float()  # out, rank
        self.weight2.data = V.cuda()
        self.weight1.data = (torch.transpose(V, 1, 0) @ self.weight).cuda()
        self.init = True

    def forward(self, input):
        if not self.init:
            self.init_lowrank(input)
        if self.pruned:
            return F.linear(
                F.linear(input, self.weight1, None),
                self.weight2,
                None,
            )
        return F.linear(
            F.linear(input, self.weight1, None) * self.get_mask(),
            self.weight2,
            None,
        )

    def hard_prune(self, calculate=True):
        if calculate:
            sorted_val, _ = torch.sort(self.zeta.abs(), 1, descending=True)
            threshold = sorted_val[0][self.active_ranks]
            self.mask.data = (
                (self.zeta.abs() >= threshold).to(self.zeta.device).to(self.zeta.dtype)
            )
            self.mask.requires_grad = False
        self.target_budget = (
            self.mask.sum()
            / (
                self.in_features
                * self.out_features
                / (self.in_features + self.out_features)
            )
        ).item()
        self.pruned = True
        self.mask_indexes = torch.nonzero(self.mask)[:, 1]
        self.weight1 = torch.nn.Parameter(self.weight1.data[self.mask_indexes, :])
        self.weight2 = torch.nn.Parameter(self.weight2.data[:, self.mask_indexes])

    def get_mask(self):
        if self.pruned:
            return self.mask
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, rank, budget):
        new_linear = DecomposeLinearEigenPrune(
            linear_module.in_features,
            linear_module.out_features,
            rank,
            budget,
            linear_module.weight,
            linear_module.bias,
        )
        new_linear.weight1.requires_grad = True
        new_linear.weight2.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


class ChannelPrune(torch.nn.Linear):
    def __init__(self, in_features, out_features, budget, weight, bias):
        super(ChannelPrune, self).__init__(
            in_features=in_features, out_features=out_features
        )
        self.weight = weight
        self.zeta = nn.Parameter(
            torch.ones(1, out_features, requires_grad=True, device=weight.device)
        )
        self.pruned = False
        self.threshold = 0
        self.target_budget = budget
        self.budget = 1.0

    def forward(self, input):
        # 간단하게 channel-wise 곱
        return F.linear(input, self.weight, None) * self.get_mask()

    def set_threshold(self):
        active_channels = int(
            math.sqrt(self.target_budget) * self.out_features
        )
        sorted_val, _ = torch.sort(self.zeta.abs(), 1, descending=True)
        self.threshold = sorted_val[0][active_channels]

    def set_budget(self):
        self.budget = ((self.zeta >= self.threshold).sum() / self.out_features).item()

    def get_mask(self):
        if self.pruned:
            self.set_threshold()
            self.set_budget()
            self.zeta.requires_grad = False
            return (
                (self.zeta >= self.threshold).to(self.zeta.device).to(self.zeta.dtype)
            )
        else:
            return self.zeta

    @staticmethod
    def from_linear(linear_module, budget):
        new_linear = ChannelPrune(
            linear_module.in_features,
            linear_module.out_features,
            budget,
            linear_module.weight,
            linear_module.bias,
        )
        new_linear.weight.requires_grad = True
        new_linear.zeta.requires_grad = True
        return new_linear


########################################
# 모듈 인젝션
########################################

class ModuleInjection:
    @staticmethod
    def make_decomposable(linear_module, budget, method="eigen"):
        """
        Make a (linear) layer decomposable.
        :param linear_module: A Linear module
        :return: a linear that is decomposed
        """
        in_channels = linear_module.in_features
        out_channels = linear_module.out_features
        kappa = in_channels * out_channels / (in_channels + out_channels)

        if method == "prune-eigen":
            new_linear = DecomposeLinearEigenPrune.from_linear(
                linear_module, linear_module.out_features, budget
            )
        elif method == "prune-svd":
            new_linear = DecomposeLinearSVDPrune.from_linear(
                linear_module, min(in_channels, out_channels), budget
            )
        elif method == "prune-channel":
            new_linear = ChannelPrune.from_linear(linear_module, budget)
        elif method == "eigen":
            if isinstance(budget, int):
                rank = budget
            else:
                rank = int(kappa * float(budget))
            new_linear = DecomposeLinearEigen.from_linear(linear_module, rank)
        elif method == "svd":
            if isinstance(budget, int):
                rank = budget
            else:
                rank = int(kappa * float(budget))
            new_linear = DecomposeLinearSVD.from_linear(linear_module, rank)
        else:
            for name, param in linear_module.named_parameters():
                param.requires_grad = True
            new_linear = linear_module
        linear_module = None
        return new_linear


########################################
# 피처 추출용 (Hook)
########################################

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, index=None, layers=None, return_outputs=False):
        super().__init__()
        self.model = model
        self.return_outputs = return_outputs
        idx = 0
        for name, l in self.model.named_modules():
            if isinstance(l, nn.Linear):
                for eligible_layer in layers:
                    if eligible_layer in name:
                        if idx == index:
                            self.model.hook = l.register_forward_hook(
                                self.save_outputs_hook()
                            )
                        idx += 1

    def save_outputs_hook(self):
        def fn(module, input, output):
            self._features = output.float()
            if not self.return_outputs:
                assert False

        return fn

    def forward(self, x):
        try:
            x = {k: x[k].to(self.model.device) for k in x}
            outputs = self.model(**x)
            return self._features, outputs["loss"]
        except Exception as E:
            return self._features