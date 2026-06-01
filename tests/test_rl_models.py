"""RL 神经网络模型测试。"""
import pytest
import numpy as np
import torch
from rl.models import StockScoreNet


class TestStockScoreNet:
    """StockScoreNet 测试。"""

    def test_forward_pass_output_shape(self):
        """前向传播应输出 (batch, 1) 形状的张量。"""
        n_factors = 10
        n_context = 4
        net = StockScoreNet(n_factors=n_factors, n_context=n_context)

        batch_size = 32
        # 模拟观测：因子 + 环境
        x = torch.randn(batch_size, n_factors + n_context)
        with torch.no_grad():
            out = net(x)

        assert out.shape == (batch_size, 1)

    def test_output_in_range_0_to_1(self):
        """输出应在 [0, 1] 范围内（sigmoid 激活）。"""
        net = StockScoreNet(n_factors=8, n_context=4)
        x = torch.randn(100, 12)
        with torch.no_grad():
            out = net(x)

        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_train_mode_gradients_flow(self):
        """训练模式下梯度应能正常反向传播。"""
        net = StockScoreNet(n_factors=5, n_context=4)
        net.train()

        x = torch.randn(16, 9, requires_grad=True)
        out = net(x)
        loss = out.mean()
        loss.backward()

        # 检查参数梯度
        for name, param in net.named_parameters():
            assert param.grad is not None, f"参数 {name} 无梯度"
            assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度为 NaN"

    def test_eval_mode_no_grad(self):
        """评估模式下不应有梯度计算。"""
        net = StockScoreNet(n_factors=5, n_context=4)
        net.eval()

        with torch.no_grad():
            out = net(torch.randn(10, 9))
        assert not out.requires_grad

    def test_handles_variable_batch_size(self):
        """应能处理不同的 batch 大小。"""
        net = StockScoreNet(n_factors=6, n_context=4)

        for bs in [1, 10, 100]:
            with torch.no_grad():
                out = net(torch.randn(bs, 10))
            assert out.shape == (bs, 1)
