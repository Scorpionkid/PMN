class EnhancedDenoisePath(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(EnhancedDenoisePath, self).__init__()

        # 注意力模块处理特征并生成注意力图
        self.attn = TE_MDTA(channels, num_heads=num_heads)

        # 内容处理分支
        self.content_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

        # 注意力引导的门控生成
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, noise_map=None, texture_mask=None):
        # 全局上下文处理
        context = self.attn(x, texture_mask)

        # 内容处理
        content = self.content_branch(x)

        # 融合原始特征和上下文特征生成门控
        gate_input = torch.cat([x, context], dim=1)
        gate = self.gate_generator(gate_input)

        # 噪声调制
        if noise_map is not None:
            # 使噪声图直接影响门控强度
            noise_weight = torch.sigmoid(4.0 * noise_map - 2.0)  # 映射到(0,1)
            gate = gate * (0.5 + 0.5 * noise_weight)  # 噪声越大，门控越强

        return x + content * gate