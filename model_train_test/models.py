import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel
from peft import get_peft_model, LoraConfig, TaskType


class Lora_ESM(nn.Module):
    def __init__(self, model_name_or_path="facebook/esm2_t12_35M_UR50D", d_model=480):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = model_name_or_path
        self.d_model = d_model

        self.peft_config = LoraConfig(
            target_modules=['query', 'out_proj', 'value', 'key', 'dense', 'regression'],
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8, lora_alpha=32, lora_dropout=0.1
        )

        self.esm = EsmModel.from_pretrained(self.model_name_or_path)
        self.lora_esm = get_peft_model(self.esm, self.peft_config)

        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(d_model // 4, 32),
            nn.BatchNorm1d(32),
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x_in):
        lora_outputs = self.lora_esm(x_in)
        last_hidden_state = lora_outputs.last_hidden_state
        out_linear = last_hidden_state.mean(dim=1)
        H = self.fc_task(out_linear)
        output = self.classifier(H)
        return output, last_hidden_state


def _make_cnn_blocks(d_model=480, cnn_num_channel=256, region_embedding_size=3,
                     cnn_kernel_size=3, cnn_padding_size=1, cnn_stride=1, pooling_size=2):
    region_cnn1 = nn.Conv1d(d_model, cnn_num_channel, region_embedding_size)
    region_cnn2 = nn.Conv1d(d_model, cnn_num_channel, region_embedding_size)
    padding1 = nn.ConstantPad1d((1, 1), 0)
    padding2 = nn.ConstantPad1d((0, 1), 0)
    relu = nn.SiLU()
    cnn1 = nn.Conv1d(cnn_num_channel, cnn_num_channel, kernel_size=cnn_kernel_size,
                     padding=cnn_padding_size, stride=cnn_stride)
    cnn2 = nn.Conv1d(cnn_num_channel, cnn_num_channel, kernel_size=cnn_kernel_size,
                     padding=cnn_padding_size, stride=cnn_stride)
    maxpooling = nn.MaxPool1d(kernel_size=pooling_size)
    return region_cnn1, region_cnn2, padding1, padding2, relu, cnn1, cnn2, maxpooling


class TransHLA2_0_BIND(nn.Module):
    def __init__(self, lora_esm: Lora_ESM, d_model=480, n_layers=4, n_head=8, d_ff=64,
                 cnn_num_channel=256):
        super().__init__()
        self.lora_esm = lora_esm

        (self.region_cnn1, self.region_cnn2, self.padding1, self.padding2, self.relu,
         self.cnn1, self.cnn2, self.maxpooling) = _make_cnn_blocks(d_model=d_model, cnn_num_channel=cnn_num_channel)

        self.epitope_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.epitope_transformer_encoder = nn.TransformerEncoder(
            self.epitope_transformer_layers, num_layers=n_layers)

        self.hla_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.hla_transformer_encoder = nn.TransformerEncoder(
            self.hla_transformer_layers, num_layers=n_layers)

        self.cross_attention_epitope_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_head, dropout=0.2) for _ in range(4)])
        self.cross_attention_hla_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_head, dropout=0.2) for _ in range(4)])

        self.bn1 = nn.BatchNorm1d(cnn_num_channel)
        self.bn2 = nn.BatchNorm1d(cnn_num_channel)
        self.fc_task = nn.Sequential(
            nn.Linear(2 * d_model + 2 * cnn_num_channel, 2 * (d_model + cnn_num_channel) // 4),
            nn.BatchNorm1d(2 * (d_model + cnn_num_channel) // 4),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(2 * (d_model + cnn_num_channel) // 4, 96),
            nn.BatchNorm1d(96),
        )
        self.classifier = nn.Linear(96, 2)

    def cnn_block1(self, x):
        return self.cnn1(self.relu(x))

    def cnn_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn1(x)
        x = px + x
        return x

    def structure_block1(self, x):
        return self.cnn2(self.relu(x))

    def structure_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = px + x
        return x

    def forward(self, epitope_in, hla_in):
        _, epitope_emb = self.lora_esm(epitope_in)
        _, hla_emb = self.lora_esm(hla_in)

        epitope_trans = self.epitope_transformer_encoder(epitope_emb.transpose(0, 1))
        hla_trans = self.hla_transformer_encoder(hla_emb.transpose(0, 1))

        for ca_e, ca_h in zip(self.cross_attention_epitope_layers, self.cross_attention_hla_layers):
            epitope_trans, _ = ca_e(epitope_trans, hla_trans, hla_trans)
            hla_trans, _ = ca_h(hla_trans, epitope_trans, epitope_trans)

        epitope_mean = epitope_trans.mean(dim=0)
        hla_mean = hla_trans.mean(dim=0)

        epitope_cnn_emb = self.region_cnn1(epitope_emb.transpose(1, 2))
        epitope_cnn_emb = self.padding1(epitope_cnn_emb)
        conv = epitope_cnn_emb + self.cnn_block1(self.cnn_block1(epitope_cnn_emb))
        while conv.size(-1) >= 2:
            conv = self.cnn_block2(conv)
        epitope_cnn_out = torch.squeeze(conv, dim=-1)
        epitope_cnn_out = self.bn1(epitope_cnn_out)

        hla_cnn_emb = self.region_cnn2(hla_emb.transpose(1, 2))
        hla_cnn_emb = self.padding1(hla_cnn_emb)
        hla_conv = hla_cnn_emb + self.structure_block1(self.structure_block1(hla_cnn_emb))
        while hla_conv.size(-1) >= 2:
            hla_conv = self.structure_block2(hla_conv)
        hla_cnn_out = torch.squeeze(hla_conv, dim=-1)
        hla_cnn_out = self.bn2(hla_cnn_out)

        representation = torch.cat((epitope_mean, hla_mean, epitope_cnn_out, hla_cnn_out), dim=1)
        reduction_feature = self.fc_task(representation)
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = F.softmax(logits_clsf, dim=1)
        return logits_clsf, reduction_feature


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from peft import get_peft_model, LoraConfig, TaskType


class TransHLA2_0_IM(nn.Module):
    def __init__(self,
                 esm_model_name: str = "facebook/esm2_t12_35M_UR50D",
                 d_model: int = 480,
                 n_layers: int = 4,
                 n_head: int = 8,
                 d_ff: int = 64,
                 cnn_num_channel: int = 256,
                 region_embedding_size: int = 3,
                 cnn_kernel_size: int = 3,
                 cnn_padding_size: int = 1,
                 cnn_stride: int = 1,
                 pooling_size: int = 2,
                 cross_layers: int = 4,
                 lora_r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        super().__init__()

        # LoRA 配置
        self.peft_config = LoraConfig(
            target_modules=['query', 'out_proj', 'value', 'key', 'dense', 'regression'],
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # 载入同一 ESM 底座，并分别套上 LoRA 适配为 epitope/hla 两个编码器
        base_esm = EsmModel.from_pretrained(esm_model_name)
        self.epitope_lora = get_peft_model(base_esm, self.peft_config)

        # 注意：需要再构建一个独立实例，否则两路将完全共享 LoRA 权重
        base_esm_hla = EsmModel.from_pretrained(esm_model_name)
        self.hla_lora = get_peft_model(base_esm_hla, self.peft_config)

        # 卷积分支
        self.region_cnn1 = nn.Conv1d(d_model, cnn_num_channel, region_embedding_size)
        self.region_cnn2 = nn.Conv1d(d_model, cnn_num_channel, region_embedding_size)
        self.padding1 = nn.ConstantPad1d((1, 1), 0)
        self.padding2 = nn.ConstantPad1d((0, 1), 0)
        self.relu = nn.SiLU()
        self.cnn1 = nn.Conv1d(cnn_num_channel, cnn_num_channel,
                              kernel_size=cnn_kernel_size, padding=cnn_padding_size, stride=cnn_stride)
        self.cnn2 = nn.Conv1d(cnn_num_channel, cnn_num_channel,
                              kernel_size=cnn_kernel_size, padding=cnn_padding_size, stride=cnn_stride)
        self.maxpooling = nn.MaxPool1d(kernel_size=pooling_size)

        # 各自的 Transformer Encoder
        self.epitope_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2, batch_first=False)
        self.epitope_transformer_encoder = nn.TransformerEncoder(
            self.epitope_transformer_layers, num_layers=n_layers)

        self.hla_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2, batch_first=False)
        self.hla_transformer_encoder = nn.TransformerEncoder(
            self.hla_transformer_layers, num_layers=n_layers)

        # Cross Attention 堆叠
        self.cross_attention_epitope_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=False) for _ in range(cross_layers)]
        )
        self.cross_attention_hla_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=False) for _ in range(cross_layers)]
        )

        # 归一化与分类头
        self.bn1 = nn.BatchNorm1d(cnn_num_channel)
        self.bn2 = nn.BatchNorm1d(cnn_num_channel)
        self.fc_task = nn.Sequential(
            nn.Linear(2 * d_model + 2 * cnn_num_channel, 2 * (d_model + cnn_num_channel) // 4),
            nn.BatchNorm1d(2 * (d_model + cnn_num_channel) // 4),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(2 * (d_model + cnn_num_channel) // 4, 96),
            nn.BatchNorm1d(96),
        )
        self.classifier = nn.Linear(96, 2)

    # CNN 辅助块
    def cnn_block1(self, x):
        return self.cnn1(self.relu(x))

    def cnn_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn1(x)
        x = px + x
        return x

    def structure_block1(self, x):
        return self.cnn2(self.relu(x))

    def structure_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = px + x
        return x

    def forward(self, epitope_in: torch.Tensor, hla_in: torch.Tensor):
        """
        epitope_in, hla_in: LongTensor [B, L]
        返回：
          - logits_clsf: [B, 2], softmax 后的概率
          - representation: [B, 2*d_model + 2*cnn_num_channel] 前馈前的融合表征
        """
        # ESM + LoRA 编码
        epitope_emb = self.epitope_lora(epitope_in).last_hidden_state  # [B, Le, d_model]
        hla_emb = self.hla_lora(hla_in).last_hidden_state              # [B, Lh, d_model]

        # 各自 transformer
        epitope_trans = self.epitope_transformer_encoder(epitope_emb.transpose(0, 1))  # [Le, B, d]
        hla_trans = self.hla_transformer_encoder(hla_emb.transpose(0, 1))              # [Lh, B, d]

        # 双向 Cross-Attention 堆叠
        for cross_attention_epitope, cross_attention_hla in zip(
            self.cross_attention_epitope_layers, self.cross_attention_hla_layers
        ):
            epitope_trans, _ = cross_attention_epitope(epitope_trans, hla_trans, hla_trans)
            hla_trans, _ = cross_attention_hla(hla_trans, epitope_trans, epitope_trans)

        # 平均池化到句级别
        epitope_mean = epitope_trans.mean(dim=0)  # [B, d_model]
        hla_mean = hla_trans.mean(dim=0)          # [B, d_model]

        # CNN 分支（对原始 token 级 embedding）
        epitope_cnn_emb = self.region_cnn1(epitope_emb.transpose(1, 2))  # [B, C, Le-2]
        epitope_cnn_emb = self.padding1(epitope_cnn_emb)
        conv = epitope_cnn_emb + self.cnn_block1(self.cnn_block1(epitope_cnn_emb))
        while conv.size(-1) >= 2:
            conv = self.cnn_block2(conv)
        epitope_cnn_out = torch.squeeze(conv, dim=-1)  # [B, C]
        epitope_cnn_out = self.bn1(epitope_cnn_out)

        hla_cnn_emb = self.region_cnn2(hla_emb.transpose(1, 2))          # [B, C, Lh-2]
        hla_cnn_emb = self.padding1(hla_cnn_emb)
        hla_conv = hla_cnn_emb + self.structure_block1(self.structure_block1(hla_cnn_emb))
        while hla_conv.size(-1) >= 2:
            hla_conv = self.structure_block2(hla_conv)
        hla_cnn_out = torch.squeeze(hla_conv, dim=-1)  # [B, C]
        hla_cnn_out = self.bn2(hla_cnn_out)

        # 融合与分类
        representation = torch.cat((epitope_mean, hla_mean, epitope_cnn_out, hla_cnn_out), dim=1)
        reduction_feature = self.fc_task(representation)
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = F.softmax(logits_clsf, dim=1)
        return logits_clsf, representation

class NoCNNModel(nn.Module):
    def __init__(self, lora_esm: Lora_ESM, d_model=480, n_layers=4, n_head=8, d_ff=64):
        super().__init__()
        self.lora_esm = lora_esm
        self.epitope_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.epitope_transformer_encoder = nn.TransformerEncoder(self.epitope_transformer_layers, num_layers=n_layers)
        self.hla_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.hla_transformer_encoder = nn.TransformerEncoder(self.hla_transformer_layers, num_layers=n_layers)

        self.cross_attention_epitope_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_head, dropout=0.2) for _ in range(4)])
        self.cross_attention_hla_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_head, dropout=0.2) for _ in range(4)])

        self.fc_task = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(d_model, 96),
            nn.BatchNorm1d(96),
        )
        self.classifier = nn.Linear(96, 2)

    def forward(self, epitope_in, hla_in):
        _, epitope_emb = self.lora_esm(epitope_in)
        _, hla_emb = self.lora_esm(hla_in)

        epitope_trans = self.epitope_transformer_encoder(epitope_emb.transpose(0, 1))
        hla_trans = self.hla_transformer_encoder(hla_emb.transpose(0, 1))

        for ca_e, ca_h in zip(self.cross_attention_epitope_layers, self.cross_attention_hla_layers):
            epitope_trans, _ = ca_e(epitope_trans, hla_trans, hla_trans)
            hla_trans, _ = ca_h(hla_trans, epitope_trans, epitope_trans)

        epitope_mean = epitope_trans.mean(dim=0)
        hla_mean = hla_trans.mean(dim=0)

        representation = torch.cat((epitope_mean, hla_mean), dim=1)
        reduction_feature = self.fc_task(representation)
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = F.softmax(logits_clsf, dim=1)
        return logits_clsf, representation


class NoTransformerModel(nn.Module):
    def __init__(self, lora_esm: Lora_ESM, d_model=480, cnn_num_channel=256):
        super().__init__()
        self.lora_esm = lora_esm
        (self.region_cnn1, self.region_cnn2, self.padding1, self.padding2, self.relu,
         self.cnn1, self.cnn2, self.maxpooling) = _make_cnn_blocks(d_model=d_model, cnn_num_channel=cnn_num_channel)

        self.bn1 = nn.BatchNorm1d(cnn_num_channel)
        self.bn2 = nn.BatchNorm1d(cnn_num_channel)
        self.fc_task = nn.Sequential(
            nn.Linear(2 * d_model + 2 * cnn_num_channel, 2 * (d_model + cnn_num_channel) // 4),
            nn.BatchNorm1d(2 * (d_model + cnn_num_channel) // 4),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(2 * (d_model + cnn_num_channel) // 4, 96),
            nn.BatchNorm1d(96),
        )
        self.classifier = nn.Linear(96, 2)

    def cnn_block1(self, x):
        return self.cnn1(self.relu(x))

    def cnn_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn1(x)
        x = px + x
        return x

    def structure_block1(self, x):
        return self.cnn2(self.relu(x))

    def structure_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = px + x
        return x

    def forward(self, epitope_in, hla_in):
        _, epitope_emb = self.lora_esm(epitope_in)
        _, hla_emb = self.lora_esm(hla_in)

        epitope_mean = epitope_emb.mean(dim=1)
        hla_mean = hla_emb.mean(dim=1)

        epitope_cnn_emb = self.region_cnn1(epitope_emb.transpose(1, 2))
        epitope_cnn_emb = self.padding1(epitope_cnn_emb)
        conv = epitope_cnn_emb + self.cnn_block1(self.cnn_block1(epitope_cnn_emb))
        while conv.size(-1) >= 2:
            conv = self.cnn_block2(conv)
        epitope_cnn_out = torch.squeeze(conv, dim=-1)
        epitope_cnn_out = self.bn1(epitope_cnn_out)

        hla_cnn_emb = self.region_cnn2(hla_emb.transpose(1, 2))
        hla_cnn_emb = self.padding1(hla_cnn_emb)
        hla_conv = hla_cnn_emb + self.structure_block1(self.structure_block1(hla_cnn_emb))
        while hla_conv.size(-1) >= 2:
            hla_conv = self.structure_block2(hla_conv)
        hla_cnn_out = torch.squeeze(hla_conv, dim=-1)
        hla_cnn_out = self.bn2(hla_cnn_out)

        representation = torch.cat((epitope_mean, hla_mean, epitope_cnn_out, hla_cnn_out), dim=1)
        reduction_feature = self.fc_task(representation)
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = F.softmax(logits_clsf, dim=1)
        return logits_clsf, representation


class NoCrossAttentionModel(nn.Module):
    def __init__(self, lora_esm: Lora_ESM, d_model=480, n_layers=4, n_head=8, d_ff=64,
                 cnn_num_channel=256):
        super().__init__()
        self.lora_esm = lora_esm
        (self.region_cnn1, self.region_cnn2, self.padding1, self.padding2, self.relu,
         self.cnn1, self.cnn2, self.maxpooling) = _make_cnn_blocks(d_model=d_model, cnn_num_channel=cnn_num_channel)

        self.epitope_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.epitope_transformer_encoder = nn.TransformerEncoder(self.epitope_transformer_layers, num_layers=n_layers)

        self.hla_transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.hla_transformer_encoder = nn.TransformerEncoder(self.hla_transformer_layers, num_layers=n_layers)

        self.bn1 = nn.BatchNorm1d(cnn_num_channel)
        self.bn2 = nn.BatchNorm1d(cnn_num_channel)
        self.fc_task = nn.Sequential(
            nn.Linear(2 * d_model + 2 * cnn_num_channel, 2 * (d_model + cnn_num_channel) // 4),
            nn.BatchNorm1d(2 * (d_model + cnn_num_channel) // 4),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(2 * (d_model + cnn_num_channel) // 4, 96),
            nn.BatchNorm1d(96),
        )
        self.classifier = nn.Linear(96, 2)

    def cnn_block1(self, x):
        return self.cnn1(self.relu(x))

    def cnn_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn1(x)
        x = px + x
        return x

    def structure_block1(self, x):
        return self.cnn2(self.relu(x))

    def structure_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = px + x
        return x

    def forward(self, epitope_in, hla_in):
        _, epitope_emb = self.lora_esm(epitope_in)
        _, hla_emb = self.lora_esm(hla_in)

        epitope_trans = self.epitope_transformer_encoder(epitope_emb.transpose(0, 1))
        hla_trans = self.hla_transformer_encoder(hla_emb.transpose(0, 1))

        epitope_mean = epitope_trans.mean(dim=0)
        hla_mean = hla_trans.mean(dim=0)

        epitope_cnn_emb = self.region_cnn1(epitope_emb.transpose(1, 2))
        epitope_cnn_emb = self.padding1(epitope_cnn_emb)
        conv = epitope_cnn_emb + self.cnn_block1(self.cnn_block1(epitope_cnn_emb))
        while conv.size(-1) >= 2:
            conv = self.cnn_block2(conv)
        epitope_cnn_out = torch.squeeze(conv, dim=-1)
        epitope_cnn_out = self.bn1(epitope_cnn_out)

        hla_cnn_emb = self.region_cnn2(hla_emb.transpose(1, 2))
        hla_cnn_emb = self.padding1(hla_cnn_emb)
        hla_conv = hla_cnn_emb + self.structure_block1(self.structure_block1(hla_cnn_emb))
        while hla_conv.size(-1) >= 2:
            hla_conv = self.structure_block2(hla_conv)
        hla_cnn_out = torch.squeeze(hla_conv, dim=-1)
        hla_cnn_out = self.bn2(hla_cnn_out)

        representation = torch.cat((epitope_mean, hla_mean, epitope_cnn_out, hla_cnn_out), dim=1)
        reduction_feature = self.fc_task(representation)
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = F.softmax(logits_clsf, dim=1)
        return logits_clsf, representation


def reinit_classifier(model: nn.Module):
    for layer in model.classifier.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)