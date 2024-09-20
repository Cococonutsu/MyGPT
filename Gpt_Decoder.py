import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding层的两个参数
        # 第一个参数:要确保能够涵盖所有数据集中的数据
        #         【vocab_size】训练数据中所有不同的字的个数
        #         【max_seq_len】训练数据中允许出现的最大的一句话的长度
        # 第二个参数:要将数据映射到的维度-->hidden_size
        self.word_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.pos_embedding = nn.Embedding(config["max_seq_len"], config["hidden_size"])

    def forward(self, x):
        word_emb = self.word_embedding(x)
        pos_idx = torch.arange(0, x.shape[1], device=x.device)
        pos_idx = pos_idx.reshape(1, -1)
        pos_idx = pos_idx.repeat(x.shape[0], 1)
        pos_emb = self.pos_embedding(pos_idx)
        final_emb = word_emb + pos_emb
        return final_emb


class Feed_Forward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config["hidden_size"], config["feed_layer_num"])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config["feed_layer_num"], config["hidden_size"])

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.k = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.v = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attention_head_num, atten_mask):
        batch, seq_len, hidden_size = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.reshape(batch, seq_len, attention_head_num, -1).transpose(1, 2)
        k = k.reshape(batch, seq_len, attention_head_num, -1).transpose(1, 2)
        v = v.reshape(batch, seq_len, attention_head_num, -1).transpose(1, 2)

        atten_mask = atten_mask.expand(-1, attention_head_num, -1, seq_len)
        look_ahead_mask = torch.triu(torch.ones_like(atten_mask), 1).to(x.device)
        mask = (atten_mask + look_ahead_mask) >= 1

        weight = (q @ k.transpose(-1, -2)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
        weight.masked_fill_(mask, -1e9)  # 该函数会将所有的True置为-1e9
        att = self.softmax(weight) @ v
        att = att.transpose(1, 2).reshape(batch, seq_len, hidden_size)
        return att


class Decoder_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_1 = Multi_Head_Attention(config)
        self.attention_2 = Multi_Head_Attention(config)
        self.feed_forward = Feed_Forward(config)
        self.layernorm = nn.LayerNorm(config["hidden_size"])
        self.config = config

    def forward(self, x, atten_mask):
        att_1_out = self.attention_1(x, self.config["attention_head_num"], atten_mask)
        add_1_out = att_1_out + x
        layer_1_out = self.layernorm(add_1_out)

        att_2_out = self.attention_2(layer_1_out, self.config["attention_head_num"], atten_mask)
        add_2_out = att_2_out + layer_1_out
        layer_2_out = self.layernorm(add_2_out)

        feed_out = self.feed_forward(layer_2_out)
        add_3_out = feed_out + layer_2_out

        return add_3_out


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.decoder_block = nn.ModuleList(
            [Decoder_Block(config) for i in range(config["decoder_block_num"])]
        )

    def forward(self, x):
        atten_mask = get_attention_mask(x)
        x = self.embedding(x)  # (batch, seq_len, hidden_size)

        for block in self.decoder_block:
            x = block(x, atten_mask)
        return x


class GPT_Model(nn.Module):
    def __init__(self, config, word_2_index):
        super().__init__()
        self.config = config
        self.word_2_index = word_2_index
        self.decoder = Decoder(config)
        self.cls = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, label=None):  # (batch, seq_len)
        decoder_out = self.decoder(x)
        pre = self.cls(decoder_out)
        if label is not None:
            loss = self.loss_func(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)

    def answer(self, input_text):
        input_idx = [self.word_2_index.get(i, self.word_2_index["<unk>"])if i != "\n" else self.word_2_index["<sep>"]
                     for i in input_text]
        input_idx = torch.tensor([input_idx], device=self.config["device"])

        a = 1
        while True:
            pre = int(self.forward(input_idx)[0][-1])
            input_idx = torch.cat([input_idx, torch.tensor([[pre]], device=input_idx.device)], dim=-1)
            if pre == self.word_2_index["<sep>"]:
                break
        return input_idx[-1]


def get_attention_mask(x):
    '''
    在计算Attention之前将<PAD>置为0
    因为在经过embedding后，<PAD>部分被置为随机数
    因此这种操作的目的时为了防止<PAD>参与计算影响softmax的结果
    :param x:
    :return:
    '''
    padding_position = (x == 0)  # x = 0表示<PAD>,经此操作后所有<PAD>部分都会被置为True（要被mask掉的位置），非<PAD>部分置为False
    padding_position = padding_position.reshape(*padding_position.shape, -1)  # batch seq_len 1(1用于后续拓展到hidden_size维度)
    padding_position = padding_position.unsqueeze(1)
    return padding_position  # batch; 1(用于后续拓展到head_num); seq_len; 1(1用于后续拓展到hidden_size维度)
