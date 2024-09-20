import torch


def load_data(file_path, num=None):
    with open(file_path, "r", encoding="utf-8") as fp:
        all_data = fp.read().split("\n\n")
    if num is None:
        return all_data[:-1]
    else:
        return all_data[:-1][:num]


def load_vocab(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        index_2_word = fp.read().split()
        word_2_index = {key: value for value, key in enumerate(index_2_word)}
        return word_2_index, index_2_word


class GptDataset():
    def __init__(self, all_data, word_2_index):
        super().__init__()
        self.all_data = all_data
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        text_data = self.all_data[index].split("\n")

        text_idx = []
        for text in text_data:
            text_idx.extend([self.word_2_index.get(i, self.word_2_index["<unk>"]) for i in text])
            text_idx.append(self.word_2_index["<sep>"])

        input_idx = text_idx[:-1]
        label_idx = text_idx[1:]

        assert len(input_idx) == len(label_idx)

        return input_idx, label_idx, len(label_idx)

    def __len__(self):
        return len(self.all_data)

    def pro_data(self, batch_data):
        batch_text_idx, batch_label_idx, batch_len_idx = zip(*batch_data)
        batch_max_len = max(batch_len_idx)
        batch_new_text_idx, batch_new_label_idx = [], []
        for text_idx, label_idx in zip(batch_text_idx, batch_label_idx):
            batch_new_text_idx.append(text_idx + [self.word_2_index["<pad>"]] * (batch_max_len - len(text_idx)))
            batch_new_label_idx.append(label_idx + [self.word_2_index["<pad>"]] * (batch_max_len - len(label_idx)))
        return torch.tensor(batch_new_text_idx), torch.tensor(batch_new_label_idx),
