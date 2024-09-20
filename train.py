import os
import torch
from torch.utils.data import DataLoader
from data import load_data, load_vocab, GptDataset
from Gpt_Decoder import GPT_Model
from tqdm import tqdm

if __name__ == "__main__":
    train_data_file = os.path.join("./data", "train.txt")
    all_data = load_data(train_data_file, num=100)

    vocab_txt_file = os.path.join("./data", "vocab.txt")
    word_2_index, index_2_word = load_vocab(vocab_txt_file)

    config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "lr": 1e-4,
        "batch_size": 10,
        "epochs": 50,
        "shuffle": True,
        "hidden_size": 768,
        "vocab_size": len(word_2_index),
        "max_seq_len": 512,
        "feed_layer_num": 1024,
        "attention_head_num": 4,
        "decoder_block_num": 5,
    }

    gptdataset = GptDataset(all_data, word_2_index)
    gptdataloader = DataLoader(gptdataset, batch_size=config["batch_size"], shuffle=config["shuffle"],
                            collate_fn=gptdataset.pro_data)

    model = GPT_Model(config, word_2_index).to(config["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        progress_bar = tqdm(gptdataloader, desc=f"Training【{epoch}】")
        for batch_data in progress_bar:
            batch_data = [tensor.to(config["device"]) for tensor in batch_data]
            batch_text, batch_label = batch_data

            loss = model.forward(batch_text, batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

    input_text = input("请输入一段话：") + "\n"
    out_idx = model.answer(input_text=input_text)
    out_text = [index_2_word[i] for i in out_idx]
    print(out_text)