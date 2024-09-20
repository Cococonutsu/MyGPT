import os
import torch
from data import load_data, load_vocab, GptDataset
from Gpt_Decoder import GPT_Model

if __name__ == "__main__":
    train_data_file = os.path.join("./data", "train.txt")
    all_data = load_data(train_data_file, num=10000)

    vocab_txt_file = os.path.join("./data", "vocab.txt")
    word_2_index, index_2_word = load_vocab(vocab_txt_file)

    config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "batch_size": 3,
        "epochs": 10,
        "shuffle": True,
        "hidden_size": 768,
        "vocab_size": len(word_2_index),
        "max_seq_len": 512,
        "feed_layer_num": 1024,
        "attention_head_num": 4,
        "decoder_block_num": 5,
    }

    model = GPT_Model(config, word_2_index).to(config["device"])
    input_text = input("请输入一段话：") + "\n"
    out_idx = model.answer(input_text=input_text)
    out_text = [index_2_word[i] for i in out_idx]
    print(out_text)

