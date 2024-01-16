import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seg_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seg_len = seg_len

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add SOS, EOS and padding to each sentence
        enc_num_padding_token = (
            self.seg_len - len(enc_input_tokens) - 2
        )  # We will add  <SOS> and <EOS> tokens
        dec_num_padding_token = (
            self.seg_len - len(dec_input_tokens) - 1
        )  # We will add <EOS> token

        # Make sur the number of paddiing tokens is not negative. If it is the sentence is too long
        if enc_num_padding_token < 0 or dec_num_padding_token < 0:
            raise Exception("The sentence is too long")

        # Add SOS and EOS tokens to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_token, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add SOS token to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_token, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only EOS token to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_token, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the length of the tensors to make sur they are all seq_len long
        assert encoder_input.shape[0] == self.seg_len
        assert decoder_input.shape[0] == self.seg_len
        assert label.shape[0] == self.seg_len

        return {
            "encoder_input": encoder_input,  # Shape: (seq_len,)
            "decoder_input": decoder_input,  # Shape: (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # Shape: (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.shape[0]
            ),  # Shape: (1, seq_len) &  (1, seq_len, seq_len)
            "label": label,  # Shape: (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
