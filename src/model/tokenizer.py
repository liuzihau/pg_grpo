# src/model/tokenizer.py
from transformers import AutoTokenizer

def load_tokenizer_leftpad(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok
