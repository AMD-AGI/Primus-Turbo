import torch
from basic_transformer import BasicTransformerModel
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [sample["text"] for sample in dataset if sample["text"].strip()][:50000]


seqlen = 4096
enc = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=seqlen,
    return_tensors="pt",
)
_token_bank = enc["input_ids"]  # shape: (N, seqlen)
_num_tokens = _token_bank.size(0)


def get_real_batch(step: int, batch_size: int):
    idx = [(step * batch_size + i) % _num_tokens for i in range(batch_size)]
    return _token_bank[idx].cuda()


def benchmark_loss(model, vocab_size=32000, steps=1000, batch_size=2, seqlen=4096, lr=1e-4):
    model = model.cuda()
    model.train()
    model = model.to(torch.bfloat16)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in tqdm(range(steps)):
        input_ids = get_real_batch(step, batch_size)
        targets = input_ids.clone()

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        loss.backward()
        optim.step()
        optim.zero_grad()

        if step % 100 == 0:
            print(f"[{step}] loss: {loss.item():.6f}")


if __name__ == "__main__":
    model = BasicTransformerModel()
    benchmark_loss(model, steps=5000)
