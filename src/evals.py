"""Sample quality: generative perplexity under an AR judge (Duo protocol) and mean unigram entropy.
Dims: B batch, L sequence length, V judge vocab."""
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_judge(name, device):
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    return tok, AutoModelForCausalLM.from_pretrained(name).to(device).eval()


def entropy(tokens_BL):
    """Mean over samples of the entropy of each sample's unigram distribution (nats)."""
    ents = [torch.special.entr(row.unique(return_counts=True)[1].float() / len(row)).sum() for row in tokens_BL]
    return torch.stack(ents).mean().item()


@torch.no_grad()
def gen_ppl(texts, tok, judge, batch_size, max_length=1024):
    """exp(mean NLL) of `texts` re-tokenized for the judge. Counts non-EOS tokens plus the run up to the first
    second EOS, as metrics.py:record_generative_perplexity in s-sahoo/duo does; padding is excluded."""
    device, eos = next(judge.parameters()).device, tok.eos_token_id
    nll, n = 0.0, 0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        x_BL, mask_BL = enc["input_ids"].to(device), enc["attention_mask"].to(device).bool()
        logits_BLV = judge(x_BL, attention_mask=mask_BL).logits
        nll_BL = F.cross_entropy(logits_BLV[:, :-1].transpose(1, 2).float(), x_BL[:, 1:], reduction="none")
        first_eos_BL = (x_BL == eos).cumsum(-1) == 1
        valid_BL = ((first_eos_BL | (x_BL != eos)) & mask_BL)[:, 1:]
        nll, n = nll + (nll_BL * valid_BL).sum().item(), n + valid_BL.sum().item()
    return math.exp(nll / n)
