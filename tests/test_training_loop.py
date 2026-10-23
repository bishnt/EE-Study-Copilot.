"""Training-loop tests on a tiny CPU model. They need torch + peft + colpali-engine (the loss)
and are skipped where those are not installed, e.g. the light CI job."""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
pytest.importorskip('colpali_engine.loss')
warnings.filterwarnings('ignore')

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from colpali_engine.loss import ColbertLoss  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from training import data as D  # noqa: E402
from training.train_colpali import (TARGET_REGEX, PairCollator, evaluate_pairs, find_lora_targets,  # noqa: E402
                                    fit)

V, DIM = 60, 16


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.ModuleDict({'q_proj': nn.Linear(DIM, DIM), 'v_proj': nn.Linear(DIM, DIM)})
        self.mlp = nn.ModuleDict({'up_proj': nn.Linear(DIM, DIM)})

    def forward(self, x):
        return x + torch.tanh(self.mlp['up_proj'](self.self_attn['v_proj'](x) + self.self_attn['q_proj'](x)))


class Tiny(nn.Module):
    """Same layer naming as ColPali: language_model.layers.N.self_attn.q_proj, a vision tower, and
    custom_text_proj."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(V, DIM)
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([Block(), Block()])
        self.vision_tower = nn.Module()
        self.vision_tower.layers = nn.ModuleList([Block()])
        self.custom_text_proj = nn.Linear(DIM, 8)

    def forward(self, input_ids, attention_mask, **_):
        x = self.embed(input_ids)
        for layer in self.language_model.layers:
            x = layer(x)
        return F.normalize(self.custom_text_proj(x), dim=-1) * attention_mask.unsqueeze(-1)


def encode(seqs):
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs])
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
    return {'input_ids': ids, 'attention_mask': mask}


class FakeProcessor:
    def process_queries(self, texts):
        return encode([[int(t) for t in s.split()] for s in texts])

    def process_images(self, images):
        return encode(images)


def toy_pairs(n_pages=12):
    """Page p is the token triple (3p+1..3p+3) plus 2 noise tokens; queries mention 2 of the 3."""
    pairs = []
    for p in range(1, n_pages + 1):
        t = [3 * p + 1, 3 * p + 2, 3 * p + 3]
        for sub in ([t[0], t[1]], [t[1], t[2]], [t[0], t[2]]):
            pairs.append({'query': ' '.join(map(str, sub + [58])), 'source': 'a.pdf',
                          'subject': 'dsp', 'page': p, 'doc': t + [57, 58]})
    return pairs


def example(pair):
    return {'query': pair['query'], 'image': pair['doc']}


def peft_tiny(r=4):
    return get_peft_model(Tiny(), LoraConfig(r=r, lora_alpha=8, target_modules=TARGET_REGEX))


# ---- LoRA targeting ---------------------------------------------------------------------------------

def test_lora_targets_language_model_and_projection_but_not_vision_tower():
    names = find_lora_targets(Tiny())
    assert 'custom_text_proj' in names
    assert 'language_model.layers.0.self_attn.q_proj' in names
    assert 'language_model.layers.1.mlp.up_proj' in names
    assert not any('vision' in n for n in names) and len(names) == 7


def test_lora_regex_failure_is_loud():
    with pytest.raises(ValueError, match='matched no Linear'):
        find_lora_targets(nn.Sequential(nn.Linear(2, 2)))


def test_peft_wrapping_matches_the_targets():
    model = peft_tiny()
    lora_layers = [n for n, m in model.named_modules() if hasattr(m, 'lora_A')]
    assert len(lora_layers) == 7 and not any('vision' in n for n in lora_layers)


# ---- collator --------------------------------------------------------------------------------------

def test_collator_prefixes_keys_and_pads():
    batch = PairCollator(FakeProcessor())([{'query': '1 2', 'image': [5, 6, 7]},
                                           {'query': '3 4 5', 'image': [8]}])
    assert set(batch) == {'query_input_ids', 'query_attention_mask', 'doc_input_ids', 'doc_attention_mask'}
    assert batch['query_input_ids'].shape == (2, 3) and batch['doc_attention_mask'].sum() == 4


# ---- the loop ---------------------------------------------------------------------------------------

def run_fit(model, pairs, **kw):
    logs = []
    out = fit(model, pairs, example, PairCollator(FakeProcessor()), ColbertLoss(),
              log=logs.append, **{'epochs': 6, 'batch_size': 4, 'lr': 1e-2, **kw})
    return out, logs


def test_training_lowers_loss_and_only_touches_lora_weights():
    model = peft_tiny()
    base = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert lora_before and all('lora' in n or 'modules_to_save' in n for n in lora_before)

    result, _ = run_fit(model, toy_pairs())
    losses = [h['train_loss'] for h in result['history']]
    assert losses[-1] < 0.7 * losses[0], losses

    for n, p in model.named_parameters():
        if n in base:
            assert torch.equal(p, base[n]), f'frozen weight changed: {n}'
    assert any(not torch.equal(p, lora_before[n]) for n, p in model.named_parameters() if n in lora_before)
    assert result['steps'] == 6 * len(D.unique_page_batches(toy_pairs(), 4, 0, 0))


def test_fit_refuses_when_nothing_is_trainable_or_too_few_pages():
    model = Tiny()
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError, match='no trainable'):
        run_fit(model, toy_pairs())
    with pytest.raises(ValueError, match='no batches'):
        run_fit(peft_tiny(), toy_pairs(n_pages=1))


def test_fit_evaluates_before_and_after_each_epoch_and_saves_only_improvements():
    scores = iter([{'recall@5': 0.5, 'mrr': 0.3}, {'recall@5': 0.6, 'mrr': 0.4},
                   {'recall@5': 0.55, 'mrr': 0.5}, {'recall@5': 0.7, 'mrr': 0.45}])
    saved = []
    result, _ = run_fit(peft_tiny(), toy_pairs(), epochs=3, eval_fn=lambda m: next(scores),
                        save_fn=lambda m, e, metrics: saved.append((e, metrics['recall@5'])))
    assert result['baseline']['recall@5'] == 0.5
    assert saved == [(1, 0.6), (3, 0.7)]          # epoch 2 got worse: not saved
    assert result['best_epoch'] == 3


def test_without_eval_every_epoch_is_saved_so_the_last_one_wins():
    saved = []
    result, _ = run_fit(peft_tiny(), toy_pairs(), epochs=2, save_fn=lambda m, e, _: saved.append(e))
    assert saved == [1, 2] and result['baseline'] is None


def test_gradient_accumulation_still_steps_once_per_group():
    result, _ = run_fit(peft_tiny(), toy_pairs(), epochs=1, grad_accum=2)
    n_batches = len(D.unique_page_batches(toy_pairs(), 4, 0, 0))
    assert result['steps'] == -(-n_batches // 2)


# ---- validation metric -------------------------------------------------------------------------------

def test_evaluate_pairs_perfect_and_broken_retrieval():
    rng = np.random.default_rng(0)
    axes = np.eye(8, dtype=np.float32)
    vectors = {('a.pdf', p): np.stack([axes[p - 1]] * 4 + [axes[(p) % 8] * 0.1]) for p in range(1, 7)}
    val = [{'query': f'q{p}', 'source': 'A.PDF', 'subject': 'dsp', 'page': p} for p in range(1, 4)]
    good = evaluate_pairs(val, vectors, lambda q: axes[[int(q[1:]) - 1]])
    assert good['recall@1'] == 1.0 and good['mrr'] == 1.0
    bad = evaluate_pairs(val, vectors, lambda q: axes[[7]])          # query points at no page
    assert bad['recall@1'] < 1.0
