"""Lazy loader for ColPali-family visual retrievers (colpali-engine).

torch and colpali_engine are imported inside the functions, so tests, CI and BM25-only
use never need them. Install them with: pip install -r requirements-visual.txt
"""
import numpy as np

# (substring of the model id, model class, processor class). Order matters:
# 'colqwen2.5' must be tested before 'colqwen2'.
_FAMILIES = (
    ('colqwen2.5', 'ColQwen2_5', 'ColQwen2_5_Processor'),
    ('colqwen2', 'ColQwen2', 'ColQwen2Processor'),
    ('colsmol', 'ColIdefics3', 'ColIdefics3Processor'),
    ('colidefics3', 'ColIdefics3', 'ColIdefics3Processor'),
    ('colpali', 'ColPali', 'ColPaliProcessor'),
)


def model_family(model_name):
    """Return (model_class_name, processor_class_name) for a Hugging Face model id."""
    lowered = model_name.lower()
    for key, model_cls, proc_cls in _FAMILIES:
        if key in lowered:
            return model_cls, proc_cls
    raise ValueError(
        f'Unsupported visual model {model_name!r}. Supported families: '
        + ', '.join(k for k, _, _ in _FAMILIES))


def load(model_name, device=None):
    """Load (model, processor). Weights download from Hugging Face on first use."""
    import torch
    from colpali_engine import models
    from colpali_engine.utils.torch_utils import get_torch_device

    model_cls, proc_cls = model_family(model_name)
    device = device or get_torch_device('auto')
    dtype = torch.float32 if str(device).startswith('cpu') else torch.bfloat16
    model = getattr(models, model_cls).from_pretrained(
        model_name, torch_dtype=dtype, device_map=device).eval()
    processor = getattr(models, proc_cls).from_pretrained(model_name)
    return model, processor


def _to_vectors(embeddings, attention_mask, dtype):
    """Split a padded batch into per-item (n_i, dim) arrays, dropping padding rows."""
    import torch
    embeddings = embeddings.to(torch.float32).cpu()
    mask = attention_mask.bool().cpu()
    return [e[m].numpy().astype(dtype) for e, m in zip(embeddings, mask)]


def embed_images(model, processor, images, batch_size=2):
    """PIL images -> list of float16 arrays, one (n_patches, dim) array per image."""
    import torch
    out = []
    for i in range(0, len(images), batch_size):
        batch = processor.process_images(images[i:i + batch_size]).to(model.device)
        with torch.no_grad():
            emb = model(**batch)
        out += _to_vectors(emb, batch['attention_mask'], np.float16)
    return out


def embed_query(model, processor, text):
    """One query string -> float32 array (n_query_vectors, dim)."""
    import torch
    batch = processor.process_queries([text]).to(model.device)
    with torch.no_grad():
        emb = model(**batch)
    return _to_vectors(emb, batch['attention_mask'], np.float32)[0]
