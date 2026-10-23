import json
from argparse import Namespace

import pytest

from training import data as D
from training.make_pairs import (clean_questions, generate_for_page, is_good_question,
                                 parse_questions, run, select_pages)
from training.train_colpali import prepare_data


def pair(q, source='a.pdf', page=1, subject='dsp'):
    return {'query': q, 'source': source, 'subject': subject, 'page': page}


def write(path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


# ---- loading / validation ---------------------------------------------------------------------

def test_load_pairs_roundtrip(tmp_path):
    f = tmp_path / 'p.jsonl'
    D.write_pairs(f, [pair('what is the region of convergence?')])
    assert D.load_pairs(f)[0]['page'] == 1


@pytest.mark.parametrize('line,msg', [
    ('{not json', 'not valid JSON'),
    (json.dumps({'query': 'what is z transform', 'source': 'a.pdf', 'page': 1}), "missing \\['subject'\\]"),
    (json.dumps(pair('too short')), 'at least 3 words'),
    (json.dumps({**pair('what is the z transform'), 'page': '4'}), 'page must be'),
    (json.dumps({**pair('what is the z transform'), 'page': 0}), 'page must be'),
])
def test_load_pairs_reports_the_bad_line(tmp_path, line, msg):
    f = tmp_path / 'p.jsonl'
    f.write_text(json.dumps(pair('a perfectly fine question here')) + '\n' + line + '\n')
    with pytest.raises(ValueError, match=f'p.jsonl:2: .*{msg}'):
        D.load_pairs(f)


def test_dedupe_ignores_case_and_punctuation():
    ps = [pair('What is the ROC?'), pair('what is the roc'), pair('what is the roc', page=2)]
    assert len(D.dedupe(ps)) == 2


# ---- guards -----------------------------------------------------------------------------------

MANIFEST = [{'file': 'a.pdf', 'subject': 'dsp', 'shareable': 'yes'},
            {'file': 'book.pdf', 'subject': 'dsp', 'shareable': 'no'}]


def test_shareable_guard_drops_non_shareable_and_can_be_overridden():
    ps = [pair('what is the z transform of a step'), pair('define the bilinear transform', 'book.pdf')]
    kept, dropped = D.shareable_guard(ps, MANIFEST)
    assert [p['source'] for p in kept] == ['a.pdf'] and dropped[0][1] == 'not_shareable'
    assert len(D.shareable_guard(ps, MANIFEST, allow_non_shareable=True)[0]) == 2


def test_shareable_guard_rejects_unknown_documents():
    with pytest.raises(ValueError, match='not in the manifest'):
        D.shareable_guard([pair('what is the z transform here', 'typo.pdf')], MANIFEST)


EVAL = [{'id': 'e1', 'question': 'What is the region of convergence of a right-sided sequence?',
         'gold': [{'source': 'A.pdf', 'page': 7}]}]


def test_eval_guard_drops_exact_and_near_duplicate_questions():
    ps = [pair('what is the region of convergence of a right sided sequence'),    # same words
          pair('Region of convergence of a right-sided sequence, what is it?'),   # near duplicate
          pair('how does the bilinear transform warp frequency')]
    kept, dropped = D.eval_guard(ps, EVAL)
    assert [p['query'] for p in kept] == ['how does the bilinear transform warp frequency']
    assert {r for _, r in dropped} == {'eval_question_overlap'}


def test_eval_guard_excludes_gold_pages_by_default_case_insensitively():
    ps = [pair('explain poles and zeros in detail', 'a.pdf', 7), pair('explain poles and zeros in detail', 'a.pdf', 8)]
    kept, dropped = D.eval_guard(ps, EVAL)
    assert [p['page'] for p in kept] == [8] and dropped[0][1] == 'eval_gold_page'
    assert len(D.eval_guard(ps, EVAL, exclude_eval_pages=False)[0]) == 2


# ---- split + batching ---------------------------------------------------------------------------

def many(n_pages, per_page=2):
    return [pair(f'question {j} about topic number {i} please', 'a.pdf', i)
            for i in range(1, n_pages + 1) for j in range(per_page)]


def test_split_is_by_page_deterministic_and_never_empty_val():
    ps = many(20)
    train, val = D.split_pairs(ps, 0.1, seed=1)
    assert {D.page_key(p) for p in train}.isdisjoint({D.page_key(p) for p in val})
    assert len(train) + len(val) == len(ps) and len(val) == 4        # 2 pages x 2 queries
    assert D.split_pairs(ps, 0.1, seed=1) == (train, val)
    assert D.split_pairs(many(3), 0.01)[1]                           # at least one val page
    assert D.split_pairs(many(1), 0.5)[1] == [] and D.split_pairs(ps, 0)[1] == []


def test_batches_never_repeat_a_page_and_cover_pairs_once():
    ps = many(10, per_page=3)
    batches = D.unique_page_batches(ps, batch_size=4, seed=0, epoch=1)
    for b in batches:
        assert len({D.page_key(ps[i]) for i in b}) == len(b) >= 2
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat)) and len(flat) >= len(ps) - 3   # only tiny remainders dropped


def test_batches_are_deterministic_but_vary_by_epoch():
    ps = many(12)
    assert D.unique_page_batches(ps, 4, 0, 1) == D.unique_page_batches(ps, 4, 0, 1)
    assert D.unique_page_batches(ps, 4, 0, 1) != D.unique_page_batches(ps, 4, 0, 2)


def test_batches_drop_lone_remainder_and_validate_size():
    ps = many(5, per_page=1)                                          # 5 pages, bs 2 -> 2,2 and a 1
    assert sorted(len(b) for b in D.unique_page_batches(ps, 2)) == [2, 2]
    with pytest.raises(ValueError):
        D.unique_page_batches(ps, 1)


# ---- prepare_data (the guards as the training script applies them) -------------------------------

def args(tmp_path, **kw):
    base = dict(pairs=str(tmp_path / 'pairs.jsonl'), eval_questions=str(tmp_path / 'q.jsonl'),
                allow_eval_pages=False, no_eval_guard=False, include_non_shareable=False,
                val_fraction=0.2, seed=0)
    return Namespace(**{**base, **kw})


def test_prepare_data_applies_every_guard(tmp_path):
    rows = many(10) + [pair('define the bilinear transform in full', 'book.pdf', 3),
                       pair('what is the region of convergence of a right sided sequence', 'a.pdf', 50),
                       pair('describe an unrelated filter structure', 'a.pdf', 7)]
    write(tmp_path / 'pairs.jsonl', rows)
    write(tmp_path / 'q.jsonl', EVAL)
    train, val, rep = prepare_data(args(tmp_path), MANIFEST)
    assert rep['dropped_not_shareable'] == 1 and rep['dropped_eval_overlap'] == 1
    # page 7 is a gold page: 2 pairs from many(10) plus the extra one are dropped; 23 - 1 - 1 - 3 = 18
    assert rep['dropped_eval_gold_page'] == 3 and rep['kept'] == 18
    assert all(p['source'] == 'a.pdf' for p in train + val)
    assert not any(D.page_key(p) == ('a.pdf', 7) for p in train + val)


def test_prepare_data_requires_eval_file_unless_explicitly_waived(tmp_path):
    write(tmp_path / 'pairs.jsonl', many(6))
    with pytest.raises(SystemExit, match='leak guard'):
        prepare_data(args(tmp_path), MANIFEST)
    logs = []
    prepare_data(args(tmp_path, no_eval_guard=True), MANIFEST, log=logs.append)
    assert 'WITHOUT the eval leak guard' in logs[0]


# ---- make_pairs ---------------------------------------------------------------------------------

def test_parse_questions_tolerates_fences_prose_and_garbage():
    assert parse_questions('```json\n["a b c d e?", "f"]\n```') == ['a b c d e?', 'f']
    assert parse_questions('Sure! ["one two three four five?"] hope that helps') == ['one two three four five?']
    assert parse_questions('no json here') == [] and parse_questions('[1, 2') == []
    assert parse_questions('[1, "x y z a b?"]') == ['x y z a b?']


def test_question_quality_filter():
    good = 'How does the bilinear transform map the s-plane to the z-plane?'
    assert is_good_question(good)
    for bad in ['Too short?', 'What does the figure shown above represent in this circuit?',
                'Explain what is on page 12 of the notes please?', 'State the ROC of this sequence.',
                'What does this page say about stability of filters?']:
        assert not is_good_question(bad), bad
    assert clean_questions([good, good.upper(), 'Too short?'], 3) == [good]
    assert len(clean_questions([f'{good} variant {i}?' for i in range(6)], 3)) == 3


class FakeLLM:
    def __init__(self, reply=None, fail=()):
        self.reply, self.fail, self.calls = reply, set(fail), []

    def complete(self, system, user, max_tokens=1024, images=None):
        self.calls.append((user, images))
        if len(self.calls) in self.fail:
            raise RuntimeError('boom')
        return self.reply or json.dumps([f'How does concept number {len(self.calls)} work in practice?',
                                         f'Why does concept number {len(self.calls)} matter for design?'])


def page(n, text='x ' * 150, shareable='yes', subject='dsp', source='a.pdf'):
    return {'id': f'p{n}', 'source': source, 'subject': subject, 'page': n, 'shareable': shareable, 'text': text}


def test_select_pages_never_includes_non_shareable_even_with_vision():
    pages = [page(1), page(2, shareable='no'), page(3, text='', shareable='no')]
    assert [p['page'] for p in select_pages(pages, vision=True)] == [1]
    assert [p['page'] for p in select_pages(pages)] == [1]


def test_select_pages_filters_sparse_done_and_subject():
    pages = [page(1), page(2, text='short'), page(3), page(4, subject='swp')]
    assert [p['page'] for p in select_pages(pages)] == [1, 3, 4]
    assert [p['page'] for p in select_pages(pages, vision=True)] == [1, 2, 3, 4]
    assert [p['page'] for p in select_pages(pages, done={('a.pdf', 1)})] == [3, 4]
    assert [p['page'] for p in select_pages(pages, subjects=['dsp'])] == [1, 3]


def test_run_writes_pairs_resumes_and_sends_only_page_text(tmp_path):
    out, llm = tmp_path / 'pairs.jsonl', FakeLLM()
    pages = [page(1, text='UNIQUE-ONE ' * 30), page(2, text='UNIQUE-TWO ' * 30)]
    written, done, failed = run(pages, out, llm, per_page=2, log=lambda *_: None)
    assert (written, done, failed) == (4, 2, 0)
    rows = D.load_pairs(out)
    assert {r['page'] for r in rows} == {1, 2} and {r['origin'] for r in rows} == {'synthetic'}
    assert 'UNIQUE-ONE' in llm.calls[0][0] and 'UNIQUE-TWO' not in llm.calls[0][0]
    # resume: nothing left to do for pages already present
    from training.make_pairs import _done_pages
    assert select_pages(pages, done=_done_pages(out)) == []


def test_run_survives_one_failure_but_stops_after_a_streak(tmp_path):
    out = tmp_path / 'p.jsonl'
    assert run([page(i) for i in range(1, 4)], out, FakeLLM(fail={2}), log=lambda *_: None) == (4, 2, 1)
    out2 = tmp_path / 'p2.jsonl'
    llm = FakeLLM(fail=set(range(1, 100)))
    _, done, failed = run([page(i) for i in range(1, 20)], out2, llm, log=lambda *_: None)
    assert done == 0 and failed == 5 and len(llm.calls) == 5          # aborted, did not hammer the API


def test_run_uses_image_for_sparse_pages_only_with_a_renderer(tmp_path):
    sparse = page(1, text='')
    img = {'label': '[a.pdf, 1]', 'media_type': 'image/png', 'data': 'AAAA'}
    llm = FakeLLM()
    assert run([sparse], tmp_path / 'a.jsonl', llm, log=lambda *_: None)[1] == 0 and llm.calls == []
    assert run([sparse], tmp_path / 'b.jsonl', llm, renderer=lambda h: img, log=lambda *_: None)[1] == 1
    assert llm.calls[0][1] == [img] and 'page image' in llm.calls[0][0]
    assert generate_for_page(FakeLLM('["bad"]'), page(1), 3) == []


# ---- CLI dry runs (no model, no API) ----------------------------------------------------------------

def setup_project(tmp_path, n_pages=12):
    import pandas as pd
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    pd.DataFrame(MANIFEST).assign(category='notes').to_csv(corpus / 'manifest.csv', index=False)
    write(tmp_path / 'pairs.jsonl', many(n_pages))
    write(tmp_path / 'q.jsonl', EVAL)
    return ['--pairs', str(tmp_path / 'pairs.jsonl'), '--manifest', str(corpus / 'manifest.csv'),
            '--eval-questions', str(tmp_path / 'q.jsonl'), '--out', str(tmp_path / 'ad')]


def test_train_dry_run_prints_plan_and_loads_no_model(tmp_path, capsys):
    from training.train_colpali import main
    assert main(setup_project(tmp_path) + ['--dry-run', '--batch-size', '4']) == 0
    out = capsys.readouterr().out
    assert 'plan:' in out and 'distinct training pages' in out and 'dropped_eval_gold_page' in out
    assert not (tmp_path / 'ad').exists()


def test_train_stops_early_when_there_are_too_few_pages_for_the_batch(tmp_path):
    from training.train_colpali import main
    with pytest.raises(SystemExit, match='distinct training pages'):
        main(setup_project(tmp_path, n_pages=5) + ['--dry-run', '--batch-size', '8'])


def test_train_rejects_batch_size_below_two(tmp_path):
    from training.train_colpali import main
    with pytest.raises(SystemExit):
        main(setup_project(tmp_path) + ['--dry-run', '--batch-size', '1'])


def test_make_pairs_dry_run_reads_the_index_and_calls_no_api(tmp_path, capsys, monkeypatch):
    import numpy as np

    from retrieval.visual import write_index
    from training.make_pairs import main
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    pages = [page(1, text='z ' * 200), page(2, text='z ' * 200, shareable='no'), page(3, text='')]
    write_index(tmp_path / 'idx', pages, [np.ones((2, 4), np.float16)] * 3, 'm')
    out = tmp_path / 'train' / 'pairs.jsonl'
    assert main(['--index', str(tmp_path / 'idx'), '--out', str(out), '--dry-run']) == 0
    text = capsys.readouterr().out
    assert '1 shareable pages to send (3 in index)' in text and not out.exists()
