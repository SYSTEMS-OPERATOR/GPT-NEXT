import importlib
import os
import sys
import types

sys.modules['tqdm'] = types.SimpleNamespace(
    trange=lambda *a, **k: range(a[0] if a else 0),
    tqdm=lambda x, **k: x,
)
sys.modules['nltk'] = types.SimpleNamespace(
    wordpunct_tokenize=lambda t: [],
    sent_tokenize=lambda t: [],
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


tokenizer_module = importlib.import_module("model.tokenizer")
BytePairTokenizer = tokenizer_module.BytePairTokenizer
create_vocab_maps = tokenizer_module.create_vocab_maps
merge_vocab = tokenizer_module.merge_vocab


def test_create_vocab_maps_ordering():
    freqs = {'b': 3, 'a': 2, 'c': 1}
    v2i, i2v = create_vocab_maps(freqs)
    assert v2i['b'] == 0
    assert v2i['a'] == 1
    assert i2v[2] == 'c'
    assert len(v2i) == len(freqs)


def test_get_byte_id_unknown():
    freqs = {
        'a': 1,
        '<unk>': 1,
        '<pad>': 1,
        '<line/>': 1,
        '</line>': 1,
        '</w>': 1,
    }
    v2i, i2v = create_vocab_maps(freqs)
    tokenizer = BytePairTokenizer(freqs, v2i, i2v)
    unk_id = v2i['<unk>']
    assert tokenizer.get_byte_id('missing') == unk_id


def test_merge_vocab():
    vocab = {'a b': 2, 'b c': 1}
    merged = merge_vocab(('a', 'b'), vocab)
    assert 'ab' in merged
    assert merged['ab'] == 2
    assert 'a b' not in merged


def test_merge_vocab_accumulates_counts():
    vocab = {'a b c': 2, 'ab c': 1}
    merged = merge_vocab(('a', 'b'), vocab)
    assert merged['ab c'] == 3


def test_save_load_roundtrip(tmp_path):
    freqs = {
        'a': 1,
        '<unk>': 1,
        '<pad>': 1,
        '<line/>': 1,
        '</line>': 1,
        '</w>': 1,
    }
    v2i, i2v = create_vocab_maps(freqs)
    tokenizer = BytePairTokenizer(freqs, v2i, i2v)
    tokenizer.save(tmp_path)
    loaded = BytePairTokenizer.load(tmp_path)
    assert loaded.vocab_to_idx == tokenizer.vocab_to_idx
    assert loaded.idx_to_vocab == tokenizer.idx_to_vocab
    assert isinstance(next(iter(loaded.idx_to_vocab.keys())), int)
