import importlib
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class DummyTokenizer:
    def get_sol(self):
        return '<line/>'

    def get_eol(self):
        return '</line>'

    def get_pad(self):
        return '<pad>'

    def get_eow(self):
        return '</w>'

    def merge_bytes(self, bytes_):
        return bytes_

    def get_byte_ids(self, tokens):
        return [0]

    def get_byte_id(self, b):
        return 0

    def __init__(self):
        self.last_byte_arg = None

    def get_byte(self, idx):
        self.last_byte_arg = type(idx)
        return 'x'


class DummyModel:
    def eval(self):
        return None

    def __call__(self, token_ids, ignore_ids):
        return None, None


class FakeTensor:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


def test_generate_sequence_passes_int_id(monkeypatch):
    fake_torch = types.SimpleNamespace(
        LongTensor=object,
        multinomial=lambda *a, **k: None,
        Tensor=object,
        no_grad=lambda: type(
            'C',
            (),
            {
                '__enter__': lambda self: None,
                '__exit__': lambda self, exc_type, exc, tb: None,
            },
        )(),
        argsort=lambda x, descending=False: x,
        full=lambda *a, **k: None,
        cat=lambda *a, **k: None,
        long=int,
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(
        sys.modules,
        'torch.nn.functional',
        types.SimpleNamespace(softmax=lambda x, dim=None: x),
    )
    monkeypatch.setitem(
        sys.modules,
        'tqdm',
        types.SimpleNamespace(trange=lambda *a, **k: range(a[0] if a else 0)),
    )

    sequencer_module = importlib.import_module('model.sequencer')
    Sequencer = sequencer_module.Sequencer

    tokenizer = DummyTokenizer()
    seq = Sequencer(DummyModel(), tokenizer, window_size=1, k=1, device='cpu')

    seq.generate_start_seq = lambda start=None: (['<line/>'], [0], [0])
    seq.gen_next_token = lambda probs, idx: FakeTensor(0)
    seq.update_token_ids = lambda idx, token_ids, next_id: (token_ids, token_ids, idx + 1)

    seq.generate_sequence(length=1)

    assert tokenizer.last_byte_arg is int
