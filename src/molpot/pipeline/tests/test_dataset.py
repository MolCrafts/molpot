import pytest
import json
from pathlib import Path

from molpot.pipeline.dataset import Dataset, DATASET_LOADER_MAP, config

class FakeFrame:
    def __init__(self, idx):
        self.idx = idx

class FakeLoader:
    def __init__(self, save_dir):
        # przyjmuje save_dir, ale nic z nim nie robi
        pass

    def load(self):
        # zwracamy 3 fikcyjne klatki
        return [FakeFrame(i) for i in range(3)]

@pytest.fixture(autouse=True)
def patch_loader_and_config(monkeypatch):
    # 1) Zamień config.processes na pusty dict, by __init__ nie wywaliło błędu
    monkeypatch.setattr(config, 'processes', {}, raising=False)
    # 2) Podmień loader dla 'dummy' na FakeLoader
    monkeypatch.setitem(DATASET_LOADER_MAP, 'dummy', FakeLoader)
    # 3) Dodaj brakującą metodę get() do FakeLoader (używaną w prepare())
    monkeypatch.setattr(FakeLoader, 'get', classmethod(lambda cls: cls), raising=False)
    # 4) Zamień download() na no-op, by nie szukało config.urls
    monkeypatch.setattr(Dataset, "download", lambda self: None, raising=False)
    yield

def test_len_before_prepare_is_zero(tmp_path):
    ds = Dataset(name="dummy", save_dir=tmp_path)
    assert len(ds) == 0

def test_get_frame_before_prepare_raises(tmp_path):
    ds = Dataset(name="dummy", save_dir=tmp_path)
    with pytest.raises(RuntimeError):
        ds.get_frame(0)

def test_prepare_and_get_frame(tmp_path):
    ds = Dataset(name="dummy", save_dir=tmp_path)
    count = ds.prepare()
    assert count == 3
    assert len(ds) == 3

    for i in range(3):
        frame = ds.get_frame(i)
        assert isinstance(frame, FakeFrame)
        assert frame.idx == i

def test_update_writes_state(tmp_path):
    ds = Dataset(name="dummy", save_dir=tmp_path)
    state = {"foo": 42, "bar": "baz"}
    ds.update(state)

    state_file = tmp_path / "state.json"
    assert state_file.exists()

    loaded = json.loads(state_file.read_text())
    assert loaded == state
