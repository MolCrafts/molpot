import pytest
import json
from pathlib import Path
import torch.nn as nn

from molpot.pipeline.dataset import Dataset, MapStyleDataset, DATASET_LOADER_MAP
from molpot.pipeline.process.base import ProcessManager, ProcessType

# --- Dummy classes for testing ---

class FakeFrame:
    def __init__(self, idx):
        self.idx = idx

class FakeLoader:
    def __init__(self, save_dir):
        pass

    def load(self):
        return [FakeFrame(i) for i in range(3)]

class FakeProcess(nn.Module):
    def __init__(self, offset: int):
        super().__init__()
        self.type = ProcessType.ONE
        self.offset = offset

    def forward(self, frame: FakeFrame) -> FakeFrame:
        frame.idx += self.offset
        return frame

# --- Fixture to patch loader, download, and ProcessManager.add ---

@pytest.fixture(autouse=True)
def patch_components(monkeypatch):
    print("\n[fixture] Patching Dataset.download, DATASET_LOADER_MAP['dummy'], ProcessManager.add")
    monkeypatch.setattr(Dataset, "download", lambda self: None, raising=False)
    monkeypatch.setitem(DATASET_LOADER_MAP, "dummy", FakeLoader)
    monkeypatch.setattr(ProcessManager, "add", ProcessManager.append, raising=False)
    yield

# --- Tests ---

def test_prepare_and_get_frame(tmp_path):
    print("\n[test_prepare_and_get_frame] Start")
    ds = Dataset(name="dummy", save_dir=tmp_path)
    print("  - Before prepare: len(ds) =", len(ds))
    assert len(ds) == 0

    print("  - Expect get_frame to raise RuntimeError before prepare()")
    with pytest.raises(RuntimeError):
        ds.get_frame(0)

    print("  - Calling prepare()")
    count = ds.prepare()
    print(f"  - prepare() returned {count}")
    assert count == 3
    print("  - After prepare: len(ds) =", len(ds))
    assert len(ds) == 3

    for i in range(3):
        f = ds.get_frame(i)
        print(f"    - Frame {i}.idx = {f.idx}")
        assert isinstance(f, FakeFrame)
        assert f.idx == i

def test_add_process_and_process_one():
    print("\n[test_add_process_and_process_one] Start")
    ds = Dataset(name="dummy", save_dir=Path("/tmp"))
    proc = FakeProcess(offset=5)
    print("  - Adding FakeProcess(offset=5)")
    ds.add_process(proc)

    frame = FakeFrame(idx=2)
    print(f"  - Before process_one: frame.idx = {frame.idx}")
    out = ds.processes.process_one(frame)
    print(f"  - After process_one: frame.idx = {out.idx}")
    assert out.idx == 7

def test_map_style_dataset_uses_processes(tmp_path):
    print("\n[test_map_style_dataset_uses_processes] Start")
    ds = MapStyleDataset(name="dummy", save_dir=tmp_path)
    print("  - After init: len(ds) =", len(ds))
    assert len(ds) == 3

    proc = FakeProcess(offset=-1)
    print("  - Adding FakeProcess(offset=-1)")
    ds.add_process(proc)

    for i in range(3):
        f = ds[i]
        print(f"    - ds[{i}].idx = {f.idx}")
        assert isinstance(f, FakeFrame)
        assert f.idx == i - 1

def test_update_writes_state(tmp_path):
    print("\n[test_update_writes_state] Start")
    ds = Dataset(name="dummy", save_dir=tmp_path)
    ds.prepare()
    state = {"foo": 42, "bar": "baz"}
    print("  - Calling update with state =", state)
    ds.update(state)

    path = tmp_path / "state.json"
    print("  - Checking state.json exists at", path)
    assert path.exists()

    loaded = json.loads(path.read_text())
    print("  - Loaded state from file:", loaded)
    assert loaded == state
