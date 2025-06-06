# tests/test_qdpi_training.py

import io
from pathlib import Path

import pytest
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from molpot.pipeline.qdpi import QDpi
import molpot as mpot
import molpot.alias as alias
from molpot.utils.element import Element


@pytest.fixture
def dummy_hdf5_bytes(tmp_path):
    p = tmp_path / "dummy_charged.hdf5"
    with h5py.File(p, "w") as f:
        m = f.create_group("mol0")
        m.create_dataset("nopbc", data=np.array([0], dtype=np.int32))
        s = m.create_group("set.000")
        coords = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float64)
        s.create_dataset("coord.npy", data=coords)
        s.create_dataset("energy.npy", data=np.array([123.45], dtype=np.float64))
        forces = np.array([[0.01, 0.02, 0.03], [0.11, 0.12, 0.13]], dtype=np.float64)
        s.create_dataset("force.npy", data=forces)
        s.create_dataset("net_charge.npy", data=np.array([-1.0], dtype=np.float64))
        m.create_dataset("type.raw", data=np.array([0, 1], dtype=np.int64))
        m.create_dataset("type.map", data=np.array([1, 6], dtype=np.int64))
    return p.read_bytes()


class DummyResponse:
    def __init__(self, data_bytes):
        self._buf = io.BytesIO(data_bytes)

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        while True:
            chunk = self._buf.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def frame_to_example(frame):
    R = frame[alias.R]
    Z = frame[alias.Z]
    E = frame[alias.E]
    Rf = R.reshape(-1)
    Zf = Z.to(torch.float32)
    x = torch.cat([Rf, Zf])
    y = E.reshape(-1)
    return x, y


class TestQDpiTraining:
    def test_training_pipeline_small(self, monkeypatch, tmp_path, dummy_hdf5_bytes):
        subset = "re_charged"
        monkeypatch.setattr(QDpi, "get_subset_data", lambda self: {"charged": {subset: "charged/re_charged.hdf5"}})
        monkeypatch.setattr("requests.get", lambda url, stream=True: DummyResponse(dummy_hdf5_bytes))
        monkeypatch.setattr(Element, "get_atomic_number", lambda sym: sym)
        for key in ("R", "F", "E", "Q", "Z"):
            monkeypatch.setattr(alias, key, key, raising=False)

        ds = QDpi(save_dir=tmp_path, subset=subset)
        frames = ds._frames
        assert len(frames) == 1
        frame = frames[0]

        x, y = frame_to_example(frame)
        assert x.ndim == 1 and x.numel() == 8
        assert y.ndim == 1 and y.numel() == 1

        class SimpleDS(torch.utils.data.Dataset):
            def __init__(self, ins, outs):
                self.ins, self.outs = ins, outs

            def __len__(self):
                return len(self.ins)

            def __getitem__(self, i):
                return self.ins[i], self.outs[i]

        tds = SimpleDS([x], [y])
        loader = DataLoader(tds, batch_size=1, shuffle=False)

        D = x.shape[0]
        model = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        for bx, by in loader:
            bx, by = bx.to(torch.float32), by.to(torch.float32)
            opt.zero_grad()
            out = model(bx).view(-1)
            loss = loss_fn(out, by)
            loss.backward()
            opt.step()
            assert isinstance(loss.item(), float)
            assert any(p.grad is not None for p in model.parameters())
            break
