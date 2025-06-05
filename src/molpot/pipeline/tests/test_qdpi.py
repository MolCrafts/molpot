import pytest
from unittest import mock
from pathlib import Path

from molpot.pipeline.qdpi import QDpi


def test_prepare_not_called_twice(monkeypatch, tmp_path):
    subset = ["ani", "re"]
    mock_download = mock.MagicMock()
    monkeypatch.setattr(QDpi, "_download", mock_download)

    ds = QDpi(save_dir=tmp_path, subset=subset)

    assert mock_download.call_count == len(subset)

    ds.prepare()

    assert mock_download.call_count == len(subset)


def test_prepare_subset_calls_download(tmp_path, caplog):
    subset = ["ani", "spice"]
    with mock.patch.object(QDpi, "_download") as mock_download:
        with caplog.at_level("INFO", logger="molpot.dataset"):
            ds = QDpi(save_dir=tmp_path, subset=subset)
            ds.prepare()

        assert mock_download.call_count == len(subset)