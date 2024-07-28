import torch

import molpot as mpot


class TestTracker:

    def test_number(self):

        tracker = mpot.Tracker()
        tracker(1)
        tracker(2)
        tracker(3)
        tracker(4)
        tracker(5)
        tracker(6)
        tracker(7)
        tracker(8)
        tracker(9)
        tracker(10)
        assert tracker.mean == 5.5
        assert tracker.variance == 8.25
        assert tracker.sample_variance == 9.166666666666666

    def test_1dtensor(self):

        tracker = mpot.Tracker()
        tracker(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        assert tracker.mean == 5.5
        assert tracker.variance == 8.25
        assert tracker.sample_variance == 9.166666666666666

    def test_2dtensor(self):

        tracker = mpot.Tracker()
        tracker(torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))
        assert tracker.mean == 5.5
        assert tracker.variance == 8.25
        assert tracker.sample_variance == 9.166666666666666