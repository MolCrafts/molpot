import pytest
import torch
from molpot.pipeline.dataloader import _compact_collate, _nested_collate
from molpot import alias, Config


class TestDataLoader:

    @pytest.mark.parametrize(
        "frames_fixture", ["gen_homogenous_frames", "gen_heterogenous_frames"]
    )
    def test_compact_collate_fn(self, request, frames_fixture):

        frame_list = request.getfixturevalue(frames_fixture)(10)

        nframes = len(frame_list)
        natom_list = torch.tensor([len(frame[alias.R]) for frame in frame_list])
        natoms = sum(natom_list)
        npairs = sum([len(frame[alias.pair_i]) for frame in frame_list])
        nbonds = sum([len(frame[alias.bond_i]) for frame in frame_list])

        frames = _compact_collate(frame_list)
        assert frames[alias.Z].shape == (natoms,)
        assert frames[alias.R].shape == (natoms, 3)

        assert frames[alias.pair_i].shape == (npairs,)
        assert frames[alias.pair_j].shape == (npairs,)

        assert frames[alias.bond_i].shape == (nbonds,)
        assert frames[alias.bond_j].shape == (nbonds,)

        # test offsets
        torch.testing.assert_close(
            frames[alias.atom_batch],
            torch.cat(
                [torch.full((t,), fill_value=i) for i, t in enumerate(natom_list)]
            ).to(Config.itype),
        )
        torch.testing.assert_close(
            frames[alias.atom_offset],
            # torch.tensor([0, 3, 6, 9, 12], dtype=Config.itype),
            torch.cumsum(
                torch.cat(
                    [
                        torch.tensor(
                            [
                                0,
                            ]
                        ),
                        natom_list[:-1],
                    ]
                ),
                dim=0,
                out=torch.zeros(len(natom_list), dtype=Config.itype),
            ).to(Config.itype),
        )

        for i, frame in enumerate(frame_list):
            pair_batch_mask = frames[alias.pair_batch] == i
            atom_offset = frames[alias.atom_offset][i]
            torch.testing.assert_close(
                frames[alias.pair_i][pair_batch_mask] - atom_offset, frame[alias.pair_i]
            )
            torch.testing.assert_close(
                frames[alias.pair_j][pair_batch_mask] - atom_offset, frame[alias.pair_j]
            )

    # NOTE: Not support nested tensor until nestedtensor support vmap
    
    # @pytest.mark.parametrize(
    #     "frames_fixture", ["gen_homogenous_frames", "gen_heterogenous_frames"]
    # )
    # def test_nested_collate_fn(self, request, frames_fixture):

    #     frame_list = request.getfixturevalue(frames_fixture)(10)

    #     frames = _nested_collate(frame_list)
    #     assert frames[alias.Z]