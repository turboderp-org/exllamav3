"""Checkpoint ownership, fallback contracts and optional CUDA transfer oracles."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.cache import recurrent_transfer as transfer
from exllamav3.modules.gated_delta_net import GDNLayerState, GDNState
from exllamav3.modules.ple import PLELayerState


def fixture(device = "cpu"):
    gdn = GDNLayerState.__new__(GDNLayerState)
    gdn.module = SimpleNamespace(conv_kernel_size = 4, fdim_qkv = 12,
                                 num_v_heads = 2, k_head_dim = 4, v_head_dim = 8)
    gdn.recurrent_state = torch.arange(192, dtype = torch.float32, device = device).view(1, 3, 2, 4, 8)
    gdn.conv_state = torch.arange(72, dtype = torch.bfloat16, device = device).view(1, 12, 6)
    ple = PLELayerState.__new__(PLELayerState)
    ple.win, ple.ctx = 4, 2
    ple.conv_state = torch.arange(48, dtype = torch.float16, device = device).view(1, 8, 6)
    ple.id_state = torch.tensor([[19, 23, 29]], dtype = torch.int64)
    layers = {"gdn": gdn, "ple": ple}
    cache = SimpleNamespace(model = SimpleNamespace(loaded_tp = False), get_all_recurrent_layers = lambda: layers)
    state = GDNState(cache, 0, position = 16, clear = False, test_state = True)
    return state, layers


class CPUChecks(unittest.TestCase):
    def test_native_ple_checkpoint_owns_cpu_ids(self):
        _, layers = fixture()
        layer = layers["ple"]
        saved = layer.stash(0)
        expected = tuple(t.clone() for t in saved)
        layer.id_state.fill_(99)
        layer.conv_state.fill_(-1)
        for actual, want in zip(saved, expected):
            self.assertTrue(torch.equal(actual, want))
        layer.unstash(0, saved)
        self.assertTrue(torch.equal(layer.id_state[0, :2], expected[1]))
        self.assertEqual(layer.id_state[0, 2].item(), 99)

    def test_layout_accounts_for_every_byte(self):
        state, layers = fixture()
        groups = {key: layer.checkpoint_tensors(0) for key, layer in layers.items()}
        entries, size, device = transfer._layout(groups, state.checkpoint_size)
        self.assertEqual(size, 432)
        self.assertIsNone(device)
        self.assertEqual(sum(n for _, items in entries for _, _, n in items), size)

    def test_cpu_only_falls_back_without_pinning(self):
        state, _ = fixture()
        with patch.object(transfer, "_get_staging", side_effect = AssertionError("must not pin")):
            self.assertIsNone(transfer.try_stash(state._checkpoint_tensors(), 16, state.checkpoint_size))

    def test_unsupported_size_and_alignment_fall_back(self):
        x = torch.zeros(16, dtype = torch.float32)
        self.assertIsNone(transfer._layout({"x": (x,)}, 63))
        with patch.object(transfer, "STAGING_BYTES", 32):
            self.assertIsNone(transfer._layout({"x": (x,)}, 64))
        self.assertIsNone(transfer._layout({"x": (torch.zeros(1, dtype = torch.uint8), x)}, 65))
        self.assertIsNone(transfer._layout({"x": (torch.empty(1, device = "meta"),)}, 4))

    def test_unsupported_layer_falls_back(self):
        state, layers = fixture()
        layers["unsupported"] = SimpleNamespace()
        self.assertIsNone(state._checkpoint_tensors())

    def test_multiple_cuda_devices_fall_back_before_allocation(self):
        # Device descriptors exercise planning without requiring a second GPU.
        def descriptor(index):
            return SimpleNamespace(layout = torch.strided, device = torch.device("cuda", index),
                                   is_cuda = True, numel = lambda: 4, element_size = lambda: 4)
        with patch.object(transfer, "_get_staging", side_effect = AssertionError("must not allocate")):
            self.assertIsNone(transfer.try_stash({"x": (descriptor(0), descriptor(1))}, 16, 32))

    def test_native_constructor_restore_and_ownership(self):
        state, layers = fixture()
        with patch("exllamav3.modules.gated_delta_net._coalesced_checkpoints_enable", True):
            saved = state.stash()
        self.assertNotIn("_coalesced_slab", saved)
        for layer in layers.values():
            for tensor in layer.checkpoint_tensors(0):
                tensor.fill_(-1)
        restored = GDNState(state.cache, 0, position = 16, stashed = saved)
        self.assertEqual(restored.checkpoint_size, saved["checkpoint_size"])
        for key, layer in layers.items():
            for target, reference in zip(layer.checkpoint_tensors(0), saved[key]):
                self.assertTrue(torch.equal(target, reference))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA; never load model weights")
class CUDAChecks(unittest.TestCase):
    def test_mixed_state_is_exact_independent_and_pageable(self):
        state, layers = fixture("cuda:0")
        # Do not let an inherited opt-in flag turn the native oracle into the candidate.
        with patch("exllamav3.modules.gated_delta_net._coalesced_checkpoints_enable", False):
            reference = state.stash()
        self.assertNotIn("_coalesced_slab", reference)
        with patch("exllamav3.modules.gated_delta_net._coalesced_checkpoints_enable", True):
            saved = state.stash()
        self.assertEqual(saved["_coalesced_slab"].numel(), state.checkpoint_size)
        self.assertFalse(saved["_coalesced_slab"].is_pinned())
        for key, layer in layers.items():
            for original, actual in zip(reference[key], saved[key]):
                self.assertTrue(torch.equal(original, actual))
            for tensor in layer.checkpoint_tensors(0):
                tensor.fill_(-1)
        restored = GDNState(state.cache, 0, position = 16, stashed = saved)
        self.assertEqual(restored.checkpoint_size, state.checkpoint_size)
        for key, layer in layers.items():
            for target, original in zip(layer.checkpoint_tensors(0), reference[key]):
                self.assertTrue(torch.equal(target.cpu(), original))

    def test_pin_failure_falls_back_and_older_snapshot_survives_reuse(self):
        state, layers = fixture("cuda:0")
        groups = state._checkpoint_tensors()
        with patch.object(transfer, "_get_staging", side_effect = RuntimeError("pin failure")):
            self.assertIsNone(transfer.try_stash(groups, 16, state.checkpoint_size))
        first = transfer.try_stash(groups, 16, state.checkpoint_size)
        preserved = first["_coalesced_slab"].clone()
        layers["ple"].id_state.fill_(77)
        transfer.try_stash(groups, 16, state.checkpoint_size)
        self.assertTrue(torch.equal(first["_coalesced_slab"], preserved))

    def test_corrupt_storage_is_rejected(self):
        state, _ = fixture("cuda:0")
        groups = state._checkpoint_tensors()
        saved = transfer.try_stash(groups, 16, state.checkpoint_size)
        saved["_coalesced_slab"] = saved["_coalesced_slab"][:-1]
        with self.assertRaises(ValueError):
            transfer.restore(groups, 16, state.checkpoint_size, saved)


if __name__ == "__main__":
    unittest.main()
