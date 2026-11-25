"""Unit tests for the loss / index math in compute_losses().

These require torch (the actual cross-entropy and slicing are exercised) but need
no model download. Skipped automatically when torch is unavailable.
"""
import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from swhisper_stage1 import compute_losses


@unittest.skipIf(torch is None, "torch not installed")
class ComputeLossesTests(unittest.TestCase):
    # Layout: prefix(n=2) | suffix(h=2) | tail(1) | target(target_len=2) -> start=5.
    N, H, TAIL, TARGET_LEN = 2, 2, 1, 2
    START = N + H + TAIL  # 5
    TOTAL = START + TARGET_LEN  # 7
    V = 8

    def _ids(self, b=1):
        base = torch.tensor([0, 1, 2, 3, 4, 5, 6])  # len == TOTAL
        return base.unsqueeze(0).expand(b, -1).clone()

    def test_uniform_logits_give_log_vocab(self):
        # All-zero logits -> every position is a uniform distribution -> NLL = log(V).
        logits = torch.zeros(1, self.TOTAL, self.V)
        ids = self._ids()
        for w in (0.0, 0.1, 0.5, 1.0):
            loss = compute_losses(logits, ids, self.START, self.TARGET_LEN,
                                  self.N, self.H, w)
            expected = math.log(self.V) * (1.0 + w)  # adv=log(V), sem=log(V)
            self.assertEqual(loss.shape, (1,))
            self.assertAlmostEqual(loss.item(), expected, places=4)

    def test_perfect_prediction_gives_near_zero(self):
        # logits[p] must predict ids[p+1]. adv reads logits[4:6]->ids[5:7];
        # sem reads logits[1:3]->ids[2:4]. Spiking exactly those alignments must
        # drive the loss to ~0; a mis-sliced implementation would not.
        logits = torch.zeros(1, self.TOTAL, self.V)
        ids = self._ids()
        big = 30.0
        for pos in (1, 2, 4, 5):
            logits[0, pos, ids[0, pos + 1]] = big
        loss = compute_losses(logits, ids, self.START, self.TARGET_LEN,
                              self.N, self.H, 1.0)
        self.assertLess(loss.item(), 1e-3)

    def test_misalignment_is_detected(self):
        # Spike the WRONG next-token at each position -> loss must be large,
        # proving the slices are not accidentally symmetric.
        logits = torch.zeros(1, self.TOTAL, self.V)
        ids = self._ids()
        for pos in (1, 2, 4, 5):
            wrong = (int(ids[0, pos + 1]) + 1) % self.V
            logits[0, pos, wrong] = 30.0
        loss = compute_losses(logits, ids, self.START, self.TARGET_LEN,
                              self.N, self.H, 1.0)
        self.assertGreater(loss.item(), 1.0)

    def test_semantic_weight_zero_isolates_target(self):
        # With w=0 the suffix term drops out; perfect target prediction but random
        # suffix logits should still give ~0.
        logits = torch.zeros(1, self.TOTAL, self.V)
        ids = self._ids()
        logits[0, 4, ids[0, 5]] = 30.0
        logits[0, 5, ids[0, 6]] = 30.0
        loss = compute_losses(logits, ids, self.START, self.TARGET_LEN,
                              self.N, self.H, 0.0)
        self.assertLess(loss.item(), 1e-3)

    def test_batch_dimension_preserved(self):
        logits = torch.zeros(4, self.TOTAL, self.V)
        ids = self._ids(b=4)
        loss = compute_losses(logits, ids, self.START, self.TARGET_LEN,
                              self.N, self.H, 0.1)
        self.assertEqual(loss.shape, (4,))
        self.assertTrue(torch.allclose(loss, loss[0].expand(4)))


if __name__ == "__main__":
    unittest.main()
