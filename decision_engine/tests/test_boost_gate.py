"""推理侧增幅饱和 gate（EP_BOOST_GATE）的配置解析测试。

只覆盖配置层：gate 默认必须关闭（β=0 时 logits 逐位不变，旧行为完全保留），
非法值必须显式报错而不是静默当 0。
"""

from __future__ import annotations

import os
import unittest

from decision_engine.planning.search_prior import StructuredNeuralPrior


class BoostGateConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get("EP_BOOST_GATE")

    def tearDown(self) -> None:
        os.environ.pop("EP_BOOST_GATE", None)
        if self._saved is not None:
            os.environ["EP_BOOST_GATE"] = self._saved

    def test_defaults_to_zero_so_behaviour_is_unchanged(self) -> None:
        os.environ.pop("EP_BOOST_GATE", None)
        prior = StructuredNeuralPrior()
        self.assertEqual(prior._boost_gate, 0.0)

    def test_reads_float(self) -> None:
        os.environ["EP_BOOST_GATE"] = "1.5"
        self.assertAlmostEqual(StructuredNeuralPrior()._boost_gate, 1.5)

    def test_negative_gate_is_allowed(self) -> None:
        """负 β 表示"满档反而该晚打"，是对照方向，不应被拒绝。"""
        os.environ["EP_BOOST_GATE"] = "-2"
        self.assertAlmostEqual(StructuredNeuralPrior()._boost_gate, -2.0)

    def test_invalid_value_raises_instead_of_silently_zero(self) -> None:
        os.environ["EP_BOOST_GATE"] = "abc"
        with self.assertRaises(ValueError):
            StructuredNeuralPrior()


if __name__ == "__main__":
    unittest.main()
