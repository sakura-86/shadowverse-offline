# -*- coding: utf-8 -*-
"""``LinearValue`` 的接口与口径测试（不依赖真实对局数据）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from decision_engine.planning.linear_value import (
    OBJECTIVE_FEATURES,
    LinearCoefficients,
    LinearValue,
    objective_feature_vector,
)
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


def _snapshot(*, me_life: int = 20, opponent_life: int = 20) -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=me_life, max_life=20, pp=1, max_pp=1),
        opponent=PlayerView(life=opponent_life, max_life=20, pp=1, max_pp=1),
        cards_by_uid={},
    )


def _coefficients(width: int = len(OBJECTIVE_FEATURES)) -> LinearCoefficients:
    return LinearCoefficients(
        weights=tuple(0.1 for _ in range(width)),
        bias=0.0,
        mean=tuple(0.0 for _ in range(width)),
        scale=tuple(1.0 for _ in range(width)),
    )


class ObjectiveFeatureVectorTest(unittest.TestCase):
    def test_extracts_diffs_and_protocol_fields(self) -> None:
        features = {
            "player": {"life": 14, "hand_count": 5, "pp": 7, "field_count": 2, "turn": 6},
            "opponent": {"life": 18, "hand_count": 9, "pp": 0, "field_count": 1},
            "pressure": {
                "player_field_attack_total": 4,
                "opponent_field_attack_total": 7,
                "opponent_guard_count": 1,
                "player_guard_count": 0,
            },
        }
        vector = objective_feature_vector(features)

        self.assertEqual(len(vector), len(OBJECTIVE_FEATURES))
        as_dict = dict(zip(OBJECTIVE_FEATURES, vector))
        self.assertEqual(as_dict["life_diff"], -4)
        self.assertEqual(as_dict["field_atk_diff"], -3)
        self.assertEqual(as_dict["hand_diff"], -4)
        self.assertEqual(as_dict["turn"], 6)

    def test_missing_fields_default_to_zero_without_raising(self) -> None:
        # 只给最少字段；缺的按 0 处理，不能抛异常中断搜索。
        vector = objective_feature_vector({"player": {"life": 20}})
        self.assertEqual(len(vector), len(OBJECTIVE_FEATURES))
        self.assertEqual(vector[OBJECTIVE_FEATURES.index("my_life")], 20)
        self.assertEqual(vector[OBJECTIVE_FEATURES.index("opp_life")], 0)


class LinearCoefficientsTest(unittest.TestCase):
    def test_validate_rejects_width_mismatch(self) -> None:
        broken = LinearCoefficients(
            weights=(0.1,),
            bias=0.0,
            mean=tuple(0.0 for _ in OBJECTIVE_FEATURES),
            scale=tuple(1.0 for _ in OBJECTIVE_FEATURES),
        )
        with self.assertRaisesRegex(ValueError, "维度"):
            broken.validate()

    def test_validate_rejects_zero_scale(self) -> None:
        values = list(1.0 for _ in OBJECTIVE_FEATURES)
        values[3] = 0.0
        broken = LinearCoefficients(
            weights=tuple(0.0 for _ in OBJECTIVE_FEATURES),
            bias=0.0,
            mean=tuple(0.0 for _ in OBJECTIVE_FEATURES),
            scale=tuple(values),
        )
        with self.assertRaisesRegex(ValueError, "scale"):
            broken.validate()

    def test_save_and_load_round_trip(self) -> None:
        coefficients = _coefficients()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.json"
            coefficients.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["features"], list(OBJECTIVE_FEATURES))
            restored = LinearCoefficients.load(path)
        self.assertEqual(restored.weights, coefficients.weights)
        self.assertEqual(restored.bias, coefficients.bias)


class LinearValueEvaluateTest(unittest.TestCase):
    def test_terminal_states_return_pm_one_without_using_weights(self) -> None:
        value = LinearValue(_coefficients())
        won = value.evaluate(_snapshot(me_life=20, opponent_life=0), player="me")
        lost = value.evaluate(_snapshot(me_life=0, opponent_life=20), player="me")

        self.assertTrue(won.terminal)
        self.assertEqual(won.value, 1.0)
        self.assertTrue(lost.terminal)
        self.assertEqual(lost.value, -1.0)

    def test_non_terminal_value_stays_in_range(self) -> None:
        value = LinearValue(_coefficients())
        result = value.evaluate(_snapshot(), player="me")
        result.validate()
        self.assertFalse(result.terminal)
        self.assertGreaterEqual(result.value, -1.0)
        self.assertLessEqual(result.value, 1.0)

    def test_perspective_flips_sign(self) -> None:
        # 同一局面从对方视角看，value 应当近似取反。
        coefficients = LinearCoefficients(
            weights=tuple(0.5 if name == "life_diff" else 0.0 for name in OBJECTIVE_FEATURES),
            bias=0.0,
            mean=tuple(0.0 for _ in OBJECTIVE_FEATURES),
            scale=tuple(1.0 for _ in OBJECTIVE_FEATURES),
        )
        value = LinearValue(coefficients)
        snapshot = _snapshot(me_life=20, opponent_life=5)
        mine = value.evaluate(snapshot, player="me")
        theirs = value.evaluate(snapshot, player="opponent")
        self.assertGreater(mine.value, 0.0)
        self.assertLess(theirs.value, 0.0)
        self.assertAlmostEqual(mine.value, -theirs.value, places=6)


if __name__ == "__main__":
    unittest.main()
