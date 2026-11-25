# -*- coding: utf-8 -*-
"""解放奥义阈值回归：进度是奥义槽 skybound_art_count，阈值为 15。

背景：薇拉（10464120）的入场曲把「解放奥义 → 本随从超进化」写成
``if_counter_at_least(key=skybound_art_count, reference_key=super_skybound_art_gauge)``。
``super_skybound_art_gauge`` 不是 snapshot 字段，旧解释器按 0 处理，
导致 13 点奥义也判定满足、无条件超进化。

这里从生成资产层面锁住两点：

1. ``card_effect_specs.json`` 的解放奥义条件统一用 ``source.skybound_art_count``；
2. ``card_templates.json`` 中薇拉的 super_evolve 操作以 ``skybound_art_count >= 15`` gated。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPECS = _ROOT / "decision_engine" / "data" / "public" / "card_effect_specs.json"
_TEMPLATES = _ROOT / "decision_engine" / "data" / "public" / "card_templates.json"


class OugiThresholdAssetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = json.loads(_SPECS.read_text(encoding="utf-8"))
        cls.templates = json.loads(_TEMPLATES.read_text(encoding="utf-8"))

    def test_super_ougi_specs_use_skybound_art_count(self) -> None:
        """解放奥义条件的 subject 必须是奥义槽，不能是 super_skybound_art_count。"""
        offenders: list[tuple[int, str, str]] = []
        for card in self.specs["cards"]:
            for spec in card.get("specs", []):
                for condition in spec.get("conditions") or ():
                    subject = str(condition.get("subject") or "")
                    if condition.get("ref") == "super_ougi_threshold" and subject != "source.skybound_art_count":
                        offenders.append((card["base_card_id"], spec["spec_id"], subject))
        self.assertEqual(offenders, [])

        # 不允许再有 spec 把 super_skybound_art_count 当进度读取。
        legacy = [
            (card["base_card_id"], spec["spec_id"])
            for card in self.specs["cards"]
            for spec in card.get("specs", [])
            if "source.super_skybound_art_count" in json.dumps(spec, ensure_ascii=False)
        ]
        self.assertEqual(legacy, [])

    def test_vira_super_evolve_is_gated_by_skybound_15(self) -> None:
        template = next(
            (item for item in self.templates["templates"]
             if int(item["identity"]["base_card_id"]) == 10464120),
            None,
        )
        self.assertIsNotNone(template)
        effects = template.get("executable_effects") or []
        super_evolve_ops = [
            operation
            for effect in effects
            for operation in effect.get("operations") or ()
            if operation.get("op") == "super_evolve"
        ]
        self.assertEqual(len(super_evolve_ops), 1)
        condition = super_evolve_ops[0]["params"].get("if_counter_at_least")
        self.assertEqual(condition, {
            "subject": "source",
            "field": "supplement_info",
            "key": "skybound_art_count",
            "value": 15,
        })

        # 入场曲的无条件部分（banish 2 个敌方随从）不能被同一个条件 gated。
        banish_ops = [
            operation
            for effect in effects
            for operation in effect.get("operations") or ()
            if operation.get("op") == "banish"
        ]
        self.assertEqual(len(banish_ops), 1)
        self.assertNotIn("if_counter_at_least", banish_ops[0].get("params") or {})


if __name__ == "__main__":
    unittest.main()
