# -*- coding: utf-8 -*-
"""MasterDataProvider 加载测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from decision_engine.cards import MasterDataProvider


class MasterDataProviderTest(unittest.TestCase):
    def test_loads_faith_fields_from_json(self):
        data = {
            "cards": [
                {
                    "base_card_id": 10354112,
                    "name": "信仰：混融的继承者·莎木·纳克雅",
                    "type_id": 7,
                    "type_name": "faith",
                    "faith_related_base_card_ids": [10354110],
                    "faith_acquired_ability_texts": [
                        {
                            "related_base_card_id": 10354110,
                            "text_key": "FAA_10354110",
                            "index": 1,
                            "raw_text": "自己选择的【模式】数+{0}。",
                            "clean_text": "自己选择的【模式】数+1。",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card_database.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            provider = MasterDataProvider.from_json(path)

        base = provider.get_base(10354112)
        self.assertIsNotNone(base)
        assert base is not None
        self.assertEqual(base.type_id, 7)
        self.assertEqual(base.card_type, "faith")
        self.assertEqual(base.faith_related_base_card_ids, (10354110,))
        self.assertEqual(len(base.faith_acquired_ability_texts), 1)
        self.assertEqual(base.faith_acquired_ability_texts[0].text_key, "FAA_10354110")
        self.assertEqual(base.faith_acquired_ability_texts[0].clean_text, "自己选择的【模式】数+1。")

    def test_missing_faith_fields_are_empty(self):
        data = {"cards": [{"base_card_id": 10001110, "name": "测试"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card_database.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            provider = MasterDataProvider.from_json(path)

        base = provider.get_base(10001110)
        self.assertIsNotNone(base)
        assert base is not None
        self.assertEqual(base.type_id, 0)
        self.assertEqual(base.faith_related_base_card_ids, ())
        self.assertEqual(base.faith_acquired_ability_texts, ())


if __name__ == "__main__":
    unittest.main()
