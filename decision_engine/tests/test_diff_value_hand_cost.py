"""Phase N1 DiffSearchValue 手牌资源成本回归测试。

固化"手牌浪费惩罚"从旧"全或无"改为"边际成本始终计入"后的关键行为：
- 空融合（消耗手牌、场面无变化）必须得负分。
- 正常出牌（消耗手牌换来场面收益）净值仍为正，不被误伤。
- 手牌无消耗的回合不受影响。

背景：旧逻辑只要整回合有任一正收益就豁免全部手牌消耗，导致
"打 1 张好牌 + 空融 3 张"被误判为好回合，模型学不到空融合是负收益。
"""

import unittest

from decision_engine.planning.diff_value import compute_diff_score, normalize_diff_score
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


def _card(uid: int, atk: int = 1, life: int = 1) -> CardView:
    return CardView(uid=uid, card_id=100 + uid, base_card_id=100 + uid,
                    card_type=CardType.FOLLOWER, atk=atk, life=life, max_life=life)


class DiffValueHandCostTest(unittest.TestCase):
    def _snapshot(self, hand, my_field, opp_field, cards, my_life=20, opp_life=20):
        me = PlayerView(life=my_life, max_life=20, pp=5, max_pp=5, deck_count=10,
                        hand=tuple(hand), field=tuple(my_field))
        opp = PlayerView(life=opp_life, max_life=20, pp=5, max_pp=5, deck_count=10,
                         field=tuple(opp_field))
        return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)

    def test_empty_fusion_consumes_hand_without_gain_is_negative(self):
        """空融合：手牌从 5 降到 2，场面无变化 → 必须负分。"""
        cards = {i: _card(i) for i in range(1, 6)}
        start = self._snapshot([1, 2, 3, 4, 5], [], [], cards)
        end = self._snapshot([1, 2], [], [], cards)  # 消耗 3 张，场面不变
        score = normalize_diff_score(compute_diff_score(start, end, player="me"))
        self.assertLess(score, 0.0, f"空融合应为负收益，实际 {score}")

    def test_no_hand_consumption_is_neutral(self):
        """手牌无消耗、场面无变化 → 中性（0）。"""
        cards = {i: _card(i) for i in range(1, 6)}
        start = self._snapshot([1, 2, 3, 4, 5], [], [], cards)
        end = self._snapshot([1, 2, 3, 4, 5], [], [], cards)
        score = normalize_diff_score(compute_diff_score(start, end, player="me"))
        self.assertEqual(score, 0.0)

    def test_normal_play_with_board_gain_stays_positive(self):
        """正常出牌：消耗 1 张手牌换来场上一个 3/3 随从 → 净值应为正，不被误伤。"""
        cards = {
            1: _card(1), 2: _card(2), 3: _card(3), 4: _card(4), 5: _card(5),
            6: _card(6, atk=3, life=3),  # 打出的随从
        }
        start = self._snapshot([1, 2, 3, 4, 6], [], [], cards)
        end = self._snapshot([1, 2, 3, 4], [6], [], cards)  # 手牌 -1，场上 +1(3/3)
        score = normalize_diff_score(compute_diff_score(start, end, player="me"))
        self.assertGreater(score, 0.0, f"正常出牌换来场面应为正收益，实际 {score}")

    def test_high_gain_fusion_not_penalized_by_hand_cost(self):
        """消耗手牌换来大量场面收益（如 combo 兑现）→ 仍明显为正。"""
        cards = {
            1: _card(1), 2: _card(2), 3: _card(3),
            10: _card(10, atk=6, life=6), 11: _card(11, atk=6, life=6),
        }
        start = self._snapshot([1, 2, 3], [], [], cards)
        # 消耗 2 张手牌，场上出现两个 6/6（高收益）
        end = self._snapshot([1], [10, 11], [], cards)
        score = normalize_diff_score(compute_diff_score(start, end, player="me"))
        self.assertGreater(score, 0.3, f"高收益操作不应被手牌成本压低，实际 {score}")


if __name__ == "__main__":
    unittest.main()
