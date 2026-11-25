# -*- coding: utf-8 -*-
"""BoardFingerprint 单元测试。

覆盖 P2 验收要点：
- 左右视角对称（me 视角 ally == opponent 视角 enemy）；
- 不含 uid：仅 uid 变化（其余属性不变）不改变指纹；
- 场面关键差异（atk/life/max_life/关键词/evolve/attacks_left/countdown）会区分；
- field 排列顺序无关（排序稳定）；
- stable / deterministic（可 hash、多次构造一致）。
"""

import unittest

from decision_engine.planning.board_fingerprint import BoardFingerprint, CardFingerprint
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    LegalView,
    PlayerView,
)


def _follower(
    uid: int,
    base_card_id: int = 1000,
    *,
    atk: int = 2,
    life: int = 2,
    max_life: int = 2,
    evolve_state: int = 0,
    attacks_left: int = 1,
    countdown: int = 0,
    stack: int = 0,
    has_guard: bool = False,
    has_storm: bool = False,
) -> CardView:
    return CardView(
        uid=uid,
        card_id=base_card_id,
        base_card_id=base_card_id,
        card_type=CardType.FOLLOWER,
        atk=atk,
        life=life,
        max_life=max_life,
        evolve_state=evolve_state,
        attacks_left=attacks_left,
        countdown=countdown,
        stack=stack,
        has_guard=has_guard,
        has_storm=has_storm,
    )


def _amulet(uid: int, base_card_id: int = 2000, *, countdown: int = 3) -> CardView:
    return CardView(
        uid=uid,
        card_id=base_card_id,
        base_card_id=base_card_id,
        card_type=CardType.COUNTDOWN_AMULET,
        countdown=countdown,
    )


def _snapshot(
    *,
    my_field: list[CardView] | None = None,
    opp_field: list[CardView] | None = None,
) -> BattleSnapshot:
    my_field = my_field or []
    opp_field = opp_field or []
    cards: dict[int, CardView] = {}
    for c in my_field + opp_field:
        cards[c.uid] = c
    me = PlayerView(life=20, max_life=20, pp=10, max_pp=10, field=tuple(c.uid for c in my_field))
    opp = PlayerView(life=20, max_life=20, pp=10, max_pp=10, field=tuple(c.uid for c in opp_field))
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, legal=LegalView())


class BoardFingerprintTest(unittest.TestCase):
    def test_perspective_symmetry(self):
        snap = _snapshot(
            my_field=[_follower(1, 1000, atk=3, life=3)],
            opp_field=[_follower(2, 2000, atk=1, life=1, has_guard=True)],
        )
        me_view = BoardFingerprint.from_snapshot(snap, "me")
        opp_view = BoardFingerprint.from_snapshot(snap, "opponent")
        # me 视角 ally(=snapshot.me field) == opponent 视角 enemy。
        self.assertEqual(me_view.ally, opp_view.enemy)
        # me 视角 enemy(=snapshot.opponent field) == opponent 视角 ally。
        self.assertEqual(me_view.enemy, opp_view.ally)

    def test_uid_only_change_does_not_distinguish(self):
        """仅 uid 不同、其余属性相同的场面，指纹应相等。"""
        snap_a = _snapshot(my_field=[_follower(1, 1000, atk=3, life=3)])
        snap_b = _snapshot(my_field=[_follower(999, 1000, atk=3, life=3)])
        fp_a = BoardFingerprint.from_snapshot(snap_a, "me")
        fp_b = BoardFingerprint.from_snapshot(snap_b, "me")
        self.assertEqual(fp_a, fp_b)

    def test_atk_life_difference_distinguishes(self):
        base = _snapshot(my_field=[_follower(1, 1000, atk=3, life=3, max_life=3)])
        atk_diff = _snapshot(my_field=[_follower(1, 1000, atk=4, life=3, max_life=3)])
        life_diff = _snapshot(my_field=[_follower(1, 1000, atk=3, life=2, max_life=3)])
        maxlife_diff = _snapshot(my_field=[_follower(1, 1000, atk=3, life=3, max_life=5)])
        fp_base = BoardFingerprint.from_snapshot(base, "me")
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(atk_diff, "me"))
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(life_diff, "me"))
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(maxlife_diff, "me"))

    def test_keyword_difference_distinguishes(self):
        base = _snapshot(my_field=[_follower(1, 1000)])
        guard = _snapshot(my_field=[_follower(1, 1000, has_guard=True)])
        storm = _snapshot(my_field=[_follower(1, 1000, has_storm=True)])
        fp_base = BoardFingerprint.from_snapshot(base, "me")
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(guard, "me"))
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(storm, "me"))

    def test_evolve_and_attacks_left_and_countdown_distinguish(self):
        base = _snapshot(my_field=[_follower(1, 1000, evolve_state=0, attacks_left=1)])
        evolved = _snapshot(my_field=[_follower(1, 1000, evolve_state=1, attacks_left=1)])
        no_attack = _snapshot(my_field=[_follower(1, 1000, evolve_state=0, attacks_left=0)])
        fp_base = BoardFingerprint.from_snapshot(base, "me")
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(evolved, "me"))
        self.assertNotEqual(fp_base, BoardFingerprint.from_snapshot(no_attack, "me"))

        amulet_a = _snapshot(my_field=[_amulet(1, 2000, countdown=3)])
        amulet_b = _snapshot(my_field=[_amulet(1, 2000, countdown=2)])
        self.assertNotEqual(
            BoardFingerprint.from_snapshot(amulet_a, "me"),
            BoardFingerprint.from_snapshot(amulet_b, "me"),
        )

    def test_field_order_independent(self):
        """field 排列顺序不同、卡集合相同，指纹应相等（排序稳定）。"""
        c1 = _follower(1, 1000, atk=2, life=2)
        c2 = _follower(2, 3000, atk=5, life=5)
        snap_a = _snapshot(my_field=[c1, c2])
        snap_b = _snapshot(my_field=[c2, c1])
        self.assertEqual(
            BoardFingerprint.from_snapshot(snap_a, "me"),
            BoardFingerprint.from_snapshot(snap_b, "me"),
        )

    def test_fingerprint_excludes_uid(self):
        """CardFingerprint 结构本身不含 uid 字段。"""
        fp = CardFingerprint.from_card(_follower(12345, 1000))
        self.assertFalse(hasattr(fp, "uid"))

    def test_hashable_and_deterministic(self):
        snap = _snapshot(
            my_field=[_follower(1, 1000, atk=3, life=3)],
            opp_field=[_amulet(2, 2000, countdown=2)],
        )
        a = BoardFingerprint.from_snapshot(snap, "me")
        b = BoardFingerprint.from_snapshot(snap, "me")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        _ = {a, b}  # 可作为 set 元素

    def test_invalid_perspective_raises(self):
        snap = _snapshot()
        with self.assertRaises(ValueError):
            BoardFingerprint.from_snapshot(snap, "spectator")  # type: ignore[arg-type]

    def test_empty_field(self):
        snap = _snapshot()
        fp = BoardFingerprint.from_snapshot(snap, "me")
        self.assertEqual(fp.ally, ())
        self.assertEqual(fp.enemy, ())


if __name__ == "__main__":
    unittest.main()
