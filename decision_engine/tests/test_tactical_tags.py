# -*- coding: utf-8 -*-
"""TacticalTags 提取测试。"""

import unittest

from decision_engine.cards.tactical_tags import TacticalTags, extract_tactical_tags


class ExtractTacticalTagsTest(unittest.TestCase):
    def test_empty_desc_returns_empty_tags(self):
        tags = extract_tactical_tags("")
        self.assertEqual(tags, TacticalTags())

    def test_extracts_guard(self):
        tags = extract_tactical_tags("【守护】")
        self.assertIn("guard", tags.keywords)

    def test_extracts_storm(self):
        tags = extract_tactical_tags("【疾驰】")
        self.assertIn("storm", tags.keywords)

    def test_extracts_rush(self):
        tags = extract_tactical_tags("【突进】\n【谢幕曲】抽取1张卡牌。")
        self.assertIn("rush", tags.keywords)
        self.assertIn("last_words", tags.timings)
        self.assertTrue(tags.has_draw)

    def test_extracts_ambush(self):
        tags = extract_tactical_tags("【潜行】")
        self.assertIn("ambush", tags.keywords)

    def test_extracts_drain(self):
        tags = extract_tactical_tags("【虹吸】")
        self.assertIn("drain", tags.keywords)

    def test_extracts_destroy_keyword(self):
        tags = extract_tactical_tags("【毁灭】")
        self.assertIn("bane", tags.keywords)

    def test_extracts_cant_select(self):
        tags = extract_tactical_tags("【灵气】")
        self.assertIn("cant_select", tags.keywords)
        self.assertNotIn("necromancy", tags.resources)

    def test_extracts_barrier(self):
        tags = extract_tactical_tags("【屏障】")
        self.assertIn("barrier", tags.keywords)

    def test_extracts_intimidate(self):
        tags = extract_tactical_tags("【威慑】")
        self.assertIn("intimidate", tags.keywords)

    def test_extracts_fanfare_timing(self):
        tags = extract_tactical_tags("【入场曲】选择对手的战场上的1个随从，对其造成3点伤害。")
        self.assertIn("fanfare", tags.timings)
        self.assertTrue(tags.has_damage)
        self.assertEqual(tags.max_damage, 3)

    def test_extracts_activation(self):
        tags = extract_tactical_tags("【启动】破坏本卡牌。")
        self.assertIn("activation", tags.timings)
        self.assertTrue(tags.has_destroy)

    def test_extracts_enhance(self):
        tags = extract_tactical_tags("【爆能强化_4】本随从+3/+3。")
        self.assertIn("enhance", tags.resources)

    def test_extracts_faith_bracket_resource(self):
        tags = extract_tactical_tags("【信仰】获得能力。")
        self.assertIn("faith", tags.resources)

    def test_extracts_faith_free_text_resource(self):
        tags = extract_tactical_tags("若信仰值为3以上，则抽取1张卡牌。")
        self.assertIn("faith", tags.resources)
        self.assertTrue(tags.has_draw)

    def test_extracts_faith_resource_without_value_word(self):
        tags = extract_tactical_tags("本局对战中自己的信仰增加。")
        self.assertIn("faith", tags.resources)

    def test_extracts_combo(self):
        tags = extract_tactical_tags("【入场曲】【连击_3】对对手的主战者造成5点伤害。")
        self.assertIn("combo", tags.resources)
        self.assertIn("fanfare", tags.timings)
        self.assertTrue(tags.has_damage)
        self.assertTrue(tags.has_leader_damage)
        self.assertEqual(tags.max_damage, 5)

    def test_extracts_heal(self):
        tags = extract_tactical_tags("【进化时】回复自己的主战者2点生命值。")
        self.assertIn("on_evolve", tags.timings)
        self.assertTrue(tags.has_heal)

    def test_extracts_summon(self):
        tags = extract_tactical_tags("【谢幕曲】召唤1个『骑士』。")
        self.assertIn("last_words", tags.timings)
        self.assertTrue(tags.has_summon)

    def test_extracts_aoe(self):
        tags = extract_tactical_tags("对战场上的所有随从造成2点伤害。")
        self.assertTrue(tags.has_damage)
        self.assertTrue(tags.has_aoe)
        self.assertEqual(tags.max_damage, 2)

    def test_extracts_bounce(self):
        tags = extract_tactical_tags("选择对手的战场上的1个随从，使其返回手牌。")
        self.assertTrue(tags.has_bounce)

    def test_extracts_draw(self):
        tags = extract_tactical_tags("抽取2张卡牌。")
        self.assertTrue(tags.has_draw)

    def test_extracts_multiple_damage_values(self):
        tags = extract_tactical_tags("对其造成2点伤害。对对手的主战者造成5点伤害。")
        self.assertEqual(tags.max_damage, 5)
        self.assertTrue(tags.has_leader_damage)

    def test_no_leader_damage_without_leader_mention(self):
        tags = extract_tactical_tags("对其造成3点伤害。")
        self.assertTrue(tags.has_damage)
        self.assertFalse(tags.has_leader_damage)

    def test_complex_card(self):
        desc = "【疾驰】\n【超进化时】选择对手的战场上的1个随从，使其返回手牌。"
        tags = extract_tactical_tags(desc)
        self.assertIn("storm", tags.keywords)
        self.assertIn("on_super_evolve", tags.timings)
        self.assertTrue(tags.has_bounce)

    def test_mode_card(self):
        desc = "【入场曲】【模式】选择1个能力发动。\n（1）使自己的所有手牌发动2次魔力增幅。\n（2）【土之秘术_1】本随从+2/+2且获得【守护】。"
        tags = extract_tactical_tags(desc)
        self.assertIn("fanfare", tags.timings)
        self.assertIn("mode", tags.timings)
        self.assertIn("earth_rite", tags.resources)
        self.assertIn("guard", tags.keywords)

    # ---- 自由文本时机（纹章/纹章效果常见模式）----

    def test_extracts_turn_end_timing(self):
        tags = extract_tactical_tags("自己的回合结束时，对对手的主战者造成1点伤害。")
        self.assertIn("turn_end", tags.timings)
        self.assertTrue(tags.has_damage)
        self.assertTrue(tags.has_leader_damage)

    def test_extracts_turn_start_timing(self):
        tags = extract_tactical_tags("自己的回合开始时，抽取1张卡牌。")
        self.assertIn("turn_start", tags.timings)
        self.assertTrue(tags.has_draw)

    def test_extracts_on_enter_timing(self):
        tags = extract_tactical_tags("自己的妖精·随从进入战场时，使其获得【疾驰】。")
        self.assertIn("on_enter", tags.timings)
        self.assertIn("storm", tags.keywords)

    def test_extracts_on_destroyed_timing(self):
        tags = extract_tactical_tags("自己的卡牌被破坏时，对对手的主战者造成1点伤害。")
        self.assertIn("on_destroyed", tags.timings)
        self.assertTrue(tags.has_damage)

    def test_extracts_on_vanish_timing(self):
        tags = extract_tactical_tags("卡牌在战场上消失时，使自己的战场上的所有随从+1/+1。")
        self.assertIn("on_vanish", tags.timings)
        self.assertTrue(tags.has_buff)

    # ---- 新效果标签 ----

    def test_extracts_buff(self):
        tags = extract_tactical_tags("本随从+3/+3。")
        self.assertTrue(tags.has_buff)

    def test_extracts_cost_reduce(self):
        tags = extract_tactical_tags("使自己的牌组中的所有卡牌的费用-1。")
        self.assertTrue(tags.has_cost_reduce)

    def test_extracts_cost_change(self):
        tags = extract_tactical_tags("使自己的牌组中的所有卡牌的费用变为5。")
        self.assertTrue(tags.has_cost_reduce)

    # ---- 真实纹章卡效果 ----

    def test_crest_curry_damage(self):
        desc = "自己的回合结束时，若对手的主战者的生命值为15或以上，则对对手的主战者造成1点伤害。"
        tags = extract_tactical_tags(desc)
        self.assertIn("turn_end", tags.timings)
        self.assertTrue(tags.has_damage)
        self.assertTrue(tags.has_leader_damage)
        self.assertEqual(tags.max_damage, 1)

    def test_crest_parfait_heal(self):
        desc = "自己的回合结束时，若自己的主战者的生命值为15或以下，则回复自己的主战者1点生命值。"
        tags = extract_tactical_tags(desc)
        self.assertIn("turn_end", tags.timings)
        self.assertTrue(tags.has_heal)

    def test_crest_supplement_draw(self):
        desc = "自己的回合结束时，若自己的手牌的张数为3张或以下，则抽取1张卡牌。"
        tags = extract_tactical_tags(desc)
        self.assertIn("turn_end", tags.timings)
        self.assertTrue(tags.has_draw)

    def test_crest_evo_aoe(self):
        desc = "自己的回合结束时，若自己的战场上有超进化后的随从，则对对手的战场上的所有随从造成2点伤害。"
        tags = extract_tactical_tags(desc)
        self.assertIn("turn_end", tags.timings)
        self.assertTrue(tags.has_damage)
        self.assertTrue(tags.has_aoe)
        self.assertEqual(tags.max_damage, 2)

    def test_crest_deep_green_combo_buff(self):
        desc = "自己的回合结束时，【连击_3】使自己的战场上的所有随从+1/+1。"
        tags = extract_tactical_tags(desc)
        self.assertIn("turn_end", tags.timings)
        self.assertIn("combo", tags.resources)
        self.assertTrue(tags.has_buff)

    def test_crest_cost_reduce_deck(self):
        desc = "自己的回合开始时，若回合数为1，则使自己的牌组中的所有费用最大的卡牌的费用-1。"
        tags = extract_tactical_tags(desc)
        self.assertIn("turn_start", tags.timings)
        self.assertTrue(tags.has_cost_reduce)


class CardResolverTacticalTagsTest(unittest.TestCase):
    def test_resolver_returns_tags_for_known_card(self):
        from decision_engine.cards import CardResolver, MasterDataProvider, BaseCardInfo

        provider = MasterDataProvider(
            base_cards={100: BaseCardInfo(base_card_id=100, skill_desc="【守护】\n【疾驰】")},
            cards={},
        )
        resolver = CardResolver(master_data=provider)
        tags = resolver.get_tactical_tags(100)
        self.assertIn("guard", tags.keywords)
        self.assertIn("storm", tags.keywords)

    def test_resolver_uses_faith_acquired_ability_texts(self):
        from decision_engine.cards import CardResolver, MasterDataProvider, BaseCardInfo, FaithAcquiredAbilityText

        provider = MasterDataProvider(
            base_cards={
                10354112: BaseCardInfo(
                    base_card_id=10354112,
                    skill_desc="信仰值起始为0。",
                    faith_acquired_ability_texts=(
                        FaithAcquiredAbilityText(
                            related_base_card_id=10354110,
                            text_key="FAA_10354110",
                            index=1,
                            clean_text="自己的随从进化时，发动1次「对对手的主战者造成1点伤害」。",
                        ),
                    ),
                )
            },
            cards={},
        )
        resolver = CardResolver(master_data=provider)
        tags = resolver.get_tactical_tags(10354112)
        self.assertIn("faith", tags.resources)
        self.assertIn("on_evolve", tags.timings)
        self.assertTrue(tags.has_leader_damage)
        self.assertEqual(tags.max_damage, 1)

    def test_resolver_returns_empty_for_unknown_card(self):
        from decision_engine.cards import CardResolver

        tags = CardResolver().get_tactical_tags(99999)
        self.assertEqual(tags, TacticalTags())


class CrestBoonSummaryTest(unittest.TestCase):
    def setUp(self):
        from decision_engine.cards import CardResolver, MasterDataProvider, BaseCardInfo, FaithAcquiredAbilityText

        self.provider = MasterDataProvider(
            base_cards={
                85401010: BaseCardInfo(
                    base_card_id=85401010,
                    name="欧丝的超辣咖喱",
                    skill_desc="自己的回合结束时，若对手的主战者的生命值为15或以上，则对对手的主战者造成1点伤害。",
                ),
                85401020: BaseCardInfo(
                    base_card_id=85401020,
                    name="欧丝的超甜芭菲",
                    skill_desc="自己的回合结束时，若自己的主战者的生命值为15或以下，则回复自己的主战者1点生命值。",
                ),
                10354112: BaseCardInfo(
                    base_card_id=10354112,
                    name="信仰：混融的继承者·莎木·纳克雅",
                    skill_desc="信仰值起始为0。自己选择【模式】时，信仰值+1。",
                    faith_related_base_card_ids=(10354110,),
                    faith_acquired_ability_texts=(
                        FaithAcquiredAbilityText(
                            related_base_card_id=10354110,
                            text_key="FAA_10354110",
                            index=1,
                            clean_text="自己选择的【模式】数+1。",
                        ),
                    ),
                ),
            },
            cards={},
        )
        self.resolver = CardResolver(master_data=self.provider)

    def test_summarize_crest_known(self):
        summary = self.resolver.summarize_crest(85401010)
        self.assertTrue(summary.known)
        self.assertEqual(summary.name, "欧丝的超辣咖喱")
        self.assertIn("turn_end", summary.tags.timings)
        self.assertTrue(summary.tags.has_damage)
        self.assertTrue(summary.tags.has_leader_damage)

    def test_summarize_crest_unknown(self):
        summary = self.resolver.summarize_crest(99999)
        self.assertFalse(summary.known)
        self.assertEqual(summary.card_id, 99999)
        self.assertEqual(summary.name, "")
        self.assertEqual(summary.tags, TacticalTags())

    def test_summarize_faith_crest(self):
        summary = self.resolver.summarize_crest(10354112)
        self.assertTrue(summary.known)
        self.assertIn("faith", summary.tags.resources)
        self.assertIn("mode", summary.tags.timings)
        self.assertEqual(summary.faith_related_base_card_ids, (10354110,))
        self.assertEqual(summary.faith_acquired_ability_texts[0].text_key, "FAA_10354110")

    def test_summarize_crests_batch(self):
        summaries = self.resolver.summarize_crests([85401010, 85401020])
        self.assertEqual(len(summaries), 2)
        self.assertTrue(summaries[0].tags.has_damage)
        self.assertTrue(summaries[1].tags.has_heal)

    def test_summarize_boon_unknown(self):
        summary = self.resolver.summarize_boon(99999)
        self.assertFalse(summary.known)
        self.assertEqual(summary.card_id, 99999)

    def test_summarize_boons_batch(self):
        summaries = self.resolver.summarize_boons([99999, 88888])
        self.assertEqual(len(summaries), 2)
        self.assertFalse(summaries[0].known)
        self.assertFalse(summaries[1].known)


if __name__ == "__main__":
    unittest.main()
