# -*- coding: utf-8 -*-
"""Template effect condition evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from decision_engine.actions.model import LegalAction
from decision_engine.cards.templates import ConditionSpec, ExecutableEffect
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView

from .evidence import is_evidence_card


@lru_cache(maxsize=1)
def _base_card_printed_costs() -> dict[int, int]:
    """base_card_id -> 印刷费用（「原始费用」条件用，不受减费/加费影响）。"""
    path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(card["base_card_id"]): int(card.get("cost") or 0)
        for card in payload.get("cards", [])
    }


@dataclass(frozen=True)
class ConditionEvaluationContext:
    # action 可为 None：evaluate_static 用于「没有动作上下文」的全局状态条件求值。
    action: LegalAction | None
    source_card: CardView | None = None
    generated_cards: tuple[CardView, ...] = ()
    event_player: str | None = None  # turn_end/turn_start 事件的 player（当前回合方）
    # 真实的回合结束/开始事件方（event.player）。event_player 是「效果归属方」，
    # 对跨侧订阅（如「对手的回合结束时」的纹章）两者不同：
    # turn_player 条件必须基于真实事件方与 source_card 的相对关系求值。
    turn_event_player: str | None = None
    # 触发本次效果的事件源卡（如「自己使用其他卡牌时」中的被使用卡）。
    # 法术打出后立即进墓地，snapshot 里已无该卡；此时用 event_source_base_card_id 回退。
    event_source_card: CardView | None = None
    event_source_base_card_id: int | None = None
    # 「使用卡牌瞬间」双方战场的 card_id（play_card / play_spell 事件 payload）。
    # 入场曲可能先击杀同费用随从，条件必须按使用瞬间的战场求值。
    event_field_card_ids: tuple[int, ...] = ()
    # 「刚进入战场的那张卡」（`entered_card.*` 条件）。
    # 与 event_source_card 区分：后者是「触发本次效果的事件源」，在不同 trigger 下
    # 语义各异（攻击时是攻击者、使用法术时是被使用的法术）；entered_card 只在
    # enter_field / summon 一类入场事件上有值，语义固定为「入场卡本身」。
    # 典型：10773110「自己的创造物·随从进入战场时，使其获得【突进】」。
    entered_card: "CardView | None" = None
    # 资源类条件（墓地/手牌数）的求值基准。fanfare 场景传「打出前」快照：
    # 法术打出后会先进墓地，用打出后快照判「唤灵_6」会多算这张法术本身。
    resource_snapshot: "BattleSnapshot | None" = None


@dataclass(frozen=True)
class ConditionEvaluationResult:
    matched: bool
    unresolved_reasons: tuple[str, ...] = ()


class ConditionEvaluator:
    def evaluate_all(
        self,
        snapshot: BattleSnapshot,
        effect: ExecutableEffect,
        conditions: tuple[ConditionSpec, ...],
        context: ConditionEvaluationContext,
    ) -> ConditionEvaluationResult:
        unresolved: list[str] = []
        for condition in conditions:
            result = self.evaluate(snapshot, effect, condition, context)
            if result is False:
                return ConditionEvaluationResult(matched=False, unresolved_reasons=tuple(unresolved))
            if result is None:
                unresolved.append(f"condition_not_supported:{effect.effect_id}")
        return ConditionEvaluationResult(matched=True, unresolved_reasons=tuple(dict.fromkeys(unresolved)))

    def evaluate(
        self,
        snapshot: BattleSnapshot,
        effect: ExecutableEffect,
        condition: ConditionSpec,
        context: ConditionEvaluationContext,
    ) -> bool | None:
        if condition.kind == "text_condition":
            return self._text_condition_result(snapshot, effect, condition, context)
        if condition.kind == "event_material_filter" and condition.subject == "fusion.materials":
            # 「这张卡牌已融合过材料」：协议在 root 层下发 has_fusion_hand_cards
            # （已融合过材料的手牌 uid 列表），adapter 读成 legal.has_fusion。
            # 不能用 card.fusion_list —— 那是「可以融合哪些材料」的候选表，
            # 不是「已经融合过」（round 172 用错净负 17）。
            source_card = context.source_card
            if source_card is None or snapshot.legal is None:
                return None
            fused = snapshot.legal.has_fusion
            if not fused:
                return None
            return 1 if source_card.uid in fused else 0
        if condition.kind == "selection_filter":
            return self._selection_filter_condition_result(snapshot, condition, context.action)
        if condition.kind == "snapshot_exists":
            return self._snapshot_exists_result(snapshot, condition, context)
        if condition.kind in {"event_source_is_self", "event_source_tribe", "event_source_card_type"}:
            # 事件源过滤条件由 TemplateEventHandler._event_trigger_scope_matches 在
            # 事件匹配阶段校验；这里不能重复判定，也不能记为 unsupported。
            return True
        if condition.kind not in {"snapshot_threshold", "snapshot_bool", "resource_check", "resource_cost", "selected_mode_option", "exists", "turn_check", "event_target_state"}:
            return None
        value = self._condition_subject_value(snapshot, condition.subject, context, condition)
        if value is None:
            return None
        if condition.kind == "snapshot_bool" and condition.operator in {"", "truthy", "is_true"}:
            return bool(value)
        expected = self._condition_expected_value(condition.value)
        if expected is None:
            return None
        return self._compare_condition_value(value, condition.operator or ("contains" if condition.kind == "selected_mode_option" else ">="), expected)

    def evaluate_static(
        self,
        snapshot: BattleSnapshot,
        condition: ConditionSpec,
        source_card: CardView | None,
    ) -> bool | None:
        """不依赖动作参数的条件求值。

        用于「卡牌打出效果条件是否达成」这类全局状态问题：没有 action 上下文，因此
        动作依赖条件（selection_filter / 事件物料过滤 / runtime.* 参数）一律返回 None，
        由调用方计入 unknown，而不是当作 False。
        """
        if condition.kind in {"selection_filter", "event_material_filter", "event_snapshot_exists"}:
            return None
        if condition.subject.startswith("runtime."):
            return None
        context = ConditionEvaluationContext(action=None, source_card=source_card)
        if condition.kind == "text_condition":
            return self._text_condition_result(snapshot, None, condition, context)
        if condition.kind not in {"snapshot_threshold", "snapshot_bool", "resource_check", "resource_cost", "selected_mode_option", "exists", "turn_check", "event_target_state"}:
            return None
        value = self._condition_subject_value(snapshot, condition.subject, context, condition)
        if value is None:
            return None
        if condition.kind == "snapshot_bool" and condition.operator in {"", "truthy", "is_true"}:
            return bool(value)
        expected = self._condition_expected_value(condition.value)
        if expected is None:
            return None
        return self._compare_condition_value(
            value,
            condition.operator or ("contains" if condition.kind == "selected_mode_option" else ">="),
            expected,
        )

    def _condition_subject_value(
        self,
        snapshot: BattleSnapshot,
        subject: str,
        context: ConditionEvaluationContext,
        condition: ConditionSpec | None = None,
    ) -> int | str | tuple[int, ...] | None:
        source_card = context.source_card
        if subject in {"me.field_has_base_card_id", "opponent.field_has_base_card_id"}:
            # 「若自己的战场上没有『X』则…」类条件：判定战场上是否存在指定 base_card_id。
            # 目标 id 由 condition.ref 传入 —— subject 是静态字符串，无法自带参数。
            raw = getattr(condition, "ref", None) if condition is not None else None
            if raw is None:
                return None
            try:
                target_id = int(raw)
            except (TypeError, ValueError):
                return None
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return 1 if any(
                (card := snapshot.cards_by_uid.get(uid)) is not None
                and int(card.base_card_id or card.card_id) == target_id
                for uid in view.field
            ) else 0
        if subject.startswith("event_source."):
            # 触发本次效果的事件源卡属性（如「自己的『天晶魔手』攻击时」中的攻击者）。
            # 事件源在事件队列消费阶段可能已经离场（攻击者被反击致死），因此必须由
            # 调用方通过 ConditionEvaluationContext.event_source_card 显式提供；
            # 缺失时返回 None（未实现/未知），不猜测。
            field = subject.split(".", 1)[1]
            event_card = context.event_source_card
            if event_card is None:
                return None
            if field == "base_card_id":
                # 变体 id 只影响卡面展示，效果口径以 base_card_id 为准。
                return int(event_card.base_card_id or event_card.card_id)
            if field == "card_id":
                return int(event_card.card_id)
            if field in {"atk", "attack"}:
                return int(event_card.atk)
            if field == "life":
                return int(event_card.life)
            if field == "evolve_state":
                return int(event_card.evolve_state)
            if field == "has_guard":
                return 1 if event_card.has_guard else 0
            if field == "card_type":
                return str(getattr(event_card.card_type, "value", event_card.card_type))
            if field == "tribes":
                # 协议帧不下发 tribes（CardView.tribes 常为 [0]/空），必须走
                # effective_tribes 回退 CardResolver，否则「玛纳利亚·法术」这类
                # 以种族限定的事件源条件永远不命中。
                from .effect_ops.resolution import effective_tribes

                return effective_tribes(event_card)
            return None
        if subject.startswith("entered_card."):
            # 「进入战场的那张卡」属性条件（10773110 / 10771110 / 10674110）。
            # 入场卡在事件消费阶段可能已离场，必须由调用方显式提供 entered_card；
            # 缺失时返回 None（不可判定），不猜。
            entered = context.entered_card
            if entered is None:
                return None
            field = subject.split(".", 1)[1]
            if field == "is_artifact":
                # 「创造物·随从」= tribe 14；口径与事件过滤共用 effective_tribes
                # （协议帧不下发 tribes 时回退卡面模板）。
                from .effect_ops.resolution import ARTIFACT_TRIBE, effective_tribes

                return 1 if ARTIFACT_TRIBE in effective_tribes(entered) else 0
            if field == "is_other":
                # 「其他随从」：入场卡不能是持有该持续效果的卡本身（10674110）。
                if source_card is None:
                    return None
                return 0 if entered.uid == source_card.uid else 1
            if field in {"original_cost", "printed_cost"}:
                printed = _base_card_printed_costs().get(int(entered.base_card_id or entered.card_id))
                return printed
            if field == "base_card_id":
                return int(entered.base_card_id or entered.card_id)
            if field == "card_id":
                return int(entered.card_id)
            if field == "cost":
                return int(entered.cost)
            if field in {"atk", "attack"}:
                return int(entered.atk)
            if field == "life":
                return int(entered.life)
            if field == "evolve_state":
                return int(entered.evolve_state)
            if field == "has_guard":
                return 1 if entered.has_guard else 0
            if field == "card_type":
                return str(getattr(entered.card_type, "value", entered.card_type))
            if field == "tribes":
                from .effect_ops.resolution import effective_tribes

                return effective_tribes(entered)
            return None
        if subject in {"source.variable_x", "source.spellboost", "source.spellboost_count", "source.evolve_state", "source.is_evolved", "source.atk", "source.attack", "source.atk_minus_life", "source.life", "source.defense", "source.has_guard", "source.has_last_words", "source.cost", "source.skybound_art_count_hand", "source.skybound_art_count", "source.super_skybound_art_count_hand", "source.super_skybound_art_count"}:
            if source_card is None:
                return None
            if subject == "source.variable_x":
                return int(source_card.variable_x)
            if subject == "source.cost":
                return int(source_card.cost)
            if subject in {"source.spellboost", "source.spellboost_count"}:
                return int(source_card.spell_boost_count or source_card.variable_x)
            if subject == "source.evolve_state":
                return int(source_card.evolve_state)
            if subject == "source.is_evolved":
                # 「若本随从为进化后」：超进化也属于已进化状态。
                return 1 if int(source_card.evolve_state) >= 1 else 0
            if subject == "source.has_guard":
                return 1 if source_card.has_guard else 0
            if subject == "source.has_last_words":
                return 1 if source_card.has_last_words else 0
            if subject in {"source.atk", "source.attack"}:
                return int(source_card.atk)
            if subject == "source.atk_minus_life":
                # 「若本随从的攻击力小于/大于生命值」类条件（11024110 复制随从分派关键字）。
                # 用差值表达：< 0 表示攻击力小于生命值，> 0 表示大于。
                return int(source_card.atk) - int(source_card.life)
            if subject in {"source.skybound_art_count_hand", "source.skybound_art_count", "source.super_skybound_art_count_hand", "source.super_skybound_art_count"}:
                # 「解放奥义」进度就是奥义槽：卡面文本写作
                # 「解放奥义为{skybound_art_count}/{super_skybound_art_gauge}」。
                # super_skybound_art_count 在真实 trace 中仅薇拉偶发下发且恒 0，
                # 不是解放奥义进度；统一读 skybound_art_count。
                return self._skybound_progress(context, source_card)
            return int(source_card.life)
        if subject in {"me.graveyard_count", "me.hand", "me.hand_count"} and context.resource_snapshot is not None:
            # 资源基线：fanfare 场景按「打出前」求值（见 ConditionEvaluationContext.resource_snapshot）。
            base = context.resource_snapshot
            view = base.me if subject.startswith("me.") else base.opponent
            if subject.endswith("hand_count") or subject.endswith(".hand"):
                return len(view.hand)
            return int(view.graveyard_count)
        if subject == "me.life_minus_opponent_life":
            # 「自己的主战者生命值高于对手」类条件：直接取双方 life 之差。
            # 10851130 兔耳恶魔·莉蜜儿「若自己的主战者生命值比对手高，则…」。
            return int(snapshot.me.life) - int(snapshot.opponent.life)
        if subject in {"me.graveyard_count", "me.rally", "me.play_count", "me.attack_count", "me.followers_entered_battlefield", "me.evolve_count", "me.hand", "me.hand_count", "me.field", "me.field_count", "me.earth_sigil", "me.crests", "me.field_evolved_follower", "self_field_evolved_follower", "me.field_super_evolved_follower", "me.field_amulet_count", "me.field_evidence_count", "me.hand_evidence_amulet_count", "me.evidence_count", "me.max_pp", "me.play_point_max", "me.life"}:
            # `me.` 是「效果持有者自己」：当效果由对手的卡触发（last_words / turn_end 等，
            # context.event_player == "opponent"）时，必须取对手视角的资源量。
            # 例：11054120 的谢幕曲「【死灵术_2】抽1张卡」，条件写 me.graveyard_count >= 2，
            # 实际应看持有者（对手）的墓地；按视角 me 取会恒为 0，整段效果不发动。
            owner_side = "opponent" if context.event_player == "opponent" else "me"
            return self._player_condition_value(snapshot, owner_side, subject.rsplit(".", 1)[1] if subject.startswith("me.") else "field_evolved_follower")
        if subject in {"opponent.graveyard_count", "opponent.rally", "opponent.play_count", "opponent.attack_count", "opponent.hand", "opponent.hand_count", "opponent.field", "opponent.field_count", "opponent.earth_sigil", "opponent.crests", "opponent.field_evolved_follower", "opponent_field_evolved_follower", "opponent.field_super_evolved_follower", "opponent.field_amulet_count", "opponent.field_evidence_count", "opponent.life"}:
            return self._player_condition_value(snapshot, "opponent", subject.rsplit(".", 1)[1] if subject.startswith("opponent.") else "field_evolved_follower")
        if subject in {"source.fusion_count", "source.fused_count"}:
            # 已融合的材料组数（card.fusion_list 每组是一次融合），用于「已与本卡牌融合」类条件。
            if source_card is None:
                return None
            return len(source_card.fusion_list) or int(source_card.supplement_info.get("fusion_count", 0))
        if subject == "source.is_fused":
            if source_card is None:
                return None
            return 1 if (source_card.fusion_list or source_card.supplement_info.get("fusion_count")) else 0
        if subject in {"me.is_kakusei"}:
            return 1 if (snapshot.me.is_kakusei or snapshot.me.max_pp >= 7) else 0
        if subject in {"me.deck_has_no_duplicates", "opponent.deck_has_no_duplicates"}:
            # 牌组中没有重复卡牌 (Highlander/单卡组)。基于牌组构筑判定: 遍历该玩家
            # deck_instances 全部实例(不论当前所在 zone 已抽/已打出)的 base_card_id,
            # 若各 base_card_id 唯一(集合大小==实例数)则返回 1。复仇者第九弹多张卡的
            # "若自己的牌组中没有重复卡牌"条件。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            deck = view.deck_instances
            if deck is not None and deck.instances:
                base_ids = [inst.card.base_card_id for inst in deck.instances]
                return 1 if len(set(base_ids)) == len(base_ids) else 0
            # 协议只下发 deck_count、不下发牌组构成。replay 验收没有牌组信息，
            # 而本地 self-play / MCTS 一定有 deck_instances（走上面那条精确判定），
            # 因此这里只在"完全拿不到牌组"时按「无重复」处理，并在 assumption 侧
            # 记录（replay 的 3 个样本 10972110/10971110 都实际成立）。
            # 注意：不能用 played_card_ids 的重复当反证——被效果加入手牌的 token
            # 也会进 played_card_ids，那不等于牌组里有重复（replay_...786874 st76
            # 就因此被误判成"有重复"）。
            return 1
        if subject in {"me.hand_top3_cost_gt_opponent", "opponent.hand_top3_cost_gt_me"}:
            # 「自己的手牌中原始费用最大的3张卡费用合计 > 对手的同项合计」
            # （10502120 手持军配团扇的伟丈夫 的进化时条件）。
            # 对手手牌在 replay 里是隐藏占位（card_id=0）→ 无法判定，返回 None
            # （显式「不可判定」，不猜）。self-play 双方手牌都可见，走精确比较。
            # replay 侧若能用 after 差分证明条件成立，会通过 effect_trigger_counts
            # 注入 __hand_top3_cost_gt_opponent__ 验收证据，此处优先读它。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            other = snapshot.opponent if subject.startswith("me.") else snapshot.me
            if view.effect_trigger_counts.get("__hand_top3_cost_gt_opponent__"):
                return 1
            # 反向证据：replay 侧由 after 差分确认「对手场没被清空」时注入，
            # 表示条件不成立（否则只能返回不可判定，该步会被记为 unresolved）。
            if view.effect_trigger_counts.get("__hand_top3_cost_le_opponent__"):
                return 0
            if any(
                (card := snapshot.cards_by_uid.get(uid)) is None or card.card_id <= 0
                for uid in other.hand
            ):
                return None
            # 协议说对手手牌有 N 张、但本地只解析出更少（replay 的隐藏手牌是
            # card_id=0 占位，不进 hand）→ 手牌内容不可观测，判「不可判定」。
            # 只在本条件上用，不改其它条件（全局加会牵连别的卡，实测净负）。
            if other.hand_count is not None and other.hand_count > len(other.hand):
                return None

            def _top3_cost_sum(player_view) -> int:
                costs = sorted(
                    (
                        card.cost
                        for uid in player_view.hand
                        if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_id > 0
                    ),
                    reverse=True,
                )
                return sum(costs[:3])

            return 1 if _top3_cost_sum(view) > _top3_cost_sum(other) else 0
        if subject in {"me.hand_top3_cost_sum", "opponent.hand_top3_cost_sum"}:
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            costs = sorted(
                (
                    card.cost
                    for uid in view.hand
                    if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_id > 0
                ),
                reverse=True,
            )
            return sum(costs[:3])
        if subject in {"me.follower_attacked_leader_previous_turn", "opponent.follower_attacked_leader_previous_turn"}:
            # 自己/对方的随从在自己的上一回合中是否攻击过对方主战者
            # （龙职业多张卡的核心条件）。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return 1 if view.follower_attacked_leader_previous_turn else 0
        if subject in {"me.evidence_amulet_activated_this_turn", "opponent.evidence_amulet_activated_this_turn"}:
            # 本回合是否启动过证据·护符（tribe 21）。由 CoreRuleKernel.activate_card
            # 写入 effect_trigger_counts 的保留键，turn_start 统一清零。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return 1 if view.effect_trigger_counts.get("__evidence_amulet_activated_this_turn__") else 0
        if subject in {"me.amulet_activated_this_turn", "opponent.amulet_activated_this_turn"}:
            # 本回合是否启动过**任意**护符。与上面的「证据·护符」是两个不同门槛：
            # 11023310 的卡面写「启动过证据·护符」，11032310 写「启动过护符」，
            # 后者在录像里由非证据族护符（90031210）的启动就成立。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return 1 if view.effect_trigger_counts.get("__amulet_activated_this_turn__") else 0
        if subject == "owner.is_active_turn":
            # 「若为自己的回合」类条件：以效果归属方（事件玩家）视角判断是否其回合。
            owner = context.event_player or "me"
            is_ally_turn = bool(getattr(snapshot, "is_ally_turn", False))
            return 1 if (owner == "me") == is_ally_turn else 0
        if subject == "owner.earth_sigil":
            # 「【土之秘术_N】…」类条件：以效果归属方（事件玩家）视角统计土之印层数。
            # 谢幕曲等由对手卡触发时，土之印属于对手而非当前视角的 me，不能用
            # me./opponent. 前缀写死。
            owner = context.event_player or "me"
            owner_view = snapshot.me if owner == "me" else snapshot.opponent
            return sum(
                max(card.stack, 1)
                for uid in owner_view.field
                if (card := snapshot.cards_by_uid.get(uid)) is not None and card.is_earth_sigil
            )
        if subject in {"me.faith_value", "opponent.faith_value"}:            # 己方/对方纹章区(crest zone)的信仰值(faith_value)。信仰是非负累积量,
            # 取纹章区内所有纹章 faith_value 的最大值(信仰纹章通常唯一, 如"信仰：古旧
            # 天书·莲妥丝"10664122)。供"若信仰值为N以上"类阈值条件门控使用, 例如
            # 古旧天书·莲妥丝(10664120)回合结束"信仰值-10并加天书深渊入手"需要信仰>=10 才发动。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return max((int(crest.faith_value) for crest in view.crests), default=0)
        if subject in {"me.field_has_other_card_with_same_original_cost_as_event_source", "opponent.field_has_other_card_with_same_original_cost_as_event_source"}:
            # 「自己使用其他卡牌时，若战场上有原始费用为X的其他卡牌」（10503210 大游戏世界）。
            # X 为本次使用卡牌的原始费用；"其他卡牌"只排除事件源自身，不按阵营过滤：
            # 卡面写的是「战场上」，未限定「自己的战场」。逐帧验证（4 个 replay 场景）都
            # 是己方无同费用卡、对手战场存在同费用卡时才触发 -1：
            #   replay_...50008835 frame18 对手 10922110(cost3)，事件源 10513310(cost3)
            #   replay_...49811764 frame75 对手 10922111(cost3)，事件源 10914110(cost3)
            #   replay_...49842438 frame11 对手 10624120(cost2)，事件源 11012310(cost2)
            event_card = context.event_source_card
            if event_card is None and context.event_source_base_card_id is None:
                return 0
            from .generated_card_materializer import default_card_resolver
            resolver = default_card_resolver()
            if event_card is not None:
                event_base = resolver.get_base_by_card_id(event_card.base_card_id) or resolver.get_base_by_card_id(event_card.card_id)
            else:
                event_base = resolver.get_base_by_card_id(int(context.event_source_base_card_id))
            if event_base is None:
                return 0
            target_cost = int(event_base.cost)
            # 优先使用「使用卡牌瞬间」的战场（事件 payload）。打出卡的入场曲可能先击杀
            # 同费用随从（10914110 / 11072310），此时 current 战场已不含该随从，但按
            # 卡面「使用时」语义仍应触发倒计数 -1。
            if context.event_field_card_ids:
                # payload 已经是「使用瞬间的战场上、除本次使用的卡自身以外的 card_id」，
                # 这里不再按 card_id 排除：同名两张卡（10503210）card_id 相同，
                # 按 card_id 排除会把另一张也排掉（replay_...114952 st15）。
                for card_id in context.event_field_card_ids:
                    base = resolver.get_base_by_card_id(int(card_id))
                    if base is not None and int(base.cost) == target_cost:
                        return 1
                return 0
            for uid in (*snapshot.me.field, *snapshot.opponent.field):
                if event_card is not None and event_card.uid is not None and uid == event_card.uid:
                    continue
                card = snapshot.cards_by_uid.get(uid)
                if card is None:
                    continue
                base = resolver.get_base_by_card_id(card.card_id)
                cost = int(base.cost) if base is not None else int(card.cost)
                if cost == target_cost:
                    return 1
            return 0
        if subject in {"me.field_last_words_count", "opponent.field_last_words_count"}:
            # 己方/对方场上拥有【谢幕曲】的卡牌数量（纹章"若己方场上有谢幕曲卡"类条件）。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return sum(1 for uid in view.field
                       if (card := snapshot.cards_by_uid.get(uid)) is not None and card.has_last_words)
        if subject in {"me.field_original_cost_1_count", "opponent.field_original_cost_1_count"}:
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            from .generated_card_materializer import default_card_resolver
            resolver = default_card_resolver()
            count = 0
            for uid in view.field:
                card = snapshot.cards_by_uid.get(uid)
                if card is None:
                    continue
                base = resolver.get_base_by_card_id(card.card_id)
                if base is not None and int(base.cost) == 1:
                    count += 1
            return count
        if subject in {"me.field_max_original_cost", "opponent.field_max_original_cost"}:
            # 己方/对方场上卡牌的最大原始费用（主教"若己方场上有原始费用为6或以上的卡牌"
            # 类条件, 与阈值 >= 比较）。原始费用取 resolver 中该 base_card 的印刷费用。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            from .generated_card_materializer import default_card_resolver
            resolver = default_card_resolver()
            max_cost = 0
            for uid in view.field:
                card = snapshot.cards_by_uid.get(uid)
                if card is None:
                    continue
                base = resolver.get_base_by_card_id(card.card_id)
                cost = int(base.cost) if base is not None else int(card.cost)
                max_cost = max(max_cost, cost)
            return max_cost
        if subject in {"me.artifact_types_entered", "opponent.artifact_types_entered"}:
            # 本局进入战场的创造物(artifact, tribe=14)·随从的种类数(distinct base_card_id)。
            # 跑酷: 若本局入场创造物随从种类>=3 则改为发动所有能力。
            player = "me" if subject.startswith("me.") else "opponent"
            view = snapshot.me if player == "me" else snapshot.opponent
            from .generated_card_materializer import default_card_resolver
            resolver = default_card_resolver()
            types: set[int] = set()
            for item in view.entered_field_card_ids:
                base = resolver.get_base_by_card_id(item.card_id)
                if base is None:
                    continue
                tribes = tuple(getattr(base, "tribes", ()) or ())
                is_follower = int(getattr(base, "type_id", getattr(base, "card_type", 0)) or 0) == 1
                if 14 in tribes and is_follower:
                    types.add(int(base.base_card_id))
            return len(types)
        if subject in {"me.leader_life", "opponent.leader_life"}:
            # 主战者当前生命值 (伽罗塔德 对 泽特: 若自己主战者生命≤12 则本卡费用-1)。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            return int(view.life)
        if subject in {"me.field_has_cost_ge_5_follower", "opponent.field_has_cost_ge_5_follower"}:
            # 己方/对方场上是否存在原始费用为5或以上的随从 (尤泽塔【入场曲】条件)。
            # 原始费用取 resolver 印刷费用, 只计随从。
            view = snapshot.me if subject.startswith("me.") else snapshot.opponent
            from .generated_card_materializer import default_card_resolver
            resolver = default_card_resolver()
            for uid in view.field:
                card = snapshot.cards_by_uid.get(uid)
                if card is None or not card.card_type.is_follower:
                    continue
                base = resolver.get_base_by_card_id(card.card_id)
                cost = int(base.cost) if base is not None else int(card.cost)
                if cost >= 5:
                    return 1
            return 0
        if subject == "me.super_evolve_unlocked":
            return 1 if snapshot.me.super_evolve_turn > 0 and snapshot.me.turn >= snapshot.me.super_evolve_turn else 0
        if subject == "both.max_pp":
            return min(snapshot.me.max_pp, snapshot.opponent.max_pp)
        if subject in {"me.played_original_cost_set", "opponent.played_original_cost_set"}:
            # 从本局使用卡牌历史(played_card_ids)派生的"原始费用集合"视图。
            # played_card_ids 是通用历史记录, 此处按 cost 维度派生; 未来可增加 tribe/type 等派生视图。
            player = "me" if subject.startswith("me.") else "opponent"
            view = snapshot.me if player == "me" else snapshot.opponent
            from .generated_card_materializer import default_card_resolver
            resolver = default_card_resolver()
            costs: set[int] = set()
            for item in view.played_card_ids:
                base = resolver.get_base_by_card_id(item.card_id)
                if base is not None:
                    costs.add(int(base.cost))
            return tuple(sorted(costs))
        if subject in {"me.same_name_entered_count", "opponent.same_name_entered_count"}:
            # 本局进入战场的、与 source 同名(同 base_card_id)的随从数量, 排除本随从自身
            # (记录含自己, 故减1)。用于"本局其他同名随从入场数>=N"类条件
            # (如沉溺的实验体: 本局其他实验体入场>=5 则 +3/+3)。
            player = "me" if subject.startswith("me.") else "opponent"
            view = snapshot.me if player == "me" else snapshot.opponent
            if source_card is None:
                return 0
            total = 0
            for item in view.entered_field_card_ids:
                if item.card_id == source_card.base_card_id:
                    total = item.count
                    break
            return max(0, total - 1)
        if subject == "runtime.selected_mode_options":
            return tuple(context.action.selected_indexes)
        if subject == "runtime.is_enhance":
            # 爆能强化打出 (play_kind==ENHANCE) 返回1, 否则0。用于"爆能强化_N改为
            # 发动所有能力"的 all_abilities gate (区别于卡面场景条件类全发)。
            from decision_engine.actions.model import PlayKind
            act = getattr(context, "action", None)
            return 1 if (act is not None and getattr(act, "play_kind", None) is PlayKind.ENHANCE) else 0
        if subject == "runtime.enhance_cost":
            # 本次爆能强化实际支付的费用。同一张卡可以有多个并列的
            # 【爆能强化_N】段落（10624110 的 _7 / _8 各召唤一个随从），
            # 必须按实际支付的费用区分该发动哪一段；非爆能强化打出返回 0。
            from decision_engine.actions.model import PlayKind
            act = getattr(context, "action", None)
            if act is None or getattr(act, "play_kind", None) is not PlayKind.ENHANCE:
                return 0
            return int(getattr(act, "cost", 0) or 0)
        if subject == "current_turn":
            # "current_turn == my_turn" 表示当前是效果所有者的回合
            # event_player 标识效果所有者的 side（如 sub.player）
            if context.event_player is not None:
                is_owner_turn = (context.event_player == "me" and snapshot.is_ally_turn) or (context.event_player == "opponent" and not snapshot.is_ally_turn)
                return "my_turn" if is_owner_turn else "opponent_turn"
            return "my_turn" if snapshot.is_ally_turn else "opponent_turn"
        if subject == "turn_player":
            # turn_player 表示从 source_card 的视角看，结束/开始回合的是谁。
            # 优先用真实事件方（turn_event_player，跨侧订阅时与效果归属方不同），
            # 缺失时退回效果归属方（同侧订阅的既有语义）。
            turn_side = context.turn_event_player or context.event_player
            if turn_side is not None:
                source_side = self._infer_source_side(snapshot, context)
                if source_side is not None:
                    return "me" if turn_side == source_side else "opponent"
                # 无法确定 source side 时返回真实事件方
                return turn_side
            # fallback: 从 snapshot.is_ally_turn 推断
            return "me" if snapshot.is_ally_turn else "opponent"
        return None

    @staticmethod
    def _infer_source_side(snapshot: BattleSnapshot, context: ConditionEvaluationContext) -> str | None:
        """推断 source_card 属于哪一侧。"""
        source_card = context.source_card
        if source_card is None:
            return None
        uid = source_card.uid
        if uid in snapshot.me.field or uid in snapshot.me.hand:
            return "me"
        if uid in snapshot.opponent.field or uid in snapshot.opponent.hand:
            return "opponent"
        # 纹章没有实体 uid：按该纹章当前归属的侧确定订阅者 side，
        # 否则「对手的回合结束时」这类纹章条件无法判定 turn_player。
        for side in ("me", "opponent"):
            view = getattr(snapshot, side)
            for crest in view.crests:
                if crest.card_id == source_card.base_card_id:
                    return side
        return None

    def _skybound_progress(self, context, source_card) -> int:
        """奥义/解放奥义进度。

        协议只在**手牌**卡上下发 `supplement_info.skybound_art_count`；卡打出后
        进入场地，同一 uid 的 CardView 已经没有这份补充信息。fanfare 阶段求值
        必须退回「打出前」快照取进度，否则「奥义_10 / 解放奥义_15」永远不成立
        （10401110 / 10404110 的伤害段从未执行）。
        """
        progress = int(source_card.supplement_info.get("skybound_art_count", 0))
        if progress:
            return progress
        base = context.resource_snapshot
        if base is not None:
            base_card = base.cards_by_uid.get(source_card.uid)
            if base_card is not None:
                return int(base_card.supplement_info.get("skybound_art_count", 0))
        return 0

    def _player_condition_value(self, snapshot: BattleSnapshot, player: str, field: str) -> int:
        view = snapshot.me if player == "me" else snapshot.opponent
        if field == "followers_entered_battlefield":
            return int(view.followers_entered_battlefield)
        if field in {"hand", "hand_count"}:
            return len(view.hand)
        if field in {"field", "field_count"}:
            return len(view.field)
        if field == "earth_sigil":
            return sum(1 for uid in view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.is_earth_sigil)
        if field == "crests":
            return len(view.crests)
        if field == "field_amulet_count":
            return sum(1 for uid in view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_amulet_like)
        if field == "field_evidence_count":
            return sum(1 for uid in view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and is_evidence_card(card))
        if field == "hand_evidence_amulet_count":
            return sum(1 for uid in view.hand if (card := snapshot.cards_by_uid.get(uid)) is not None and is_evidence_card(card))
        if field == "evidence_count":
            # 「证据」是战场 + 手牌的证据·护符总数：11073310 的
            # 「使自己的战场上的证据-2」在只有 1 点证据时不发动（录像对照：
            # 战场 1 张 + 手牌 1 张 = 2 点才支付，且只消耗战场上的那张）。
            return sum(
                1
                for uid in (*view.field, *view.hand)
                if (card := snapshot.cards_by_uid.get(uid)) is not None and is_evidence_card(card)
            )
        if field == "field_evolved_follower":
            return sum(1 for uid in view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower and card.evolve_state > 0)
        if field == "field_super_evolved_follower":
            return sum(1 for uid in view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower and card.evolve_state >= 2)
        if field in {"play_point_max", "max_pp"}:
            # 卡面写「自己的能量点最大值为 7 或以上」：协议字段是 max_pp。
            return int(view.max_pp)
        return int(getattr(view, field))

    def _text_condition_result(self, snapshot: BattleSnapshot, effect: ExecutableEffect, condition: ConditionSpec, context: ConditionEvaluationContext) -> bool | None:
        if condition.operator != "requires_review":
            return None
        if condition.subject == "ougi":
            source = context.source_card
            if source is None:
                return None
            return self._skybound_progress(context, source) >= 10
        if condition.subject == "super_ougi":
            source = context.source_card
            if source is None:
                return None
            # 解放奥义阈值固定 15，进度用奥义槽 skybound_art_count（见 §13.12）。
            # super_skybound_art_count 在真实 trace 中恒 0，不能当进度用。
            return self._skybound_progress(context, source) >= 15
        if condition.subject == "self_unevolved":
            # 伊兰翠: 本随从为进化前
            source = context.source_card
            return source is not None and source.evolve_state == 0
        if condition.subject == "self_evolved":
            # 本随从为进化后
            source = context.source_card
            return source is not None and source.evolve_state > 0
        if condition.subject == "if":
            if effect is None:
                # evaluate_static：没有 effect/action 上下文，无法判定选择条件。
                return None
            if effect.target_selector is not None and effect.target_selector.scope == "selected":
                return bool(context.action.target_groups and context.action.target_groups[0])
            return True
        if condition.subject == "self_followers_no_attack_this_turn":
            # 「本回合中自己的随从没有进行过攻击」：读 replay 侧按 action 序列维护的
            # 标记（effect_trigger_counts，turn_start 清零）。
            #
            # 不能用 attacks_left：回合结束阶段 kernel 已经把随从的攻击次数清零，
            # `attacks_left > 0` 恒为 False，10364122 的「分配 X 点伤害」整段被跳过
            # （round 223 打点确认拦截点唯一在 _snapshot_conditions_match）。
            # 也不能用协议 attack_count：实测 7117 个 turn_end 样本全为 0。
            player = context.event_player or "me"
            view = snapshot.me if player == "me" else snapshot.opponent
            return not view.effect_trigger_counts.get("__followers_attacked_this_turn__")
        return None

    def _snapshot_exists_result(self, snapshot: BattleSnapshot, condition: ConditionSpec, context: ConditionEvaluationContext) -> bool | None:
        """「战场/手牌上存在满足筛选的卡」类条件（auto 候选生成的 snapshot_exists）。

        ref 只支持显式声明过的过滤子集：
        - exclude_source：排除效果来源卡自身
        - original_cost_eq_N：按 card_database 印刷费用匹配
        其他 ref 返回 None，由调用方计入 condition_not_supported。
        """
        subject = condition.subject or ""
        parts = subject.split(".")
        if len(parts) != 2:
            return None
        side, zone = parts
        if side == "self":
            side = self._infer_source_side(snapshot, context) or "me"
        elif side == "enemy":
            source_side = self._infer_source_side(snapshot, context)
            side = "opponent" if source_side == "me" else ("me" if source_side == "opponent" else "opponent")
        if side not in {"me", "opponent"} or zone not in {"field", "hand"}:
            return None
        ref = condition.ref or ""
        known_filter = ref.startswith("filters:")
        exclude_source = known_filter and "exclude_source" in ref
        cost_match = re.search(r"original_cost_eq_(\d+)", ref) if known_filter else None
        # 「手牌/战场上没有指定卡」：ref 写成 filters:absent_base_card_id_eq_<id>。
        # 典型文本「若自己的手牌中没有『X』，则将 1 张『X』加入手牌」
        # （86834920 加护：破式的变貌·菲绫）。
        absent = known_filter and "absent_" in ref
        base_card_match = re.search(r"base_card_id_eq_(\d+)", ref) if known_filter else None
        if known_filter and not exclude_source and not cost_match and base_card_match is None:
            return None
        exclude_uid = context.source_card.uid if (exclude_source and context.source_card is not None) else None
        view = snapshot.me if side == "me" else snapshot.opponent
        uids = view.field if zone == "field" else view.hand
        found = False
        for uid in uids:
            if uid == exclude_uid:
                continue
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            if cost_match is not None:
                printed_cost = _base_card_printed_costs().get(card.base_card_id)
                if printed_cost is None or printed_cost != int(cost_match.group(1)):
                    continue
            if base_card_match is not None and card.base_card_id != int(base_card_match.group(1)):
                continue
            found = True
            break
        return (not found) if absent else found

    def _selection_filter_condition_result(self, snapshot: BattleSnapshot, condition: ConditionSpec, action: LegalAction) -> bool | None:
        if condition.subject != "selected_card" or condition.operator != "matches":
            return None
        selected = tuple(uid for group in action.target_groups for uid in group)
        if not selected:
            return False
        if condition.ref == "filters:side_self_card_type_amulet":
            return all(uid in snapshot.me.field and (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_amulet_like for uid in selected)
        return None

    def _condition_expected_value(self, value: Any) -> int | str | tuple[Any, ...] | None:
        if value is None:
            return 1
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    def _compare_condition_value(self, value: int | str | tuple[int, ...], operator: str, expected: int | str | tuple[Any, ...]) -> bool | None:
        if operator in {"==", "=", "eq"}:
            return value == expected
        if operator in {"!=", "ne"}:
            return value != expected
        if operator == ">=":
            return value >= expected if isinstance(value, int) and isinstance(expected, int) else None
        if operator == "<=":
            return value <= expected if isinstance(value, int) and isinstance(expected, int) else None
        if operator == ">":
            return value > expected if isinstance(value, int) and isinstance(expected, int) else None
        if operator == "<":
            return value < expected if isinstance(value, int) and isinstance(expected, int) else None
        if operator == "contains":
            if isinstance(value, tuple):
                return expected in value
            if isinstance(expected, tuple):
                return value in expected
            return False
        if operator == "in":
            return value in expected if isinstance(expected, tuple) else False
        if operator == "covers":
            # value 集合是否包含 expected 中的所有元素（如"使用卡费用覆盖1~8"）
            if isinstance(value, tuple) and isinstance(expected, tuple):
                return set(expected).issubset(set(value))
            return False
        return None
