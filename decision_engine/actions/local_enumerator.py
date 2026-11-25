# -*- coding: utf-8 -*-
"""本地基础动作枚举器。"""

from __future__ import annotations

import logging
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.cards import UnifiedCardTemplate, load_card_templates_by_base_card_id
from decision_engine.cards.effect_specs import CardEffectSpecBundle, EffectSpec, EffectSpecOperation, EffectSpecTarget, load_effect_spec_bundles
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, ChoiceSkillInfo, ModeSkillInfo, SelectFusionOption, SelectSkillOption

_logger = logging.getLogger(__name__)


MAX_FIELD_SIZE = 5


@lru_cache(maxsize=1)
def _effect_spec_bundles() -> dict[int, CardEffectSpecBundle]:
    path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_effect_specs.json"
    return {
        bundle.base_card_id: bundle
        for bundle in load_effect_spec_bundles(path)
    }


@lru_cache(maxsize=1)
def _card_templates_by_base_card_id() -> dict[int, UnifiedCardTemplate]:
    path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_templates.json"
    return load_card_templates_by_base_card_id(path)


class LocalActionEnumerator:
    """只根据 BattleSnapshot 中可观察基础状态枚举本地动作。"""

    def enumerate(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        if not snapshot.is_ally_turn or snapshot.is_turn_ended:
            return []
        actions: list[LegalAction] = []
        actions.extend(self._enumerate_plays(snapshot))
        actions.extend(self._enumerate_attacks(snapshot))
        actions.extend(self._enumerate_evolves(snapshot))
        actions.extend(self._enumerate_activations(snapshot))
        actions.extend(self._enumerate_special_actions(snapshot))
        actions.extend(self._enumerate_fusion(snapshot))
        actions.extend(self._enumerate_extra_pp(snapshot))
        actions.append(self._action(kind=ActionKind.TURN_END, source_uid=0))
        return actions

    def _action(
        self,
        *,
        kind: ActionKind,
        source_uid: int,
        play_kind: PlayKind | None = None,
        cost: int | None = None,
        target_groups: tuple[tuple[int, ...], ...] = (),
        selected_indexes: tuple[int, ...] = (),
        material_uids: tuple[int, ...] = (),
        variant: str | None = None,
        target_group_specs: tuple[str, ...] = (),
    ) -> LegalAction:
        return LegalAction(
            kind=kind,
            source_uid=source_uid,
            play_kind=play_kind,
            cost=cost,
            target_groups=target_groups,
            selected_indexes=selected_indexes,
            material_uids=material_uids,
            variant=variant,
            legality_source="local",
            confidence="core_rules",
            target_group_specs=target_group_specs,
        )

    def _enumerate_plays(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        actions: list[LegalAction] = []
        field_full = len(snapshot.me.field) >= MAX_FIELD_SIZE
        for uid in sorted(snapshot.me.hand):
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            # 出牌优先级（强制）：
            # 1. Enhance：PP >= enhance_cost 时强制以最高可负担 enhance 费用打出
            # 2. Normal：否则以原始费用打出
            # 3. Accelerate / Crystal：否则以对应方式打出
            enhance_played = False
            if card.can_enhance and card.enhance_costs:
                # 多个 enhance 费用时选择可负担的最高费用
                affordable = [c for c in sorted(card.enhance_costs, reverse=True) if self._can_play_card(snapshot, card, c, field_full)]
                if affordable:
                    best_enhance_cost = affordable[0]
                    actions.extend(self._actions_with_parameters(snapshot, card, "play", kind=ActionKind.PLAY, source_uid=uid, play_kind=PlayKind.ENHANCE, cost=best_enhance_cost))
                    enhance_played = True
            if not enhance_played:
                if self._can_play_card(snapshot, card, card.cost, field_full):
                    actions.extend(self._actions_with_parameters(snapshot, card, "play", kind=ActionKind.PLAY, source_uid=uid, play_kind=PlayKind.NORMAL, cost=card.cost))
                else:
                    # accelerate — 作为法术打出，不触发 fanfare 目标选择
                    for cost in sorted(card.accelerate_costs):
                        if cost <= snapshot.me.pp:
                            actions.append(self._action(kind=ActionKind.PLAY, source_uid=uid, play_kind=PlayKind.ACCELERATE, cost=cost, target_groups=(), selected_indexes=()))
                    # crystal — 作为护符打出，不触发 fanfare 目标选择
                    for cost in sorted(card.crystal_costs):
                        if cost <= snapshot.me.pp and not field_full:
                            actions.append(self._action(kind=ActionKind.PLAY, source_uid=uid, play_kind=PlayKind.CRYSTAL, cost=cost, target_groups=(), selected_indexes=()))
        # diff 检查
        self._check_play_diff(snapshot, actions)
        return actions

    def _check_play_diff(self, snapshot: BattleSnapshot, actions: list[LegalAction]) -> None:
        legal = snapshot.legal
        if legal is None:
            return
        local_play_uids = frozenset(a.source_uid for a in actions if a.play_kind is PlayKind.NORMAL)
        local_enhance_uids = frozenset(a.source_uid for a in actions if a.play_kind is PlayKind.ENHANCE)
        local_accel_uids = frozenset(a.source_uid for a in actions if a.play_kind is PlayKind.ACCELERATE)
        local_crystal_uids = frozenset(a.source_uid for a in actions if a.play_kind is PlayKind.CRYSTAL)
        if legal.can_play and local_play_uids != legal.can_play:
            _logger.warning("LocalActionEnumerator play diff: local=%s protocol=%s", sorted(local_play_uids), sorted(legal.can_play))
        if legal.can_enhance and local_enhance_uids != legal.can_enhance:
            _logger.warning("LocalActionEnumerator enhance diff: local=%s protocol=%s", sorted(local_enhance_uids), sorted(legal.can_enhance))
        if legal.can_accelerate and local_accel_uids != legal.can_accelerate:
            _logger.warning("LocalActionEnumerator accelerate diff: local=%s protocol=%s", sorted(local_accel_uids), sorted(legal.can_accelerate))
        if legal.can_crystal and local_crystal_uids != legal.can_crystal:
            _logger.warning("LocalActionEnumerator crystal diff: local=%s protocol=%s", sorted(local_crystal_uids), sorted(legal.can_crystal))

    def _can_play_card(self, snapshot: BattleSnapshot, card: CardView, cost: int, field_full: bool) -> bool:
        if cost > snapshot.me.pp:
            return False
        if not card.card_type.is_spell and field_full:
            return False
        return True

    def _enumerate_attacks(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        actions: list[LegalAction] = []
        guard_uids = tuple(uid for uid in snapshot.opponent.field if self._is_guard(snapshot, uid))
        attackable_guard_uids = tuple(uid for uid in guard_uids if self._is_attackable_follower(snapshot, uid))
        for uid in sorted(snapshot.me.field):
            card = snapshot.cards_by_uid.get(uid)
            if card is None or not self._can_attack(card):
                continue
            candidates = attackable_guard_uids if guard_uids else tuple(target_uid for target_uid in snapshot.opponent.field if self._is_attackable_follower(snapshot, target_uid))
            if card.attack_targets:
                allowed = frozenset(card.attack_targets)
                candidates = tuple(target_uid for target_uid in candidates if target_uid in allowed)
            for target_uid in candidates:
                actions.append(
                    self._action(
                        kind=ActionKind.ATTACK_FOLLOWER,
                        source_uid=uid,
                        target_groups=((target_uid,),),
                        target_group_specs=("attack_follower",),
                    )
                )
            if card.can_attack_leader and (not guard_uids or card.can_ignore_guard):
                # 主战者用 unique_id 作 target；为 0 时没有目标，specs 也为空
                leader_groups = ((snapshot.opponent.unique_id,),) if snapshot.opponent.unique_id else ()
                actions.append(
                    self._action(
                        kind=ActionKind.ATTACK_LEADER,
                        source_uid=uid,
                        target_groups=leader_groups,
                        target_group_specs=("attack_leader",) if leader_groups else (),
                    )
                )
        return actions

    def _is_guard(self, snapshot: BattleSnapshot, uid: int) -> bool:
        card = snapshot.cards_by_uid[uid]
        return card.has_guard and self._is_follower(card)

    def _is_attackable_guard(self, snapshot: BattleSnapshot, uid: int) -> bool:
        card = snapshot.cards_by_uid[uid]
        return card.has_guard and self._is_attackable_follower(snapshot, uid)

    def _is_attackable_follower(self, snapshot: BattleSnapshot, uid: int) -> bool:
        card = snapshot.cards_by_uid[uid]
        return self._is_follower(card) and self._can_be_attacked(card)

    def _is_follower(self, card: CardView) -> bool:
        return card.card_type.is_follower

    def _can_be_attacked(self, card: CardView) -> bool:
        if card.has_ambush or card.has_cant_be_attacked:
            return False
        return True

    def _can_attack(self, card: CardView) -> bool:
        return card.attacks_left > 0 and card.can_attack and not card.has_cant_attack

    def _enumerate_evolves(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        actions: list[LegalAction] = []
        me = snapshot.me
        # 使用 PlayerView.evolve_turn / super_evolve_turn 判断解禁
        # 这些字段在协议端由服务器下发，在本地 self-play 中由 LocalMatchSetup 初始化
        # evolve_turn=0 表示未设置（兼容旧 fixture），此时 fallback 到 can_evolve 字段
        evolve_unlocked = me.evolve_turn > 0 and me.turn >= me.evolve_turn
        super_evolve_unlocked = me.super_evolve_turn > 0 and me.turn >= me.super_evolve_turn

        # 纯本地推导：EP > 0、回合已解禁、本回合未手动进化
        if me.ep > 0 and not me.is_evolved_this_turn:
            for uid in sorted(me.field):
                card = snapshot.cards_by_uid.get(uid)
                if card is None or card.evolve_state != 0:
                    continue
                if card.card_type != CardType.FOLLOWER:
                    continue
                # 优先使用 evolve_turn 判断；如果未设置则 fallback 到 can_evolve 字段
                if not evolve_unlocked and not card.can_evolve:
                    continue
                actions.extend(self._actions_with_parameters(snapshot, card, "evolve", kind=ActionKind.EVOLVE, source_uid=uid))
        # 纯本地推导：SEP > 0、回合已解禁、本回合未手动进化
        if me.sep > 0 and not me.is_evolved_this_turn:
            for uid in sorted(me.field):
                card = snapshot.cards_by_uid.get(uid)
                if card is None or card.evolve_state != 0:
                    continue
                if card.card_type != CardType.FOLLOWER:
                    continue
                if not super_evolve_unlocked and not card.can_super_evolve:
                    continue
                actions.extend(self._actions_with_parameters(snapshot, card, "super_evolve", kind=ActionKind.SUPER_EVOLVE, source_uid=uid))
        # diff 检查：协议存在时对比本地枚举与协议
        self._check_evolve_diff(snapshot, actions)
        return actions

    def _check_evolve_diff(self, snapshot: BattleSnapshot, actions: list[LegalAction]) -> None:
        legal = snapshot.legal
        if legal is None:
            return
        local_evolve_uids = frozenset(a.source_uid for a in actions if a.kind is ActionKind.EVOLVE)
        local_super_evolve_uids = frozenset(a.source_uid for a in actions if a.kind is ActionKind.SUPER_EVOLVE)
        protocol_evolve = legal.can_evolve
        protocol_super_evolve = legal.can_super_evolve
        if protocol_evolve and local_evolve_uids != protocol_evolve:
            _logger.warning(
                "LocalActionEnumerator evolve diff: local=%s protocol=%s",
                sorted(local_evolve_uids), sorted(protocol_evolve),
            )
        if protocol_super_evolve and local_super_evolve_uids != protocol_super_evolve:
            _logger.warning(
                "LocalActionEnumerator super_evolve diff: local=%s protocol=%s",
                sorted(local_super_evolve_uids), sorted(protocol_super_evolve),
            )

    def _enumerate_activations(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        actions: list[LegalAction] = []
        # 纯本地推导
        for uid in sorted(snapshot.me.field):
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            if not self._can_locally_activate(snapshot, card):
                continue
            actions.extend(self._actions_with_parameters(snapshot, card, "activation", kind=ActionKind.ACTIVATE, source_uid=uid))
        # diff 检查
        self._check_activate_diff(snapshot, actions)
        return actions

    def _check_activate_diff(self, snapshot: BattleSnapshot, actions: list[LegalAction]) -> None:
        legal = snapshot.legal
        if legal is None:
            return
        local_uids = frozenset(a.source_uid for a in actions)
        protocol_uids = legal.can_activate
        if protocol_uids and local_uids != protocol_uids:
            _logger.warning(
                "LocalActionEnumerator activate diff: local=%s protocol=%s",
                sorted(local_uids), sorted(protocol_uids),
            )

    def _can_locally_activate(self, snapshot: BattleSnapshot, card: CardView) -> bool:
        if card.activation_used_this_turn:
            return False
        if not card.has_activation:
            return False
        cost = self._activation_cost(card)
        # 费用未知时默认 0（大部分 activate 是 0 费启动能力；
        # 有 PP 消耗的 activate 应通过 EffectSpec pay_pp 声明费用）
        effective_cost = cost if cost is not None else 0
        if effective_cost > snapshot.me.pp:
            return False
        return True

    def _activation_cost(self, card: CardView) -> int | None:
        effect_cost = self._activation_effect_spec_cost(card)
        if effect_cost is not None:
            return effect_cost
        template = _card_templates_by_base_card_id().get(card.base_card_id)
        if template is None:
            return None
        for mode in template.play_modes:
            if getattr(mode.kind, "value", mode.kind) != "activation":
                continue
            cost = mode.cost
            if cost is None:
                return None
            if cost.kind == "fixed" and cost.value is not None:
                return int(cost.value)
        return None

    def _activation_effect_spec_cost(self, card: CardView) -> int | None:
        bundle = _effect_spec_bundles().get(card.base_card_id)
        if bundle is None:
            return None
        costs: list[int] = []
        for spec in bundle.specs:
            if spec.trigger.event not in {"activate", "activation"}:
                continue
            for operation in spec.operations:
                if operation.op != "pay_pp" or operation.amount is None:
                    continue
                if operation.amount.kind == "fixed" and operation.amount.value is not None:
                    costs.append(int(operation.amount.value))
        return max(costs) if costs else None

    def _enumerate_special_actions(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        actions: list[LegalAction] = []
        for uid in sorted(snapshot.me.field):
            card = snapshot.cards_by_uid.get(uid)
            if card is not None and card.can_special_action:
                actions.extend(self._actions_with_parameters(snapshot, card, "special_action", kind=ActionKind.SPECIAL_ACTION, source_uid=uid))
        return actions

    def _enumerate_fusion(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        actions: list[LegalAction] = []
        hand_uids = frozenset(snapshot.me.hand)
        legal = snapshot.legal
        fused_sources: set[int] = set()
        if legal is not None:
            for uid in sorted(legal.can_fusion & hand_uids):
                card = snapshot.cards_by_uid.get(uid)
                if card is None or self._fusion_used_this_turn(card):
                    continue
                option = legal.select_fusion_map.get(uid)
                if option is None:
                    continue
                for material_uids in self._protocol_fusion_groups(option):
                    actions.append(self._action(kind=ActionKind.FUSION, source_uid=uid, material_uids=material_uids))
                fused_sources.add(uid)
        for uid in sorted(snapshot.me.hand):
            if uid in fused_sources:
                continue
            card = snapshot.cards_by_uid.get(uid)
            if card is None or self._fusion_used_this_turn(card):
                continue
            # 模板 fusion 的无条件 pay_pp 是动作合法性的一部分；不能枚举出
            # 预测阶段必然因 PP 不足而失败的融合分支。
            if not self._can_pay_unconditional_fusion_cost(snapshot, card):
                continue
            template = _card_templates_by_base_card_id().get(card.base_card_id)
            fusion_modes = tuple(
                mode for mode in (template.play_modes if template is not None else ())
                if getattr(mode.kind, "value", mode.kind) == "fusion"
            )
            if fusion_modes:
                for mode in fusion_modes:
                    if mode.fusion is not None:
                        materials = self._template_fusion_materials(snapshot, card, mode.fusion)
                        actions.extend(
                            self._action(kind=ActionKind.FUSION, source_uid=uid, material_uids=group)
                            for group in materials
                        )
                continue
            if card.select_fusion is not None:
                for material_uids in self._fusion_material_groups(card.select_fusion, hand_uids, uid):
                    actions.append(self._action(kind=ActionKind.FUSION, source_uid=uid, material_uids=material_uids))
                continue
            for group in card.fusion_list:
                material_uids = tuple(material_uid for material_uid in group if material_uid in hand_uids and material_uid != uid)
                if len(material_uids) == len(group):
                    actions.append(self._action(kind=ActionKind.FUSION, source_uid=uid, material_uids=material_uids))
        return actions

    def _actions_with_parameters(self, snapshot: BattleSnapshot, card: CardView, mode: str, **kwargs) -> list[LegalAction]:
        params = self._action_parameters(snapshot, card, mode)
        return [
            self._action(target_groups=groups, selected_indexes=indexes, target_group_specs=specs, **kwargs)
            for groups, indexes, specs in params
        ]

    @staticmethod
    def _with_group_specs(
        params: list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]],
        group_spec: str,
    ) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]]]:
        """给非 effect-spec 路径的 target_groups 补上等长的来源标识。"""
        return [(groups, indexes, (group_spec,) * len(groups)) for groups, indexes in params]

    def _action_parameters(
        self, snapshot: BattleSnapshot, card: CardView, mode: str
    ) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]]]:
        legal_options = snapshot.legal.select_skill_map.get(mode, {}).get(card.uid, ()) if snapshot.legal is not None else ()
        if legal_options:
            return self._with_group_specs(self._expand_select_skill_options(legal_options), "protocol_select")
        options = card.select_skill_map.get(mode, ())
        if options:
            return self._with_group_specs(self._expand_select_skill_options(options), "local_select")
        effect_params = self._effect_spec_parameters(snapshot, card, mode)
        if effect_params is not None:
            return effect_params
        mode_info = card.mode_skill_map.get(mode)
        if mode_info is not None:
            expanded = self._expand_mode_skill(mode_info)
            if expanded:
                return self._with_group_specs(expanded, "mode_skill")
        choice_info = card.choice_skill_map.get(mode)
        if choice_info is not None:
            expanded = self._expand_choice_skill(choice_info)
            if expanded:
                return self._with_group_specs(expanded, "choice_skill")
        return [((), (), ())]

    def _has_effect_spec_trigger(self, card: CardView, trigger_events: set[str]) -> bool:
        bundle = _effect_spec_bundles().get(card.base_card_id)
        if bundle is None:
            return False
        return any(spec.trigger.event in trigger_events for spec in bundle.specs)

    def _effect_spec_parameters(
        self, snapshot: BattleSnapshot, card: CardView, mode: str
    ) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]]] | None:
        trigger_events = {
            "play": {"fanfare", "on_spell_play"},
            "activation": {"activation", "activate"},
            "evolve": {"manual_evolve"},
            "super_evolve": {"manual_evolve", "manual_super_evolve"},
        }.get(mode)
        if trigger_events is None:
            return None
        bundle = _effect_spec_bundles().get(card.base_card_id)
        if bundle is None:
            return None
        # 收集每个 spec 的目标候选（各自独立的选择步骤）
        per_spec_candidates: list[list[tuple[int, ...]]] = []
        # 与 per_spec_candidates 等长：每个 slot 对应的 spec_id，供 target_group_specs 使用。
        # 位置会因条件不满足的 spec 被跳过而漂移，spec_id 不会。
        per_spec_ids: list[str] = []
        mode_indexes: list[int] = []
        has_required_parameters = False
        has_blocked_required_target = False
        for spec in bundle.specs:
            if spec.trigger.event not in trigger_events:
                continue
            indexes = tuple(index for operation in spec.operations if (index := self._mode_option_index(operation)) is not None)
            if indexes:
                # 模式选择(mode option)始终可选：选择哪个能力是玩家的决定，不受该
                # 能力目标候选是否存在的影响。若选中的能力目标为空(如「使随机守护进化前
                # 其他随从进化」而场上无守护随从)，效果空转但入场/发动不受阻。
                mode_indexes.extend(indexes)
                continue
            if spec.target is None:
                # spec.target 为空时，检查 operations 中是否有 selected_* 的 target_ref
                # 例如 evolve/super_evolve spec 可能在 operation.target_ref 中编码目标选择
                synthesized = self._synthesize_target_from_operations(spec, None)
                if synthesized is None:
                    continue
                expanded = self._expand_effect_target(snapshot, card, synthesized)
                if expanded is None:
                    continue
                has_real_selection = any(groups != () for groups, _ in expanded)
                if has_real_selection:
                    has_required_parameters = True
                    candidates = [groups[0] for groups, _ in expanded if groups]
                    per_spec_candidates.append(candidates)
                    per_spec_ids.append(spec.spec_id)
                elif synthesized.kind.startswith("selected") and not expanded:
                    has_blocked_required_target = True
                continue
            expanded = self._expand_effect_target(snapshot, card, spec.target)
            if expanded is None:
                continue
            # 只有需要用户选择的目标（非 all_* 的空参数）才标记为 required
            has_real_selection = any(groups != () for groups, _ in expanded)
            if has_real_selection:
                has_required_parameters = True
                # 提取这个 spec 的候选 uid 列表（每个 expanded entry 的第一个 group）
                candidates = [groups[0] for groups, _ in expanded if groups]
                per_spec_candidates.append(candidates)
                per_spec_ids.append(spec.spec_id)
            elif spec.target.kind.startswith("selected") and not expanded:
                # selected_* 目标但候选池为空 → 阻止打出
                has_blocked_required_target = True
        mode_options = self._expand_mode_option_indexes(snapshot, card, tuple(dict.fromkeys(mode_indexes)))
        if mode_options:
            has_required_parameters = True
        # 「无合法目标」规则：仅法术牌(spell)在打出且必选目标候选为空时禁止打出；
        # 随从(follower)/护符(amulet)的所有操作(play/evolve/activation)均不阻塞——
        # 无目标时仍可正常打出/进化/启动，其目标性效果因无目标而空转，非目标效果
        # (如抽牌、召唤)照常发动。启动/进化(activation/evolve)对任何卡都不阻塞。
        if has_blocked_required_target and mode == "play" and card.card_type.is_spell:
            return []
        if not has_required_parameters:
            return None
        # 生成最终 target_groups：
        # - 如果只有 1 个 spec 需要选目标：每个候选生成 1 个 action，格式 ((uid,),)
        # - 如果有多个 spec 各自需要选目标：笛卡尔积生成组合 ((spec1_uid,), (spec2_uid,), ...)
        # 第三个元素是 target_group_specs：与 target_groups 等长的 spec_id 序列。
        spec_ids = tuple(per_spec_ids)
        if per_spec_candidates:
            import itertools
            if len(per_spec_candidates) == 1:
                # 单步选择：保持原格式 target_groups = ((uid,),)
                effect_params = [((candidates,), (), spec_ids) for candidates in per_spec_candidates[0]]
            else:
                # 多步选择：笛卡尔积
                combos = list(itertools.product(*per_spec_candidates))
                effect_params = [(combo, (), spec_ids) for combo in combos]
        else:
            effect_params = []
        if mode_options:
            # mode 选择没有 target_groups（选的是能力索引，不是实体），
            # 因此 target_group_specs 为空。
            effect_params.extend((groups, indexes, ()) for groups, indexes in mode_options)
        if effect_params:
            return list(dict.fromkeys(effect_params))
        return [((), (), ())]

    def _expand_mode_option_indexes(self, snapshot: BattleSnapshot, card: CardView, indexes: tuple[int, ...]) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
        if not indexes:
            return []
        select_count = self._mode_select_count(snapshot, card)
        if select_count <= 1:
            return [((), (index,)) for index in indexes]
        if len(indexes) < select_count:
            # spec 声明的选择数大于模板候选数：保留单选动作，让预测侧显式报
            # count mismatch 暴露数据缺口，不在这里静默扩选或丢弃合法动作。
            return [((), (index,)) for index in indexes]
        return [((), tuple(group)) for group in combinations(indexes, select_count)]

    def _mode_select_count(self, snapshot: BattleSnapshot, card: CardView) -> int:
        legal = snapshot.legal
        if legal is not None:
            info = legal.mode_skill_map.get(card.uid)
            if info is not None and info.select_count > 0:
                return info.select_count
        # 本地自对弈 / self-play 没有协议 mode 信息：从 EffectSpec 的
        # mode_choice_effect.selection_count 推导。否则「选择 N 个能力」的卡会被
        # 枚举成单选动作，预测时必然 selected_mode_options count mismatch。
        return self._spec_mode_select_count(card)

    @staticmethod
    def _spec_mode_select_count(card: CardView) -> int:
        bundle = _effect_spec_bundles().get(card.base_card_id)
        if bundle is None:
            return 1
        counts = [
            int(operation.params.get("selection_count") or 0)
            for spec in bundle.specs
            for operation in spec.operations
            if operation.op == "mode_choice_effect"
            and operation.params.get("ability_resolution_mode") != "all_abilities"
        ]
        counts = [count for count in counts if count > 0]
        return max(counts) if counts else 1

    def _mode_option_index(self, operation: EffectSpecOperation) -> int | None:
        value = operation.params.get("mode_option")
        if not isinstance(value, dict) or "option_index" not in value:
            return None
        try:
            return int(value["option_index"])
        except (TypeError, ValueError):
            return None

    def _synthesize_target_from_operations(self, spec: EffectSpec, action: LegalAction | None) -> EffectSpecTarget | None:
        """从 operations 的 target_ref 合成一个 EffectSpecTarget（当 spec.target 为 None 时）。

        某些 evolve/super_evolve spec 将目标选择编码在 operation.target_ref 中
        （如 "selected_enemy_follower"），而非 spec 顶层的 target 字段。
        """
        selected_kinds = {
            "selected_enemy_follower", "selected_self_follower", "selected_any_follower",
            "selected_target", "selected_any_card", "selected_self_card",
            "selected_self_hand_card", "selected_opponent_hand_card",
            "selected_filtered_cards", "selected_enemy_follower_or_leader",
            "selected_enemy_card",
        }
        for operation in spec.operations:
            if not self._operation_mode_selected_for_spec(operation, action):
                continue
            ref = operation.target_ref or ""
            if ref in selected_kinds:
                # 根据 kind 推断 zone 和 card_type
                zone = "hand" if "hand" in ref else "field"
                card_type = "follower" if "follower" in ref else None
                side = "enemy" if "enemy" in ref else ("self" if "self" in ref else None)
                # 合成的 implicit 选择一定是「要选 1 个」：显式写 count=1，
                # 与 spec 顶层 target 的 count=None（内部引用、不产生用户选择）区分开。
                return EffectSpecTarget(kind=ref, count=1, zone=zone, card_type=card_type, side=side, scope="selected")
        return None

    @staticmethod
    def _operation_mode_selected_for_spec(operation: EffectSpecOperation, action: LegalAction | None) -> bool:
        """检查 operation 在当前 action 的 mode_option 下是否应被选中。"""
        mode_option = operation.params.get("mode_option")
        if not isinstance(mode_option, dict) or mode_option.get("option_index") is None:
            return True
        if action is None:
            return True
        try:
            return int(mode_option["option_index"]) in action.selected_indexes
        except (TypeError, ValueError):
            return False

    def _expand_effect_target(self, snapshot: BattleSnapshot, card: CardView, target: EffectSpecTarget) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] | None:
        if target.kind == "mode_choice":
            count = target.count or 0
            if count <= 0:
                return []
            return [((), (index,)) for index in range(count)]
        if target.kind in {"selected_enemy_follower", "selected_self_follower", "selected_any_follower", "selected_target", "selected_any_card", "selected_self_card", "selected_self_hand_card", "selected_opponent_hand_card", "selected_filtered_cards", "selected_enemy_follower_or_leader", "selected_enemy_card", "selected_self_unevolved_other_follower"}:
            if target.count is None:
                # count=None 表示该 spec 借 selected_* kind 指向一个已知对象（内部引用），
                # 由 operation.target_ref / params 自行解析，不产生用户选择。
                # 例：「【超进化时】使自己的『纹章：X』的倒计数+N」的目标是纹章区
                # 指定 card_id，不是玩家要选的卡。若按默认 1 处理，会按 _target_pool
                # 的兜底池枚举出非法目标（实测 11024120 / 11064110 会多出
                # 「选择对方随从」的 5 个动作）。
                # 显式合成路径（_synthesize_target_from_operations）已写 count=1，
                # 因此这里不会误伤 evolve/super_evolve 的隐式目标选择。
                return [((), ())]
            pools = self._target_pool(snapshot, card, target)
            if not pools:
                return []
            select_count = target.count
            if select_count <= 1:
                return [(((uid,),), ()) for uid in pools]
            if len(pools) < select_count:
                # 目标不足时选全部（optional 或"最多N个"场景）
                return [((pools,), ())] if pools else []
            return [((combo,), ()) for combo in combinations(pools, select_count)]
        if target.kind in {"all_self_followers", "all_enemy_followers", "all_self_hand_cards", "all_enemy_hand_cards", "all_any_followers", "all_self_cards", "all_self_unevolved_followers", "all_self_shikigami_followers"}:
            pools = self._target_pool(snapshot, card, target)
            if not pools:
                return []
            return [((), ())]
        if target.kind == "all_enemy_leaders":
            return [((), ())]
        if target.kind in {"last_added_group", "last_summoned", "last_entered_self_follower", "source_card"}:
            # 运行时动态目标，不需要用户选择
            return [((), ())]
        return None

    def _target_pool(self, snapshot: BattleSnapshot, card: CardView, target: EffectSpecTarget) -> tuple[int, ...]:
        if target.kind in {"selected_self_card", "selected_self_follower", "selected_self_hand_card", "all_self_followers", "all_self_cards", "all_self_unevolved_followers", "all_self_shikigami_followers", "selected_self_unevolved_other_follower"}:
            pool = snapshot.me.hand if target.zone == "hand" else snapshot.me.field
        elif target.kind in {"selected_enemy_follower", "selected_enemy_follower_or_leader", "all_enemy_followers", "all_enemy_leaders"}:
            # 随从/主战者类目标一定在战场; zone=null 不应退化到 hand
            pool = snapshot.opponent.field
        elif target.kind in {"selected_enemy_card", "all_enemy_hand_cards"}:
            # "卡"泛指: 仅当明确 zone==hand (或 all_enemy_hand_cards) 时取手牌, 否则场上
            pool = snapshot.opponent.hand if (target.zone == "hand" or target.kind == "all_enemy_hand_cards") else snapshot.opponent.field
        elif target.kind in {"selected_any_follower", "all_any_followers"}:
            pool = (*snapshot.me.field, *snapshot.opponent.field)
        elif target.kind in {"selected_any_card", "all_any_cards"}:
            # 双方战场上的任意卡牌 (随从 + 护符); zone==hand 时取双方手牌
            pool = (*snapshot.me.hand, *snapshot.opponent.hand) if target.zone == "hand" else (*snapshot.me.field, *snapshot.opponent.field)
        elif target.kind == "selected_filtered_cards":
            # 根据 side 和 zone 确定搜索范围
            if target.side == "self":
                pool = snapshot.me.hand if target.zone == "hand" else snapshot.me.field
            elif target.side == "enemy":
                pool = snapshot.opponent.hand if target.zone == "hand" else snapshot.opponent.field
            else:
                # side=null: 搜索双方场面/手牌
                pool = (*snapshot.me.field, *snapshot.opponent.field) if target.zone != "hand" else (*snapshot.me.hand, *snapshot.opponent.hand)
        elif target.kind == "selected_target":
            pool = target.selector_refs and snapshot.me.field or snapshot.opponent.field
        else:
            pool = snapshot.me.field
        result: list[int] = []
        for uid in pool:
            card_view = snapshot.cards_by_uid.get(uid)
            if card_view is None:
                continue
            if target.card_type is not None and not card_view.card_type.matches(target.card_type):
                # 与 target_selector_materializer 共用 CardType.matches 的语义。
                # 原来只处理 "follower"，其余取值（"crest" 等）被静默忽略，
                # 会把不匹配的卡放进候选（见 docs/plan/ENTITY_TRANSFORMER_DESIGN.md §3.4）。
                continue
            # relation: other_than_source → 排除 source card 自身
            if target.relation == "other_than_source" and uid == card.uid:
                continue
            # 灵气（cant_select）与潜伏（ambush）：对方能力无法选择该随从。
            # 潜伏只阻止「被指定」，不阻止被随机效果命中，也不阻止其攻击（攻击后
            # 自动失去潜伏，见 CoreRuleKernel），因此这里与 cant_select 同口径排除。
            # 不一致会让「选择对手 1 个随从」在对手只剩潜伏随从时仍给出非法目标
            # （replay_...1219490 st36：11002110 进化时对潜伏中的 0/2 打了 2 点，录像没有伤害）。
            if (card_view.has_cant_select or card_view.has_ambush) and target.kind in {"selected_enemy_follower", "selected_enemy_follower_or_leader", "selected_enemy_card", "selected_any_follower", "selected_target", "selected_any_card", "selected_filtered_cards"}:
                continue
            if target.kind == "selected_self_hand_card" and uid == card.uid:
                continue
            if target.kind == "all_self_unevolved_followers" and card_view.evolve_state != 0:
                continue
            # selected_self_unevolved_other_follower: 未进化 + 排除源随从自身
            # (欧丝【进化时】选择自己战场上进化前的其他随从使其进化)
            if target.kind == "selected_self_unevolved_other_follower" and (card_view.evolve_state != 0 or uid == card.uid or not card_view.card_type.is_follower):
                continue
            if target.kind == "all_self_shikigami_followers" and not self._is_shikigami(card_view):
                continue
            if not self._matches_effect_target_filters(card_view, target.filters):
                continue
            result.append(uid)
        if target.kind == "all_enemy_leaders":
            return (0,)
        # selected_enemy_follower_or_leader: 在随从候选之外追加对手主战者。
        # 对手主战者统一用哨兵 uid=0 表示 (与 all_enemy_leaders、materializer
        # selected_enemy_follower_or_leader 的 (0,) 分支、local_predictor uid==0
        # 判定一致); 用真实 unique_id 会导致 materializer 判定 "target must be
        # on opponent field" (主战者不在 field 集合内)。
        if target.kind == "selected_enemy_follower_or_leader":
            if 0 not in result:
                result.append(0)
        return tuple(result)

    def _matches_effect_target_filters(self, card: CardView, filters: tuple[dict[str, object], ...]) -> bool:
        for item in filters:
            field = item.get("field")
            operator = str(item.get("operator", "==") or "==")
            value = item.get("value")
            # exclude filter 由 _target_pool 通过 relation 处理，这里跳过
            if item.get("exclude"):
                continue
            # uid != source_uid: 排除 source card 自身
            if field == "uid" and operator == "!=" and value == "source_uid":
                continue  # 由 _target_pool 的 source_uid 排除逻辑处理
            if field == "tribes" and operator == "contains":
                if int(value) not in (card.tribes or ()):
                    return False
                continue
            if operator != "==":
                return False
            if field == "base_card_id":
                if card.base_card_id != int(value):
                    return False
            elif field == "card_type":
                if card.card_type.value != value:
                    return False
            elif field in {"cost", "original_cost"}:
                if card.cost != int(value):
                    return False
            elif field == "atk":
                if card.atk != int(value):
                    return False
            elif field == "life":
                if card.life != int(value):
                    return False
            elif field in {"side", "zone"}:
                continue
            elif field == "has_spell_boost":
                if bool(card.has_spell_boost or card.variable_x > 0) != bool(value):
                    return False
            else:
                return False
        return True

    def _expand_mode_skill(self, mode_info: ModeSkillInfo) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
        if mode_info.mode_count <= 0 or mode_info.select_count <= 0:
            return []
        candidates = mode_info.candidate_indexes or tuple(range(mode_info.mode_count))
        if mode_info.select_count >= len(candidates):
            return [((), tuple(candidates))]
        return [((), combo) for combo in combinations(candidates, mode_info.select_count)]

    def _expand_choice_skill(self, choice_info: ChoiceSkillInfo) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
        if choice_info.choice_count <= 0 or not choice_info.choice_id_list:
            return []
        count = len(choice_info.choice_id_list)
        if choice_info.choice_count >= count:
            return [((), tuple(range(count)))]
        return [((), combo) for combo in combinations(range(count), choice_info.choice_count)]

    def _expand_select_skill_options(self, options: tuple[SelectSkillOption, ...]) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
        per_option_choices: list[list[tuple[int, ...]]] = []
        selected_indexes = tuple(index for option in options for index in option.selected_indexes)
        for option in options:
            forced = option.force_target_uids
            remaining_targets = tuple(uid for uid in option.target_uids if uid not in forced)
            remaining_count = max(option.select_count - len(forced), 0)
            if option.select_count > 0 and len(remaining_targets) > remaining_count:
                per_option_choices.append([tuple((*forced, *combo)) for combo in combinations(remaining_targets, remaining_count)])
            else:
                per_option_choices.append([tuple((*forced, *remaining_targets))])
        return [(tuple(combo), selected_indexes) for combo in product(*per_option_choices)]

    @staticmethod
    def _fusion_used_this_turn(card: CardView) -> bool:
        return bool(card.supplement_info.get("fusion_used_this_turn"))

    @staticmethod
    def _can_pay_unconditional_fusion_cost(snapshot: BattleSnapshot, card: CardView) -> bool:
        """检查本地可推导的 fusion 无条件 PP 支付。

        只累计无条件、固定数额 ``pay_pp``：条件/变量费用仍由严格预测器处理，
        避免枚举器自行猜测复杂规则；这里仅排除已知必定非法的动作。
        """

        bundle = _effect_spec_bundles().get(card.base_card_id)
        if bundle is None:
            return True
        required = 0
        for effect in bundle.specs:
            if effect.trigger.event != "fusion" or effect.conditions:
                continue
            for operation in effect.operations:
                amount = operation.amount
                if operation.op == "pay_pp" and amount is not None and amount.kind == "fixed" and amount.value is not None:
                    required += int(amount.value)
        return snapshot.me.pp >= required

    def _template_fusion_materials(self, snapshot: BattleSnapshot, source: CardView, spec) -> tuple[tuple[int, ...], ...]:
        candidates = []
        for uid in snapshot.me.hand:
            if uid == source.uid:
                continue
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            if spec.allowed_class_ids and card.class_id not in spec.allowed_class_ids:
                continue
            if spec.allowed_tribes and not set(spec.allowed_tribes).intersection(card.tribes):
                continue
            candidates.append(uid)
        if len(candidates) < spec.min_material_count:
            return ()
        max_count = spec.max_material_count or len(candidates)
        max_count = min(max_count, len(candidates))
        return tuple(
            group
            for count in range(spec.min_material_count, max_count + 1)
            for group in combinations(candidates, count)
        )

    def _fusion_material_groups(self, option: SelectFusionOption, hand_uids: frozenset[int], source_uid: int) -> tuple[tuple[int, ...], ...]:
        if option.material_groups:
            return tuple(
                tuple(group)
                for group in option.material_groups
                if all(material_uid in hand_uids and material_uid != source_uid for material_uid in group)
            )
        material_uids = tuple(uid for uid in option.material_uids if uid in hand_uids and uid != source_uid)
        if option.select_material_count is None:
            return (material_uids,) if material_uids else ()
        if option.select_material_count <= 0:
            return ((),)
        return tuple(tuple(group) for group in combinations(material_uids, option.select_material_count))

    def _protocol_fusion_groups(self, option) -> tuple[tuple[int, ...], ...]:
        if option.material_groups:
            return tuple(tuple(group) for group in option.material_groups)
        if option.select_material_count is None:
            return (tuple(option.material_uids),) if option.material_uids else ((),)
        if option.select_material_count <= 0:
            return ((),)
        return tuple(tuple(group) for group in combinations(option.material_uids, option.select_material_count))

    def _enumerate_extra_pp(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        extra_pp = snapshot.me.extra_pp
        if extra_pp is None or extra_pp.extra_pp <= 0 or extra_pp.is_used_extra_pp_this_turn:
            return []
        if any(self._would_unlock_with_extra_pp(snapshot, uid) for uid in snapshot.me.hand):
            return [self._action(kind=ActionKind.EXTRA_PP, source_uid=0)]
        return []

    def _would_unlock_with_extra_pp(self, snapshot: BattleSnapshot, uid: int) -> bool:
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            return False
        pp_after_extra = snapshot.me.pp + 1
        field_full = len(snapshot.me.field) >= MAX_FIELD_SIZE
        if card.cost == pp_after_extra and not (card.card_type is not CardType.SPELL and field_full):
            return True
        for cost in card.enhance_costs:
            if card.can_enhance and cost == pp_after_extra and not (card.card_type is not CardType.SPELL and field_full):
                return True
        for cost in card.accelerate_costs:
            if cost == pp_after_extra:
                return True
        for cost in card.crystal_costs:
            if cost == pp_after_extra and not field_full:
                return True
        return False
