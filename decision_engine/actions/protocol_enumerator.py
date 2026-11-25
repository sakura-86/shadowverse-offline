# -*- coding: utf-8 -*-
"""协议合法动作枚举器。

该枚举器只消费 BattleSnapshot.legal / LegalView 中已经由协议层给出的合法性，
不在本地复刻游戏规则，也不解析卡牌效果。
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations, product

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.state.snapshot import BattleSnapshot, CardView, ChoiceSkillInfo, LegalView, ModeSkillInfo, SelectFusionOption


class ProtocolActionEnumerator:
    """从协议 LegalView 枚举精确合法动作。

    主路径只消费协议层给出的合法性。但当协议 LegalView 未提供某张卡的 play
    目标选择映射（select_skill_map/mode_skill_map/choice_skill_map 均缺失），
    而该卡的 effect spec 明确要求必选目标（如「入场曲选择自己 1 张手牌返回牌组」，
    optional=false）时，若直接 fallback 到无目标打出会产出非法动作。此时借助本地
    effect-spec 目标推导补出必选目标（有候选则强制选择，候选为空则不枚举该 play），
    保证「有手牌可选时强制选择一张手牌」。
    """

    def __init__(self, *, target_inference: "LocalActionEnumerator | None" = None):
        # 延迟导入避免循环依赖；仅用于 play 必选目标的推导，不复刻规则合法性。
        if target_inference is None:
            from decision_engine.actions.local_enumerator import LocalActionEnumerator
            target_inference = LocalActionEnumerator()
        self._target_inference = target_inference

    def enumerate(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        """枚举当前协议视图给出的合法动作。"""
        legal = snapshot.legal
        if legal is None:
            return []
        if not snapshot.is_ally_turn or snapshot.is_turn_ended:
            return []

        actions: list[LegalAction] = []
        hand_uids = frozenset(snapshot.me.hand)
        field_uids = frozenset(snapshot.me.field)
        actions.extend(self._enumerate_attack_leader(snapshot, legal.can_attack_leader & field_uids, legal.attack_targets))
        actions.extend(self._enumerate_attack_followers(legal.can_attack_field & field_uids, legal.attack_targets, snapshot))
        actions.extend(self._enumerate_plays(snapshot, legal, legal.can_play & hand_uids, PlayKind.NORMAL))
        actions.extend(self._enumerate_plays(snapshot, legal, legal.can_enhance & hand_uids, PlayKind.ENHANCE))
        actions.extend(self._enumerate_plays(snapshot, legal, legal.can_accelerate & hand_uids, PlayKind.ACCELERATE))
        actions.extend(self._enumerate_plays(snapshot, legal, legal.can_crystal & hand_uids, PlayKind.CRYSTAL))
        actions.extend(self._enumerate_simple(legal, legal.can_evolve & field_uids, ActionKind.EVOLVE))
        actions.extend(self._enumerate_simple(legal, legal.can_super_evolve & field_uids, ActionKind.SUPER_EVOLVE))
        actions.extend(self._enumerate_simple(legal, legal.can_activate & field_uids, ActionKind.ACTIVATE))
        actions.extend(self._enumerate_special_action(legal, field_uids))
        actions.extend(self._enumerate_extra_pp(legal, hand_uids, field_uids))
        actions.extend(self._enumerate_fusion(legal, hand_uids))
        actions.extend(self._enumerate_end_turn(legal))
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
            legality_source="protocol",
            confidence="exact",
        )

    def _enumerate_attack_leader(
        self,
        snapshot: BattleSnapshot,
        source_uids: Iterable[int],
        attack_targets: dict[int, frozenset[int]],
    ) -> list[LegalAction]:
        actions: list[LegalAction] = []
        opponent_uid = snapshot.opponent.unique_id
        if not opponent_uid:
            return actions
        for source_uid in sorted(source_uids):
            if opponent_uid not in attack_targets.get(source_uid, frozenset()):
                continue
            actions.append(self._action(
                kind=ActionKind.ATTACK_LEADER,
                source_uid=source_uid,
                target_groups=((opponent_uid,),),
            ))
        return actions

    def _enumerate_attack_followers(
        self,
        source_uids: Iterable[int],
        attack_targets: dict[int, frozenset[int]],
        snapshot: BattleSnapshot,
    ) -> list[LegalAction]:
        """枚举合法的攻击随从动作。

        协议 attack_targets_map 有时会包含 leader uid 作为 target（当对手场上
        没有可攻击的随从时），这里需要过滤掉不在对手 field 中的 target。
        """
        actions: list[LegalAction] = []
        # 有效 attack_follower 目标：必须在对手场上（排除 leader uid 等）
        valid_targets = frozenset(snapshot.opponent.field)
        for source_uid in sorted(source_uids):
            for target_uid in sorted(attack_targets.get(source_uid, frozenset())):
                if target_uid not in valid_targets:
                    continue
                actions.append(self._action(
                    kind=ActionKind.ATTACK_FOLLOWER,
                    source_uid=source_uid,
                    target_groups=((target_uid,),),
                ))
        return actions

    def _enumerate_plays(
        self,
        snapshot: BattleSnapshot,
        legal: LegalView,
        source_uids: Iterable[int],
        play_kind: PlayKind,
        variant: str | None = None,
    ) -> list[LegalAction]:
        actions: list[LegalAction] = []
        for source_uid in sorted(source_uids):
            params = self._select_skill_parameters(legal, "play", source_uid)
            # 协议未提供目标选择（唯一项为无目标 [((), ())]）时，尝试从 effect spec
            # 推导必选目标，避免对「入场曲需选手牌」等卡产出非法的无目标打出。
            if params == [((), ())]:
                inferred = self._infer_play_target_params(snapshot, source_uid)
                if inferred is not None:
                    params = inferred
            for target_groups, selected_indexes in params:
                actions.append(self._action(
                    kind=ActionKind.PLAY,
                    source_uid=source_uid,
                    play_kind=play_kind,
                    cost=self._play_cost(snapshot.cards_by_uid.get(source_uid), play_kind),
                    target_groups=target_groups,
                    selected_indexes=selected_indexes,
                    variant=variant,
                ))
        return actions

    def _infer_play_target_params(
        self,
        snapshot: BattleSnapshot,
        source_uid: int,
    ) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] | None:
        """借助本地 effect-spec 推导 play 的必选目标参数。

        返回值语义与 LocalActionEnumerator._effect_spec_parameters 一致：
        - None：该卡无必选目标（保持协议 fallback 的无目标打出）；
        - []：必须选目标但候选池为空 → 不枚举该 play（调用方得到空列表）；
        - 非空列表：每项为一个带目标的 (target_groups, selected_indexes)。
        """
        card = snapshot.cards_by_uid.get(source_uid)
        if card is None:
            return None
        return self._target_inference._effect_spec_parameters(snapshot, card, "play")

    def _enumerate_simple(self, legal: LegalView, source_uids: Iterable[int], kind: ActionKind, variant: str | None = None) -> list[LegalAction]:
        mode = {
            ActionKind.EVOLVE: "evolve",
            ActionKind.SUPER_EVOLVE: "super_evolve",
            ActionKind.ACTIVATE: "activation",
        }.get(kind, "")
        actions: list[LegalAction] = []
        for source_uid in sorted(source_uids):
            for target_groups, selected_indexes in self._select_skill_parameters(legal, mode, source_uid):
                actions.append(self._action(
                    kind=kind,
                    source_uid=source_uid,
                    target_groups=target_groups,
                    selected_indexes=selected_indexes,
                    variant=variant,
                ))
        return actions

    def _enumerate_special_action(self, legal: LegalView, field_uids: frozenset[int]) -> list[LegalAction]:
        actions: list[LegalAction] = []
        for source_uid in sorted(legal.can_special_action & field_uids):
            for target_groups, selected_indexes in self._select_skill_parameters(legal, "special_action", source_uid):
                actions.append(self._action(
                    kind=ActionKind.SPECIAL_ACTION,
                    source_uid=source_uid,
                    target_groups=target_groups,
                    selected_indexes=selected_indexes,
                ))
        return actions

    def _enumerate_fusion(self, legal: LegalView, hand_uids: frozenset[int]) -> list[LegalAction]:
        actions = []
        for source_uid in sorted(legal.can_fusion & hand_uids):
            option = legal.select_fusion_map.get(source_uid)
            if option is None:
                continue
            for material_uids in self._fusion_material_groups(option):
                if material_uids:
                    actions.append(self._action(kind=ActionKind.FUSION, source_uid=source_uid, material_uids=material_uids))
        return actions

    def _enumerate_end_turn(self, legal: LegalView) -> list[LegalAction]:
        if not legal.can_end_turn:
            return []
        return [self._action(kind=ActionKind.TURN_END, source_uid=0)]

    def _enumerate_extra_pp(self, legal: LegalView, hand_uids: frozenset[int], field_uids: frozenset[int]) -> list[LegalAction]:
        has_ex_pp_only = (
            ((legal.can_play_with_ex_pp & hand_uids) - legal.can_play)
            | ((legal.can_activate_with_ex_pp & field_uids) - legal.can_activate)
        )
        if not has_ex_pp_only:
            return []
        return [self._action(kind=ActionKind.EXTRA_PP, source_uid=0)]

    def _select_skill_parameters(
        self,
        legal: LegalView,
        mode: str,
        source_uid: int,
    ) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
        options = legal.select_skill_map.get(mode, {}).get(source_uid, ())
        if options:
            return self._expand_select_skill_options(options)

        mode_map = (
            legal.super_evolve_mode_skill_map if mode == "super_evolve"
            else legal.mode_skill_map
        )
        mode_info = mode_map.get(source_uid)
        if mode_info and mode_info.mode_count > 0 and mode_info.select_count > 0:
            candidates = mode_info.candidate_indexes or tuple(range(mode_info.mode_count))
            if mode_info.select_count >= len(candidates):
                return [((), tuple(candidates))]
            return [((), combo) for combo in combinations(candidates, mode_info.select_count)]

        choice_info = legal.choice_skill_map.get(mode, {}).get(source_uid)
        if choice_info and choice_info.choice_count > 0 and choice_info.choice_id_list:
            n = len(choice_info.choice_id_list)
            if choice_info.choice_count >= n:
                return [((), tuple(range(n)))]
            return [((), combo) for combo in combinations(range(n), choice_info.choice_count)]

        return [((), ())]

    def _expand_select_skill_options(
        self,
        options: tuple[SelectSkillOption, ...],
    ) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
        """展开 SelectSkillOption 序列为目标组合。

        当某个 option 有 select_count > 0 且候选目标多于 select_count 时，
        按组合枚举每种选择；否则直接使用全部候选作为已选。
        """
        needs_expansion = any(
            option.select_count > 0 and len(option.target_uids) > option.select_count
            for option in options
        )
        if not needs_expansion:
            target_groups = tuple(option.target_uids for option in options)
            selected_indexes = tuple(index for option in options for index in option.selected_indexes)
            return [(target_groups, selected_indexes)]

        # 对每个 option 产出候选列表，然后对所有 option 做笛卡尔积
        per_option_choices: list[list[tuple[int, ...]]] = []
        for option in options:
            if option.select_count > 0 and len(option.target_uids) > option.select_count:
                per_option_choices.append(
                    [combo for combo in combinations(option.target_uids, option.select_count)]
                )
            else:
                per_option_choices.append([option.target_uids])

        results: list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = []
        for combo in product(*per_option_choices):
            target_groups = tuple(combo)
            selected_indexes = tuple(index for option in options for index in option.selected_indexes)
            results.append((target_groups, selected_indexes))
        return results

    def _fusion_material_groups(self, option: SelectFusionOption | None) -> tuple[tuple[int, ...], ...]:
        if option is None:
            return ((),)
        if option.material_groups:
            return tuple(tuple(group) for group in option.material_groups)
        if option.select_material_count is None:
            return (tuple(option.material_uids),) if option.material_uids else ((),)
        if option.select_material_count <= 0:
            return ((),)
        return tuple(tuple(group) for group in combinations(option.material_uids, option.select_material_count))

    def _play_cost(self, card: CardView | None, play_kind: PlayKind) -> int | None:
        if card is None:
            return None
        if play_kind is PlayKind.NORMAL:
            return card.cost
        if play_kind is PlayKind.ENHANCE:
            return max(card.enhance_costs) if card.enhance_costs else None
        if play_kind is PlayKind.ACCELERATE:
            return max(card.accelerate_costs) if card.accelerate_costs else None
        if play_kind is PlayKind.CRYSTAL:
            return max(card.crystal_costs) if card.crystal_costs else None
        return None
