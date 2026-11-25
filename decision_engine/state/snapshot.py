# -*- coding: utf-8 -*-
"""
新架构状态视图。

本模块只描述可观察局面与协议级状态，不承载卡牌效果 DSL。
卡牌效果应由 cards/CardResolver 与 simulation/effect_predictor 提供。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any

from decision_engine.state.deck_instance import DeckInstanceState
from decision_engine.state.hand_knowledge import HandKnowledgeState


class CardType(Enum):
    """卡牌类型，对齐 BaseCardMaster.Type 1~7。"""

    FOLLOWER = "follower"                    # type_id=1: 随从
    AMULET = "amulet"                        # type_id=2: 护符（无倒计时）
    COUNTDOWN_AMULET = "countdown_amulet"    # type_id=3: 倒计时护符
    SPELL = "spell"                          # type_id=4: 法术
    CREST = "crest"                          # type_id=5: 纹章
    COUNTDOWN_CREST = "countdown_crest"      # type_id=6: 倒计时纹章
    FAITH = "faith"                          # type_id=7: 信仰

    # ── 行为谓词 ──

    @property
    def is_follower(self) -> bool:
        """能上场战斗的单位。"""
        return self is CardType.FOLLOWER

    @property
    def is_spell(self) -> bool:
        """从手牌使用后进墓地、不进场。"""
        return self is CardType.SPELL

    @property
    def is_amulet_like(self) -> bool:
        """场上的护符类（占场地位、不能攻击）。"""
        return self in _AMULET_LIKE

    @property
    def is_field_card(self) -> bool:
        """能从手牌打出后进入 field 的类型。"""
        return self in _FIELD_CARDS

    @property
    def is_crest_like(self) -> bool:
        """存在于 crest zone 的类型（不进 field）。"""
        return self in _CREST_LIKE

    @property
    def has_countdown(self) -> bool:
        """具有倒计时行为（回合开始 -1）。"""
        return self in _HAS_COUNTDOWN

    def matches(self, name: str | None) -> bool:
        """按 spec / selector 的 ``card_type`` 字符串匹配本类型。

        取值约定与 ``TargetSelector.card_type`` 一致：

        - ``card`` / ``any`` / 空值：不过滤（任意卡牌类型）
        - ``follower`` / ``spell``：走行为谓词
        - ``amulet`` / ``countdown_amulet``：护符类
        - ``crest`` / ``countdown_crest`` / ``faith``：纹章区类型（不占场地位）
        - ``boon``：兼容护符与纹章两侧

        未识别的取值返回 ``False``（不匹配任何类型），调用方会因此得到空候选池，
        而不是把不匹配的卡放进候选。
        """
        normalized = (name or "").strip().lower()
        if normalized in {"", "card", "any"}:
            return True
        if normalized == self.value:
            return True
        if normalized == "follower":
            return self.is_follower
        if normalized == "spell":
            return self.is_spell
        if normalized in {"amulet", "countdown_amulet"}:
            return self.is_amulet_like
        if normalized in {"crest", "countdown_crest", "faith"}:
            return self.is_crest_like
        if normalized == "boon":
            return self.is_amulet_like or self.is_crest_like
        return False


# 行为分组集合（枚举定义完成后初始化）
_AMULET_LIKE: frozenset["CardType"] = frozenset()
_FIELD_CARDS: frozenset["CardType"] = frozenset()
_CREST_LIKE: frozenset["CardType"] = frozenset()
_HAS_COUNTDOWN: frozenset["CardType"] = frozenset()


def _init_card_type_sets() -> None:
    global _AMULET_LIKE, _FIELD_CARDS, _CREST_LIKE, _HAS_COUNTDOWN
    _AMULET_LIKE = frozenset({CardType.AMULET, CardType.COUNTDOWN_AMULET})
    _FIELD_CARDS = frozenset({CardType.FOLLOWER, CardType.AMULET, CardType.COUNTDOWN_AMULET})
    _CREST_LIKE = frozenset({CardType.CREST, CardType.COUNTDOWN_CREST, CardType.FAITH})
    _HAS_COUNTDOWN = frozenset({CardType.COUNTDOWN_AMULET, CardType.COUNTDOWN_CREST})


_init_card_type_sets()


@dataclass(frozen=True)
class PlayerBuffView:
    """主战者 buff 协议事实。"""

    max_life: int = 0
    temp_shield: bool = False
    damage_cut: int = 0
    increase_damage: int = 0
    raw: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class CardBuffView:
    """卡牌 buff 协议事实。"""

    cost: int = 0
    atk: int = 0
    max_life_diff: int = 0
    quick: bool = False
    rush: bool = False
    temp_shield: bool = False
    attack_limit: int = 0
    add_dead_by_reanimate: bool = False
    add_tribe: tuple[int, ...] = ()
    raw: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class SelectSkillOption:
    """协议选择技能的目标候选。

    只承载 BattleRootMpo.SelectSkillMap 暴露的动作参数候选，
    不解释目标背后的卡牌效果。
    """

    select_count: int = 0
    target_uids: tuple[int, ...] = ()
    force_target_uids: tuple[int, ...] = ()
    selected_indexes: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModeSkillInfo:
    """协议 mode skill 选择规格。

    mode_count 个选项中选 select_count 个；candidate_indexes 非空时限定可选范围。
    """

    mode_group: int = 0
    mode_count: int = 0
    select_count: int = 0
    candidate_indexes: tuple[int, ...] = ()


@dataclass(frozen=True)
class ChoiceSkillInfo:
    """协议 choice skill 选择规格。"""

    choice_count: int = 0
    choice_id_list: tuple[int, ...] = ()
    choice_type: int = 0


@dataclass(frozen=True)
class SelectFusionOption:
    """协议融合材料候选。

    material_groups 是 mock/fixture 可直接给出的保守材料组；
    material_uids + select_material_count 用于按数量组合材料 uid。
    """

    select_material_count: int | None = None
    material_uids: tuple[int, ...] = ()
    material_groups: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class CardView:
    """手牌或场上卡牌的协议风格视图。"""

    uid: int
    card_id: int = 0
    base_card_id: int = 0
    name: str = ""
    class_id: int = 0
    cost: int = 0
    atk: int = 0
    life: int = 0
    max_life: int = 0
    card_type: CardType = CardType.FOLLOWER

    # 协议显式下发的关键字字段名（如 ("has_guard",)）。用于区分「协议明确下发 False」
    # 与「协议没下发」：前者是失去关键字的明确状态，后者才允许用模板静态关键字补全。
    protocol_keyword_fields: tuple[str, ...] = ()
    has_guard: bool = False
    has_storm: bool = False
    has_rush: bool = False
    has_drain: bool = False
    has_bane: bool = False
    has_ambush: bool = False
    has_cant_attack: bool = False
    has_cant_be_attacked: bool = False
    has_cant_select: bool = False
    has_last_words: bool = False
    has_activation: bool = False
    activation_used_this_turn: bool = False

    is_direct_damage: bool = False
    is_buff: bool = False
    buff_targets: str = ""

    can_attack: bool = False
    can_attack_leader: bool = False
    can_evolve: bool = False
    can_super_evolve: bool = False
    attacks_left: int = 0
    countdown: int = 0
    stack: int = 0
    evolve_state: int = 0

    enhance_costs: tuple[int, ...] = ()
    accelerate_costs: tuple[int, ...] = ()
    crystal_costs: tuple[int, ...] = ()
    fusion_list: tuple[tuple[int, ...], ...] = ()
    select_fusion: SelectFusionOption | None = None

    can_special_action: bool = False
    select_skill_map: dict[str, tuple[SelectSkillOption, ...]] = dataclass_field(default_factory=dict)
    mode_skill_map: dict[str, ModeSkillInfo] = dataclass_field(default_factory=dict)
    choice_skill_map: dict[str, ChoiceSkillInfo] = dataclass_field(default_factory=dict)

    tribes: tuple[int, ...] = ()
    buff: CardBuffView | None = None
    has_spell_boost: bool = False
    spellboost_reduces_cost: bool = False
    spell_boost_count: int = 0
    variable_x: int = 0
    style_id: int = 0
    supplement_info: dict[str, int] = dataclass_field(default_factory=dict)
    cant_action_type: int = 0
    activated_random_once_indexes: tuple[int, ...] = ()
    current_run_in_order_count: int = 0
    can_enhance: bool = False
    can_accelerate: bool = False
    is_changed_ability: bool = False
    attack_targets: tuple[int, ...] = ()
    is_earth_sigil: bool = False
    enhanced_cost: int = 0
    has_damage_cut: bool = False
    has_induction: bool = False
    has_reduce_damage: bool = False
    has_cant_destroy: bool = False
    has_super_evolve_buff: bool = False
    is_remove_field_at_turn_change: bool = False
    has_temp_shield: bool = False
    can_ignore_guard: bool = False
    attack_limit: int = 0
    is_same_name_token: bool = False
    content_unknown: int = 0


# ---- Phase 3 协议状态事实容器 ------------------------------------------------


@dataclass(frozen=True)
class CrestView:
    """纹章协议事实。不解释效果，只存协议字段。"""

    card_id: int = 0
    countdown: int = 0
    faith_value: int = 0
    variable_x: int = 0
    unique_id: int = 0
    style_id: int = 0
    supplement_info: dict[str, int] = dataclass_field(default_factory=dict)
    buff: dict[str, Any] = dataclass_field(default_factory=dict)
    is_show_skill_notice_effect: bool = False
    is_show_skill_notice_icon: bool = False
    activated_random_once_indexes: tuple[int, ...] = ()
    current_run_in_order_count: int = 0


@dataclass(frozen=True)
class ExtraCrestView:
    """额外纹章协议事实。"""

    card_id: int = 0
    unique_id: int = 0
    countdown: int = 0
    variable_x: int = 0
    style_id: int = 0
    supplement_info: dict[str, int] = dataclass_field(default_factory=dict)
    buff: dict[str, Any] = dataclass_field(default_factory=dict)
    is_show_skill_notice_effect: bool = False
    is_show_skill_notice_icon: bool = False
    activated_random_once_indexes: tuple[int, ...] = ()
    current_run_in_order_count: int = 0


@dataclass(frozen=True)
class BoonView:
    """恩惠协议事实。

    ``supplement_info`` 与其他卡牌视图一致：协议侧是
    ``Dictionary<string, int>``（见 ``BattleBoonMpo.SupplementInfo``）。
    """

    card_id: int = 0
    unique_id: int = 0
    supplement_info: dict[str, int] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class CardIdCount:
    """card_id + count 对，用于 played/destroyed 历史。"""

    card_id: int = 0
    count: int = 0


@dataclass(frozen=True)
class DeckPoolCard:
    """已知牌组池中的单张牌组条目。"""

    card_id: int = 0
    count: int = 0
    base_card_id: int = 0
    name: str = ""
    cost: int = 0
    card_type: CardType | None = None


@dataclass(frozen=True)
class DeckPoolView:
    """玩家已知牌组池的 hidden-knowledge 视图。"""

    source: str = ""
    cards: tuple[DeckPoolCard, ...] = ()
    observed_total: int = 0
    remaining_count: int = 0
    consumed_hidden_draws: int = 0
    status: str = "available"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtraPpView:
    """Extra PP 协议状态。"""

    extra_pp: int = 0
    preparation_extra_pp: int = 0
    extra_pp_state: int = 0
    restore_extra_pp_turn: int = 0
    is_used_extra_pp_this_turn: bool = False


@dataclass(frozen=True)
class PlayerView:
    """玩家状态视图。"""

    life: int
    max_life: int
    pp: int
    max_pp: int
    ep: int = 0
    sep: int = 0
    max_ep: int = 0
    max_sep: int = 0
    deck_count: int = 0
    result: int = 0
    is_deck_out_win: bool = False
    turn: int = 0
    is_first_side: bool = False
    is_kakusei: bool = False
    evolve_turn: int = 0
    super_evolve_turn: int = 0
    is_evolved_this_turn: bool = False
    manual_evolve_count: int = 0
    evolve_count: int = 0
    followers_entered_battlefield: int = 0
    hand: tuple[int, ...] = ()
    # 协议下发的**手牌张数**（含不可见的隐藏手牌）。replay 里对手手牌是
    # card_id=0 占位、不会进 hand，因此 len(hand) 可能小于真实张数。
    # 依赖「对手手牌内容」的条件据此判「不可观测」，而不是当成 0 张。
    hand_count: int | None = None
    field: tuple[int, ...] = ()
    # 卡牌登记顺序：本局内卡牌**打出/入场**（进入场地区域）或**纹章出现**的先后
    # （uid 序列，早→晚）。手牌里的卡不算登记——它还没有参与任何时点结算；
    # 离场后再次打出按新的出场时刻重新登记。
    # 同一时点触发多个裁定时，场地区域与纹章区域**合并**后按此顺序结算。
    # 不能由单帧推导：场地数组是卡位顺序（卡位会被复用）、纹章在独立数组里，
    # 两者之间没有可比较的键（unique_id 非单调分配，实测同一局首手牌 uid 为
    # 39/3/38/9）。因此顺序必须由跨 step 观察累积
    # （`advance_persistent_registration_order`）。
    persistent_registration_order: tuple[int, ...] = ()
    graveyard_count: int = 0
    rally: int = 0
    play_count: int = 0
    attack_count: int = 0
    unique_id: int = 0

    crests: tuple[CrestView, ...] = ()
    extra_crests: tuple[ExtraCrestView, ...] = ()
    boons: tuple[BoonView, ...] = ()
    extra_pp: ExtraPpView | None = None
    deck_pool: DeckPoolView | None = None
    deck_instances: DeckInstanceState | None = None
    played_card_ids: tuple[CardIdCount, ...] = ()
    destroyed_card_ids: tuple[CardIdCount, ...] = ()
    # 本局进入战场的随从 base_card_id 计数（通用历史，供"本局第N个入场的XX"
    # "本局其他同名随从入场数>=N"等条件查询；对称 played_card_ids）
    entered_field_card_ids: tuple[CardIdCount, ...] = ()
    # 本回合是否有己方随从攻击过对方主战者（攻击时置位，回合开始归档）。
    follower_attacked_leader_this_turn: bool = False
    # 上一个己方回合中是否有己方随从攻击过对方主战者（回合开始时由
    # follower_attacked_leader_this_turn 归档而来）。供龙职业"若自己的随从
    # 在自己的上一回合中攻击过主战者"类条件查询。
    follower_attacked_leader_previous_turn: bool = False
    deck_skill_card_ids: tuple[CardIdCount, ...] = ()
    battle_log_info_map: dict[int, Any] = dataclass_field(default_factory=dict)
    buff: PlayerBuffView | None = None
    # 本回合内持续效果的触发计数，用于 per_turn_limit 限制
    # key: effect_id, value: 本回合已触发次数
    effect_trigger_counts: dict[str, int] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class LegalView:
    """协议合法动作集合视图。

    第一轮重构中 puzzle fallback 可以留空；协议接入后由 BattleRootMpo.Can* 填充。
    """

    can_play: frozenset[int] = frozenset()
    can_attack_leader: frozenset[int] = frozenset()
    can_attack_field: frozenset[int] = frozenset()
    attack_targets: dict[int, frozenset[int]] = dataclass_field(default_factory=dict)
    can_evolve: frozenset[int] = frozenset()
    can_super_evolve: frozenset[int] = frozenset()
    can_activate: frozenset[int] = frozenset()
    can_fusion: frozenset[int] = frozenset()
    can_enhance: frozenset[int] = frozenset()
    can_accelerate: frozenset[int] = frozenset()
    can_crystal: frozenset[int] = frozenset()
    can_special_action: frozenset[int] = frozenset()
    can_end_turn: bool = False
    can_super_evolve_with_skill: frozenset[int] = frozenset()
    can_play_with_ex_pp: frozenset[int] = frozenset()
    can_activate_with_ex_pp: frozenset[int] = frozenset()
    attacked_cards: frozenset[int] = frozenset()
    has_activation: frozenset[int] = frozenset()
    has_fusion: frozenset[int] = frozenset()
    usage_ep_map: dict[int, int] = dataclass_field(default_factory=dict)
    select_skill_map: dict[str, dict[int, tuple[SelectSkillOption, ...]]] = dataclass_field(default_factory=dict)
    select_fusion_map: dict[int, SelectFusionOption] = dataclass_field(default_factory=dict)
    mode_skill_map: dict[int, ModeSkillInfo] = dataclass_field(default_factory=dict)
    super_evolve_mode_skill_map: dict[int, ModeSkillInfo] = dataclass_field(default_factory=dict)
    choice_skill_map: dict[str, dict[int, ChoiceSkillInfo]] = dataclass_field(default_factory=dict)


class ConditionState(str, Enum):
    """卡牌「打出效果条件」的达成状态。

    这是本地推导的状态事实，不是协议字段：协议只提供 SkillEffectYellowCards（已达成），
    没有提供「该卡是否绑定条件」。推导规则见 execution/condition_progress.py。
    """

    NONE = "none"                # 已确认：该卡打出效果不绑定条件
    SATISFIED = "satisfied"      # 全部条件可判定且成立
    UNSATISFIED = "unsatisfied"  # 至少一个条件可判定且不成立
    UNKNOWN = "unknown"          # 存在不可判定条件，或该卡条件未建模


@dataclass(frozen=True)
class ConditionProgress:
    """单张卡的打出条件进度。

    由 ConditionProgressProjector 在每次状态转移后物化；特征编码只读取，不再求值，
    因此 self-play 不需要为特征额外模拟或搜索。
    """

    state: ConditionState = ConditionState.NONE
    satisfied: int = 0
    unsatisfied: int = 0
    unknown: int = 0
    total: int = 0
    # 该卡是否绑定了打出条件（由 ConditionPlan 决定）。UNKNOWN + has=False 表示
    # 「未建模/缺模板」，UNKNOWN + has=True 表示「有条件但当前无法判定」。
    has_conditional_effect: bool = False
    # 留牌潜在价值的另外两类静态来源（与「效果条件」维度独立）：
    # 本卡牌被舍弃时发动；在手牌中因条件降费。
    has_discard_payoff: bool = False
    has_hand_cost_reduction: bool = False


@dataclass(frozen=True)
class BattleSnapshot:
    """决策系统的核心状态输入。"""

    me: PlayerView
    opponent: PlayerView
    cards_by_uid: dict[int, CardView]
    hand_knowledge: HandKnowledgeState = dataclass_field(default_factory=HandKnowledgeState.empty)
    legal: LegalView | None = None
    source: str = "puzzle"
    is_ally_turn: bool = False
    is_turn_ended: bool = False
    skill_effect_yellow_cards: frozenset[int] = frozenset()
    # 卡牌「打出效果条件」进度（本地推导）。键为 cards_by_uid 的 uid。
    # 只由 execution/condition_progress.py 物化，特征编码只读。
    condition_progress_by_uid: dict[int, ConditionProgress] = dataclass_field(default_factory=dict)
