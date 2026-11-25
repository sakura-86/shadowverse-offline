# -*- coding: utf-8 -*-
"""PersistentEffectRegistry：事件订阅注册表。

卡进入场上/纹章区时注册自己的 persistent effects，
EventQueue dispatch 时通过 get_subscribers 查询响应者，
不再遍历全场搜索。

两种使用模式：
- replay validation: 每 step 通过 from_snapshot 重建（counter=0）
- self-play / MCTS: 游戏开始时创建，增量维护，turn_start 时 reset counters
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from decision_engine.cards.templates import ExecutableEffect, PersistentEffect, UnifiedCardTemplate
from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView

from .events import ExecutionEvent


def persistent_zone_uids(view: PlayerView) -> tuple[int, ...]:
    """当前处于持久效果区域（场地 / 纹章 / 额外纹章 / 恩惠）的 uid，按协议区域顺序。

    这个顺序只是「单帧可见顺序」，**不是**登记顺序：它把场地整体排在纹章之前。
    只用于判定「当前哪些 uid 会参与结算」。
    """
    uids = list(view.field)
    uids.extend(crest.unique_id for crest in view.crests if crest.unique_id)
    uids.extend(crest.unique_id for crest in view.extra_crests if crest.unique_id)
    uids.extend(boon.unique_id for boon in view.boons if boon.unique_id)
    return tuple(uids)


def advance_persistent_registration_order(
    previous: Sequence[int],
    view: PlayerView,
) -> tuple[int, ...]:
    """推进卡牌登记顺序：只登记**已打出/已出现**的卡，手牌不算登记。

    登记事件是「卡牌进入场景」——打出/召唤/放置到场地区域，或纹章（含额外纹章、
    恩惠）出现。手牌里的卡**不登记**：它还没有参与任何时点结算，登记时间应当
    是它真正出场的时刻。

    离场（破坏 / 弹回手牌 / 返回牌组）会把该项移出登记序列：再次打出时按新的
    出场时刻重新登记，与「登记时点是出场」一致；这依赖调用方在每个 step 的终态
    都推进一次（否则离场项不会被移除）。
    """
    present = persistent_zone_uids(view)
    present_set = set(present)
    kept = [uid for uid in previous if uid in present_set]
    known = set(kept)
    kept.extend(uid for uid in present if uid not in known)
    return tuple(kept)


def trigger_count_key(effect_id: str, subscriber_uid: int | None) -> str:
    """per_turn 计数在 snapshot 里的 key：身份是「订阅实例」= 卡实例 + effect_id。

    同名卡可能同时在场（两把『同悦之钟』10863210：回合结束时抽 1 张并回复 1 点），
    每张卡各自有一次「每回合 1 次」。只按 effect_id 记计数会让第二张看到第一张
    的计数而被 per_turn_limit 过滤，整张卡的效果完全不执行
    （replay_...1735553820 st77：两个同悦之钟只抽 1 张、只回复 1 点）。

    订阅者没有 uid（合成订阅）时无法区分实例，退回 effect_id。
    """
    if subscriber_uid is None:
        return effect_id
    return f"{effect_id}@{subscriber_uid}"


@dataclass
class PersistentSubscription:
    """一条订阅记录：某张卡/纹章对某个事件的响应。"""

    subscriber_uid: int | None
    """订阅者 uid。纹章无实体 uid 时为 None。"""

    subscriber_base_card_id: int

    subscriber_card: CardView
    """执行效果时用作 source_card 上下文。"""

    player: str
    """订阅者属于 'me' / 'opponent'。"""

    effect: ExecutableEffect
    """要执行的效果。"""

    trigger_event: str
    """订阅的事件名（如 'damage', 'enter_field'）。"""

    scope: str
    """来源区域：'field' / 'crest'。"""

    target_relation: str = "any"
    """事件 target 与订阅者的关系：'self'=自身受伤, 'ally'=同侧, 'any'=任意。"""

    per_turn_limit: int | None = None

    counter: int = 0
    """当前回合已触发次数。"""

    per_turn_limits_map: dict[str, int] = field(default_factory=dict)
    """effect_id → limit 映射（传递给 handler 用）。"""


_VARIANT_BASE_CACHE: dict[int, dict[int, int]] = {}


def _variant_base_index(templates: Mapping[int, UnifiedCardTemplate]) -> dict[int, int]:
    """card_id → base_card_id 索引（含纹章的进化/超进化变体）。

    纹章在协议里可能以变体 id 出现（如 10954113 是 10954112 的变体），
    而模板只按 base_card_id 建索引。不做归一会直接查不到模板，
    该纹章的持续效果就永远不注册（replay_...1484307 st51）。
    """
    key = id(templates)
    cached = _VARIANT_BASE_CACHE.get(key)
    if cached is not None and len(cached) >= len(templates):
        return cached
    index: dict[int, int] = {}
    for base, tmpl in templates.items():
        base_id = int(base)
        index[base_id] = base_id
        identity = getattr(tmpl, "identity", None)
        for card_id in getattr(identity, "card_ids", ()) or ():
            try:
                index.setdefault(int(card_id), base_id)
            except (TypeError, ValueError):
                continue
    _VARIANT_BASE_CACHE[key] = index
    return index


class PersistentEffectRegistry:
    """事件订阅注册表。"""

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[PersistentSubscription]] = {}

    def register(
        self,
        uid: int | None,
        card: CardView,
        player: str,
        template: UnifiedCardTemplate,
        scope: str = "field",
        *,
        source_zone: str | None = None,
        only_events: frozenset[str] | None = None,
    ) -> None:
        """注册一张卡的所有 persistent effects。

        source_zone 指定时只注册该来源区域的 persistent（如手牌中发动的持续效果）。
        only_events 指定时只注册这些触发事件（用于「本步离场的卡只补离场触发」）。
        """
        for persistent in template.persistent_effects:
            if persistent.trigger is None or not persistent.executable_effects:
                continue
            if source_zone is not None and persistent.source_zone != source_zone:
                continue
            if only_events is not None and persistent.trigger.event not in only_events:
                continue
            target_relation = self._infer_target_relation(persistent, scope)
            for exe_effect in persistent.executable_effects:
                sub = PersistentSubscription(
                    subscriber_uid=uid,
                    subscriber_base_card_id=card.base_card_id,
                    subscriber_card=card,
                    player=player,
                    effect=exe_effect,
                    trigger_event=persistent.trigger.event,
                    scope=scope,
                    target_relation=target_relation,
                    per_turn_limit=persistent.per_turn_limit,
                    per_turn_limits_map={exe_effect.effect_id: persistent.per_turn_limit} if persistent.per_turn_limit else {},
                )
                self._subscriptions.setdefault(persistent.trigger.event, []).append(sub)

    def unregister(self, uid: int) -> None:
        """卡离场时注销所有订阅。"""
        for event_name in list(self._subscriptions):
            subs = self._subscriptions[event_name]
            self._subscriptions[event_name] = [s for s in subs if s.subscriber_uid != uid]
            if not self._subscriptions[event_name]:
                del self._subscriptions[event_name]

    def get_subscribers(
        self,
        event: ExecutionEvent,
        snapshot: BattleSnapshot,
    ) -> list[PersistentSubscription]:
        """返回响应此 event 的所有订阅（已过滤 side/scope/limit）。"""
        # event name alias 映射: runtime event name -> spec trigger event name
        _EVENT_ALIASES: dict[str, tuple[str, ...]] = {
            "attack_follower": ("on_attack", "attack_follower"),
            "attack_leader": ("on_attack", "attack_leader"),
            "turn_end": ("on_turn_end", "turn_end"),
            "turn_start": ("on_turn_start", "turn_start"),
            # 运行时入场事件名为 enter_field/summon，需匹配以 on_self_follower_entered_field
            # 等 trigger event 名注册的 persistent 订阅（如纹章"自己的随从进入战场时"）。
            # 与 effect_trigger_resolver._EVENT_TO_TRIGGER_ALIASES 保持一致。
            # on_enter_field 是 spec 生成端对「任意随从进入战场」的写法（10674110 /
            # 10771110 / 10773110），此前不在别名表里，三条持续效果从未被触发。
            "enter_field": ("enter_field", "on_enter_field", "on_self_follower_entered_field", "on_self_entered_field", "on_enemy_follower_entered_field"),
            # 进入战场是同一次入场，但 kernel.summon_card 会同时发 summon 与 enter_field
            # 两个事件；若 summon 也映射到 on_self_follower_entered_field / on_self_entered_field，
            # 同一张卡入场会让这类持续效果触发两次（10754120 马克米朗的 1 点伤害双结算）。
            # 因此 summon 只匹配以 summon 命名 trigger 的订阅，入场效果统一由 enter_field 触发。
            "summon": ("summon",),
            # 「…在战场上**消失**时」只对应 banish（消失），不含 destroy / bounce /
            # return_to_deck：卡面用「消失」指的就是 banish 这个动作。
            # 此前把五类离场事件都别名过去，导致「自己的卡牌在战场上消失时」
            # 类效果在普通战斗破坏后也触发（replay_...1227396 st98：多召唤一个
            # 骸骨士兵，而且是 synthetic uid）。
            "destroy": ("destroy",),
            # 「对手的随从在战场上消失时」（11064120 亚教偶像）与 on_self_card_left_field
            # 是镜像语义：同为 banish 事件，但要求事件归属侧与订阅者**相反**，
            # 由 `_matches_event` 区分。destroy 不参与（卡面「消失」只指 banish）。
            "banish": ("banish", "on_self_card_left_field", "on_enemy_card_left_field"),
            "bounce": ("bounce",),
            "return_to_deck": ("return_to_deck",),
            "activate": ("activate", "activation", "on_self_amulet_activated"),
            # 「自己使用法术时」持续效果（如 90021210 海盗旗倒计数-1）注册在
            # on_spell_play，运行时事件名为 play_spell。
            # 同时映射到 play_card：卡面「自己使用其他卡牌时」的“卡牌”包含法术，
            # kernel 对法术只发 play_spell 不发 play_card（rule_kernel.play_spell_from_hand）。
            # 10503210 大游戏世界依赖这一点把倒计数 -1。
            "play_spell": ("play_spell", "on_spell_play", "play_card"),
            # 「自己抽到卡牌时」持续效果（如纹章 10564122 按抽到卡费用决定召唤侧）：
            # 运行时事件名为 draw，spec 触发名为 on_draw_card。
            # 只映射可见的 draw；draw_hidden 表示协议未暴露抽到的具体卡，
            # 无法判定费用等条件，不能靠猜测触发。
            "draw": ("draw", "on_draw_card"),
        }
        lookup_keys = _EVENT_ALIASES.get(event.name, (event.name,))
        candidates: list[PersistentSubscription] = []
        for key in lookup_keys:
            candidates.extend(self._subscriptions.get(key, []))
        # destroyed_by_damage 也匹配注册为 destroy 的订阅
        if event.name == "destroyed_by_damage":
            candidates.extend(self._subscriptions.get("destroy", []))
        if not candidates:
            return []
        result: list[PersistentSubscription] = []
        for sub in candidates:
            if sub.per_turn_limit is not None and sub.counter >= sub.per_turn_limit:
                continue
            if self._matches_event(sub, event, snapshot):
                result.append(sub)
        return result

    def increment_counter(self, sub: PersistentSubscription) -> None:
        """效果成功执行后递增计数器。"""
        sub.counter += 1

    def reset_turn_counters(self) -> None:
        """回合开始时重置所有 per_turn 计数器。"""
        for subs in self._subscriptions.values():
            for sub in subs:
                sub.counter = 0

    @property
    def subscription_count(self) -> int:
        return sum(len(subs) for subs in self._subscriptions.values())

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BattleSnapshot,
        templates: Mapping[int, UnifiedCardTemplate],
    ) -> PersistentEffectRegistry:
        """从 snapshot 重建 registry（replay validation 每 step 调用）。"""
        registry = cls()
        registry._register_snapshot(snapshot, templates)
        return registry

    @property
    def subscriber_uids(self) -> frozenset[int]:
        return frozenset(
            sub.subscriber_uid
            for subs in self._subscriptions.values()
            for sub in subs
            if sub.subscriber_uid is not None
        )

    def register_departed(
        self,
        previous: BattleSnapshot,
        current: BattleSnapshot,
        templates: Mapping[int, UnifiedCardTemplate],
    ) -> None:
        """为本步**离开战场**的卡补注册「离场时」订阅。

        效果执行可能在发出事件之前就把卡移出战场（11023310 打出后先移除旧证据
        护符，再发 BANISH）。这类卡在本步仍需要响应它自己的
        ``on_self_card_left_field``（11003210 回复自己的主战者 2 点）。

        只补这一个事件：卡已经不在场上，其它持久效果不该继续生效（实测按整份
        订阅补注册会多结算一批留场才触发的效果，8 组回放净负 30 步）。
        """
        only_events = frozenset({"on_self_card_left_field"})
        variant_index = _variant_base_index(templates)
        registered = self.subscriber_uids
        for player, view in [("me", previous.me), ("opponent", previous.opponent)]:
            current_view = current.me if player == "me" else current.opponent
            current_uids = (
                set(current_view.field)
                | {crest.unique_id for crest in current_view.crests if crest.unique_id}
                | {extra.unique_id for extra in current_view.extra_crests if extra.unique_id}
                | {boon.unique_id for boon in current_view.boons if boon.unique_id}
            )
            for uid in view.field:
                if uid in current_uids or uid in registered:
                    continue
                card = previous.cards_by_uid.get(uid)
                if card is None or card.base_card_id not in templates:
                    continue
                tmpl = templates[card.base_card_id]
                if tmpl.persistent_effects:
                    self.register(uid, card, player, tmpl, scope="field", source_zone="field", only_events=only_events)
            for crest in view.crests:
                uid = crest.unique_id
                if not uid or uid in current_uids or uid in registered:
                    continue
                base_id = variant_index.get(crest.card_id, crest.card_id)
                if base_id not in templates:
                    continue
                tmpl = templates[base_id]
                if tmpl.persistent_effects:
                    self.register(uid, CardView(uid=uid, card_id=base_id, base_card_id=base_id), player, tmpl, scope="crest", only_events=only_events)
            for extra in view.extra_crests:
                uid = extra.unique_id
                if not uid or uid in current_uids or uid in registered:
                    continue
                base_id = variant_index.get(extra.card_id, extra.card_id)
                if base_id not in templates:
                    continue
                tmpl = templates[base_id]
                if tmpl.persistent_effects:
                    self.register(uid, CardView(uid=uid, card_id=base_id, base_card_id=base_id), player, tmpl, scope="extra_crest", only_events=only_events)
            for boon in view.boons:
                uid = boon.unique_id
                if not uid or uid in current_uids or uid in registered:
                    continue
                base_id = variant_index.get(boon.card_id, boon.card_id)
                if base_id not in templates:
                    continue
                tmpl = templates[base_id]
                if tmpl.persistent_effects:
                    self.register(uid, CardView(uid=uid, card_id=base_id, base_card_id=base_id), player, tmpl, scope="boon", only_events=only_events)

    def _register_snapshot(
        self,
        snapshot: BattleSnapshot,
        templates: Mapping[int, UnifiedCardTemplate],
    ) -> None:
        variant_index = _variant_base_index(templates)
        for player, view in [("me", snapshot.me), ("opponent", snapshot.opponent)]:
            # 场地与纹章**合并**按登记顺序注册：同一时点触发多个裁定时，事件队列按
            # 注册顺序结算，因此注册顺序必须等于卡牌进入持久效果区域的先后，而不是
            # 「先场地后纹章」（后者是早期按区域遍历得到的错误顺序：
            # replay_...0006 st65 里纹章 10364122 登记早于随从 10564110，
            # 却因区域遍历被排到它后面，导致 3 点分配打了主战者）。
            order = advance_persistent_registration_order(view.persistent_registration_order, view)
            field_cards = {uid: snapshot.cards_by_uid.get(uid) for uid in view.field}
            crests_by_uid = {crest.unique_id: crest for crest in view.crests if crest.unique_id}
            # 额外纹章区（协议 extra_crests，卡牌 type=12 "特性"）与恩惠区
            # （协议 boons：加护、权能、特性、异能）与纹章同属持续效果区域。
            # 它们也参与「登记顺序」：`persistent_zone_uids` 已经把它们的 uid
            # 计入登记序列，注册也必须按同一顺序进行，否则一个登记更早的额外纹章
            # 会被排到所有场地/纹章之后结算（与 §29 的「按登记顺序」不一致）。
            extras_by_uid = {extra.unique_id: extra for extra in view.extra_crests if extra.unique_id}
            boons_by_uid = {boon.unique_id: boon for boon in view.boons if boon.unique_id}
            for uid in order:
                field_card = field_cards.get(uid)
                if field_card is not None:
                    if field_card.base_card_id in templates:
                        tmpl = templates[field_card.base_card_id]
                        if tmpl.persistent_effects:
                            # 只注册「从场地发动」的持续效果。source_zone="hand"
                            # 的持续效果（「在手牌中发动」）不属于场地订阅：
                            # 不过滤会让同一张卡在手牌里减费的效果在它出场后
                            # 继续以场地订阅身份触发（replay_...1246399 st125：
                            # 场上的 10954120 每回合结束都被扣 1 费）。
                            self.register(uid, field_card, player, tmpl, scope="field", source_zone="field")
                    continue
                crest = crests_by_uid.get(uid)
                if crest is not None:
                    base_id = variant_index.get(crest.card_id, crest.card_id)
                    if base_id in templates:
                        tmpl = templates[base_id]
                        if tmpl.persistent_effects:
                            crest_card = CardView(uid=crest.unique_id or 0, card_id=base_id, base_card_id=base_id)
                            # subscriber_uid 用纹章 unique_id：纹章的谢幕曲事件
                            # （倒计数归零）以它为 source_uid，据此区分「自身触发」与
                            # 「同侧其他卡的谢幕曲」。
                            self.register(crest.unique_id or None, crest_card, player, tmpl, scope="crest")
                    continue
                extra = extras_by_uid.get(uid)
                if extra is not None:
                    base_id = variant_index.get(extra.card_id, extra.card_id)
                    if base_id in templates:
                        tmpl = templates[base_id]
                        if tmpl.persistent_effects:
                            extra_card = CardView(uid=extra.unique_id or 0, card_id=base_id, base_card_id=base_id)
                            self.register(extra.unique_id or None, extra_card, player, tmpl, scope="extra_crest")
                    continue
                boon = boons_by_uid.get(uid)
                if boon is not None:
                    base_id = variant_index.get(boon.card_id, boon.card_id)
                    if base_id in templates:
                        tmpl = templates[base_id]
                        if tmpl.persistent_effects:
                            boon_card = CardView(uid=boon.unique_id or 0, card_id=base_id, base_card_id=base_id)
                            self.register(boon.unique_id or None, boon_card, player, tmpl, scope="boon")
            # 手牌中发动的持续效果（source_zone="hand"）：只在 hand 中订阅，
            # 例如「自己【启动】证据·护符时，本随从的攻击力与生命值交换。本随从+1/+0」。
            for uid in view.hand:
                card = snapshot.cards_by_uid.get(uid)
                if card and card.base_card_id in templates:
                    tmpl = templates[card.base_card_id]
                    if tmpl.persistent_effects:
                        self.register(uid, card, player, tmpl, scope="hand", source_zone="hand")
        # 从 snapshot 恢复 per_turn 计数器
        for player, view in [("me", snapshot.me), ("opponent", snapshot.opponent)]:
            if view.effect_trigger_counts:
                for subs in self._subscriptions.values():
                    for sub in subs:
                        if sub.player != player:
                            continue
                        key = trigger_count_key(sub.effect.effect_id, sub.subscriber_uid)
                        if key in view.effect_trigger_counts:
                            sub.counter = view.effect_trigger_counts[key]

    def _matches_event(
        self,
        sub: PersistentSubscription,
        event: ExecutionEvent,
        snapshot: BattleSnapshot,
    ) -> bool:
        """粗过滤：event 是否匹配订阅的 side/scope/target_relation。"""
        # 「…在战场上消失时」有两种卡面语义，用 spec 的 `event_source_is_self` 条件区分
        # （与 template_event_handler 的直接路由段同口径）：
        #   * 有 `event_source_is_self` ——「**本卡牌**在战场上消失时」（11003210），
        #     只有订阅者自己离场才触发；
        #   * 没有该条件 ——「**自己的卡牌**在战场上消失时」（11052110 / 11053110 /
        #     11054110），同侧任意卡离场都触发。
        # 此前这里一律按「自身」处理，导致后三张卡的效果永远不触发
        # （replay_...1301724 st57：怨灵离场应召唤骸骨士兵，本地没召唤）。
        if sub.effect.trigger is not None and sub.effect.trigger.event == "on_self_card_left_field":
            left_uid = event.target_uid if event.target_uid is not None else event.source_uid
            allow_self = any(
                getattr(condition, "kind", "") == "event_source_is_self"
                for condition in (sub.effect.conditions or ())
            )
            if allow_self:
                return sub.subscriber_uid is not None and left_uid == sub.subscriber_uid
            # 没有 `event_source_is_self` 时是「自己的**其他**卡牌在战场上消失时」：
            # 订阅者自己离场不算（与 template_event_handler 直接路由段的
            # `source_uid == left_uid and not allow_self -> False` 一致）。
            if sub.subscriber_uid is not None and left_uid == sub.subscriber_uid:
                return False
            return event.player is not None and event.player == sub.player
        if sub.effect.trigger is not None and sub.effect.trigger.event == "on_enemy_card_left_field":
            # 「对手的随从在战场上消失时」：与 on_self_card_left_field 镜像，事件归属侧
            # 必须与订阅者相反（11064120 亚教偶像：对手的随从消失时召唤复制随从）。
            return event.player is not None and event.player != sub.player
        # damage event: 检查 target 与订阅者的关系
        if event.name == "damage":
            # leader damage (target_uid=None) 不触发场上卡/纹章的 persistent
            if event.target_uid is None:
                return False
            if sub.target_relation == "self":
                # 只响应自己受伤
                return event.target_uid == sub.subscriber_uid
            elif sub.target_relation == "ally":
                # 响应同侧任意随从受伤
                if sub.player == "me":
                    return event.target_uid in snapshot.me.field
                else:
                    return event.target_uid in snapshot.opponent.field
            return True

        # enter_field/summon: 同侧入场
        if event.name in {"enter_field", "summon"}:
            # 入场卡自身的持续订阅不响应「自己入场」这个事件。
            # 这类效果语义是「己方其他随从入场时…」，触发者必须是别的随从；
            # 「自己入场时…」应使用 on_self_entered_field trigger 而非 enter_field。
            if sub.subscriber_uid is not None and event.source_uid == sub.subscriber_uid:
                return False
            if event.player is not None:
                return event.player == sub.player
            if event.source_uid is not None:
                if sub.player == "me":
                    return event.source_uid in snapshot.me.field
                else:
                    return event.source_uid in snapshot.opponent.field
            return True

        # play_card: 「自己使用其他卡牌时」不响应自身被打出的事件。
        # play_spell 走同一分支：法术也是「使用卡牌」，同样按 side 过滤并排除自身。
        if event.name in {"play_card", "play_spell"}:
            if sub.subscriber_uid is not None and event.source_uid == sub.subscriber_uid:
                return False
            if event.player is not None:
                return event.player == sub.player
            return True

        # destroy/destroyed_by_damage: 同侧
        if event.name in {"destroy", "destroyed_by_damage"}:
            if event.player is not None:
                return event.player == sub.player
            return True

        # last_words: 场上的卡按同侧匹配（同一方的谢幕曲订阅需要响应；
        # 卡自身的模板 last_words 走直接路由段，不依赖这里）。
        # 但**纹章/额外纹章**的谢幕曲只在自身倒计数归零时触发：那个事件的
        # source_uid 就是纹章的 unique_id。若按同侧匹配，同侧任何卡死亡都会
        # 误触发纹章的谢幕曲（replay_...1431478 st80 因此凭空加了一张卡）。
        if event.name == "last_words" and sub.scope in {"crest", "extra_crest"}:
            return sub.subscriber_uid is not None and event.source_uid == sub.subscriber_uid

        # attack: 同侧攻击者
        if event.name in {"attack_leader", "attack_follower"}:
            if event.player is not None:
                return event.player == sub.player
            return True

        # turn_start/turn_end: 默认同侧（「自己的回合结束时」）。
        # 显式声明 turn_player 条件的持续效果可以跨侧（「对手的回合结束时」，
        # 如 10734112 纹章），此时 side 由该条件求值决定，这里不拦截。
        if event.name in {"turn_start", "turn_end"}:
            if any(condition.subject == "turn_player" for condition in sub.effect.conditions):
                return True
            if event.player is not None:
                return event.player == sub.player
            return True

        # 默认：如果 event 有 player 字段，检查同侧
        if event.player is not None:
            return event.player == sub.player
        return True

    @staticmethod
    def _infer_target_relation(persistent: PersistentEffect, scope: str) -> str:
        """从 persistent spec 推断 target_relation。"""
        # 纹章的 damage trigger → 响应同侧 ally 受伤
        if scope == "crest" and persistent.trigger and persistent.trigger.event == "damage":
            return "ally"
        # 场上卡的 damage trigger → 响应自身受伤
        if scope == "field" and persistent.trigger and persistent.trigger.event == "damage":
            return "self"
        # 其他：默认 ally（同侧）
        return "ally"
