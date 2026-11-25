# -*- coding: utf-8 -*-
"""Local execution evidence context."""

from __future__ import annotations

from dataclasses import dataclass

from decision_engine.state.snapshot import CardView


@dataclass(frozen=True)
class DrawBinding:
    card: CardView
    source: str = "recorded_after_hand"
    instance_id: int | None = None


@dataclass(frozen=True)
class RandomBranchBinding:
    source: str
    effect_id: str
    operation_index: int
    cards: tuple[CardView, ...]
    assumption: str = "random_branch:source_pool"


@dataclass(frozen=True)
class LocalExecutionContext:
    draw_bindings: tuple[DrawBinding, ...] = ()
    # 抽牌方是 opponent 时的绑定池。replay 的视角可能被 opponent_as_me 互换：
    # 此时真正的抽牌方是 opponent，而 draw_bindings 只装了当前 me 侧的新卡
    # （replay_...4741709 st44：谢幕曲抽牌拿到 card=None，整次抽牌 no-op）。
    opponent_draw_bindings: tuple[DrawBinding, ...] = ()
    after_hand_bindings: tuple[DrawBinding, ...] = ()
    replay_mode: bool = False
    random_branch_bindings: tuple[RandomBranchBinding, ...] = ()
    branch_seed: int | None = None
    # 当前视角下手牌不可观测的玩家（"me"/"opponent"）。默认对手手牌是隐藏信息；
    # replay 视角互换后当前视角的 me 就是录像对手，由 replay 工具改成 {"me"}。
    # 任何「加入手牌」都不能物化到不可观测的手牌，否则会生成幽灵卡。
    hidden_hand_players: frozenset[str] = frozenset({"opponent"})
    # 「谢幕曲：召唤1个与本随从同名的随从」在 replay 中的证据绑定：after 帧新增的
    # 同名卡。直接沿用录像 CardView（uid/card_id/name/关键字），避免本地召出后
    # 字段（变体 card_id、name、attack_limit）与录像不一致。
    summon_copy_bindings: tuple[CardView, ...] = ()

    def drop_draw_bindings(self, count: int) -> "LocalExecutionContext":
        if count <= 0:
            return self
        return LocalExecutionContext(
            draw_bindings=self.draw_bindings[count:],
            opponent_draw_bindings=self.opponent_draw_bindings,
            after_hand_bindings=self.after_hand_bindings,
            replay_mode=self.replay_mode,
            random_branch_bindings=self.random_branch_bindings,
            branch_seed=self.branch_seed,
            hidden_hand_players=self.hidden_hand_players,
            summon_copy_bindings=self.summon_copy_bindings,
        )
