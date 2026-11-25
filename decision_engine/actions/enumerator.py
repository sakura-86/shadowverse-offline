# -*- coding: utf-8 -*-
"""动作枚举器接口。"""

from __future__ import annotations

from typing import Protocol

from decision_engine.actions.model import LegalAction
from decision_engine.state.snapshot import BattleSnapshot


class ActionEnumerator(Protocol):
    """合法动作枚举器协议。"""

    def enumerate(self, snapshot: BattleSnapshot) -> list[LegalAction]:
        """枚举当前局面的合法动作。"""
        ...
