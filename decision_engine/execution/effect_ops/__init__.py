"""effect_ops：EffectInterpreter 的 operation 执行器子包。

拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from .base import OP_REGISTRY, EffectInterpreterError, OpHandler, register_op

# 导入各 op 模块以触发 @register_op 注册（副作用导入）
from . import counter_ops  # noqa: F401,E402
from . import crest_ops  # noqa: F401,E402
from . import stat_ops  # noqa: F401,E402
from . import evolve_ops  # noqa: F401,E402
from . import resource_ops  # noqa: F401,E402
from . import control_ops  # noqa: F401,E402
from . import combat_ops  # noqa: F401,E402
from . import legacy_ops  # noqa: F401,E402

__all__ = [
    "OP_REGISTRY",
    "EffectInterpreterError",
    "OpHandler",
    "register_op",
]
