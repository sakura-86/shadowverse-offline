# -*- coding: utf-8 -*-
"""已冻结模块清单：AlphaZero 路线的否定性结果。

冻结的含义
==========

冻结**不是删除**。这些模块的代码、测试和诊断价值全部保留：

- ``decision_engine/tests/`` 允许继续 import 它们（回归证据、行为锁定）；
- ``tools/`` 允许继续 import 它们（历史实验复现）；
- **``decision_engine`` 主路径不允许 import 它们。**

主路径指 ``runtime`` / ``training`` / ``execution`` / ``cards`` / ``state`` /
``actions`` / ``protocol`` / ``simulation`` / ``sampling`` / ``strategy`` /
``meta``，以及 ``planning`` 内的非冻结模块。该约束由
``decision_engine/tests/test_import_isolation.py`` 静态扫描（AST）强制，不是文档约定。

为什么冻结
==========

依据 19 节诊断（``docs/blog/诊断历程_SelfPlay不涨与价值信号_面试材料.md``）：

- ``§18.5`` PUCT 在 128 sims 下有效视野只有 **2~3 个动作**，而一个回合有 6~15 个
  动作——它看不到回合结束，然后用一个不准的 value 猜后面；
- ``§19`` 决策点由「新信息到达」定义（每回合 3.92 个，增幅法 4.77 个），与
  「树一层 = 一个微动作」的粒度对不上；
- ``§20.2`` AlphaZero 的五个前提（完全信息 / 树层=决策 / 确定转移 / value 可学 /
  零和对称）影之诗只满足一个；
- 代理 ``value target`` 与 neural value 自举回路（§1、§3、§16、§17）会持续
  产出「代理指标涨、真实胜率不涨」的结果。

保留的锚点
==========

``turn_planner.plan_turn``（严格配对镜像 12:0 完胜 PUCT(128 sims)，仅用 37 个
节点）是当前唯一有数据支撑的强基线，**不冻结**，是后续所有候选的唯一验收锚点。
它依赖的 ``turn_evaluation`` / ``tactical_summary`` 一并保留。

解冻条件
========

先满足 ``§20.5`` 的前三条（option 搜索单位、抽牌 chance node、对手手牌 belief），
并有「对 ``plan_turn`` 的配对胜率」作为新证据——而不是只拿 self-play 内部对照。
"""

from __future__ import annotations

#: 冻结模块名（``decision_engine.planning`` 下的末段名）。
FROZEN_MODULES: frozenset[str] = frozenset(
    {
        # --- PUCT 搜索本体与由其驱动的自对弈采样（§18.5 / §19） ---
        "puct",
        "self_play_search",
        "strict_self_play",
        # --- 建立在 PUCT 之上的 macro / 快照排序搜索 ---
        "macro_driven_search",
        "macro_turn_search",
        "macro_ab_selector",
        "macro_frontier",
        "fixed_snapshot_ranking",
        # --- proof search（标记树爆炸，背景段落已判定走不通） ---
        "deterministic_proof_search",
        "proof_number_search",
        "local_proof_game",
        "proof_job",
        "proof_diagnostics",
        # --- 已证伪的 value target 与回填（§3 / §4 / §8） ---
        "value_target",
        "value_target_backfill",
        # --- PUCT 叶评估势函数与 PP 本位势（§14-§17） ---
        "diff_value",
        "incremental_value",
    }
)

#: 每个冻结模块的一句话理由（便于 code review 与面试讲解时直接引用）。
FROZEN_REASONS: dict[str, str] = {
    "puct": "§18.5 有效视野仅 2~3 个动作，看不到回合结束",
    "self_play_search": "以 PUCT 为采样器生成训练数据，继承其视野缺陷",
    "strict_self_play": "PUCT + value 臂的自对弈样本生成，同上",
    "macro_driven_search": "PUCT 的 macro 包装，仍以 PUCT 节点为单位",
    "macro_turn_search": "macro 搜索依赖 PUCT 节点与叶评估",
    "macro_ab_selector": "macro 搜索臂 selector，依赖 macro_turn_search",
    "macro_frontier": "macro 搜索的 frontier / 预算管理，依赖 macro_turn_search 的 TurnLine 类型",
    "fixed_snapshot_ranking": "依赖 diff_value + puct 的快照排序",
    "deterministic_proof_search": "proof search 标记树爆炸，算力不可行",
    "proof_number_search": "依赖 deterministic_proof_search",
    "local_proof_game": "依赖 fixed_snapshot_ranking（PUCT 链）",
    "proof_job": "依赖 proof search 全链",
    "proof_diagnostics": "依赖 proof search 全链",
    "value_target": "§3 代理目标，§19 证其信号缺陷非主要矛盾",
    "value_target_backfill": "§8 TD(λ) 回填，服务于已证伪的监督 value 目标",
    "diff_value": "§16 势函数，代理指标连续三次误导",
    "incremental_value": "§14-§17 PP 本位势，未能在锚点上稳定证明",
}

#: 主路径包（相对仓库根的 ``decision_engine/``）。
PRIMARY_PATH_PACKAGES: tuple[str, ...] = (
    "runtime",
    "training",
    "execution",
    "cards",
    "state",
    "actions",
    "protocol",
    "simulation",
    "sampling",
    "strategy",
    "meta",
)

#: 跨包 / 非 ``planning`` 命名空间的冻结文件。
#:
#: 这些文件**属于**冻结单元（不是"主路径被豁免"）：它们本身依赖冻结链，随该路线
#: 一起废弃。主路径扫描时整体跳过，删除条件写在各条理由里。
FROZEN_MODULE_PATHS: dict[str, str] = {
    "decision_engine/runtime/puct_selector.py": (
        "runtime PUCT 决策器本体：已从 runtime/__init__ 导出移除，不再作为 runtime "
        "决策入口；仅 tools/protocol/run_practice_battle_capture.py 的历史采集脚本"
        "直接引用。删除条件：该采集脚本迁移到 plan_turn 或废弃后一并删除。"
    ),
    "decision_engine/training/proof_examples.py": (
        "proof search 的训练样本导出：属于已冻结的 proof 路线（标记树爆炸）。"
        "仅 tests/test_proof_examples.py 与 tools/protocol 下的诊断脚本引用。"
        "删除条件：proof 路线清理时一并删除。"
    ),
}

#: 显式登记的主路径例外。**当前为空——主路径不应有任何冻结依赖。**
#:
#: 约定：新增例外必须写明「为什么不是主路径」+「删除条件」。静默豁免视为违规。
ALLOWED_FROZEN_IMPORTERS: dict[str, str] = {}


def is_frozen(module_name: str) -> bool:
    """``module_name`` 是否为 ``decision_engine.planning`` 下的冻结模块末段名。"""
    return module_name in FROZEN_MODULES
