# planning 层：冻结边界与锚点

本目录同时承载两类东西，**必须区分**：

| 类别 | 状态 | 谁能 import |
|---|---|---|
| 锚点 + 规划基础设施 | 有效 | 任何人 |
| 已冻结的 AlphaZero 路线 | 冻结 | 只有 `tests/`、`tools/` |

冻结清单与理由在 [`frozen.py`](frozen.py)，由
[`tests/test_import_isolation.py`](../tests/test_import_isolation.py) 静态强制。

## 锚点：`turn_planner.plan_turn`

严格配对镜像下 **12:0** 完胜 `PUCT(128 sims)`，单回合只用 **37 个节点**（PUCT 用 63 个
却只看得到回合前 2~3 步）。

**但它不是完整对局策略，不能当对称 opponent。** 它的评分只看净血差，不奖励铺场；
放到 opponent 侧实测每回合直接 `turn_end`，导致 plan-vs-plan 24 对全判候选胜。
12:0 那次的对手是 policy 模型，plan 从未在 opponent 侧，所以这个缺陷没暴露。

现状：锚点需要先补"铺场层"（把"打随从"从净血差 0 提到正分），或用完整对局策略替换，
才能用于验收。在此之前，`tools/run_anchor_match.py` 的对局结果不能写进结论。

依赖链（全部有效）：

```text
turn_planner ──▶ turn_evaluation ──▶ (terminal scoring)
     └─────────▶ tactical_summary

turn_plan_arm ──▶ search_ab_match ──▶ execution / actions
     └─────────▶ turn_planner

turn_plan_value ──▶ search_value, turn_evaluation
search_prior / search_value / entity_prior / entity_features   （网络接口层）
board_fingerprint / future_resource_signature                   （局面表征）
```

## 冻结了什么，为什么

依据 [`docs/blog/诊断历程_SelfPlay不涨与价值信号_面试材料.md`](../../docs/blog/诊断历程_SelfPlay不涨与价值信号_面试材料.md)：

| 组 | 模块 | 理由 |
|---|---|---|
| PUCT 本体 | `puct`, `self_play_search`, `strict_self_play` | §18.5 有效视野仅 2~3 个动作，看不到回合结束 |
| macro / 排序 | `macro_driven_search`, `macro_turn_search`, `macro_ab_selector`, `macro_frontier`, `fixed_snapshot_ranking` | 建立在 PUCT 节点与叶评估上 |
| proof search | `deterministic_proof_search`, `proof_number_search`, `local_proof_game`, `proof_job`, `proof_diagnostics` | 标记树爆炸，算力不可行 |
| value target | `value_target`, `value_target_backfill` | §3 代理目标；§8 TD(λ) 服务于已证伪的监督目标 |
| 势函数 | `diff_value`, `incremental_value` | §14-§17 代理指标连续三次误导 |

跨包冻结文件（不属于主路径，但依赖冻结链）：

- `decision_engine/runtime/puct_selector.py` — 已从 `runtime/__init__` 导出移除
- `decision_engine/training/proof_examples.py` — proof 路线的样本导出

## 解冻条件

不是"再调一下参数"，而是换搜索单位。按
[`§20.5`](../../docs/blog/诊断历程_SelfPlay不涨与价值信号_面试材料.md)：

1. 搜索单位换成 **option / SMDP**（从当前决策点到下一次信息到达）；
2. 抽牌做 **chance node**（expectimax，而不是采样单条分支）；
3. 对手手牌做 **belief**（CFR 或公开牌推断）；
4. 有新证据的形式是**对 `plan_turn` 的配对胜率**，不是 self-play 内部对照。

前三条都满足之前，不要解冻 PUCT。

## 加新模块时

1. 新模块默认属于"有效"类，不要 import 冻结模块；
2. 如果确实需要，先问"这是不是又把旧路线引回来了"，并解释为什么不违反 `frozen.py`；
3. 跑 `python -m unittest decision_engine.tests.test_import_isolation`。
