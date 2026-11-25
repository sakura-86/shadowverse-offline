# Recorded scenario fixtures

本目录保存协议回放用的稳定 JSON fixture，用于 `load_recorded_scenario`、`ReplayOracle` 和 `ProtocolDrivenPlannerLoop` 测试。

## 边界

Recorded scenario 是轻量的 BattleRootMpo-like 回放样例格式，不是完整 `BattleRootMpo` schema：

- 不描述真实 API 请求或响应的完整字段。
- 不承诺覆盖 MessagePack / MasterMemory / LZ4Block 等序列化细节。
- 不执行卡牌效果，也不作为本地规则引擎的补丁入口。
- `initial_root` 和 `transitions[].next_root` 只需包含 `BattleRootMpoAdapter` 当前测试需要的 mock 字段。
- `transitions[].action` 必须与 `ActionEncoder.encode(...)` 的输出一致，用来做精确匹配。

## 顶层字段

必填：

- `schema_version`: 整数。当前只支持 `1`。
- `initial_root`: 对象。起始 BattleRootMpo-like dict，会交给 `BattleRootMpoAdapter.from_dict`。
- `transitions`: 数组。按录制动作描述可回放的状态推进。

可选：

- `metadata`: 对象。fixture id、标题、来源说明等测试元数据。
- `note`: 字符串。人类可读备注。
- `source`: 字符串。样例来源说明，例如 `protocol_mock`。
- `expected_goal`: 对象。测试可选读取的期望目标描述，不由 loader 执行判定。

## transition 字段

必填：

- `action`: 对象。稳定 request-like 动作 dict，必须等于 `ActionEncoder.encode(LegalAction | PlannedAction)` 的结果。
- `next_root`: 对象。该动作之后的 BattleRootMpo-like dict。

可选：

- `note`: 字符串。该 transition 的备注。

## root 字段约定

`initial_root` / `next_root` 当前使用 `BattleRootMpoAdapter` 支持的最小字段子集，例如：

- `Players`: 两个 player 对象。
- player 中的 `Life`、`MaxLife`、`PP`、`MaxPP`、`EP`、`SEP`、`HandCards`、`FieldCards`。
- card 中的 `UniqueId`、`CardId`、`BaseCardId`、`Name`、`Cost`、`Atk`、`Life`、`MaxLife`、`CardType`。
- 协议合法动作字段，如 `CanAttackLeaderCards`、`CanPlayCards`、`CanAttackFieldCards`、`AttackTargetsMap` 等。

新增 fixture 时应保持样例最小化，只表达协议合法动作和录制后的下一状态，不在 JSON 中塞入未被 adapter 消费的效果语义。

## Pending evidence scenario

`*_pending.json` fixture 用于固定复杂动作字段的候选证据形态，例如 `ChoiceSkillMap`、`CanModeSkillCards`、`CanSpecialActionFieldCards`。

这些样例必须带有 `metadata.status = "pending"` 与 `metadata.requires_runtime_confirmation = true`，表示：

- 字段来自 BattleRootMpo / 客户端消费路径的候选证据，但尚未有足够 runtime scenario 确认具体动作编码。
- 只能用于验证 recorded scenario schema、fixture 加载、adapter 对未知字段的保守忽略行为。
- 不能被当成已支持动作映射；除非同一个 fixture 同时包含当前已支持的 `Can*` 字段，否则 `ProtocolActionEnumerator` 不应从 pending 字段凭空产生动作。
- 不在 pending fixture 中模拟效果、按卡名写规则，或连接真实 API / MessagePack / Hook 项目。

## Puzzle dump current-turn fixtures

`puzzle_dump/` 子目录中的 fixture 来自 `puzzle_dump/*.json` 裸 BattleRoot-like 解谜快照。每个输入快照是独立谜题起点，不是连续 replay。

这些样例必须满足：

- `metadata.source_kind = "puzzle_dump_current_turn_fixture"`。
- `metadata.status = "pending"` 与 `metadata.requires_runtime_confirmation = true`。
- `metadata.turn_limit = "current_turn"`。
- `transitions = []`，因为原始 dump 不包含 `action_request` / `action_response`。
- `initial_root` 是用于搜索起点的规范化 root；若修正过回合状态，必须在 `metadata.root_patches` 中记录。
- `expected_goal.kind` 必须显式表达目标，例如 `lethal` 或 `board_clear`，不得从描述文本猜测。

这些 fixture 只验证 schema、`BattleRootMpoAdapter.from_dict`、协议动作空间枚举和 ReplayAudit pending 分类；不验证动作序列、LocalPredictor transition、谜题解法或斩杀线。不得把相邻 puzzle/state 文件强行连接成 before/after transition。
