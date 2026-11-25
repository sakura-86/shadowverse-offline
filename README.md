# Shadowverse Offline Engine

从完整卡牌对战 AI 项目中剥离出来的**离线可运行子集**：本地规则引擎、PUCT 搜索、PPO 自博弈训练与评测。
不含游戏客户端、反编译代码、SvHook 通信、回放采集与美术资产。

## 这是什么

完整项目（私有）是一个卡牌对战 Agent：本地规则引擎把卡牌效果规格化为可执行 `EffectSpec`，PUCT 搜索负责决策，Entity Transformer + PPO 负责学习，配对评测负责验证。规模约 94k 行 Python、397 个文件、13 个模块、2,200+ 单元测试、169 个 checkpoint；8 组真实对局共 18,767 个动作，逐步一致率 98.93%。

本仓库保留**不依赖游戏运行时**的部分（235 个 Python 文件、约 52k 行、93 个测试）：

- `actions/` —— 动作模型与合法动作枚举
- `cards/` —— 卡牌主数据、模板、效果规格
- `state/` —— `BattleSnapshot` 状态视图
- `simulation/` —— 预测器接口
- `execution/` —— 本地规则引擎：`rule_kernel`、`effect_interpreter`、`local_predictor`、事件队列、`local_self_play`、MCTS 回滚搜索
- `planning/` —— PUCT、评估函数、搜索价值
- `training/` —— 特征编码、tensorization、Entity Transformer、PPO/GAE、rollout
- `tests/` —— 离线测试（规则、状态、搜索、训练组件）

移除的部分：

- `protocol/`、`runtime/` —— SvHook 命令编码、实机 agent loop
- `execution/replay_*`、`prediction_diff`、`raw_transition_alignment`、`puzzle_*` —— 真实回放采集与对账
- 依赖未公开卡牌或实机数据的测试

## 快速开始

```bash
pip install -e .
python -m pytest decision_engine/tests -q
```

依赖：Python 3.10+、NumPy、PyTorch（CPU 即可跑测试）。

## 数据

`decision_engine/data/public/` 只包含**游戏官网已公开**的内容：

| 文件 | 内容 |
|---|---|
| `card_database.json` | 1–10 弹 + 衍生 token，共 1,591 张卡 |
| `card_templates.json` | 对应的可执行效果模板 |
| `card_effect_specs.json` | 已整理的效果规格（530 条） |
| `card_patches.json` | 手工效果补丁 |
| `decks/` | 190 个公开卡组 |

`data/generated/`（从游戏解包数据生成、包含未公开卡）**不在本仓库中**，也不应被提交。

## 与完整项目的差异

| 能力 | 完整项目 | 本仓库 |
|---|---|---|
| 本地规则引擎 / 自博弈 | ✅ | ✅ |
| PUCT / PPO / 评测 | ✅ | ✅ |
| SvHook 实机通信 | ✅ | ❌ |
| 真实回放采集与对账 | ✅ | ❌ |
| 未公开卡牌 / 11 弹+ | ✅ | ❌ |
| 依赖实机数据的测试 | ✅ | 已移除 |

## 已知限制

- 不能直接驱动游戏客户端；`local_self_play` 只在本地规则引擎内运行。
- 卡池限定 1–10 弹 + token，11 弹及之后的卡不可用。
- 部分依赖未公开卡牌数据的回归测试未包含在本仓库。
- 训练产物（数据集、checkpoint）不随仓库分发，需自行运行生成。

## License

MIT
