# -*- coding: utf-8 -*-
# =====================================================================================
# 【学习版 · 带详细中文注释】structured_policy_value.py
# =====================================================================================
# 本文件是原 structured_policy_value.py 的学习副本，用于逐行理解：
#   (A) 算法原理：这是一个 AlphaZero 风格的「策略-价值双头神经网络」(Policy-Value Net)
#   (B) Python 语法：针对有 Java/C++ 经验但没写过 Python 的读者，
#       在每个语法结构「首次出现」处做对照讲解。
#
# 阅读顺序建议：从上往下读。神经网络部分（class 定义）是核心算法，
# 后半段（_collate_* / train_* / _write_*）是训练流程和落盘逻辑。
#
# ⚠️ 这是临时学习文件，不参与项目运行，读完可删除。
# =====================================================================================


# -------------------------------------------------------------------------------------
# 【语法1】文件顶部的 """ ... """ 三引号字符串
# -------------------------------------------------------------------------------------
# 在 Java/C++ 里注释用 // 或 /* */。Python 里：
#   - 单行注释用 #
#   - 一个「模块 / 类 / 函数」的第一条语句如果是三引号字符串，它就是「文档字符串」(docstring)，
#     类似 Java 的 Javadoc。它是真实的字符串对象，运行时可通过 __doc__ 读取。
# 下面这段就是「模块级 docstring」，描述整个网络的架构。
"""Structured Policy-Value Network with card-level encoders.

Architecture:
    State Encoder（状态编码器）:
        scalar_state (schema-defined d) -> MLP -> state_scalar_emb
        hand_cards[9] -> card_encoder -> mean_pool -> hand_emb
        field_cards[5] -> card_encoder -> mean_pool -> field_emb
        opp_field_cards[5] -> card_encoder -> mean_pool -> opp_field_emb
        state_emb = MLP(concat(state_scalar_emb, hand_emb, field_emb, opp_field_emb))

    Action Encoder（动作编码器）:
        action_scalar -> MLP -> action_scalar_emb
        source_card -> card_encoder -> source_emb
        target_cards[5] -> card_encoder -> mean_pool -> target_emb
        action_emb = MLP(concat(...))

    Card Encoder（共享的单卡编码器）:
        card_id -> embedding(vocab_size, embed_dim)
        card_stats (25d) -> Linear -> stat_emb
        card_emb = ReLU(Linear(concat(embedding, stat_emb)))

    Heads（输出头）:
        policy: 对每个动作算 MLP(concat(state_emb, action_emb)) -> logit
        value : MLP(state_emb) -> tanh -> 标量（当前局面对我方的胜率评估，范围 -1~1）
"""


# -------------------------------------------------------------------------------------
# 【语法2】from __future__ import annotations
# -------------------------------------------------------------------------------------
# 这是 Python 的「未来特性开关」。作用：让本文件里所有类型注解(type hint)
# 都被当成「字符串」延迟求值，而不是在定义时立刻求值。
# 好处：可以在类型注解里写「还没定义的类名」（前向引用），比如函数签名里写
#   model: "StructuredPolicyValueNet"  也能引用后面才定义的类。
# 类比：C++ 里可以前向声明 class Foo; 再在别处用 Foo*。这行让 Python 也能前向引用类型。
# 注意：它必须写在文件最顶部（docstring 之后、其它 import 之前）。
from __future__ import annotations


# -------------------------------------------------------------------------------------
# 【语法3】import 语句
# -------------------------------------------------------------------------------------
# import json         -> 导入标准库 json 模块，用法 json.dumps(...)，类似 Java 的 import。
# from X import A, B  -> 只把 X 模块里的 A、B 名字导入当前命名空间，用时直接写 A / B。
#                        类似 Java 的 import static，或 C++ 的 using X::A;
import json
# dataclass / field 是「数据类」工具（下面【语法6】详解）。
from dataclasses import dataclass, field
# Path 是面向对象的文件路径类，替代手写字符串拼路径。类似 Java 的 java.nio.file.Path。
from pathlib import Path
# Random 是随机数生成器类（可指定种子，保证可复现）。
from random import Random
# typing 里是「类型注解」用的泛型标记（纯给人和工具看，运行时不强制）：
#   Any      = 任意类型（相当于放弃类型检查）
#   Mapping  = 只读的「键值映射」接口（dict 是它的一种实现），类比 Java 的 Map 接口
#   Sequence = 只读的「序列」接口（list/tuple 都是），类比 Java 的 List 接口
from typing import Any, Mapping, Sequence

# 下面是导入本项目内部模块里的具体对象（卡牌词表、特征 schema 版本、维度常量）。
from decision_engine.training.card_vocabulary import CardVocabulary, save_card_vocabulary
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION
from decision_engine.training.tensorization import ACTION_VECTOR_DIM
from decision_engine.training.structured_tensorization import CARD_STAT_DIM


# -------------------------------------------------------------------------------------
# 【语法4】try / except ImportError —— 可选依赖的优雅降级
# -------------------------------------------------------------------------------------
# try/except 就是异常处理，类似 Java 的 try/catch。
# 这里的意图：PyTorch 是重型依赖，不一定装了。所以尝试导入，
#   - 成功 -> 标志位 _TORCH_AVAILABLE = True
#   - 失败(抛 ImportError) -> 置 False，让后面代码可以「没装 torch 也能 import 本文件」，
#     只是在真正训练时再报错。
# 注意变量名前的下划线 _TORCH_AVAILABLE：Python 没有 private 关键字，
#   「单下划线前缀」是约定俗成的「模块内部/私有」标记（仅约定，不强制）。
# import torch.nn as nn  -> as 是起别名，等价 C++ 的 namespace 别名。
#   nn = neural network 模块；F = functional（无状态的函数式算子，如 relu/softmax）。
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# -------------------------------------------------------------------------------------
# 【语法5】装饰器 @dataclass  +  【语法6】数据类
# -------------------------------------------------------------------------------------
# @dataclass 是一个「装饰器」(decorator)。装饰器就是「一个接收类/函数、返回增强版类/函数」
#   的语法糖，写在定义上方，以 @ 开头。类比：Java 的注解 @Override（但 Python 装饰器
#   真的会执行并改写被装饰对象，比注解强大）。
# @dataclass 的作用：自动为这个类生成 __init__ 构造器、__repr__、__eq__ 等样板方法，
#   你只需声明「字段名: 类型 = 默认值」。类比 Java 的 record，或 Lombok 的 @Data。
# frozen=True：生成「不可变」对象（构造后字段不能改），类似 Java record 或 final 字段。
#
# 这个类是「网络超参数配置」——把所有可调的维度/开关集中成一个对象传递。
@dataclass(frozen=True)
class StructuredModelConfig:
    """Configuration for StructuredPolicyValueNet."""
    # 【语法6续】字段声明形式：  名字: 类型 = 默认值
    #   int 是类型注解；= 32 是默认值。有默认值就相当于构造时可省略该参数。

    # ---- 单卡编码器 CardEncoder 相关维度 ----
    card_embed_dim: int = 32       # 卡牌 id 的 embedding 向量维度
    card_stat_hidden: int = 32     # 卡牌数值特征(攻击/血量等)投影后的隐藏维度
    card_emb_dim: int = 64         # 单卡编码器最终输出维度

    # ---- 状态编码器维度 ----
    state_scalar_hidden: int = 128 # 标量状态(PP/EP/血量等)编码隐藏维度
    state_fusion_hidden: int = 256 # 融合所有状态成分后的隐藏维度（= 最终 state_emb 维度）

    # ---- 动作编码器维度 ----
    action_scalar_hidden: int = 64
    action_fusion_hidden: int = 128

    # ---- 两个输出头维度 ----
    policy_hidden: int = 128       # 策略头隐藏层
    value_hidden: int = 128        # 价值头隐藏层

    # ---- 训练相关 ----
    value_loss_weight: float = 1.0 # 价值损失在总损失里的权重（policy_loss + w * value_loss）
    value_head_dropout: float = 0.0# 价值头 dropout 比例(0 表示不用)，防过拟合
    history_enabled: bool = False  # 是否给价值头喂「历史回合」时序信息
    history_length: int = 4        # 历史窗口长度
    seed: int = 42                 # 随机种子，保证结果可复现

    # ---- 集合注意力池化（可选，启用后取代 mean pooling）----
    use_set_attention: bool = False # 是否用自注意力聚合一组卡牌（捕捉卡间 combo 关系）
    num_attention_heads: int = 2    # 多头注意力的头数
    attention_dropout: float = 0.0



# =====================================================================================
# 神经网络模块定义
# =====================================================================================
# -------------------------------------------------------------------------------------
# 【语法7】if _TORCH_AVAILABLE: 里嵌套 class 定义
# -------------------------------------------------------------------------------------
# Python 里 class、def 都是「运行时执行的语句」，不是编译期声明。
# 所以可以把 class 放在 if 里：只有装了 torch 时才定义这些类。
# 这在 Java/C++ 里做不到（类是编译期结构）。这里正是【语法4】降级策略的配套：
# 没装 torch 就不定义网络类，避免 import 阶段报错。
if _TORCH_AVAILABLE:

    # ---------------------------------------------------------------------------------
    # 【概念】nn.Module —— PyTorch 所有网络层/模型的基类
    # ---------------------------------------------------------------------------------
    # 类比：Java 里所有 Servlet 继承 HttpServlet。你继承 nn.Module，
    #   在 __init__ 里声明「子层」(参数)，在 forward() 里写「前向计算」。
    # PyTorch 会自动追踪 forward 中的运算，反向传播时自动求梯度(autograd)。
    #
    # 【语法8】class 子类(父类): 表示继承。等价 Java 的 class A extends B。
    class SetAttentionPooling(nn.Module):
        """对「一组卡牌向量」做自注意力，再做带 mask 的均值池化。

        原理：集合里每张卡都去"注意"其它所有卡，从而建模卡与卡之间的两两关系
        （比如 combo 协同），然后再聚合成一个代表整组的向量。
        """

        # -----------------------------------------------------------------------------
        # 【语法9】def __init__(self, ...): 构造器
        # -----------------------------------------------------------------------------
        # __init__ 是构造方法（类比 Java 构造器）。名字前后双下划线的方法叫「魔术方法/
        #   dunder 方法」，由 Python 在特定时机自动调用。
        # 【语法10】self：每个实例方法的第一个参数必须显式写 self，代表「当前对象」，
        #   等价 Java/C++ 里隐式的 this，但 Python 要求你手动写出来。
        #   调用 obj.method(x) 时，Python 自动把 obj 作为 self 传进去。
        def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.0):
            # 【语法11】super().__init__() 调用父类构造器，等价 Java 的 super()。
            #   继承 nn.Module 时必须先调它，否则参数注册机制不工作。
            super().__init__()
            # nn.MultiheadAttention：PyTorch 内置的多头自注意力层。
            #   batch_first=True 表示张量形状是 [批大小, 序列长, 特征维]。
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            # LayerNorm：层归一化，稳定训练。
            self.norm = nn.LayerNorm(dim)

        # -----------------------------------------------------------------------------
        # 【概念】forward() —— 前向传播
        # -----------------------------------------------------------------------------
        # 你不会直接调 obj.forward(x)，而是调 obj(x)。因为 nn.Module 定义了 __call__
        #   魔术方法，会转而调用 forward。类比：C++ 重载 operator()，让对象像函数一样调用。
        # 【语法12】-> torch.Tensor 是「返回值类型注解」，等价 Java 方法的返回类型，
        #   但 Python 里只是提示，运行时不强制。
        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            x   : [B, N, dim]  一批(B个)集合，每个集合 N 张卡，每张卡 dim 维向量
            mask: [B, N] 布尔   True=该位置是真卡, False=填充占位(padding)
            返回 : [B, dim]     每个集合聚合成一个 dim 维向量
            """
            # 【语法13】多重赋值/解包：一次把 x.shape 这个三元组拆给三个变量。
            #   类比 C++17 结构化绑定 auto [B,N,dim] = ...; 或 Python 的 tuple unpacking。
            B, N, dim = x.shape

            # mask.any(dim=1)：沿第1维(N)做「逻辑或」，得到 [B]，表示每个集合是否至少有一张真卡。
            any_valid = mask.any(dim=1)  # [B]
            # 特殊情况：某些集合可能完全为空（比如场上没随从）。注意力对全 padding 会出 NaN，
            #   所以这里分流处理：全空集合直接给零向量，非空集合走注意力。
            # 【语法14】not / and / or 是逻辑运算符（对应 C++ 的 ! && ||）。
            #   .all() 返回「是否全为 True」。
            if not any_valid.all():
                # 先建一个全零结果张量，形状 [B, dim]。
                result = torch.zeros(B, dim, device=x.device, dtype=x.dtype)
                if any_valid.any():
                    # nonzero(...)[0] 取出「有真卡的那些集合」的下标。
                    valid_idx = any_valid.nonzero(as_tuple=True)[0]
                    # 【语法15】张量的「花式索引」：x[valid_idx] 一次取出多行，
                    #   类似只保留非空的那批集合。
                    valid_x = x[valid_idx]
                    valid_mask = mask[valid_idx]
                    key_padding_mask = ~valid_mask  # ~ 是按位取反：把「有效」翻成「需忽略的padding」
                    # 自注意力：query=key=value=valid_x（自己注意自己）。返回(输出, 注意力权重)，
                    #   用 _ 接住不需要的第二个返回值。
                    attn_out, _ = self.attn(valid_x, valid_x, valid_x, key_padding_mask=key_padding_mask)
                    # 残差连接 + LayerNorm：normed = LayerNorm(输入 + 注意力输出)。
                    normed = self.norm(valid_x + attn_out)
                    # 带 mask 的均值池化：只对真卡求和再除以真卡数。
                    mask_exp = valid_mask.unsqueeze(-1).float()          # [n,N,1]
                    pooled = (normed * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
                    result[valid_idx] = pooled  # 把算好的塞回对应位置
                return result

            # 正常路径：所有集合都至少有一张真卡。
            key_padding_mask = ~mask  # [B, N]，True 表示该位置是 padding 要被注意力忽略
            attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm(x + attn_out)  # 残差 + 归一化

            # 带 mask 的均值池化（把 padding 位置乘 0 排除掉）
            mask_expanded = mask.unsqueeze(-1).float()   # [B,N,1]，unsqueeze 是插入一个新维度
            masked_sum = (x * mask_expanded).sum(dim=1)  # [B, dim]
            count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B,1]，clamp 防止除以 0
            return masked_sum / count


    # ---------------------------------------------------------------------------------
    # CardEncoder：把「一张卡」编码成一个固定长度向量。整个网络里到处复用（shared）。
    # ---------------------------------------------------------------------------------
    class CardEncoder(nn.Module):
        """共享的单卡编码器：id embedding + 数值特征MLP → card_emb."""

        def __init__(self, vocab_size: int, config: StructuredModelConfig):
            super().__init__()
            # nn.Embedding：把「离散 id」映射成「稠密向量」的查表层。
            #   vocab_size = 词表大小(有多少种卡)；每个 id 学一个 card_embed_dim 维向量。
            #   padding_idx=0：id=0 视为「空位」，其向量恒为 0 且不参与训练。
            #   类比：NLP 里把单词 id 变词向量，这里把卡牌 id 变卡向量。
            self.embedding = nn.Embedding(vocab_size, config.card_embed_dim, padding_idx=0)
            # nn.Linear(in, out)：全连接层，做 y = xW + b 的线性变换。
            #   这里把卡牌数值特征(攻击/生命/关键字等 CARD_STAT_DIM 维)投影到隐藏维度。
            self.stat_proj = nn.Linear(CARD_STAT_DIM, config.card_stat_hidden)
            # nn.Sequential：把多个层「串成一条流水线」，前一层输出喂给后一层。
            #   类比：把若干函数复合成一个。这里是 Linear -> ReLU。
            self.fuse = nn.Sequential(
                nn.Linear(config.card_embed_dim + config.card_stat_hidden, config.card_emb_dim),
                nn.ReLU(),  # ReLU 激活函数：max(0, x)，引入非线性
            )

        def forward(self, card_ids: torch.Tensor, card_stats: torch.Tensor) -> torch.Tensor:
            """
            card_ids  : [...] 整数张量（每个元素是一张卡的 id）
            card_stats: [..., CARD_STAT_DIM] 浮点张量（这张卡的数值特征）
            返回       : [..., card_emb_dim]
            """
            emb = self.embedding(card_ids)              # id -> 向量
            stat_emb = F.relu(self.stat_proj(card_stats))  # 数值特征投影 + ReLU
            # torch.cat([a, b], dim=-1)：沿最后一维「拼接」两个向量（concatenate）。
            #   然后送入 fuse 流水线得到最终单卡向量。
            return self.fuse(torch.cat([emb, stat_emb], dim=-1))



    # ---------------------------------------------------------------------------------
    # 主网络：StructuredPolicyValueNet
    # ---------------------------------------------------------------------------------
    # 【算法总览】这是一个「策略-价值双头网络」，输入是「当前局面 + 一组候选动作」，输出：
    #   1) policy(策略)：每个候选动作的 logit(分数)，softmax 后就是「该走哪一步」的概率分布。
    #   2) value(价值)：一个 -1~1 的标量，评估「当前局面对我方有多好」(接近 +1 = 快赢了)。
    # 训练目标来自 PUCT/MCTS 自我对弈：
    #   - policy 学习 MCTS 的访问次数分布（哪些动作被搜索器认为更值得走）
    #   - value  学习对局最终胜负结果
    class StructuredPolicyValueNet(nn.Module):
        """Structured policy-value network with card-level encoding."""

        # -----------------------------------------------------------------------------
        # 【语法16】函数签名里的单独一个 *  ——  强制关键字参数
        # -----------------------------------------------------------------------------
        # def __init__(self, *, vocab_size: int, ...):
        # 星号 * 后面的所有参数都变成「只能用关键字传」(keyword-only)，
        #   即调用时必须写 StructuredPolicyValueNet(vocab_size=100, ...)，
        #   不能按位置写 StructuredPolicyValueNet(100, ...)。
        # 好处：调用处可读性强、防止参数顺序传错。Java/C++ 没有这个特性。
        def __init__(self, *, vocab_size: int, state_scalar_width: int, action_scalar_width: int, config: StructuredModelConfig):
            super().__init__()
            # 把配置和词表大小存成实例属性。
            # 【语法17】Python 不需要在类里预先声明字段，直接 self.xxx = ... 就创建了属性。
            #   类比 JS 对象动态加属性；与 Java/C++ 必须先在类体里声明字段不同。
            self.config = config
            self.vocab_size = vocab_size

            # 共享单卡编码器（手牌/场上/动作源/目标都复用它，参数共享 = 更省、泛化更好）
            self.card_encoder = CardEncoder(vocab_size, config)

            # 是否启用集合注意力池化（否则用简单均值池化）
            if config.use_set_attention:
                self.card_set_attention = SetAttentionPooling(
                    dim=config.card_emb_dim,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                )
                self.crest_set_attention = SetAttentionPooling(
                    dim=config.card_emb_dim,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                )
            else:
                # 【语法18】None 是「空对象」单例，等价 Java 的 null / C++ 的 nullptr。
                #   后面用 `if xxx is not None` 判断是否启用。
                self.card_set_attention = None
                self.crest_set_attention = None

            # 纹章(crest)编码器：结构比普通卡简单（只有 3 维数值特征）
            self.crest_embedding = nn.Embedding(vocab_size, config.card_embed_dim, padding_idx=0)
            self.crest_stat_proj = nn.Linear(3, config.card_stat_hidden)  # CREST_STAT_DIM=3
            self.crest_fuse = nn.Sequential(
                nn.Linear(config.card_embed_dim + config.card_stat_hidden, config.card_emb_dim),
                nn.ReLU(),
            )

            # 标量状态编码器：把 PP/EP/血量/回合数等「整局面级」数值编码
            #   结构：Linear -> ReLU -> Linear -> ReLU （两层 MLP，MLP=多层感知机）
            self.state_scalar_encoder = nn.Sequential(
                nn.Linear(state_scalar_width, config.state_scalar_hidden),
                nn.ReLU(),
                nn.Linear(config.state_scalar_hidden, config.state_scalar_hidden),
                nn.ReLU(),
            )

            # 状态融合层：把「标量向量 + 6 组卡牌集合向量」拼起来再压缩成 state_emb。
            #   6 = 我方手牌、我方场上、对方手牌、对方场上、我方纹章、对方纹章。
            state_cat_dim = config.state_scalar_hidden + 6 * config.card_emb_dim
            self.state_fusion = nn.Sequential(
                nn.Linear(state_cat_dim, config.state_fusion_hidden),
                nn.ReLU(),
                nn.Linear(config.state_fusion_hidden, config.state_fusion_hidden),
                nn.ReLU(),
            )

            # 可选：历史时序编码（只喂给价值头，让它感知「最近几回合怎么走过来的」）
            if config.history_enabled:
                # 位置嵌入：给历史里第 0/1/2/3 个回合各一个可学习的位置向量。
                self.history_position_embedding = nn.Embedding(config.history_length, state_scalar_width)
                # nn.GRU：门控循环单元，一种处理「序列」的 RNN，能把变长历史压成一个向量。
                self.history_encoder = nn.GRU(state_scalar_width, config.state_fusion_hidden, batch_first=True)
                self.history_fusion = nn.Sequential(
                    nn.Linear(2 * config.state_fusion_hidden, config.state_fusion_hidden),
                    nn.ReLU(),
                )

            # 动作标量编码器：编码「动作本身的数值特征」（费用、类型 one-hot 等）
            self.action_scalar_encoder = nn.Sequential(
                nn.Linear(action_scalar_width, config.action_scalar_hidden),
                nn.ReLU(),
                nn.Linear(config.action_scalar_hidden, config.action_scalar_hidden),
                nn.ReLU(),
            )

            # 动作融合层：拼接「动作标量 + 源卡向量 + 目标卡池化向量」
            action_cat_dim = config.action_scalar_hidden + 2 * config.card_emb_dim
            self.action_fusion = nn.Sequential(
                nn.Linear(action_cat_dim, config.action_fusion_hidden),
                nn.ReLU(),
                nn.Linear(config.action_fusion_hidden, config.action_fusion_hidden),
                nn.ReLU(),
            )

            # 策略头：输入 = 拼接(状态向量, 动作向量)，输出 = 1 个 logit(该动作分数)
            self.policy_head = nn.Sequential(
                nn.Linear(config.state_fusion_hidden + config.action_fusion_hidden, config.policy_hidden),
                nn.ReLU(),
                nn.Linear(config.policy_hidden, 1),
            )

            # 价值头：输入 = 状态向量，输出 = 1 个标量，最后 Tanh 压到 [-1, 1]。
            # 【语法19】这里用「先建 list 再按条件 append/extend，最后 *list 展开」的方式
            #   动态拼装网络层，因为 dropout 是可选的。
            value_layers = [
                nn.Linear(config.state_fusion_hidden, config.value_hidden),
                nn.ReLU(),
            ]
            if config.value_head_dropout > 0:
                value_layers.append(nn.Dropout(config.value_head_dropout))  # list.append 加一个元素
            value_layers.extend([                                            # list.extend 加多个元素
                nn.Linear(config.value_hidden, 1),
                nn.Tanh(),  # 双曲正切，把输出压到 (-1, 1)，正好当胜率评估
            ])
            # 【语法20】*value_layers 是「解包」(unpacking)：把 list 里的元素
            #   逐个作为独立参数传给 nn.Sequential(层1, 层2, ...)。
            #   类比 C++ 的可变参数展开 / JS 的扩展运算符 ...arr。
            self.value_head = nn.Sequential(*value_layers)



        # -----------------------------------------------------------------------------
        # 内部辅助方法：编码「一组卡牌」→ 一个向量
        # -----------------------------------------------------------------------------
        # 方法名前缀单下划线 _encode_card_set：约定「内部方法」，外部不该直接调（同【语法4】）。
        def _encode_card_set(self, card_ids: torch.Tensor, card_stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            card_ids  : [B, N]                每批 B 个集合，每集合最多 N 张卡的 id
            card_stats: [B, N, CARD_STAT_DIM] 对应的数值特征
            mask      : [B, N] 布尔            True=真卡, False=padding
            返回       : [B, card_emb_dim]     每个集合聚合成一个向量
            """
            # 先逐卡编码（共享 card_encoder）：[B, N] -> [B, N, card_emb_dim]
            card_embs = self.card_encoder(card_ids, card_stats)

            # 如果启用了注意力池化，用它；否则退回均值池化。
            if self.card_set_attention is not None:
                return self.card_set_attention(card_embs, mask)

            # 兜底：带 mask 的均值池化（把 padding 位乘 0 排除，再除以真卡数）
            mask_expanded = mask.unsqueeze(-1).float()          # [B,N,1]
            masked_sum = (card_embs * mask_expanded).sum(dim=1) # [B, card_emb_dim]
            count = mask_expanded.sum(dim=1).clamp(min=1.0)     # [B,1]
            return masked_sum / count

        def _encode_crest_set(self, crest_ids: torch.Tensor, crest_stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """编码一组纹章。逻辑同上，只是用 crest 专属的 embedding/proj/fuse。"""
            emb = self.crest_embedding(crest_ids)
            stat_emb = F.relu(self.crest_stat_proj(crest_stats))
            crest_embs = self.crest_fuse(torch.cat([emb, stat_emb], dim=-1))

            if self.crest_set_attention is not None:
                return self.crest_set_attention(crest_embs, mask)

            mask_expanded = mask.unsqueeze(-1).float()
            masked_sum = (crest_embs * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            return masked_sum / count

        # -----------------------------------------------------------------------------
        # encode_state：把整个局面编码成一个 state_emb 向量
        # -----------------------------------------------------------------------------
        # 参数 batch 是一个「字典」(Mapping[str, Tensor])：键是字符串特征名，值是张量。
        # 【语法21】字典取值 batch["state_scalar"]，类比 Java Map.get("state_scalar")，
        #   区别是 Python 用方括号，且 key 不存在会抛 KeyError。
        def encode_state(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            """把状态编码成 state_emb，返回 [B, state_fusion_hidden]。"""
            # 标量状态编码
            scalar_emb = self.state_scalar_encoder(batch["state_scalar"])  # [B, state_scalar_hidden]

            # 四组卡牌集合分别编码（我方/对方 × 手牌/场上）
            hand_emb      = self._encode_card_set(batch["hand_card_ids"],      batch["hand_card_stats"],      batch["hand_card_mask"])
            field_emb     = self._encode_card_set(batch["field_card_ids"],     batch["field_card_stats"],     batch["field_card_mask"])
            opp_hand_emb  = self._encode_card_set(batch["opp_hand_card_ids"],  batch["opp_hand_card_stats"],  batch["opp_hand_card_mask"])
            opp_field_emb = self._encode_card_set(batch["opp_field_card_ids"], batch["opp_field_card_stats"], batch["opp_field_card_mask"])

            # 两组纹章集合编码
            crest_emb     = self._encode_crest_set(batch["crest_card_ids"],     batch["crest_stats"],     batch["crest_mask"])
            opp_crest_emb = self._encode_crest_set(batch["opp_crest_card_ids"], batch["opp_crest_stats"], batch["opp_crest_mask"])

            # 拼接全部成分，再过融合层压成最终 state_emb
            state_cat = torch.cat([scalar_emb, hand_emb, field_emb, opp_hand_emb, opp_field_emb, crest_emb, opp_crest_emb], dim=-1)
            return self.state_fusion(state_cat)

        # -----------------------------------------------------------------------------
        # encode_value_state：（可选）给价值头额外融合历史时序信息
        # -----------------------------------------------------------------------------
        def encode_value_state(self, state_emb: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            """把当前状态与「回合边界历史标量」融合，仅供 value 头使用。"""
            # 没开历史就原样返回。
            if not self.config.history_enabled:
                return state_emb
            history = batch["history_state_scalar"]  # [B, L, state_width]
            history_mask = batch["history_mask"]     # [B, L]
            # torch.arange(n)：生成 0,1,...,n-1 的张量（类似 Python 的 range）。
            positions = torch.arange(self.config.history_length, device=history.device)
            # 给每个历史位置加上「位置嵌入」。unsqueeze(0) 在最前插入 batch 维以便广播。
            history = history + self.history_position_embedding(positions).unsqueeze(0)
            # 把无效历史位置乘 0 抹掉。
            history = history * history_mask.unsqueeze(-1).float()
            # 过 GRU 得到每个时间步的编码。第二个返回值(最终隐藏态)不需要，用 _ 接住。
            encoded, _ = self.history_encoder(history)
            # torch.where(条件, A, B)：逐元素三元选择，类似 C 的 cond ? a : b（向量化版）。
            #   这里在找「每个样本最后一个有效历史位置的下标」。
            last_positions = torch.where(
                history_mask,
                torch.arange(self.config.history_length, device=history.device).unsqueeze(0),
                torch.zeros_like(history_mask, dtype=torch.long),
            ).max(dim=1).values
            # 用花式索引取出「每个样本在最后有效位置」的 GRU 输出。
            history_emb = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_positions]
            # 若整段历史都是空的，把 history_emb 归零。
            history_emb = history_emb * history_mask.any(dim=1, keepdim=True).float()
            # 融合当前 state_emb 与历史 emb。
            return self.history_fusion(torch.cat([state_emb, history_emb], dim=-1))

        # -----------------------------------------------------------------------------
        # encode_actions：把「一批候选动作」编码成向量
        # -----------------------------------------------------------------------------
        # 关键技巧：批里每个样本有 N 个候选动作，[B, N, ...]。为了复用只吃 2D 输入的
        #   card_encoder，先把 B、N 两维「压平」成 B*N，算完再 reshape 回 [B, N, ...]。
        def encode_actions(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            """编码动作，返回 [B, max_actions, action_fusion_hidden]。"""
            # action_scalars 形状 [B, N, action_scalar_width]
            B, N, _ = batch["action_scalars"].shape
            # .view(...) 是「重塑形状」，不复制数据（类似 reshape）。-1 表示「该维自动算」。
            action_scalar_flat = batch["action_scalars"].view(B * N, -1)              # [B*N, width]
            action_scalar_emb = self.action_scalar_encoder(action_scalar_flat).view(B, N, -1)

            # 动作的「源卡」（比如出的那张牌 / 进攻的那个随从）
            source_ids_flat = batch["action_source_ids"].view(B * N)
            source_stats_flat = batch["action_source_stats"].view(B * N, -1)
            source_emb = self.card_encoder(source_ids_flat, source_stats_flat).view(B, N, -1)

            # 动作的「目标卡集合」（最多 T 个目标），编码后做均值池化成一个向量
            T = batch["action_target_ids"].shape[2]  # MAX_ACTION_TARGETS
            target_ids_flat = batch["action_target_ids"].view(B * N * T)
            target_stats_flat = batch["action_target_stats"].view(B * N * T, -1)
            target_embs = self.card_encoder(target_ids_flat, target_stats_flat).view(B, N, T, -1)
            target_mask = batch["action_target_mask"].unsqueeze(-1).float()  # [B, N, T, 1]
            target_sum = (target_embs * target_mask).sum(dim=2)              # [B, N, card_emb_dim]
            target_count = target_mask.sum(dim=2).clamp(min=1.0)            # [B, N, 1]
            target_pool = target_sum / target_count

            # 拼接三部分并过动作融合层
            action_cat = torch.cat([action_scalar_emb, source_emb, target_pool], dim=-1)
            action_emb_flat = self.action_fusion(action_cat.view(B * N, -1)).view(B, N, -1)
            return action_emb_flat

        # -----------------------------------------------------------------------------
        # forward：完整前向传播，产出 (策略 logits, 价值)
        # -----------------------------------------------------------------------------
        # 【语法22】返回类型 tuple[torch.Tensor, torch.Tensor]：返回一个二元组。
        #   Python 函数「返回多个值」本质是返回一个 tuple，调用处可解包接收。
        def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            """
            前向传播。返回:
                logits: [B, max_actions]  已 mask 掉非法动作
                value : [B]
            """
            state_emb = self.encode_state(batch)      # [B, state_fusion_hidden]
            action_emb = self.encode_actions(batch)   # [B, max_actions, action_fusion_hidden]

            # 策略：对每个动作，把「同一个 state_emb」和「各自的 action_emb」拼接后过策略头。
            B, N, _ = action_emb.shape
            # unsqueeze(1) 插入动作维；expand(-1, N, -1) 把 state 沿动作维「广播」复制 N 份
            #   （不真复制内存，只是视图）。这样每个动作都能看到同一个状态。
            state_expanded = state_emb.unsqueeze(1).expand(-1, N, -1)     # [B, N, state_dim]
            policy_input = torch.cat([state_expanded, action_emb], dim=-1)  # [B, N, state+action]
            logits = self.policy_head(policy_input.view(B * N, -1)).view(B, N)  # [B, N]

            # 屏蔽非法动作：把 padding/非法位置的 logit 设成极小值 -1e9，
            #   这样后续 softmax 后它们概率≈0。masked_fill 按布尔 mask 填值。
            action_mask = batch["action_mask"]  # [B, N] bool
            logits = logits.masked_fill(~action_mask, -1.0e9)

            # 价值：可选融合历史，再过价值头。squeeze(-1) 去掉最后长度为1的维 -> [B]。
            value_state_emb = self.encode_value_state(state_emb, batch)
            value = self.value_head(value_state_emb).squeeze(-1)  # [B]

            return logits, value



# =====================================================================================
# 训练循环相关
# =====================================================================================
# 【语法23】上面这行 # ─────── 只是普通注释里画的分隔线，无语法意义。


@dataclass(frozen=True)
class StructuredTrainingResult:
    # 训练结束后返回的结果对象：checkpoint 路径 + 指标 + 各 artifact 文件路径。
    checkpoint_path: Path
    train_metrics: dict[str, Any]
    # 【语法24】默认值是「可变对象」的坑：不能直接写 = {}！
    #   因为默认值只在定义时创建一次，多个实例会共享同一个 dict（Java/C++ 没这问题）。
    #   dataclass 的解法：field(default_factory=dict)，表示「每次构造时调用 dict() 新建一个」。
    artifact_paths: dict[str, Path] = field(default_factory=dict)


# -------------------------------------------------------------------------------------
# _collate_structured：把「一批 Python 字典样本」打包成「PyTorch 张量字典」
# -------------------------------------------------------------------------------------
# 「collate」= 整理/归拢。神经网络吃的是规整的定长张量，但每个样本的手牌数/动作数不等长，
#   所以这里做两件事：
#   1) 求出这批里各维度的最大长度（max_actions / max_hand ...）
#   2) 分配全零张量做「padding 容器」，再把每个样本的真实数据填进去，同时记录 mask。
# 这就是深度学习里经典的「变长序列 padding + mask」套路。
def _collate_structured(rows: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, torch.Tensor]:
    """把一批结构化张量化结果整理成张量。"""
    B = len(rows)  # len() 取长度，类比 Java 的 .size()/.length
    # 用第一个样本推断各特征维度。
    first = rows[0]  # 序列按下标取元素，0 基。
    state_width = len(first["state_vector"])
    # 【语法25】三元表达式：  值A if 条件 else 值B
    #   等价 C/Java 的  条件 ? A : B，但 Python 把条件写中间。
    #   这里：若第一个样本有动作向量就取其宽度，否则用默认常量 ACTION_VECTOR_DIM。
    action_width = len(first["action_scalar_vectors"][0]) if first["action_scalar_vectors"] else ACTION_VECTOR_DIM
    # 【语法26】生成器表达式 + max()：
    #   (len(row["..."]) for row in rows) 是「生成器」，惰性逐个产出每个样本的动作数，
    #   max(...) 取最大。类比 Java Stream: rows.stream().mapToInt(...).max()。
    #   注意用的是圆括号（生成器），若用方括号 [] 则是「列表推导式」会一次性建整个列表。
    max_actions = max(len(row["action_scalar_vectors"]) for row in rows)
    card_stat_dim = len(first["hand_card_stats"][0]) if first["hand_card_stats"] else CARD_STAT_DIM
    max_hand = len(first["hand_card_ids"])
    max_field = len(first["field_card_ids"])
    max_opp_field = len(first["opp_field_card_ids"])
    # 【语法27】dict.get(key, 默认)：取键，不存在返回默认（不抛异常），类比 Java Map.getOrDefault。
    #   这里 first.get("opp_hand_card_ids", ()) 取不到就给空元组 ()，再 `or 9` 兜底成 9。
    #   `A or B`：A 为「假值」(空/0/None)时取 B，否则取 A —— Python 常见的默认值写法。
    max_opp_hand = len(first.get("opp_hand_card_ids", ())) or 9
    max_crests = len(first["crest_card_ids"])
    max_targets = len(first["action_target_card_ids"][0]) if first["action_target_card_ids"] else 5

    # ---- 分配全零张量作为 padding 容器 ----
    # torch.zeros(形状..., dtype=数据类型, device=设备)：
    #   dtype=torch.float32(浮点) / torch.long(64位整数, 给 id) / torch.bool(掩码)
    #   device="cpu" 或 "cuda"：张量放哪块硬件上算。
    state_scalar = torch.zeros(B, state_width, dtype=torch.float32, device=device)
    hand_card_ids = torch.zeros(B, max_hand, dtype=torch.long, device=device)
    hand_card_stats = torch.zeros(B, max_hand, card_stat_dim, dtype=torch.float32, device=device)
    hand_card_mask = torch.zeros(B, max_hand, dtype=torch.bool, device=device)
    field_card_ids = torch.zeros(B, max_field, dtype=torch.long, device=device)
    field_card_stats = torch.zeros(B, max_field, card_stat_dim, dtype=torch.float32, device=device)
    field_card_mask = torch.zeros(B, max_field, dtype=torch.bool, device=device)
    opp_field_card_ids = torch.zeros(B, max_opp_field, dtype=torch.long, device=device)
    opp_field_card_stats = torch.zeros(B, max_opp_field, card_stat_dim, dtype=torch.float32, device=device)
    opp_field_card_mask = torch.zeros(B, max_opp_field, dtype=torch.bool, device=device)
    opp_hand_card_ids = torch.zeros(B, max_opp_hand, dtype=torch.long, device=device)
    opp_hand_card_stats = torch.zeros(B, max_opp_hand, card_stat_dim, dtype=torch.float32, device=device)
    opp_hand_card_mask = torch.zeros(B, max_opp_hand, dtype=torch.bool, device=device)

    crest_card_ids = torch.zeros(B, max_crests, dtype=torch.long, device=device)
    crest_stats_t = torch.zeros(B, max_crests, 3, dtype=torch.float32, device=device)  # CREST_STAT_DIM=3
    crest_mask_t = torch.zeros(B, max_crests, dtype=torch.bool, device=device)
    opp_crest_card_ids = torch.zeros(B, max_crests, dtype=torch.long, device=device)
    opp_crest_stats_t = torch.zeros(B, max_crests, 3, dtype=torch.float32, device=device)
    opp_crest_mask_t = torch.zeros(B, max_crests, dtype=torch.bool, device=device)

    action_scalars = torch.zeros(B, max_actions, action_width, dtype=torch.float32, device=device)
    action_mask = torch.zeros(B, max_actions, dtype=torch.bool, device=device)
    action_source_ids = torch.zeros(B, max_actions, dtype=torch.long, device=device)
    action_source_stats = torch.zeros(B, max_actions, card_stat_dim, dtype=torch.float32, device=device)
    action_target_ids = torch.zeros(B, max_actions, max_targets, dtype=torch.long, device=device)
    action_target_stats = torch.zeros(B, max_actions, max_targets, card_stat_dim, dtype=torch.float32, device=device)
    action_target_mask = torch.zeros(B, max_actions, max_targets, dtype=torch.bool, device=device)

    history_length = len(first.get("history_state_vectors", ())) or 4
    history_state_scalar = torch.zeros(B, history_length, state_width, dtype=torch.float32, device=device)
    history_mask = torch.zeros(B, history_length, dtype=torch.bool, device=device)

    policy = torch.zeros(B, max_actions, dtype=torch.float32, device=device)  # 训练目标：动作概率分布
    value = torch.zeros(B, dtype=torch.float32, device=device)                # 训练目标：胜负标签
    value_mask = torch.zeros(B, dtype=torch.bool, device=device)             # 该样本有没有 value 标签

    # ---- 遍历每个样本，把真实数据填进上面的容器 ----
    # 【语法28】enumerate(rows)：同时给出「下标 b」和「元素 row」。
    #   等价 Java 里手动 for(int b=0; b<n; b++){ row=rows.get(b); }。
    for b, row in enumerate(rows):
        # 标量状态
        sv = row["state_vector"]
        # 【语法29】切片赋值 state_scalar[b, :len(sv)]：
        #   `:len(sv)` 是切片，表示「第 0 到 len(sv)-1 列」。把该样本标量填进第 b 行前段，
        #   后面没填的保持 0（就是 padding）。
        state_scalar[b, :len(sv)] = torch.tensor(sv, dtype=torch.float32)

        # 手牌：zip 把三个等长序列「按位配对」一起遍历。
        # 【语法30】zip(a, b, c)：把多个可迭代对象「打包」成元组序列 (a0,b0,c0),(a1,b1,c1)...
        #   再配合 enumerate 得到下标 i 与元组 (cid,stats,m)。这里 (cid,stats,m) 是解包。
        for i, (cid, stats, m) in enumerate(zip(row["hand_card_ids"], row["hand_card_stats"], row["hand_card_mask"])):
            hand_card_ids[b, i] = int(cid)                                  # 显式转 int
            hand_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            hand_card_mask[b, i] = bool(m)                                  # 显式转 bool

        # 场上随从（我方）
        for i, (cid, stats, m) in enumerate(zip(row["field_card_ids"], row["field_card_stats"], row["field_card_mask"])):
            field_card_ids[b, i] = int(cid)
            field_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            field_card_mask[b, i] = bool(m)

        # 场上随从（对方）
        for i, (cid, stats, m) in enumerate(zip(row["opp_field_card_ids"], row["opp_field_card_stats"], row["opp_field_card_mask"])):
            opp_field_card_ids[b, i] = int(cid)
            opp_field_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            opp_field_card_mask[b, i] = bool(m)

        # 对方手牌（完美信息模式下才有；用 .get 兜底空元组，避免 KeyError）
        for i, (cid, stats, m) in enumerate(zip(row.get("opp_hand_card_ids", ()), row.get("opp_hand_card_stats", ()), row.get("opp_hand_card_mask", ()))):
            opp_hand_card_ids[b, i] = int(cid)
            opp_hand_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            opp_hand_card_mask[b, i] = bool(m)

        # 纹章（我方 / 对方）
        for i, (cid, stats, m) in enumerate(zip(row["crest_card_ids"], row["crest_stats"], row["crest_mask"])):
            crest_card_ids[b, i] = int(cid)
            crest_stats_t[b, i] = torch.tensor(stats, dtype=torch.float32)
            crest_mask_t[b, i] = bool(m)
        for i, (cid, stats, m) in enumerate(zip(row["opp_crest_card_ids"], row["opp_crest_stats"], row["opp_crest_mask"])):
            opp_crest_card_ids[b, i] = int(cid)
            opp_crest_stats_t[b, i] = torch.tensor(stats, dtype=torch.float32)
            opp_crest_mask_t[b, i] = bool(m)

        # 历史回合标量（从旧到新）
        for i, vector in enumerate(row.get("history_state_vectors", ())):
            history_state_scalar[b, i] = torch.tensor(vector, dtype=torch.float32)
        for i, present in enumerate(row.get("history_mask", ())):
            history_mask[b, i] = bool(present)

        # 动作：逐个填充。这里用 range 而非 zip，因为要嵌套处理目标。
        n_actions = len(row["action_scalar_vectors"])
        # 【语法31】range(n)：产出 0..n-1 的整数序列，类比 for(int i=0;i<n;i++)。
        for i in range(n_actions):
            action_scalars[b, i] = torch.tensor(row["action_scalar_vectors"][i], dtype=torch.float32)
            action_mask[b, i] = True  # 这一格是真动作
            action_source_ids[b, i] = int(row["action_source_card_ids"][i])
            action_source_stats[b, i] = torch.tensor(row["action_source_card_stats"][i], dtype=torch.float32)
            # 该动作的目标卡（最多 max_targets 个），越界的保持 0/False
            for j in range(max_targets):
                if j < len(row["action_target_card_ids"][i]):
                    action_target_ids[b, i, j] = int(row["action_target_card_ids"][i][j])
                    action_target_stats[b, i, j] = torch.tensor(row["action_target_card_stats"][i][j], dtype=torch.float32)
                    action_target_mask[b, i, j] = bool(row["action_target_mask"][i][j])

            policy[b, i] = row["policy_target"][i]  # 该动作的目标概率(来自 MCTS 访问次数)

        # value 标签是可选的（有些样本没打分）
        vt = row.get("value_target")
        if vt is not None:
            value[b] = float(vt)
            value_mask[b] = True  # 标记「这个样本参与 value 损失计算」

    # 返回一个大字典，键名与网络 forward 里用的键一一对应。
    # 【语法32】字典字面量 { "键": 值, ... }，类比 Java Map.of(...) 或 JSON 对象。
    return {
        "state_scalar": state_scalar,
        "hand_card_ids": hand_card_ids,
        "hand_card_stats": hand_card_stats,
        "hand_card_mask": hand_card_mask,
        "field_card_ids": field_card_ids,
        "field_card_stats": field_card_stats,
        "field_card_mask": field_card_mask,
        "opp_field_card_ids": opp_field_card_ids,
        "opp_field_card_stats": opp_field_card_stats,
        "opp_field_card_mask": opp_field_card_mask,
        "opp_hand_card_ids": opp_hand_card_ids,
        "opp_hand_card_stats": opp_hand_card_stats,
        "opp_hand_card_mask": opp_hand_card_mask,
        "crest_card_ids": crest_card_ids,
        "crest_stats": crest_stats_t,
        "crest_mask": crest_mask_t,
        "opp_crest_card_ids": opp_crest_card_ids,
        "opp_crest_stats": opp_crest_stats_t,
        "opp_crest_mask": opp_crest_mask_t,
        "action_scalars": action_scalars,
        "action_mask": action_mask,
        "action_source_ids": action_source_ids,
        "action_source_stats": action_source_stats,
        "action_target_ids": action_target_ids,
        "action_target_stats": action_target_stats,
        "action_target_mask": action_target_mask,
        "history_state_scalar": history_state_scalar,
        "history_mask": history_mask,
        "policy": policy,
        "value": value,
        "value_mask": value_mask,
    }



# -------------------------------------------------------------------------------------
# _batches：把数据切成一批批喂给网络（生成器函数）
# -------------------------------------------------------------------------------------
# 【语法33】yield —— 生成器函数
# 只要函数体里出现 yield，这个函数就变成「生成器函数」。调用它不会立刻执行，
#   而是返回一个「惰性迭代器」；每次 for 循环拿一个值时才执行到下一个 yield 处暂停。
# 好处：不用一次性把所有批次都建出来占内存，边算边给（惰性/流式）。
#   类比：Java 的 Iterator + 惰性求值，或 C# 的 yield return。
def _batches(items, batch_size: int):
    """按 batch_size 产出批次。list/tuple 用切片；其它可迭代对象流式累积。"""
    # isinstance(x, (A, B))：判断 x 是否是 A 或 B 类型，类比 Java 的 instanceof。
    if isinstance(items, (list, tuple)):
        for i in range(0, len(items), batch_size):  # range(start, stop, step)：带步长
            # items[i:i+batch_size] 是切片，取一段连续子列表。
            yield items[i:i + batch_size]
        return  # 生成器里的 return 表示「到此结束，不再产出」（相当于停止迭代）
    # 若是惰性数据源（不能切片），就手动累积到 batch_size 再吐出。
    # 【语法34】变量注解 batch: list = []：给变量标注类型（可选，仅提示）。
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []  # 重置，开始攒下一批
    if batch:  # 最后一批不足 batch_size 也要吐出。空 list 是「假值」，非空为「真」。
        yield batch


# -------------------------------------------------------------------------------------
# _load_warm_start：热启动——从上一轮训练的 checkpoint 继承权重
# -------------------------------------------------------------------------------------
# 「warm start」= 不从随机初始化开始训练，而是加载上次训练好的权重接着练。
# 这里还处理了「词表增长」：新版本可能加了新卡（vocab 变大），要求老卡的 embedding 行号
#   必须保持不变（append-only），这样老权重能对齐，新卡的 embedding 行是新初始化的。
# 【语法35】函数签名里 model: "StructuredPolicyValueNet" 用了字符串前向引用
#   （得益于文件顶部的 from __future__ import annotations）。
# 参数里 Path | None：这是「联合类型」，表示「Path 或 None」，类比 Java 的 @Nullable Path。
#   竖线 | 是 Python 3.10+ 的联合类型写法。
def _load_warm_start(
    *,
    model: "StructuredPolicyValueNet",
    initial_checkpoint: Path | None,
    vocabulary: CardVocabulary | None,
    state_scalar_width: int,
    action_scalar_width: int,
    config: StructuredModelConfig,
    device: str,
) -> dict[str, Any]:
    """加载兼容的权重与「只增不改」的 embedding 行。"""
    # 没给初始 checkpoint 就不热启动，返回一个说明字典。
    if initial_checkpoint is None:
        return {"enabled": False, "reason": "no_initial_checkpoint"}
    if vocabulary is None:
        # raise 抛异常，类比 Java 的 throw。ValueError 是内置异常类型。
        raise ValueError("warm-start requires an explicit CardVocabulary")
    if not initial_checkpoint.exists():  # Path.exists() 判断文件是否存在
        # f"...{变量}..." 是 f-string（格式化字符串），花括号里可嵌表达式。见下方【语法36】。
        raise FileNotFoundError(f"initial checkpoint not found: {initial_checkpoint}")

    # torch.load 读取保存的 checkpoint（一个字典）。weights_only=True 是安全选项。
    checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=True)
    # 校验：老 checkpoint 的特征宽度必须和当前一致，否则结构对不上。
    # checkpoint.get(...) or 0 ：取不到或为 None 时兜底 0，再转 int 比较。
    if int(checkpoint.get("state_scalar_width") or 0) != state_scalar_width:
        raise ValueError("warm-start state scalar width mismatch")
    if int(checkpoint.get("action_scalar_width") or 0) != action_scalar_width:
        raise ValueError("warm-start action scalar width mismatch")

    # 从 checkpoint 里重建「上一次的配置对象」。
    # 【语法37】字典推导式  {k: v for k, v in ... if 条件}
    #   类比列表推导式，但产出 dict。这里过滤出「属于 StructuredModelConfig 合法字段」的键值。
    #   __dataclass_fields__ 是 dataclass 自动生成的字段元信息表。
    #   **{...} 是「字典解包」：把字典展开成 key=value 形式的关键字参数传给构造器。
    previous_config = StructuredModelConfig(**{
        key: value for key, value in dict(checkpoint.get("config") or {}).items()
        if key in StructuredModelConfig.__dataclass_fields__
    })
    # 结构性字段：这些必须完全一致，否则老权重无法复用。
    architecture_fields = (
        "card_embed_dim", "card_stat_hidden", "card_emb_dim", "state_scalar_hidden",
        "state_fusion_hidden", "action_scalar_hidden", "action_fusion_hidden",
        "policy_hidden", "value_hidden", "value_head_dropout", "history_enabled",
        "history_length", "use_set_attention", "num_attention_heads", "attention_dropout",
    )
    # 【语法38】any(生成器)：只要有一个为 True 就返回 True，类比 Java Stream.anyMatch。
    #   getattr(obj, name)：按「字符串名字」动态取属性，类比 Java 反射 obj.getClass().getField(name)。
    if any(getattr(previous_config, field) != getattr(config, field) for field in architecture_fields):
        raise ValueError("warm-start model architecture mismatch")

    # 【语法39】函数体内部的 import：Python 允许在函数里 import（延迟导入），
    #   常用于打破循环依赖或只在需要时才加载重模块。
    from decision_engine.training.card_vocabulary import build_legacy_card_vocabulary, load_card_vocabulary

    # 找上一次训练的词表文件。Path 支持用 / 运算符拼路径（重载了 __truediv__）。
    previous_vocab_path = initial_checkpoint.parent / "card_vocabulary.json"
    if previous_vocab_path.exists():
        previous_vocabulary = load_card_vocabulary(previous_vocab_path)
        previous_vocabulary_mode = "base_card_id"
    else:
        # 兼容老格式：从卡牌数据库按顺序重建「遗留词表」。
        # Path(__file__)：__file__ 是当前文件路径的内置变量。.resolve() 取绝对路径，
        #   .parents[1] 取「上上级目录」（parents[0]=父目录, [1]=祖父目录）。
        database_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
        previous_vocabulary = build_legacy_card_vocabulary(database_path)
        if previous_vocabulary.vocab_size != int(checkpoint.get("vocab_size") or 0):
            raise ValueError("legacy warm-start checkpoint is incompatible with current card database")
        previous_vocabulary_mode = "legacy_database_order"

    # 关键约束：老词表里每张卡的行号，必须在新词表里保持不变（append-only）。
    # 【语法40】dict.items() 返回 (键,值) 序列，用于遍历字典。
    for base_card_id, old_index in previous_vocabulary.base_card_id_to_index.items():
        if vocabulary.base_card_id_to_index.get(base_card_id) != old_index:
            raise ValueError("warm-start vocabulary must preserve every prior base_card_id index")

    previous_state = checkpoint.get("model_state_dict")  # 老模型的权重字典(参数名->张量)
    if not isinstance(previous_state, Mapping):
        raise ValueError("warm-start checkpoint lacks model_state_dict")
    current_state = model.state_dict()  # 当前(随机初始化的)模型权重字典
    # 集合字面量 {a, b}：这是 set（无序不重复集合），注意和 dict 区分（dict 是 {k:v}）。
    embedding_keys = {"card_encoder.embedding.weight", "crest_embedding.weight"}
    inherited_keys = 0
    # 逐个参数从老权重拷贝到当前权重。
    for key, value in previous_state.items():
        if key not in current_state:  # `in` 判断成员，类比 Java containsKey
            raise ValueError(f"warm-start checkpoint has unknown parameter: {key}")
        if key in embedding_keys:
            # embedding 允许「行数增加」(加了新卡)，但每行维度必须一致，且不能变少。
            # value.shape[1:] 是「除第0维外的形状」，切片从下标1到末尾。
            if value.shape[1:] != current_state[key].shape[1:] or value.shape[0] > current_state[key].shape[0]:
                raise ValueError(f"warm-start embedding shape mismatch: {key}")
            # 只拷贝老词表覆盖的前 N 行；新卡的行保持随机初始化。
            # copy_ 结尾的下划线：PyTorch 约定「原地(in-place)操作」，直接改张量内容。
            current_state[key][:value.shape[0]].copy_(value)
        elif value.shape != current_state[key].shape:  # elif = else if
            raise ValueError(f"warm-start parameter shape mismatch: {key}")
        else:
            current_state[key].copy_(value)  # 形状一致，整块拷贝
        inherited_keys += 1  # Python 没有 ++，用 += 1
    model.load_state_dict(current_state)  # 把拼好的权重装回模型
    # 返回热启动的元信息（记录到训练日志/artifact）。
    return {
        "enabled": True,
        "initial_checkpoint": str(initial_checkpoint),
        "previous_vocabulary_digest": previous_vocabulary.digest,
        "previous_vocabulary_mode": previous_vocabulary_mode,
        "vocabulary_digest": vocabulary.digest,
        "inherited_base_cards": len(previous_vocabulary.base_card_id_to_index),
        "added_base_cards": vocabulary.vocab_size - previous_vocabulary.vocab_size,
        "inherited_parameter_keys": inherited_keys,
        "optimizer_state_inherited": False,
    }



# -------------------------------------------------------------------------------------
# train_structured_policy_value：训练主函数（整个文件的入口逻辑）
# -------------------------------------------------------------------------------------
# 【算法：标准的监督训练循环】
#   for 每个 epoch(把整个数据集过一遍):
#       for 每个 batch:
#           前向 -> 算损失 -> 反向传播求梯度 -> 优化器更新权重
#       在验证集上评估 -> 记录指标 -> Early Stopping(验证不再变好就提前停)
#   最后：恢复最优权重 -> 保存 checkpoint 和各种 artifact。
def train_structured_policy_value(
    *,
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],  # {"train": [...], "val": [...]}
    output_dir: Path,
    run_id: str,
    vocab_size: int,
    vocabulary: CardVocabulary | None = None,
    initial_checkpoint: Path | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,       # 1e-3 = 0.001，科学计数法
    device: str = "cpu",
    config: StructuredModelConfig | None = None,
) -> StructuredTrainingResult:
    """训练结构化 policy-value 模型。"""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for structured training")

    # config 为空就用默认配置。`A or B`：A 是 None(假值)时取 B。
    cfg = config or StructuredModelConfig()
    # 固定随机种子，保证每次训练可复现（权重初始化、shuffle 顺序都确定）。
    torch.manual_seed(cfg.seed)

    from decision_engine.training.sharded_dataset import ShardedRowDataset

    raw_train = dataset.get("train", ())
    raw_val = dataset.get("val", ())
    # 惰性分片数据集保持流式；普通序列一次性转成 list（放进内存）。
    train_rows = raw_train if isinstance(raw_train, ShardedRowDataset) else list(raw_train)
    val_rows = raw_val if isinstance(raw_val, ShardedRowDataset) else list(raw_val)
    if len(train_rows) == 0:
        raise ValueError("Need at least one training sample")

    # 推断特征宽度：惰性数据集用 first_row() 取一条样，普通序列取 [0]。
    _fr = train_rows.first_row() if isinstance(train_rows, ShardedRowDataset) else train_rows[0]
    state_scalar_width = len(_fr["state_vector"])
    action_scalar_width = len(_fr["action_scalar_vectors"][0]) if _fr["action_scalar_vectors"] else ACTION_VECTOR_DIM

    # 实例化模型并搬到目标设备。.to(device) 把所有参数移到 cpu/cuda。
    model = StructuredPolicyValueNet(
        vocab_size=vocab_size,
        state_scalar_width=state_scalar_width,
        action_scalar_width=action_scalar_width,
        config=cfg,
    ).to(device)

    if vocabulary is not None and vocabulary.vocab_size != vocab_size:
        raise ValueError("vocabulary vocab_size must match trainer vocab_size")
    # 尝试热启动（若给了 initial_checkpoint）。
    warm_start = _load_warm_start(
        model=model,
        initial_checkpoint=initial_checkpoint,
        vocabulary=vocabulary,
        state_scalar_width=state_scalar_width,
        action_scalar_width=action_scalar_width,
        config=cfg,
        device=device,
    )

    # 优化器：Adam，负责根据梯度更新权重。model.parameters() 是所有可训练参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = Random(cfg.seed)  # 独立的随机器，用于 shuffle

    output_dir.mkdir(parents=True, exist_ok=True)  # 建输出目录，parents=True 递归建，exist_ok 已存在不报错
    epoch_reports: list[dict[str, Any]] = []  # 记录每个 epoch 的指标

    # ---- Early Stopping（早停）状态 ----
    best_val_loss = float("inf")  # 正无穷大，作为「初始最优」，任何真实损失都比它小
    best_epoch = 0
    best_state_dict = None
    patience = 5          # 容忍验证损失连续多少个 epoch 不改善
    patience_counter = 0

    # ============ 主训练循环 ============
    for epoch in range(1, epochs + 1):  # 从 1 到 epochs
        # 每个 epoch 前打乱训练数据顺序（避免模型记住样本顺序）。
        if isinstance(train_rows, ShardedRowDataset):
            train_rows.shuffle(rng)
        else:
            rng.shuffle(train_rows)  # 原地打乱 list
        model.train()  # 切到「训练模式」（启用 dropout 等）
        train_losses: list[dict[str, float]] = []

        for batch_rows in _batches(train_rows, batch_size):
            batch = _collate_structured(batch_rows, device=device)  # 打包成张量
            optimizer.zero_grad()          # 清空上一步累积的梯度（PyTorch 梯度默认累加）
            logits, values = model(batch)  # 前向传播（等价 model.forward(batch)）

            # ---- 策略损失：带软标签的交叉熵 ----
            # log_softmax：先 softmax 转概率再取对数，数值更稳定。
            log_probs = F.log_softmax(logits, dim=-1)
            # 交叉熵 = -Σ 目标概率 * log(预测概率)。这里目标是 MCTS 的访问分布(软标签)。
            #   .sum(dim=-1) 对动作维求和，.mean() 对 batch 求平均。
            #   前面的负号是交叉熵定义里的。
            policy_loss = -(batch["policy"] * log_probs).sum(dim=-1).mean()

            # ---- 价值损失：只对「有 value 标签」的样本算 MSE(均方误差) ----
            value_loss = torch.tensor(0.0, device=device)
            if batch["value_mask"].any():
                # values[mask]：用布尔 mask 筛出有标签的样本，再算预测与真值的 MSE。
                value_loss = F.mse_loss(values[batch["value_mask"]], batch["value"][batch["value_mask"]])

            # 总损失 = 策略损失 + 权重 * 价值损失
            total_loss = policy_loss + cfg.value_loss_weight * value_loss
            total_loss.backward()  # 反向传播：自动对所有参数求梯度
            optimizer.step()       # 用梯度更新权重

            # .item()：把「单元素张量」转成 Python 浮点数（脱离计算图，省内存）。
            train_losses.append({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "total_loss": total_loss.item(),
            })

        # ---- 本 epoch 训练损失汇总（对所有 batch 求平均）----
        # sum(生成器) / len(...)：先累加再除，得平均。
        avg_policy = sum(d["policy_loss"] for d in train_losses) / len(train_losses)
        avg_value = sum(d["value_loss"] for d in train_losses) / len(train_losses)
        avg_total = sum(d["total_loss"] for d in train_losses) / len(train_losses)

        # ---- 验证集评估（分批，按样本数加权聚合，避免一次性张量化 OOM）----
        val_metrics = {}
        if len(val_rows) > 0:
            model.eval()  # 切到「评估模式」（关闭 dropout）
            eval_batch_size = max(batch_size, 256)
            policy_loss_sum = 0.0
            policy_sample_count = 0
            value_sq_error_sum = 0.0
            value_sample_count = 0
            # 【语法41】with torch.no_grad(): —— 上下文管理器
            #   with 块进入时做「准备」、退出时自动「收尾」，类比 Java 的 try-with-resources
            #   或 C++ 的 RAII。torch.no_grad() 表示「块内不追踪梯度」，评估时省内存加速。
            with torch.no_grad():
                for val_chunk in _batches(val_rows, eval_batch_size):
                    val_batch = _collate_structured(val_chunk, device=device)
                    val_logits, val_values = model(val_batch)
                    val_log_probs = F.log_softmax(val_logits, dim=-1)
                    per_sample_ce = -(val_batch["policy"] * val_log_probs).sum(dim=-1)
                    policy_loss_sum += per_sample_ce.sum().item()      # 累加损失总和
                    policy_sample_count += per_sample_ce.shape[0]      # 累加样本数
                    mask = val_batch["value_mask"]
                    if mask.any():
                        diff = val_values[mask] - val_batch["value"][mask]
                        value_sq_error_sum += (diff * diff).sum().item()
                        value_sample_count += int(mask.sum().item())
            val_policy_loss = policy_loss_sum / max(1, policy_sample_count)
            val_value_loss = value_sq_error_sum / value_sample_count if value_sample_count else 0.0
            val_metrics = {"val_policy_loss": val_policy_loss, "val_value_loss": val_value_loss}

        # 【语法42】字典里的 **val_metrics：把 val_metrics 的键值「展开合并」进这个新字典。
        #   类比 JS 的 {...obj}。这样有验证集时报告里就含 val_* 字段，没有则不含。
        report = {"epoch": epoch, "train_policy_loss": avg_policy, "train_value_loss": avg_value, "train_total_loss": avg_total, **val_metrics}
        epoch_reports.append(report)

        # ---- Early Stopping 判定 ----
        if val_rows:
            val_total = val_metrics.get("val_policy_loss", 0) + cfg.value_loss_weight * val_metrics.get("val_value_loss", 0)
            if val_total < best_val_loss:
                # 变得更好：记录最优，深拷贝一份权重快照。
                best_val_loss = val_total
                best_epoch = epoch
                # {k: v.clone() for ...}：字典推导式，clone() 复制张量避免后续被改。
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                # 没变好：耐心计数 +1，超过 patience 就提前停。
                patience_counter += 1
                if patience_counter >= patience:
                    # print + f-string；:.4f 是格式说明符，保留 4 位小数。
                    print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_total: {best_val_loss:.4f})")
                    break  # 跳出 epoch 循环

        # 每 10 个 epoch 或最后一个打印一次进度。% 是取模。
        if epoch % 10 == 0 or epoch == epochs:
            vp_str = f"{val_metrics.get('val_policy_loss', 0):.4f}" if val_metrics else "?"
            print(f"  epoch {epoch}: train_policy={avg_policy:.4f} train_value={avg_value:.4f} val_policy={vp_str}")

    # 训练结束：若用过早停，恢复到「最优 epoch」的权重（而非最后一轮）。
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"  Restored best model from epoch {best_epoch}")

    # ---- 保存 checkpoint ----
    if vocabulary is not None:
        save_card_vocabulary(output_dir / "card_vocabulary.json", vocabulary)
    checkpoint_path = output_dir / "model.pt"
    # torch.save(对象, 路径)：把字典(含模型权重和元信息)序列化落盘。
    torch.save({
        "schema": "StructuredPolicyValueCheckpoint",
        "version": 1,
        "run_id": run_id,
        "vocab_size": vocab_size,
        "card_identity_kind": "base_card_id",
        "card_vocabulary_digest": vocabulary.digest if vocabulary is not None else None,
        "state_scalar_width": state_scalar_width,
        "action_scalar_width": action_scalar_width,
        "config": {
            "card_embed_dim": cfg.card_embed_dim,
            "card_stat_hidden": cfg.card_stat_hidden,
            "card_emb_dim": cfg.card_emb_dim,
            "state_scalar_hidden": cfg.state_scalar_hidden,
            "state_fusion_hidden": cfg.state_fusion_hidden,
            "action_scalar_hidden": cfg.action_scalar_hidden,
            "action_fusion_hidden": cfg.action_fusion_hidden,
            "policy_hidden": cfg.policy_hidden,
            "value_hidden": cfg.value_hidden,
            "value_head_dropout": cfg.value_head_dropout,
            "history_enabled": cfg.history_enabled,
            "history_length": cfg.history_length,
            "use_set_attention": cfg.use_set_attention,
            "num_attention_heads": cfg.num_attention_heads,
            "attention_dropout": cfg.attention_dropout,
            "value_loss_weight": cfg.value_loss_weight,
        },
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)

    # ---- 汇总训练指标 ----
    # best_epoch - 1 是「最优 epoch 在列表里的下标」（epoch 从 1 计，列表从 0 计）。
    metrics = {
        "run_id": run_id,
        "epochs": epoch_reports,
        "final_train_policy_loss": epoch_reports[-1]["train_policy_loss"],  # [-1] 是「最后一个」
        "final_train_value_loss": epoch_reports[-1]["train_value_loss"],
        "final_val_policy_loss": epoch_reports[best_epoch - 1].get("val_policy_loss") if best_epoch > 0 else epoch_reports[-1].get("val_policy_loss"),
        "final_val_value_loss": epoch_reports[best_epoch - 1].get("val_value_loss") if best_epoch > 0 else epoch_reports[-1].get("val_value_loss"),
        "best_epoch": best_epoch,
        "early_stopped": patience_counter >= patience,
        "vocab_size": vocab_size,
        "card_identity_kind": "base_card_id",
        "card_vocabulary_digest": vocabulary.digest if vocabulary is not None else None,
        "warm_start": warm_start,
        "state_scalar_width": state_scalar_width,
        "action_scalar_width": action_scalar_width,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }

    # ---- 产出 artifact bundle（runtime 加载模型所需的多个 json）----
    artifact_paths = _write_artifact_bundle(
        output_dir=output_dir,
        run_id=run_id,
        metrics=metrics,
        vocab_size=vocab_size,
        state_scalar_width=state_scalar_width,
        action_scalar_width=action_scalar_width,
        train_samples=len(train_rows),
        val_samples=len(val_rows),
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        # dataset.get(...) or "" 兜底空串；str(...) 转字符串。
        input_dir=str(dataset.get("_source_path", "") or ""),
        vocabulary=vocabulary,
        warm_start=warm_start,
    )

    # 构造并返回结果对象（dataclass）。
    return StructuredTrainingResult(checkpoint_path=checkpoint_path, train_metrics=metrics, artifact_paths=artifact_paths)



# -------------------------------------------------------------------------------------
# _write_artifact_bundle：训练完成后落盘一整套 runtime 需要的元数据 json
# -------------------------------------------------------------------------------------
# 这部分没有新算法，纯粹是「把训练结果整理成若干标准 json 文件」，供 runtime 加载模型时读取。
# 语法上都是前面见过的：dataclass 参数、字典字面量、Path / 运算符拼路径。
def _write_artifact_bundle(
    *,
    output_dir: Path,
    run_id: str,
    metrics: dict[str, Any],
    vocab_size: int,
    state_scalar_width: int,
    action_scalar_width: int,
    train_samples: int,
    val_samples: int,
    epochs: int,
    checkpoint_path: Path,
    input_dir: str,
    vocabulary: CardVocabulary | None,
    warm_start: Mapping[str, Any],
) -> dict[str, Path]:
    """训练完成后自动产出完整 artifact bundle（5 个 json 文件）。"""
    paths: dict[str, Path] = {}  # 收集各文件路径返回

    # 1. train_metrics.json —— 训练指标
    train_metrics_path = output_dir / "train_metrics.json"
    _write_json(train_metrics_path, metrics)
    paths["train_metrics"] = train_metrics_path

    # 2. model_artifact.json —— 模型描述（runtime 按此了解输入/输出宽度、词表等）
    model_artifact_path = output_dir / "model_artifact.json"
    model_artifact = {
        "schema": "PolicyValueModelArtifact",
        "version": 1,
        "run_id": run_id,
        "backend": "torch_structured",
        "model_kind": "policy_value_dual_head",
        "input_tensorization_schema": "StructuredTensorization",
        "dataset_schema": "StructuredPolicyValueDataset",
        "state_vector_width": state_scalar_width,
        "action_vector_width": action_scalar_width,
        "policy_head_width": action_scalar_width,
        "value_head_output": "scalar",
        "feature_schema_version": TRAINING_FEATURE_SCHEMA_VERSION,
        "metadata": {
            "checkpoint": str(checkpoint_path.name),  # Path.name 取文件名(不含目录)
            "candidate": True,
            "source": run_id,
            "vocab_size": vocab_size,
            "card_identity_kind": "base_card_id",
            "card_vocabulary_file": "card_vocabulary.json" if vocabulary is not None else None,
            "card_vocabulary_digest": vocabulary.digest if vocabulary is not None else None,
            "warm_start": dict(warm_start),  # dict(...) 复制一份，避免引用外部对象
        },
    }
    _write_json(model_artifact_path, model_artifact)
    paths["model_artifact"] = model_artifact_path

    # 3. run_manifest.json —— 训练运行清单
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        "schema": "PolicyValueTrainingRunManifest",
        "version": 1,
        "run_id": run_id,
        "input_dir": input_dir,
        "output_dir": str(output_dir),
        "tensor_dataset_summary": {
            "schema": "TensorDatasetSummary",
            "total_samples": train_samples + val_samples,
            "state_vector_width": state_scalar_width,
            "action_vector_width": action_scalar_width,
        },
        "training_summary": {
            "backend": "torch_structured",
            "candidate": True,
            "epochs": epochs,
            "vocab_size": vocab_size,
            "train_samples": train_samples,
            "val_samples": val_samples,
        },
    }
    _write_json(run_manifest_path, run_manifest)
    paths["run_manifest"] = run_manifest_path

    # 4. candidate_report.json —— 候选模型质量报告
    candidate_report_path = output_dir / "candidate_report.json"
    candidate_report = {
        "schema": "PolicyValueCandidateRunReport",
        "version": 1,
        "run_id": run_id,
        "run_dir": str(output_dir),
        "backend": "torch_structured",
        "artifact_status": "trained",
        "quality_summary": {
            "policy_loss_final": metrics.get("final_train_policy_loss"),
            "value_loss_final": metrics.get("final_train_value_loss"),
            "val_policy_loss_final": metrics.get("final_val_policy_loss"),
            "val_value_loss_final": metrics.get("final_val_value_loss"),
            "train_samples": train_samples,
            "val_samples": val_samples,
        },
        "runtime_loader_contract": {
            "schema": "PolicyValueRuntimeLoaderContract",
            "version": 1,
            "required_files": ["run_manifest.json", "candidate_report.json", "promotion_decision.json", "model_artifact.json"],
            "backend": "torch_structured",
            "artifact_schema": "PolicyValueModelArtifact",
            "artifact_version": 1,
            "artifact_status": "trained",
        },
    }
    _write_json(candidate_report_path, candidate_report)
    paths["candidate_report"] = candidate_report_path

    # 5. promotion_decision.json —— 晋升决策（新训练出的模型默认 pending，待评估后才 promote）
    promotion_decision_path = output_dir / "promotion_decision.json"
    promotion_decision = {
        "schema": "PolicyValuePromotionDecisionReport",
        "version": 1,
        "candidate_run_id": run_id,
        "baseline_run_id": "",
        "status": "pending",
        "warnings": ["auto_generated_pending_evaluation"],
        "evaluation_summary": {
            "backend": "torch_structured",
            "artifact_status": "trained",
        },
    }
    _write_json(promotion_decision_path, promotion_decision)
    paths["promotion_decision"] = promotion_decision_path

    return paths


# -------------------------------------------------------------------------------------
# _write_json：把 Python 对象写成 JSON 文件的小工具
# -------------------------------------------------------------------------------------
# 【语法43】返回类型 -> None：表示「不返回有意义的值」（相当于 Java/C++ 的 void）。
def _write_json(path: Path, payload: Any) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    # json.dumps(obj, ...)：把 Python 对象序列化成 JSON 字符串。
    #   ensure_ascii=False：允许直接写中文（不转成 \uXXXX 转义）。
    #   indent=2：缩进 2 空格美化。末尾 + "\n" 补一个换行。
    # Path.write_text(内容, encoding=...)：一次性写文本文件（自动开关文件句柄）。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# =====================================================================================
# 【全文小结 · 算法 & 语法】
# =====================================================================================
# 算法层面（AlphaZero 风格 policy-value 网络）：
#   1. CardEncoder：单卡 = id embedding + 数值特征 MLP，全网络共享，是最小编码单元。
#   2. 集合聚合：一组卡(手牌/场上/纹章/动作目标) -> 逐卡编码 -> (注意力或)带 mask 均值池化
#      -> 一个定长向量。mask 用来排除 padding 空位。
#   3. 状态编码：标量状态 + 6 组卡集合向量 拼接融合 -> state_emb。
#   4. 动作编码：动作标量 + 源卡 + 目标卡池化 拼接融合 -> 每个动作一个 action_emb。
#   5. 双头输出：
#        policy = MLP(concat(state_emb, 各 action_emb)) -> 每个动作 logit -> mask 非法 -> softmax
#        value  = MLP(state_emb)(可选融合历史) -> tanh -> [-1,1] 胜率评估
#   6. 训练：policy 用软标签交叉熵(学 MCTS 访问分布)，value 用 MSE(学胜负)，
#      Adam 优化 + Early Stopping + 保存最优权重 + warm-start 继承旧权重。
#
# 关键 Python 语法（首次出现处已标 【语法N】）：
#   - 三引号 docstring、# 注释、from __future__ import annotations（前向引用）
#   - import / from...import / as 别名 / 函数内延迟 import
#   - try/except 异常、raise 抛异常
#   - @dataclass 装饰器、field(default_factory=...)、frozen 不可变
#   - self / __init__ / super().__init__() / __call__(对象当函数调)
#   - 类型注解(int/None/联合类型 A|B/前向引用字符串)，仅提示不强制
#   - 强制关键字参数 (*,) 、解包 *list / **dict、多返回值 tuple 解包
#   - 列表/字典/集合字面量与推导式、生成器表达式、yield 生成器函数
#   - 三元表达式 A if cond else B、`A or B` 兜底、enumerate/zip/range
#   - 切片 x[a:b] / x[-1] / x[1:]、字典 get/items/in、isinstance
#   - with 上下文管理器(torch.no_grad)、f-string 与 :.4f 格式说明符
#   - 张量常用操作：view/unsqueeze/expand/cat/masked_fill/clamp/sum/mean/copy_（原地）
# =====================================================================================
