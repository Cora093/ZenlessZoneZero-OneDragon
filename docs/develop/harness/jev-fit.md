# Jev（TypeSafe System One）适配评估

> 只读评估结论。**不引入 SDK/依赖，不改业务代码。**
> 评估对象：本仓库运行时产品 + 开发期 AI harness。
> Jev 事实来源：TypeSafe 官方博文（2026-09-15）、厂商口径延迟/定价；以官方为准。

## 仓库结论

**部分适合。**

本仓是面向 Windows 玩家的绝区零本地自动化工具（CV + 状态机 + YAML），运行时没有 LLM、没有 tool-calling Agent、也没有「补丁 apply 前评分」这类 harness。Jev 的典型工程用法（工具风险门控、模型路由、PR 评分、工单分类）大部分落在**仓外**的编码助手上，而不是产品主循环里。

唯一对得上的产品决策面，是迷失之地 / 零号空洞里「多选一」的藏品与鸣徽选择；即便如此，现有 YAML 优先级已经可配置、可复现，云端 70–500ms 决策层会引入网络、密钥、误判与合规成本，不值得作为产品依赖。

## 一句话结论

**不要把 Jev 接进运行时；若要试，只在仓外编码助手或离线回放上做对照实验。**

## 仓库画像（与 Jev 的错位）

| 维度 | 本仓现状 | Jev 典型前提 |
|---|---|---|
| 目标用户 | Windows 玩家，本地跑日常/战斗/空洞 | 开发者把决策嵌进软件/Agent |
| 运行时智能 | OCR、YOLO、模板匹配、本地 ONNX | 文本/程序状态 → typed 决策 |
| 控制流 | `Operation` 节点图、`ConditionalOperator` YAML、`ApplicationFactory` | 脆弱 if-else 的智能分支 |
| Agent / 工具调用 | **无**。`src/` 中无 OpenAI/LLM/LangChain | 对 tool/shell/edit 做门控 |
| 实时性 | 闪避检测间隔 **20ms**（`auto_battle_operator.check_dodge_interval`） | 端到端约 70–500ms（厂商口径，美西） |
| 网络 | 可选通知推送、资源下载；主循环可离线 | `POST https://api.typesafe.ai/v1/systemone` |
| 决策可复现 | YAML 优先级 + BFS 寻路，测试仓可回归 | 概率输出，同一输入也可能随阈值策略变 |

技能文档 `skills/zzz-one-dragon-player/SKILL.md` 写过「迷失之地：LLM 辅助」。对照代码，实际是 YOLO 检测 + `get_artifact_by_priority` / 四层本地规则，**运行时没有 LLM**。

## 关键决策点

有决策点，但多数已被确定性规则覆盖；**没有** Jev 擅长的 Agent 工具门控。

### 不该交给 Jev（硬否决）

1. **自动战斗 / 闪避**  
   `src/zzz_od/auto_battle/auto_battle_operator.py`：黄红光检测 20ms 量级。Jev 延迟高一个数量级，且输入是像素不是程序状态。继续用本地 YOLO `FlashClassifier` + YAML `config/auto_battle_state_handler/`。

2. **画面识别与路由**  
   `ScreenContext` / `round_by_goto_screen`：模板 + OCR + Floyd 路由。这是 CV 问题，不是 Choice/Score 问题。

3. **空洞寻路**  
   `src/zzz_od/hollow_zero/hollow_map/hollow_pathfinding.py`：BFS。确定最短路不需要概率模型。

4. **一条龙任务编排**  
   `ZOneDragonApp` + 用户在 GUI 里勾选顺序。调度策略是用户配置，不是分类路由。

5. **OCR 模糊匹配**  
   `str_utils.find_most_similar` / LCS。输入是错字字符串对封闭词表，本地编辑距离已够。

### 形态接近、但仍不建议产品化

这些点「长得像」Choice/Score，但现有实现已经是 typed、可配置、可测试的规则。

1. **迷失之地藏品/武备选择**  
   - `LostVoidChooseCommon.select_by_layers`：NEW → 同流派 → YAML 优先级 → 兜底点击  
   - `LostVoidChooseGear.choose_gear`：追新模式 / `get_artifact_by_priority`  
   - `UpdatePriorityOperation`：从藏品名提取动态优先级  

2. **零号空洞鸣徽/事件**  
   - `resonium_utils.choose_resonium_by_priority`  
   - `HollowRunner._special_event_handlers` 把 OCR 事件名映射到确定 `ZOperation`  
   - `hollow_zero_challenge_config.resonium_priority` / `event_priority`

3. **GitHub Issue 标签**  
   `.github/workflows/issue-automation.yml`：从 Issue 表单字段用正则抽取类型/范围。表单已经把分类变成枚举，Jev 增量很小。

4. **开发期 AI harness（仓外）**  
   `docs/develop/harness/README.md` 方向 A 已落地 AGENTS.md / skills；方向 B（游戏 MCP）仍规划中。Jev 若有价值，是给 Claude Code 等**外部 Agent** 做 tool 门控，而不是写进 `src/`。

## 若仍要试验：具体插入点与 typed questions

以下仅作对照实验设计，**不是实现清单**。策略默认 **shadow（只记录、不改点击）**。

### 1) `LostVoidChooseCommon` / `LostVoidChooseGear`

程序状态：OCR 到的候选藏品名、已持有流派、`challenge_config.artifact_priority`、追新开关。

| 类型 | 问题 | 策略 |
|---|---|---|
| Choice | 当前应点选哪一件？选项 = 可见藏品 display_name | shadow；仅当与 YAML 第一优先一致才可考虑 auto |
| Score | 该藏品对当前流派的契合度 0–5 | review：与 YAML 排序 Spearman 相关低则人工看回放 |
| Noul | 此刻应走「追新」而不是优先级？ | block 覆盖用户配置；最多当日志特征 |

### 2) `resonium_utils.choose_resonium_by_priority` + `ChooseResonium`

程序状态：3 个鸣徽的 category/name/level，用户 `resonium_priority`。

| 类型 | 问题 | 策略 |
|---|---|---|
| Choice | 选哪个鸣徽？ | shadow vs YAML |
| Score | 该鸣徽与当前构筑的匹配度 0–5 | review |
| Noul | 三个选项是否都未命中用户优先级、应走兜底？ | auto 仅当 Noul 高置信且与现有 `choose_default` 同向 |

### 3) 规划中的游戏 MCP（harness B-1/B-2）

若将来把截图 / `Application.execute` 暴露给编码 Agent，Jev 更适合做**仓外**门控，而不是游戏内决策。

| 类型 | 问题 | 策略 |
|---|---|---|
| Choice | 该 MCP 调用：auto / ask / deny？选项含 `run_application`、`send_key`、`screenshot` | `send_key` / 启动游戏：ask 或 block；只读截图/OCR：auto |
| Score | 这次调用的爆炸半径 0–5 | ≥4 review |
| Noul | 该参数是否会在非 1080p 窗口外点击？ | true → block |

### 4) Issue 自动化（弱推荐）

仅当表单字段缺失、标题无法归类时。

| 类型 | 问题 | 策略 |
|---|---|---|
| Choice | Bug / 增强 / 等待适配 / 无效反馈 / 无法判断 | 「无法判断」→ 不打标签（review）；「无效」→ 必须人工，禁止 auto 关单 |
| Noul | 描述是否足以复现？ | false → 自动评论补日志，不关单 |

## 推荐接入方式

**无（产品仓库不接入）。**

对照：

| 方式 | 结论 |
|---|---|
| Python SDK / HTTP 进 `src/` | 否。会成为首个运行时云端模型依赖 |
| LangChain TypeSafeClassifier | 否。本仓无 LangChain |
| 社区 `jev-mcp` | **仅个人级**（`CLAUDE.local.md`），按 `setup/ai_coding.md` 三级晋升不要进 AGENTS.md |
| GitHub Action 调 HTTP | Issue 分类 ROI 低；若做，独立 workflow + 密钥，失败必须 noop |

## 风险与成本

- **API Key**：玩家发行版无法合理分发 TypeSafe key；开发者密钥进客户端等于公开额度。
- **延迟**：厂商 70–500ms 且服务在美西。闪避不可用；空洞选物可忍受，但国内网络抖动可能变成秒级，叠加 OCR 已有等待。
- **输入形态**：Jev 吃的是结构化程序状态（官方 Doom demo 也强调「不是图像」）。本仓状态在像素里，仍要先 OCR/YOLO。Jev 不替代识别。
- **误判**：YAML 优先级是用户显式意图。模型改选等于无视配置。任何 auto 覆盖都必须默认关闭。
- **职责边界**：Jev 不写代码、不跑 CV。编码仍用 Claude Code / Copilot；识别仍用本地模型。
- **合规 / 发行**：向境外 API 发送游戏内文本（藏品名、事件 OCR）对玩家工具不合适；也不应把截图当 state。
- **可测试性**：`zzz-od-test` 依赖截图回归。云端概率决策会打穿现有「同图同结果」假设。
- **早期产品**：2026-09 首发、early access。不适合绑进给玩家用的稳定路径。

## MVP（不改 `src/`）

1. **离线对照，不上线**  
   从已有迷失之地/鸣徽截图（测试仓或 debug 图）跑现有 OCR + YAML 排序，把「候选列表 + 用户优先级」存成 JSON。用 HTTP 调 Jev（开发者自己的 key），只比较 Choice 是否与 YAML 第一名一致。样本个位数即可判断有没有信号。

2. **成功标准先写死**  
   例如：与 YAML 一致率显著高于「随机选可见项」，且高置信分歧能被人工标成「YAML 不合理」。达不到就停。禁止把 Jev 接进 `LostVoidChooseCommon`。

3. **编码助手保持个人级**  
   若有人用 Claude Code 改本仓，可在本地加 `jev-mcp` 做 shell/edit 门控。证明对团队有用再考虑写进 `docs/develop/setup/ai_coding.md`；**不要**写进运行时或 `AGENTS.md`。

## 替代方案（更贴合本仓）

| 需求 | 更合理的做法 |
|---|---|
| 藏品/鸣徽选得更好 | 继续打磨 YAML 优先级、动态优先级、追新规则；用测试截图回归 |
| 识别不准 | 更新本地 ONNX（闪光 / 空洞事件 / 迷失之地检测），不是云端决策模型 |
| 战斗决策 | 保持 `ConditionalOperator` + 状态模板；Jev 延迟不可接受 |
| 让 AI 更好改这个仓 | 完成 harness A-2/A-3（skills、术语表、devtools），而不是加决策 API |
| 将来 Agent 驱动游戏 | 先做 MCP B-1 只读工具；门控用 Claude Code 权限或后续再评 Jev |
| Issue 分流 | 保持 GitHub 表单 + 现有 script；缺字段用模板必填，不必上模型 |

## 明确未做的事

- 未改 `src/`、未加依赖、未调用 TypeSafe API。
- 本文只是评估记录，不构成接入计划。
