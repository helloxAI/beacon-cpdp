# 数据集说明

本目录包含四个公开软件缺陷预测基准数据集集合，均以 ARFF 格式存储，粒度为**类级别（class-level）**。

---

## 一、数据集总览

| 数据集组 | 项目来源 | 特征数 | 特征类型 | 标签属性 | 标签格式 |
|---------|---------|--------|---------|---------|---------|
| NASA | NASA 嵌入式/航天软件 | 37 | McCabe 复杂度 + Halstead 软件科学度量 | `Defective` | `{Y, N}` |
| AEEEM | 开源 Eclipse 生态项目 | 61 | 多工具 OO 度量（CK/LDHH/WCHU）+ 代码变更熵 | `class` | `{buggy, clean}` |
| PROMISE | 开源 Java 项目 | 20 | CK 面向对象度量 | `defects` | 数值（0 或正整数） |
| JIRA | Apache/大型开源项目 | 65 | 代码静态度量 + 过程度量（提交历史） | `RealBugCount` | 数值（0 或正整数） |

---

## 二、各数据集组详细说明

### 2.1 NASA 数据集

来源：NASA Metrics Data Program，包含 NASA 内部 C/C++ 项目，粒度为函数/模块级。

**特征构成（37 个）：**
- **McCabe 复杂度指标**：`CYCLOMATIC_COMPLEXITY`、`ESSENTIAL_COMPLEXITY`、`DESIGN_COMPLEXITY`、`BRANCH_COUNT`、`DECISION_COUNT` 等
- **Halstead 软件科学指标**：`HALSTEAD_VOLUME`、`HALSTEAD_DIFFICULTY`、`HALSTEAD_EFFORT`、`HALSTEAD_ERROR_EST`、`HALSTEAD_LENGTH` 等
- **代码行数**：`LOC_BLANK`、`LOC_COMMENTS`、`LOC_EXECUTABLE`、`LOC_TOTAL` 等

**数据集统计：**

| 文件 | 实例数 | 缺陷数 | 缺陷率 |
|------|--------|--------|--------|
| CM1.arff | 327 | 42 | 12.8% |
| MW1.arff | 253 | 27 | 10.7% |
| PC1.arff | 705 | 61 | 8.6% |
| PC3.arff | 1077 | 134 | 12.4% |
| PC4.arff | 1287 | 177 | 13.7% |

**特点：**
- 数据集规模较小（253–1287 个实例）
- 缺陷率较低且集中（8.6%–13.8%），类别不平衡明显
- 特征为底层过程式代码度量，不含 OO 特征
- 标签为二值：`Y`（有缺陷）/ `N`（无缺陷）

---

### 2.2 AEEEM 数据集

来源：AEEEM（An Extensive Empirical Evaluation of Machine Learning for Bug Prediction）基准，包含 Eclipse 生态系统的开源 Java 项目。

**特征构成（61 个）：**
- **CK OO 度量**（前缀 `ck_oo_`）：`wmc`、`dit`、`noc`、`cbo`、`rfc`、`lcom`、`fanIn`、`fanOut`、`loc` 等
- **LDHH 度量**（前缀 `LDHH_`）：与 CK 类似但采用不同工具计算
- **WCHU 度量**（前缀 `WCHU_`）：第三套 OO 度量工具
- **代码变更熵特征**：`CvsEntropy`、`CvsWEntropy`、`CvsLogEntropy`、`CvsLinEntropy`、`CvsExpEntropy`（反映代码文件历史变更的混乱程度）
- **历史 Bug 统计**：`numberOfBugsFoundUntil:`、`numberOfCriticalBugsFoundUntil:` 等

**数据集统计：**

| 文件 | 实例数 | 缺陷数 | 缺陷率 |
|------|--------|--------|--------|
| EQ.arff | 324 | 129 | 39.8% |
| JDT.arff | 997 | 206 | 20.7% |
| Lucene.arff | 691 | 64 | 9.3% |
| Mylyn.arff | 1862 | 245 | 13.2% |
| PDE.arff | 1497 | 209 | 14.0% |

**特点：**
- 特征维度最高（61 个），包含来自三套工具的冗余/互补 OO 度量
- 涵盖代码变更历史熵（process metrics），信息更丰富
- 缺陷率差异最大（9.3%–39.8%），EQ 数据集缺陷比例显著偏高
- 标签为字符串：`buggy` / `clean`

---

### 2.3 PROMISE 数据集

来源：PROMISE 软件工程数据仓库，涵盖多个知名开源 Java 项目（Ant、Camel、Lucene、Log4j 等）。

**特征构成（20 个）：**
标准 CK 面向对象度量套件：
`wmc`（加权方法数）、`dit`（继承深度）、`noc`（子类数）、`cbo`（类耦合度）、`rfc`（响应特征集）、`lcom`（内聚缺失度）、`ca`（传入耦合）、`ce`（传出耦合）、`npm`（公共方法数）、`lcom3`、`loc`（代码行数）、`dam`、`moa`、`mfa`、`cam`、`ic`、`cbm`、`amc`（平均方法复杂度）、`max_cc`（最大圈复杂度）、`avg_cc`（平均圈复杂度）

**数据集统计：**

| 文件 | 实例数 | 缺陷数 | 缺陷率 |
|------|--------|--------|--------|
| ant-1.7.arff | 745 | 166 | 22.3% |
| camel-1.4.arff | 872 | 145 | 16.6% |
| ivy-2.0.arff | 352 | 40 | 11.4% |
| jedit-4.0.arff | 306 | 75 | 24.5% |
| log4j-1.0.arff | 135 | 34 | 25.2% |
| poi-2.0.arff | 314 | 37 | 11.8% |
| tomcat.arff | 858 | 77 | 9.0% |
| velocity-1.6.arff | 229 | 78 | 34.1% |
| xalan-2.4.arff | 723 | 110 | 15.2% |
| xerces-1.3.arff | 453 | 69 | 15.2% |
| **PROMISE_Combo_Large** | **4987** | **831** | **16.7%** |

**特点：**
- 特征数最少（20 个），仅使用经典 CK 度量，无过程度量
- 数据集规模跨度大（135–872 个实例），log4j 和 velocity 规模较小
- 缺陷率分布广（9.0%–34.1%），velocity 缺陷率偏高
- 标签为数值型：`0`（无缺陷）/ 正整数（缺陷数）
- `PROMISE_Combo_Large.arff` 为所有子数据集的合并版本（4987 个实例）

---

### 2.4 JIRA 数据集

来源：从 Apache JIRA 缺陷追踪系统与代码仓库挖掘得到，涵盖 Apache 生态系统中的大型Java开源项目（ActiveMQ、Derby、Groovy、HBase、Hive、JRuby、Wicket 等）。

**特征构成（65 个）：**
- **代码静态度量**（前 ~55 个）：包含 Understand 工具提取的 McCabe/Halstead 系列指标，如 `CountLine`、`SumCyclomatic`、`MaxCyclomatic`、`AvgCyclomatic`、`CountDeclMethod*`、`CountInput/Output/Path`、`MaxNesting`、`PercentLackOfCohesion`、`MaxInheritanceTree`、`CountClassCoupled` 等
- **过程度量**（最后 ~10 个）：
  - `COMM`：提交次数
  - `ADEV`：活跃开发者数
  - `DDEV`：独立开发者数
  - `Added_lines` / `Del_lines`：添加/删除行数
  - `OWN_LINE` / `OWN_COMMIT`：代码所有权（行级/提交级）
  - `MINOR_COMMIT` / `MINOR_LINE`：次要贡献者占比
  - `MAJOR_COMMIT` / `MAJOR_LINE`：主要贡献者占比

**数据集统计：**

| 文件 | 实例数 | 缺陷数 | 缺陷率 |
|------|--------|--------|--------|
| activemq-5.0.0.arff | 1884 | 293 | 15.6% |
| derby-10.5.1.1.arff | 2705 | 383 | 14.2% |
| groovy-1_6_BETA_1.arff | 821 | 70 | 8.5% |
| hbase-0.94.0.arff | 1059 | 218 | 20.6% |
| hive-0.9.0.arff | 1416 | 283 | 20.0% |
| jruby-1.1.arff | 731 | 87 | 11.9% |
| wicket-1.3.0-beta2.arff | 1763 | 130 | 7.4% |
| **JIRA_Combo_Large** | **10379** | **1464** | **14.1%** |
| **JIRA_Combo_Source** | **6236** | **1018** | **16.3%** |
| **JIRA_Combo_Target** | **1559** | **246** | **15.8%** |

**特点：**
- 数据集规模最大（731–2705 个实例），derby 规模最大
- 同时包含代码静态度量和过程度量，特征信息最为全面
- 缺陷率相对稳定（7.4%–20.6%），集中在 14%–16% 附近
- 标签为数值型 `RealBugCount`，需二值化（>0 为缺陷）
- 提供三种 Combo 文件：`Combo_Large`（全量合并）、`Combo_Source`（源域子集）、`Combo_Target`（目标域子集），用于跨项目场景

---

## 三、跨数据集对比

| 维度 | NASA | AEEEM | PROMISE | JIRA |
|------|------|-------|---------|------|
| 总实例数 | 3649 | 5371 | 4987（不含Combo） | 8379（不含Combo） |
| 特征数 | 37 | 61 | 20 | 65 |
| 度量类型 | 过程式代码度量 | OO + 变更熵 | OO（CK）| 代码 + 过程度量 |
| 平均缺陷率 | 11.6% | 19.4% | 18.5% | 14.0% |
| 标签类型 | 二值 {Y/N} | 二值 {buggy/clean} | 数值（需二值化）| 数值（需二值化）|
| 适用场景 | 函数级/模块级 SDP | 类级 SDP、跨项目 SDP | 类级 SDP、跨项目 SDP | 类级 SDP、结合过程信息 |

---

## 四、数据集两两公共特征分析

### 结论：精确名称匹配 = 0

四组数据集的属性名称**完全不重叠**，因为它们来自不同的度量工具，命名规范各异：
- NASA：全大写下划线（McCabe Metrics 工具）
- PROMISE：全小写 CK 缩写（`wmc`、`dit`、`cbo`...）
- JIRA：驼峰命名的 Understand 工具输出
- AEEEM：三套工具前缀（`ck_oo_`、`LDHH_`、`WCHU_`）

### NASA ∩ PROMISE（概念上 ~3 个）

| 概念 | NASA | PROMISE |
|------|------|---------|
| 圈复杂度 | `CYCLOMATIC_COMPLEXITY` | `avg_cc` / `max_cc` |
| 代码行数 | `LOC_TOTAL` / `LOC_EXECUTABLE` | `loc` |
| 方法复杂度均值 | `CYCLOMATIC_DENSITY` | `amc` |

### NASA ∩ JIRA（概念上 ~8 个）

| 概念 | NASA | JIRA |
|------|------|------|
| 圈复杂度 | `CYCLOMATIC_COMPLEXITY` | `SumCyclomatic` / `AvgCyclomatic` / `MaxCyclomatic` |
| 本质复杂度 | `ESSENTIAL_COMPLEXITY` | `SumEssential` / `AvgEssential` |
| 代码行数 | `LOC_TOTAL` / `LOC_EXECUTABLE` | `CountLine` / `CountLineCode` |
| 注释行数 | `LOC_COMMENTS` | `CountLineComment` |
| 空白行数 | `LOC_BLANK` | `CountLineBlank` |
| 注释比例 | `PERCENT_COMMENTS` | `RatioCommentToCode` |
| 方法数 | `PARAMETER_COUNT` | `CountDeclMethod` |
| 条件/分支数 | `BRANCH_COUNT` / `CONDITION_COUNT` | `CountInput_*` |

### NASA ∩ AEEEM（概念上 ~2 个）

| 概念 | NASA | AEEEM |
|------|------|-------|
| 代码行数 | `LOC_TOTAL` | `ck_oo_numberOfLinesOfCode` / `LDHH_numberOfLinesOfCode` |
| 方法数 | `PARAMETER_COUNT` | `ck_oo_numberOfMethods` / `LDHH_numberOfMethods` |

> NASA 以过程式度量为主，AEEEM 以 OO 度量为主，本质不同，交集最少。

### PROMISE ∩ AEEEM（概念上 ~9 个，最多）

AEEEM 的 `ck_oo_*` 系列与 PROMISE 的 CK 度量在**概念上完全对应**（均来自 CK 方法论）：

| 概念 | PROMISE | AEEEM (`ck_oo_`) |
|------|---------|----------------|
| WMC（加权方法数）| `wmc` | `ck_oo_wmc` |
| DIT（继承深度）| `dit` | `ck_oo_dit` |
| NOC（子类数）| `noc` | `ck_oo_noc` |
| CBO（耦合度）| `cbo` | `ck_oo_cbo` |
| RFC（响应集）| `rfc` | `ck_oo_rfc` |
| LCOM（内聚缺失）| `lcom` | `ck_oo_lcom` |
| FanIn | `ca` | `ck_oo_fanIn` |
| FanOut | `ce` | `ck_oo_fanOut` |
| LOC | `loc` | `ck_oo_numberOfLinesOfCode` |

### PROMISE ∩ JIRA（概念上 ~5 个）

| 概念 | PROMISE | JIRA |
|------|---------|------|
| 代码行数 | `loc` | `CountLineCode` |
| 圈复杂度均/最大值 | `avg_cc` / `max_cc` | `AvgCyclomatic` / `MaxCyclomatic` |
| 方法数 | `wmc`（间接） | `CountDeclMethod` |
| 耦合度 | `cbo` | `CountClassCoupled` |
| 继承深度 | `dit` | `MaxInheritanceTree` |

### AEEEM ∩ JIRA（概念上 ~6 个）

| 概念 | AEEEM | JIRA |
|------|-------|------|
| WMC/圈复杂度 | `ck_oo_wmc` | `SumCyclomatic` |
| LOC | `ck_oo_numberOfLinesOfCode` | `CountLineCode` |
| CBO/耦合度 | `ck_oo_cbo` | `CountClassCoupled` |
| FanIn | `ck_oo_fanIn` | `CountClassBase` |
| FanOut | `ck_oo_fanOut` | `CountClassDerived` |
| 内聚度 | `ck_oo_lcom` | `PercentLackOfCohesion` |

### 汇总

| 数据集对 | 精确匹配 | 概念对应数 | 主要共同度量类型 |
|---------|---------|-----------|----------------|
| NASA ∩ PROMISE | 0 | ~3 | 圈复杂度、代码行数 |
| NASA ∩ JIRA | 0 | ~8 | 圈复杂度、代码行、注释、本质复杂度 |
| NASA ∩ AEEEM | 0 | ~2 | 代码行数、方法数 |
| **PROMISE ∩ AEEEM** | **0** | **~9** | **CK OO 全套度量（同源不同名）** |
| PROMISE ∩ JIRA | 0 | ~5 | LOC、复杂度、耦合度、继承 |
| AEEEM ∩ JIRA | 0 | ~6 | WMC、LOC、耦合、内聚、扇入/出 |

**PROMISE ∩ AEEEM 概念重叠最多**，因为 AEEEM 的 `ck_oo_*` 系列本质上就是 CK 度量的另一实现，与 PROMISE 同源异名。各组之间**没有任何精确名称匹配**，跨数据集迁移时需要特征对齐或使用特征无关的方法。

---

## 五、跨项目缺陷预测（CPDP）常用配对示例

本项目主要支持以下跨数据集/跨项目迁移场景：

- **NASA 内部迁移**：如 `CM1 → PC4`、`PC1 → PC3`
- **PROMISE 项目间迁移**：如 `ant → camel`、`xalan → xerces`
- **AEEEM 项目间迁移**：如 `JDT → PDE`、`Mylyn → Lucene`
- **JIRA 项目间迁移**：如 `derby → activemq`、`hive → hbase`
