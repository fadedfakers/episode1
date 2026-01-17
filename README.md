# 华数杯 - 十国AI竞争力面板数据集

## 项目概述

本项目将多源异构数据整合为一个可用于数学建模的**面板数据集 (Panel Data)**，覆盖10个目标国家2010-2025年的AI竞争力相关指标。

### 目标国家
- 美国 (United States)
- 中国 (China)
- 英国 (United Kingdom)
- 德国 (Germany)
- 日本 (Japan)
- 韩国 (South Korea)
- 法国 (France)
- 加拿大 (Canada)
- 印度 (India)
- 阿联酋 (United Arab Emirates)

---

## 最终输出

| 文件 | 说明 |
|------|------|
| `final_model_data_v4_ready.csv` | 最终面板数据，无缺失值，可直接用于建模 |
| `fix_data_final_v4.py` | 数据处理脚本 |

---

## 数据字段说明

| 字段名 | 含义 | 单位 | 数据来源 |
|--------|------|------|----------|
| `Country` | 国家名称 | - | - |
| `Year` | 年份 | - | - |
| `GERD_USD_PPP` | 研发总支出 (GERD) | 百万美元 (PPP) | OECD MSTI |
| `Total_Generation_TWh` | 总发电量 | TWh | Ember |
| `Renewable_Generation_TWh` | 可再生能源发电量 | TWh | Ember |
| `Broadband_Penetration` | 宽带渗透率/光纤占比 | % | OECD Broadband |
| `Supercomputer_TFlops` | 超级计算机算力 | TFlops | TOP500 |
| `AI_Publication_Share` | AI论文占比/AI研究活跃度 | 比例值 | Stanford AI Index |
| `AI_Patent_Share` | AI专利占比/AI创新能力 | 比例值 | Stanford AI Index |
| `Commercial_Score` | 商业化能力评分 | 0-100 | Tortoise Index |

---

## 数据处理方法

### 1. 数据清洗 (Data Cleaning)

#### 1.1 国家名称标准化
将各数据源中的不同国家名称统一映射为标准名：

```python
COUNTRY_MAP = {
    "USA": "United States",
    "US": "United States",
    "CHN": "China",
    "People's Republic of China": "China",
    "GBR": "United Kingdom",
    "UK": "United Kingdom",
    "DEU": "Germany",
    "JPN": "Japan",
    "KOR": "South Korea",
    "Korea": "South Korea",
    "FRA": "France",
    "CAN": "Canada",
    "IND": "India",
    "ARE": "United Arab Emirates",
    "UAE": "United Arab Emirates"
}
```

#### 1.2 文件格式处理
- **Ember 电力数据**：文件扩展名为 `.csv` 但实际为 Excel 格式，脚本自动检测并使用 `pd.read_excel()` 读取
- **OECD 数据**：处理多层嵌套的列结构，提取 `REF_AREA`, `TIME_PERIOD`, `OBS_VALUE` 等关键字段
- **Stanford AI Index**：支持三种数据格式自动识别：
  - 长格式 (Long Format)：含 `Label`/`Entity` 列
  - 宽格式 (Wide Format)：国家名作为列名
  - 静态累计表：只有总计数，无时间序列

---

### 2. 缺失值补全 (Imputation)

#### 2.1 时间序列插值
对于有部分年份数据的指标，使用**双向线性插值**：

```python
df.groupby('Country')[col].transform(
    lambda x: x.interpolate(limit_direction='both')
)
```

#### 2.2 基于评价指标的代理填补

对于完全缺失的国家/指标，使用 **Tortoise Index 评分**作为代理变量进行估算：

| 缺失数据 | 填补方法 |
|----------|----------|
| 印度/阿联酋 的 GERD | 使用列均值填补 (约 653.69 百万美元) |
| 中国/阿联酋 的宽带数据 | 使用 Tortoise 基建评分比例估算 (中国≈26.4%, 阿联酋≈11.6%) |
| G7国家的 AI 指标 | 使用 Stanford `fig_1.3.3.csv` 中的累计 AI 模型数量，归一化为相对美国的比例 |

#### 2.3 静态分数填补 (Static Score Imputation)

对于缺乏时间序列数据的国家（如加拿大、法国、德国、日本、韩国、英国的 AI 指标），使用累计数据的相对比例：

```
国家_AI_Share = 国家_累计模型数 / 美国_累计模型数

示例：
- 加拿大: 61 / 558 ≈ 0.109
- 英国: 72 / 558 ≈ 0.129
- 德国: 28 / 558 ≈ 0.050
```

---

### 3. 数据来源明细

| 数据源 | 文件路径 | 提取内容 |
|--------|----------|----------|
| OECD MSTI | `OECD_MSTI, 主要科技指标.csv` | GERD (研发支出) |
| OECD 宽带 | `OECD_宽带与电信.csv` | 光纤宽带渗透率 |
| Ember | `基础设施/ember_十国发电量.csv` | 总发电量、可再生能源发电量 |
| TOP500 | `基础设施/TOP500 TOP500List(已求和).xlsx` | 超级计算机算力 (Rmax) |
| Stanford AI Index | `The 2025 AI Index Report/1. Research and Development/` | AI论文占比、AI专利占比 |
| Tortoise | `Tortoise_核心得分.xlsx` | 商业化评分、政策评分 |

---

## 数据质量报告

### 最终数据统计

| 指标 | 值 |
|------|-----|
| 总行数 | 160 (10国 × 16年) |
| 总列数 | 10 |
| 缺失值 | 0 |
| 时间跨度 | 2010-2025 |

### 各国数据完整性

| 国家 | GERD | 电力 | 宽带 | 超算 | AI论文 | AI专利 |
|------|------|------|------|------|--------|--------|
| 🇺🇸 美国 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | ✅原始 |
| 🇨🇳 中国 | ✅原始 | ✅原始 | 🔶估算 | ✅原始 | ✅原始 | ✅原始 |
| 🇬🇧 英国 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | 🔶估算 | 🔶估算 |
| 🇩🇪 德国 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | 🔶估算 | 🔶估算 |
| 🇯🇵 日本 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | 🔶估算 | 🔶估算 |
| 🇰🇷 韩国 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | 🔶估算 | 🔶估算 |
| 🇫🇷 法国 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | 🔶估算 | 🔶估算 |
| 🇨🇦 加拿大 | ✅原始 | ✅原始 | ✅原始 | ✅原始 | 🔶估算 | 🔶估算 |
| 🇮🇳 印度 | 🔶估算 | ✅原始 | 🔶估算 | ✅原始 | ✅原始 | ✅原始 |
| 🇦🇪 阿联酋 | 🔶估算 | ✅原始 | 🔶估算 | ✅原始 | 🔶估算 | 🔶估算 |

**图例**：✅原始数据 | 🔶估算/插值数据

---

## 使用方法

### 直接使用已处理数据

```python
import pandas as pd

# 读取最终数据
df = pd.read_csv('final_model_data_v4_ready.csv')

# 设置多级索引
df = df.set_index(['Country', 'Year'])

# 查看数据
print(df.head(20))
```

### 重新运行数据处理

```bash
python fix_data_final_v4.py
```

---

## 建模建议

1. **熵权法/TOPSIS**：所有指标均为正向指标（越大越好），可直接使用
2. **回归分析**：`Commercial_Score` 可作为因变量，其他为自变量
3. **聚类分析**：建议先进行 Z-score 标准化
4. **时间序列预测**：数据已按 Country-Year 排序，可直接用于 ARIMA/Prophet

---

## 注意事项

1. **AI 指标的估算值**：G7 国家（除美国外）的 AI_Publication_Share 和 AI_Patent_Share 是基于累计模型数量的比例估算，可能与实际时间序列趋势有偏差
2. **印度/阿联酋的 GERD**：由于 OECD 不包含这两国数据，使用了列均值填补，建议在敏感性分析中单独处理
3. **单位一致性**：GERD 为 PPP 口径的百万美元，已消除汇率影响

---

## 文件结构

```
华数杯/
├── final_model_data_v4_ready.csv  # 最终输出数据
├── fix_data_final_v4.py           # 数据处理脚本
├── README.md                      # 本说明文档
├── OECD_MSTI, 主要科技指标.csv    # OECD 研发数据
├── OECD_宽带与电信.csv            # OECD 宽带数据
├── Tortoise_核心得分.xlsx         # Tortoise 评分
├── The 2025 AI Index Report/      # Stanford AI Index 数据
│   └── 1. Research and Development/
│       ├── fig_1.1.6.csv          # AI 论文数据
│       ├── fig_1.3.3.csv          # AI 模型累计数据
│       └── ...
├── 基础设施/
│   ├── ember_十国发电量.csv       # 电力数据 (实际为xlsx)
│   ├── TOP500 TOP500List(已求和).xlsx  # 超算数据
│   └── ...
└── 人才数量/
    └── CSRanking faculty数量.xlsx # CS教职数量
```

---

*文档生成时间：2026-01-17*
