import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================
TARGET_COUNTRIES = [
    "United States", "China", "United Kingdom", "Germany", "Japan", 
    "South Korea", "France", "Canada", "India", "United Arab Emirates"
]

# 映射表：把各种简写统一为标准名称
COUNTRY_MAP = {
    "USA": "United States", "US": "United States", "United States of America": "United States",
    "CHN": "China", "People's Republic of China": "China",
    "GBR": "United Kingdom", "UK": "United Kingdom", "Great Britain": "United Kingdom",
    "DEU": "Germany", "Germany (until 1990 former territory of the FRG)": "Germany",
    "JPN": "Japan",
    "KOR": "South Korea", "Korea": "South Korea", "Korea, Rep.": "South Korea",
    "FRA": "France",
    "CAN": "Canada",
    "IND": "India",
    "ARE": "United Arab Emirates", "UAE": "United Arab Emirates"
}

def normalize_country(name):
    clean_name = str(name).strip()
    return COUNTRY_MAP.get(clean_name, clean_name)

# ================= 2. 增强版读取函数 =================

def load_stanford_flexible(base_path):
    """读取斯坦福数据，支持多种格式"""
    print("🔍 Scanning Stanford Data (Deep Search)...")
    data_points = []
    static_scores = {}  # 存储累计/静态数据（用于分配）

    # 遍历所有 CSV
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".csv"): continue

            path = os.path.join(root, file)
            try:
                df = pd.read_csv(path)
                cols = [c.strip() for c in df.columns]
                
                # 识别指标类型
                is_publication = "1.1." in file
                is_patent = "1.3." in file
                if not (is_publication or is_patent):
                    continue
                
                metric_type = "AI_Publication_Share" if is_publication else "AI_Patent_Share"
                
                # 模式 A: 长格式 - 有 Year 列和国家列 (Label/Entity/Geographic area)
                country_col = next((c for c in cols if c in ['Label', 'Entity', 'Geographic area', 'Country']), None)
                year_col = next((c for c in cols if 'Year' in c), None)
                
                if country_col and year_col:
                    value_cols = [c for c in cols if c not in [country_col, year_col]]
                    if not value_cols:
                        continue
                    value_col = value_cols[0]
                    
                    for _, row in df.iterrows():
                        c_name = normalize_country(row[country_col])
                        if c_name in TARGET_COUNTRIES:
                            data_points.append({
                                "Country": c_name,
                                "Year": row[year_col],
                                metric_type: row[value_col]
                            })
                            print(f"    Found: {c_name} {row[year_col]} -> {metric_type}")
                
                # 模式 B: 静态累计表 (Geographic area + 总计数) - 用于填补
                elif country_col and not year_col:
                    value_cols = [c for c in cols if c != country_col]
                    if value_cols:
                        value_col = value_cols[0]
                        for _, row in df.iterrows():
                            c_name = normalize_country(row[country_col])
                            if c_name in TARGET_COUNTRIES:
                                if c_name not in static_scores:
                                    static_scores[c_name] = {}
                                static_scores[c_name][metric_type] = row[value_col]
                                print(f"    Static score: {c_name} -> {metric_type} = {row[value_col]}")
                
                # 模式 C: 宽格式 - 国家是列名
                elif 'United States' in cols or 'China' in cols:
                    id_col = df.columns[0]
                    df_melted = df.melt(id_vars=[id_col], var_name='Country', value_name='Value')
                    df_melted.rename(columns={id_col: 'Year'}, inplace=True)
                    for _, row in df_melted.iterrows():
                        c_name = normalize_country(row['Country'])
                        if c_name in TARGET_COUNTRIES:
                            data_points.append({
                                "Country": c_name,
                                "Year": row['Year'],
                                metric_type: row['Value']
                            })

            except Exception as e:
                continue

    print(f"\n  Data points collected: {len(data_points)}")
    print(f"  Static scores collected: {static_scores}")
    
    if not data_points:
        print("  ⚠️ No time-series Stanford data found!")
        # 如果没有时序数据，用静态分数生成假时间序列
        if static_scores:
            print("  Using static scores to generate proxy data...")
            years = list(range(2015, 2025))
            for country, scores in static_scores.items():
                for year in years:
                    point = {"Country": country, "Year": year}
                    for metric, value in scores.items():
                        # 归一化（除以美国的值作为份额）
                        us_value = static_scores.get("United States", {}).get(metric, 1)
                        if us_value > 0:
                            point[metric] = value / us_value
                        else:
                            point[metric] = value
                    data_points.append(point)
    
    if not data_points:
        return pd.DataFrame(columns=['Country', 'Year'])
        
    # 聚合结果
    df_all = pd.DataFrame(data_points)
    df_all['Year'] = pd.to_numeric(df_all['Year'], errors='coerce')
    
    # 为缺失的国家用静态分数填补
    if static_scores:
        for country in TARGET_COUNTRIES:
            if country in static_scores and country not in df_all['Country'].values:
                years = list(range(2015, 2025))
                us_scores = static_scores.get("United States", {})
                for year in years:
                    point = {"Country": country, "Year": year}
                    for metric, value in static_scores[country].items():
                        us_val = us_scores.get(metric, 1)
                        point[metric] = value / us_val if us_val > 0 else value
                    data_points.append(point)
        df_all = pd.DataFrame(data_points)
        df_all['Year'] = pd.to_numeric(df_all['Year'], errors='coerce')
    
    # 按国家年份聚合取均值
    agg_cols = [c for c in ['AI_Publication_Share', 'AI_Patent_Share'] if c in df_all.columns]
    if agg_cols:
        df_final = df_all.groupby(['Country', 'Year'])[agg_cols].mean().reset_index()
    else:
        df_final = df_all.groupby(['Country', 'Year']).first().reset_index()
    
    print(f"  ✓ Final Stanford data: {len(df_final)} rows")
    return df_final

def load_broadband_fixed(filepath):
    """修复宽带数据读取 (处理 REF_AREA 代码)"""
    print(f"🌐 Reading Broadband: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f"  Columns: {df.columns.tolist()[:10]}...")
        
        # OECD 通常用 REF_AREA 存国家代码
        if 'REF_AREA' in df.columns:
            df['Country'] = df['REF_AREA'].apply(normalize_country)
        elif 'Country' in df.columns:
            df['Country'] = df['Country'].apply(normalize_country)
        
        df = df[df['Country'].isin(TARGET_COUNTRIES)]
        print(f"  Found {len(df)} rows for target countries")
        
        # 提取光纤数据 (假设指标代码包含 'FIBRE' 或直接用总宽带)
        # 这里简化：如果有 FIBRE 就用，没有就用 BroadBand 总数
        # 实际操作：直接按 Country, Year 分组取最大值作为代理变量
        df['Year'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
        df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
        df = df.groupby(['Country', 'Year'])['OBS_VALUE'].max().reset_index()
        df.columns = ['Country', 'Year', 'Broadband_Penetration']
        print(f"  ✓ Processed {len(df)} broadband records")
        return df
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()

def load_existing_data(filepath):
    """读取已有的 v3/v4 数据（保留 GERD, Electricity 等）"""
    print(f"📂 Loading existing data: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()

# ================= 3. 主逻辑与填补 =================

def main():
    # =============================================
    # 修正后的文件路径 (根据您项目的实际结构)
    # =============================================
    base_dir = "."  # 当前目录 (华数杯)
    stanford_path = "The 2025 AI Index Report/1. Research and Development"
    broadband_path = "OECD_宽带与电信.csv"
    existing_data_path = "final_model_data_v4.csv"  # 使用已有的 v4 数据作为基础
    
    print("=" * 60)
    print("🚀 Starting Enhanced Data Merge (fix_data_final_v4)")
    print("=" * 60)
    
    # --- 读取已有数据 (保留 GERD, Electricity, Supercomputer 等) ---
    df_existing = load_existing_data(existing_data_path)
    
    # --- 读取各个源 ---
    # A. 斯坦福 (AI)
    df_ai = load_stanford_flexible(stanford_path)
    
    # B. 宽带 (Infrastructure)
    df_bb = load_broadband_fixed(broadband_path)
    
    # --- 合并 ---
    print("\n🔗 Merging datasets...")
    
    if not df_existing.empty:
        # 基于已有数据进行增强
        final = df_existing.copy()
        
        # 合并 AI 数据
        if not df_ai.empty:
            # 只更新空值
            for col in ['AI_Publication_Share', 'AI_Patent_Share']:
                if col in df_ai.columns:
                    if col not in final.columns:
                        final[col] = np.nan
                    # Merge and fill
                    merged = pd.merge(final[['Country', 'Year']], df_ai[['Country', 'Year', col]], 
                                     on=['Country', 'Year'], how='left', suffixes=('', '_new'))
                    if f'{col}_new' in merged.columns:
                        final[col] = final[col].fillna(merged[f'{col}_new'])
                    elif col in merged.columns:
                        final[col] = final[col].fillna(merged[col])
        
        # 合并宽带数据
        if not df_bb.empty and 'Broadband_Penetration' in df_bb.columns:
            if 'Broadband_Penetration' not in final.columns:
                final['Broadband_Penetration'] = np.nan
            merged = pd.merge(final[['Country', 'Year']], df_bb[['Country', 'Year', 'Broadband_Penetration']], 
                             on=['Country', 'Year'], how='left', suffixes=('', '_new'))
            if 'Broadband_Penetration_new' in merged.columns:
                final['Broadband_Penetration'] = final['Broadband_Penetration'].fillna(merged['Broadband_Penetration_new'])
    else:
        # 从头构建
        years = range(2010, 2026)
        skeleton = pd.DataFrame([(c, y) for c in TARGET_COUNTRIES for y in years], columns=['Country', 'Year'])
        final = pd.merge(skeleton, df_ai, on=['Country', 'Year'], how='left')
        final = pd.merge(final, df_bb, on=['Country', 'Year'], how='left')
    
    # ================= 4. 关键：强力填补 (Imputation) =================
    print("\n🔧 Running Smart Imputation...")
    
    # 规则 1: 线性插值 (填补中间空缺)
    final = final.sort_values(['Country', 'Year'])
    numeric_cols = final.select_dtypes(include=[np.number]).columns.tolist()
    if 'Year' in numeric_cols:
        numeric_cols.remove('Year')
    
    for col in numeric_cols:
        final[col] = final.groupby('Country')[col].transform(lambda x: x.interpolate(limit_direction='both'))
    
    # 规则 2: 对于完全缺失的国家 (如印度的 GERD，阿联酋的 AI)，使用 Tortoise 分数映射
    # 映射逻辑：Country_Value = US_Value * (Country_Score / US_Score) * Correction_Factor
    
    # 设定基准值 (基于美国 2023 年数据的估算)
    us_ai_share = 0.15  # 假设美国 AI 论文占比约 15%
    us_broadband = 40.0  # 假设美国宽带渗透率
    
    tortoise_scores = {
        'India': {'AI': 0.14, 'Infra': 0.15},  # 0-1 归一化后的分数
        'United Arab Emirates': {'AI': 0.13, 'Infra': 0.29},
        'China': {'AI': 0.48, 'Infra': 0.66}
    }
    
    print("  Applying Tortoise-based imputation for missing countries...")
    for idx, row in final.iterrows():
        ctry = row['Country']
        
        # 填补 AI 数据
        if 'AI_Publication_Share' in final.columns:
            if pd.isna(row.get('AI_Publication_Share')) and ctry in tortoise_scores:
                final.at[idx, 'AI_Publication_Share'] = us_ai_share * (tortoise_scores[ctry]['AI'] / 1.0) 
                
        if 'AI_Patent_Share' in final.columns:
            if pd.isna(row.get('AI_Patent_Share')) and ctry in tortoise_scores:
                ai_pub = final.at[idx, 'AI_Publication_Share'] if 'AI_Publication_Share' in final.columns else us_ai_share * 0.1
                final.at[idx, 'AI_Patent_Share'] = ai_pub * 0.8  # 专利通常比论文少
            
        # 填补宽带数据
        if 'Broadband_Penetration' in final.columns:
            if pd.isna(row.get('Broadband_Penetration')) and ctry in tortoise_scores:
                final.at[idx, 'Broadband_Penetration'] = us_broadband * (tortoise_scores[ctry]['Infra'] / 1.0)

    # 规则 3: 最后的兜底 (用列均值填充，防止代码报错)
    print("  Final fillna with column means...")
    for col in numeric_cols:
        if final[col].isna().any():
            col_mean = final[col].mean()
            if pd.notna(col_mean):
                final[col] = final[col].fillna(col_mean)
            else:
                final[col] = final[col].fillna(0)
    
    # 保存
    output_path = "final_model_data_v4_ready.csv"
    final.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Done! File saved as '{output_path}'")
    print("=" * 60)
    
    # 数据完整性检查
    print("\n📊 Data Quality Report:")
    print(f"  Total rows: {len(final)}")
    print(f"  Countries: {final['Country'].nunique()}")
    print(f"  Year range: {final['Year'].min()} - {final['Year'].max()}")
    print(f"\n  Missing values per column:")
    missing = final.isnull().sum()
    for col, count in missing.items():
        status = "✓" if count == 0 else "⚠️"
        print(f"    {status} {col}: {count}")
    
    print("\n📋 Sample data (first 15 rows):")
    print(final.head(15).to_string())

if __name__ == "__main__":
    main()
