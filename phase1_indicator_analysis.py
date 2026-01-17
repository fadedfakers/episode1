"""
Phase 1: Indicator Construction & Interaction Analysis
2026 ICM Contest - Problem B: AI Development Competitiveness

This script performs:
1. Data Preparation & Validation
2. Indicator Mapping & Z-Score Normalization
3. Correlation Analysis & Granger Causality Tests
4. Mechanism Path Diagram (Network Visualization)

Author: AI Assistant
Date: 2026-01-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings('ignore')

# Set global plot style for publication quality
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 8),
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'final_model_data_v4_ready.csv'

# Dimension Mapping (Conceptual Framework)
DIMENSION_MAPPING = {
    'Infrastructure': [
        'Total_Generation_TWh',
        'Renewable_Generation_TWh', 
        'Broadband_Penetration',
        'Supercomputer_TFlops'
    ],
    'Innovation': [
        'AI_Publication_Share',
        'AI_Patent_Share'
    ],
    'Application': [
        'Commercial_Score'
    ],
    'Policy_Resources': [
        'GERD_USD_PPP'
    ]
}

# Flatten to get all feature columns
FEATURE_COLUMNS = []
for cols in DIMENSION_MAPPING.values():
    FEATURE_COLUMNS.extend(cols)

# Short labels for visualization
SHORT_LABELS = {
    'GERD_USD_PPP': 'R&D Investment',
    'Total_Generation_TWh': 'Total Electricity',
    'Renewable_Generation_TWh': 'Renewable Energy',
    'Broadband_Penetration': 'Broadband',
    'Supercomputer_TFlops': 'Supercomputer',
    'AI_Publication_Share': 'AI Publications',
    'AI_Patent_Share': 'AI Patents',
    'Commercial_Score': 'Commercialization'
}

# Color mapping for dimensions
DIMENSION_COLORS = {
    'Infrastructure': '#3498db',      # Blue
    'Innovation': '#e74c3c',          # Red
    'Application': '#2ecc71',         # Green
    'Policy_Resources': '#9b59b6'     # Purple
}

def get_dimension(column):
    """Get the dimension category for a given column."""
    for dim, cols in DIMENSION_MAPPING.items():
        if column in cols:
            return dim
    return 'Unknown'

def get_node_color(column):
    """Get color based on dimension."""
    dim = get_dimension(column)
    return DIMENSION_COLORS.get(dim, '#95a5a6')


# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

def load_and_prepare_data(filepath):
    """Load CSV and set MultiIndex."""
    print("=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded data from '{filepath}'")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Set MultiIndex
    df = df.set_index(['Country', 'Year'])
    print(f"  MultiIndex set: ['Country', 'Year']")
    
    # Verify no NaN values
    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        print(f"  ✓ Data Integrity Check: No missing values detected")
    else:
        print(f"  ⚠ Warning: {nan_count} missing values found!")
        print(df.isna().sum())
    
    # Display basic statistics
    print(f"\n📊 Countries: {df.index.get_level_values('Country').unique().tolist()}")
    print(f"📅 Year Range: {df.index.get_level_values('Year').min()} - {df.index.get_level_values('Year').max()}")
    
    return df


# ============================================================================
# STEP 2: INDICATOR MAPPING & STANDARDIZATION
# ============================================================================

def standardize_data(df):
    """Perform Z-Score normalization on all numeric features."""
    print("\n" + "=" * 70)
    print("STEP 2: INDICATOR MAPPING & STANDARDIZATION")
    print("=" * 70)
    
    # Print dimension mapping
    print("\n📐 Dimension Mapping (Conceptual Framework):")
    for dim, cols in DIMENSION_MAPPING.items():
        print(f"  • {dim}: {cols}")
    
    # Z-Score Normalization
    scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
    
    print(f"\n✓ Z-Score Normalization applied to {len(FEATURE_COLUMNS)} features")
    print("  Formula: z = (x - μ) / σ")
    
    # Display normalization statistics
    print("\n📈 Post-Normalization Statistics (should be μ≈0, σ≈1):")
    stats = df_norm[FEATURE_COLUMNS].describe().loc[['mean', 'std']]
    print(stats.round(4).to_string())
    
    return df_norm, scaler


# ============================================================================
# STEP 3: INTERACTION ANALYSIS
# ============================================================================

def correlation_analysis(df_norm):
    """Calculate and visualize Pearson correlation matrix."""
    print("\n" + "=" * 70)
    print("STEP 3A: CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Calculate correlation matrix
    corr_matrix = df_norm[FEATURE_COLUMNS].corr(method='pearson')
    
    print("\n📊 Pearson Correlation Matrix:")
    print(corr_matrix.round(3).to_string())
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use short labels
    labels = [SHORT_LABELS.get(col, col) for col in FEATURE_COLUMNS]
    
    # Create mask for upper triangle (optional, for cleaner look)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=None,  # Show full matrix
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title('Indicator Correlation Matrix\n(Z-Score Normalized Data)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("\n✓ Heatmap saved as 'correlation_heatmap.png'")
    
    # Identify strong correlations
    print("\n🔗 Strong Correlations (|r| > 0.65):")
    strong_corr = []
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(i+1, len(FEATURE_COLUMNS)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.65:
                col1 = SHORT_LABELS.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i])
                col2 = SHORT_LABELS.get(FEATURE_COLUMNS[j], FEATURE_COLUMNS[j])
                strong_corr.append((FEATURE_COLUMNS[i], FEATURE_COLUMNS[j], r))
                print(f"  • {col1} ↔ {col2}: r = {r:.3f}")
    
    return corr_matrix, strong_corr


def granger_causality_analysis(df_norm, country='China'):
    """Perform Granger Causality Tests for specific country."""
    print("\n" + "=" * 70)
    print(f"STEP 3B: GRANGER CAUSALITY ANALYSIS (Case Study: {country})")
    print("=" * 70)
    
    # Extract country-specific time series
    try:
        df_country = df_norm.loc[country].sort_index()
    except KeyError:
        print(f"  ⚠ Country '{country}' not found in data!")
        return None
    
    print(f"\n📅 Time series length for {country}: {len(df_country)} years")
    
    # Define causality pairs to test
    causality_pairs = [
        ('AI_Patent_Share', 'GERD_USD_PPP', 'R&D Investment → AI Patents'),
        ('Commercial_Score', 'Supercomputer_TFlops', 'Supercomputer → Commercialization'),
        ('AI_Patent_Share', 'AI_Publication_Share', 'AI Publications → AI Patents'),
        ('Commercial_Score', 'AI_Patent_Share', 'AI Patents → Commercialization')
    ]
    
    max_lag = 3
    results_summary = []
    
    for y_col, x_col, description in causality_pairs:
        print(f"\n{'─' * 60}")
        print(f"Testing: {description}")
        print(f"  H0: {SHORT_LABELS.get(x_col, x_col)} does NOT Granger-cause {SHORT_LABELS.get(y_col, y_col)}")
        print(f"{'─' * 60}")
        
        # Prepare data for Granger test (requires 2D array: [y, x])
        test_data = df_country[[y_col, x_col]].dropna()
        
        if len(test_data) < max_lag + 3:
            print(f"  ⚠ Insufficient data points ({len(test_data)}) for lag={max_lag}")
            continue
        
        try:
            # Run Granger causality test
            gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            
            print(f"\n  {'Lag':<6} {'F-stat':<12} {'P-value':<12} {'Significance'}")
            print(f"  {'─' * 45}")
            
            for lag in range(1, max_lag + 1):
                # Get F-test results (most commonly used)
                f_stat = gc_result[lag][0]['ssr_ftest'][0]
                p_value = gc_result[lag][0]['ssr_ftest'][1]
                
                # Determine significance
                if p_value < 0.01:
                    sig = "*** (p<0.01)"
                elif p_value < 0.05:
                    sig = "**  (p<0.05)"
                elif p_value < 0.10:
                    sig = "*   (p<0.10)"
                else:
                    sig = "    (n.s.)"
                
                print(f"  {lag:<6} {f_stat:<12.4f} {p_value:<12.4f} {sig}")
                
                results_summary.append({
                    'Pair': description,
                    'Lag': lag,
                    'F_stat': f_stat,
                    'P_value': p_value,
                    'Significant': p_value < 0.05
                })
                
        except Exception as e:
            print(f"  ⚠ Error in Granger test: {str(e)[:50]}...")
            continue
    
    # Create summary DataFrame
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        print("\n📋 Granger Causality Summary Table:")
        print(results_df.to_string(index=False))
        results_df.to_csv('granger_causality_results.csv', index=False)
        print("\n✓ Results saved to 'granger_causality_results.csv'")
        return results_df
    
    return None


# ============================================================================
# STEP 4: MECHANISM PATH DIAGRAM (NETWORK VISUALIZATION)
# ============================================================================

def create_mechanism_diagram(corr_matrix, strong_corr, threshold=0.65):
    """Create directed network graph based on correlations and theoretical flow."""
    print("\n" + "=" * 70)
    print("STEP 4: MECHANISM PATH DIAGRAM")
    print("=" * 70)
    
    # Define theoretical causal flow (from -> to)
    # Based on the conceptual model: Policy/Infra → Innovation → Application
    theoretical_flow = {
        # Policy/Resources influence
        'GERD_USD_PPP': ['AI_Publication_Share', 'AI_Patent_Share', 'Commercial_Score'],
        # Infrastructure influence
        'Total_Generation_TWh': ['Supercomputer_TFlops'],
        'Renewable_Generation_TWh': ['Total_Generation_TWh'],
        'Broadband_Penetration': ['AI_Publication_Share', 'Commercial_Score'],
        'Supercomputer_TFlops': ['AI_Publication_Share', 'AI_Patent_Share'],
        # Innovation influence
        'AI_Publication_Share': ['AI_Patent_Share'],
        'AI_Patent_Share': ['Commercial_Score'],
    }
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for col in FEATURE_COLUMNS:
        G.add_node(
            col,
            label=SHORT_LABELS.get(col, col),
            dimension=get_dimension(col),
            color=get_node_color(col)
        )
    
    # Add edges based on correlation strength AND theoretical flow
    print("\n🔗 Network Edges (|correlation| > threshold AND theoretical flow):")
    edge_weights = []
    
    for source, targets in theoretical_flow.items():
        for target in targets:
            if source in corr_matrix.columns and target in corr_matrix.columns:
                corr_value = corr_matrix.loc[source, target]
                if abs(corr_value) > threshold:
                    G.add_edge(source, target, weight=abs(corr_value), corr=corr_value)
                    edge_weights.append(abs(corr_value))
                    print(f"  • {SHORT_LABELS.get(source, source)} → {SHORT_LABELS.get(target, target)}: r={corr_value:.3f}")
    
    # Also add strong correlations not in theoretical flow (as dashed lines)
    for col1, col2, r in strong_corr:
        if not G.has_edge(col1, col2) and not G.has_edge(col2, col1):
            # Add as undirected (bidirectional for visualization)
            if abs(r) > threshold:
                G.add_edge(col1, col2, weight=abs(r), corr=r, style='dashed')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define positions using hierarchical layout (left to right: Input → Process → Output)
    pos = {
        # Layer 1: Policy/Resources (Left)
        'GERD_USD_PPP': (0, 0.5),
        # Layer 2: Infrastructure (Center-Left)
        'Total_Generation_TWh': (1, 0.8),
        'Renewable_Generation_TWh': (1, 0.6),
        'Broadband_Penetration': (1, 0.4),
        'Supercomputer_TFlops': (1, 0.2),
        # Layer 3: Innovation (Center-Right)
        'AI_Publication_Share': (2, 0.6),
        'AI_Patent_Share': (2, 0.4),
        # Layer 4: Application (Right)
        'Commercial_Score': (3, 0.5),
    }
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [2000 for _ in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax
    )
    
    # Draw node labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels,
        font_size=9,
        font_weight='bold',
        ax=ax
    )
    
    # Draw edges with varying width based on correlation strength
    edges = G.edges(data=True)
    
    # Separate solid and dashed edges
    solid_edges = [(u, v) for u, v, d in edges if d.get('style') != 'dashed']
    dashed_edges = [(u, v) for u, v, d in edges if d.get('style') == 'dashed']
    
    # Edge widths based on correlation
    solid_widths = [G[u][v]['weight'] * 4 for u, v in solid_edges] if solid_edges else []
    dashed_widths = [G[u][v]['weight'] * 3 for u, v in dashed_edges] if dashed_edges else []
    
    # Edge colors based on positive/negative correlation
    def get_edge_color(u, v):
        corr = G[u][v].get('corr', 0)
        return '#2ecc71' if corr > 0 else '#e74c3c'  # Green for positive, Red for negative
    
    solid_colors = [get_edge_color(u, v) for u, v in solid_edges] if solid_edges else []
    dashed_colors = [get_edge_color(u, v) for u, v in dashed_edges] if dashed_edges else []
    
    # Draw solid edges (theoretical + strong correlation)
    if solid_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=solid_edges,
            width=solid_widths,
            edge_color=solid_colors,
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
    
    # Draw dashed edges (strong correlation only, no theoretical basis)
    if dashed_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=dashed_edges,
            width=dashed_widths,
            edge_color=dashed_colors,
            alpha=0.5,
            style='dashed',
            arrows=True,
            arrowsize=15,
            ax=ax
        )
    
    # Add edge labels (correlation values)
    edge_labels = {(u, v): f"{d['corr']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color='gray',
        ax=ax
    )
    
    # Add legend for dimensions
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=DIMENSION_COLORS['Policy_Resources'], 
                   markersize=12, label='Policy/Resources'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=DIMENSION_COLORS['Infrastructure'], 
                   markersize=12, label='Infrastructure'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=DIMENSION_COLORS['Innovation'], 
                   markersize=12, label='Innovation'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=DIMENSION_COLORS['Application'], 
                   markersize=12, label='Application'),
        plt.Line2D([0], [0], color='#2ecc71', linewidth=2, label='Positive Correlation'),
        plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Negative Correlation'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Add layer labels
    ax.text(0, 1.0, 'POLICY\nINPUT', ha='center', va='bottom', fontsize=10, 
            fontweight='bold', color='gray')
    ax.text(1, 1.0, 'INFRASTRUCTURE', ha='center', va='bottom', fontsize=10, 
            fontweight='bold', color='gray')
    ax.text(2, 1.0, 'INNOVATION', ha='center', va='bottom', fontsize=10, 
            fontweight='bold', color='gray')
    ax.text(3, 1.0, 'APPLICATION\nOUTPUT', ha='center', va='bottom', fontsize=10, 
            fontweight='bold', color='gray')
    
    ax.set_title('AI Development Mechanism Path Diagram\n(Based on Correlation Analysis, threshold=0.65)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mechanism_path_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\n✓ Mechanism diagram saved as 'mechanism_path_diagram.png'")
    print(f"  • Nodes: {G.number_of_nodes()}")
    print(f"  • Edges: {G.number_of_edges()}")
    
    return G


# ============================================================================
# BONUS: TEMPORAL TREND VISUALIZATION
# ============================================================================

def plot_temporal_trends(df, countries=['United States', 'China', 'United Kingdom']):
    """Plot temporal trends for key indicators across selected countries."""
    print("\n" + "=" * 70)
    print("BONUS: TEMPORAL TREND ANALYSIS")
    print("=" * 70)
    
    key_indicators = ['GERD_USD_PPP', 'AI_Patent_Share', 'Commercial_Score', 'Supercomputer_TFlops']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(countries)))
    
    for idx, indicator in enumerate(key_indicators):
        ax = axes[idx]
        
        for i, country in enumerate(countries):
            try:
                data = df.loc[country][indicator]
                ax.plot(data.index, data.values, marker='o', markersize=4,
                       label=country, color=colors[i], linewidth=2)
            except KeyError:
                continue
        
        ax.set_title(SHORT_LABELS.get(indicator, indicator), fontsize=11, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Trends of Key Indicators (2010-2025)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('temporal_trends.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✓ Temporal trend plot saved as 'temporal_trends.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "█" * 70)
    print("  PHASE 1: INDICATOR CONSTRUCTION & INTERACTION ANALYSIS")
    print("  2026 ICM Contest - Problem B: AI Development Competitiveness")
    print("█" * 70)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data(INPUT_FILE)
    
    # Step 2: Standardize data
    df_norm, scaler = standardize_data(df)
    
    # Step 3A: Correlation analysis
    corr_matrix, strong_corr = correlation_analysis(df_norm)
    
    # Step 3B: Granger causality analysis
    granger_results = granger_causality_analysis(df_norm, country='China')
    
    # Also run for USA for comparison
    print("\n" + "=" * 70)
    print("ADDITIONAL: GRANGER CAUSALITY FOR UNITED STATES")
    print("=" * 70)
    granger_results_usa = granger_causality_analysis(df_norm, country='United States')
    
    # Step 4: Mechanism path diagram
    G = create_mechanism_diagram(corr_matrix, strong_corr, threshold=0.65)
    
    # Bonus: Temporal trends
    plot_temporal_trends(df)
    
    # Final summary
    print("\n" + "█" * 70)
    print("  ANALYSIS COMPLETE")
    print("█" * 70)
    print("\n📁 Output Files Generated:")
    print("  1. correlation_heatmap.png      - Indicator correlation matrix")
    print("  2. granger_causality_results.csv - Granger test results")
    print("  3. mechanism_path_diagram.png   - Network visualization")
    print("  4. temporal_trends.png          - Time series trends")
    print("\n✅ Phase 1 analysis completed successfully!")
    
    return df, df_norm, corr_matrix, G


if __name__ == "__main__":
    df, df_norm, corr_matrix, G = main()
