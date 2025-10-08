#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Performance Radar Chart Visualization - Figure 3 (All Models)
Create radar chart showing performance comparison between all models
across total score and all evaluation dimensions (excluding TCM Design).
雷达图 - 所有模型
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from math import pi
warnings.filterwarnings('ignore')

# Set font and chart style (support English display)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
plt.style.use('default')

# Define color scheme for all 6 models
COLORS = {
    'grok': '#B07AA1',      # Purple
    'deepseek': '#59A14F',  # Green
    'expert': '#F28E2B',    # Orange
    'gemini': '#E15759',    # Red
    'claude': '#4E79A7',    # Blue
    'o3': '#76B7B2',        # Teal
    'grid': '#E0E0E0',      # Grid color
    'text': '#212121',      # Text color
    'background': '#FAFAFA' # Background color
}

# Model display name mapping
MODEL_NAMES = {
    'grok': 'Grok4',
    'deepseek': 'Deepseek-r1-0528',
    'expert': 'Expert',
    'gemini': 'Gemini-2.5-pro',
    'claude': 'claude-opus-4-20250514',
    'o3': 'ChatGPT-5'
}

# Dimension name mapping (English) - Remove TCM Design
DIMENSION_NAMES = {
    'weighted_score': 'Total Score',
    'clinical_safety': 'C & S',
    'scientific_evidence': 'S & E',
    'individual_solution': 'I & C',
    'clarity_education': 'C & L'
}


def safe_numeric_convert(series):
    """Safely convert series to numeric type"""
    return pd.to_numeric(series, errors='coerce')


def load_and_process_data():
    """Load and process evaluation data - Chinese evaluation group only"""
    print("🔄 Loading Chinese evaluation group data...")

    try:
        # Try to read Chinese evaluation group data file first
        file_path = '中文评估组_数据.xlsx'
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Successfully loaded Chinese evaluation data, {len(df)} records")
        except:
            # Fallback: read from consolidated file and filter Chinese data
            print("���� Trying to load from consolidated file and filter Chinese data...")
            consolidated_file = '专家评估数据汇总.xlsx'
            try:
                df_all = pd.read_excel(consolidated_file, sheet_name='全部数据')
                # Filter for Chinese evaluation group
                chinese_experts = ['c1566', 'c3562', 'c8655']
                df = df_all[df_all['expert_id'].isin(chinese_experts)].copy()
                print(f"✅ Filtered Chinese evaluation data, {len(df)} records")
            except:
                df_all = pd.read_excel(consolidated_file)
                chinese_experts = ['c1566', 'c3562', 'c8655']
                df = df_all[df_all['expert_id'].isin(chinese_experts)].copy()
                print(f"✅ Filtered Chinese evaluation data from default sheet, {len(df)} records")

        print(f"📊 Contains experts: {df['expert_id'].nunique()}")
        print(f"📋 Contains cases: {df['case_id'].nunique()}")
        print(f"📂 Contains categories: {df['category'].nunique()}")
        print(f"🤖 Contains evaluators: {df['evaluator'].nunique()}")

        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def calculate_radar_data(df):
    """Calculate mean scores for radar chart dimensions for all models"""
    print("\n📊 Calculating radar chart data for all models...")

    # Include all target evaluators
    target_evaluators = ['grok', 'deepseek', 'expert', 'gemini', 'claude', 'o3']
    df_filtered = df[df['evaluator'].isin(target_evaluators)].copy()

    if df_filtered.empty:
        print("❌ No data found for target evaluators")
        return None

    print(f"📋 Found evaluators: {df_filtered['evaluator'].unique()}")

    # Define dimensions for radar chart (without TCM)
    dimensions = ['weighted_score', 'clinical_safety', 'scientific_evidence',
                 'individual_solution', 'clarity_education']

    radar_data = {}

    for evaluator in target_evaluators:
        evaluator_data = df_filtered[df_filtered['evaluator'] == evaluator]

        if len(evaluator_data) == 0:
            print(f"⚠️ No data found for {evaluator}")
            continue

        scores = {}

        for dim in dimensions:
            if dim in evaluator_data.columns:
                dim_scores = safe_numeric_convert(evaluator_data[dim])
                dim_scores = dim_scores.dropna()

                if len(dim_scores) > 0:
                    scores[dim] = dim_scores.mean()
                else:
                    scores[dim] = 0.0
            else:
                scores[dim] = 0.0

        radar_data[evaluator] = scores

        # Print statistics
        print(f"\n📈 {MODEL_NAMES[evaluator]} Performance:")
        for dim in dimensions:
            print(f"   {DIMENSION_NAMES[dim]}: {scores[dim]:.2f}")

    return radar_data


def create_all_models_radar_chart(radar_data, save_path='Figure3_all_models_radar_chart222.png'):
    """Create radar chart comparing all 6 models"""
    print("\n🎯 Creating radar chart for all models...")

    if not radar_data:
        print("❌ No radar data available")
        return

    # Get dimensions
    dimensions = list(DIMENSION_NAMES.keys())
    dimension_labels = [DIMENSION_NAMES[dim] for dim in dimensions]

    # Number of dimensions
    N = len(dimensions)

    # Calculate angles for each dimension
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Create figure and polar subplot with compact size and layout
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(projection='polar'), constrained_layout=True)

    # Clean white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Make first axis at the top and go clockwise for better readability
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Plot data for each evaluator
    model_order = ['grok', 'gemini', 'o3', 'deepseek', 'claude', 'expert']  # Order by performance
    line_styles = ['-', '-', '-', '-', '-', '-']

    for i, evaluator in enumerate(model_order):
        if evaluator not in radar_data:
            continue
        # Get scores for this evaluator
        values = [radar_data[evaluator][dim] for dim in dimensions]
        values += values[:1]  # Complete the circle

        color = COLORS[evaluator]
        label = MODEL_NAMES[evaluator]
        linestyle = line_styles[i % len(line_styles)]

        ax.plot(angles, values, linestyle=linestyle, linewidth=2.0, label=label,
                color=color, markersize=4, marker='o', markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=0.8, alpha=0.9)
        ax.fill(angles, values, alpha=0.06, color=color)

    # Customize the chart
    ax.set_xticks(angles[:-1])
    # Format dimension labels to prevent overlap
    formatted_labels = []
    for label in dimension_labels:
        if len(label) > 12:
            words = label.split(' ')
            if len(words) > 1:
                mid = len(words) // 2
                formatted_labels.append('\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])]))
            else:
                formatted_labels.append(label)
        else:
            formatted_labels.append(label)

    ax.set_xticklabels(formatted_labels, fontsize=11, fontweight='bold')

    # Set y-axis (radial axis) properties: tighter ticks and subtle grid
    ax.set_ylim(2.5, 5)
    ax.set_yticks([3, 4, 5])
    ax.set_yticklabels(['3.0', '4.0', '5.0'], fontsize=13, alpha=0.9,
                       fontfamily='Times New Roman', fontweight='bold', color='#333333')
    ax.grid(True, alpha=0.25, linewidth=0.9, color='#9E9E9E')

    # Remove default outer circle visibility for cleaner look
    ax.spines['polar'].set_visible(False)

    # Minimal internal padding by keeping everything inside axes
    # No title to save vertical space (paper captions will describe the figure)

    # Legend inside the plot to avoid extra blank canvas
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1), fontsize=12,
                        frameon=False, ncol=1, labelspacing=0.4, handlelength=2.0)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    # Tighten layout
    plt.tight_layout(pad=0.2)

    # Save figure (high-resolution PNG + vector formats for publication)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.03)
    base, ext = (save_path.rsplit('.', 1) + ['png'])[:2]
    pdf_path = f"{base}.pdf"
    svg_path = f"{base}.svg"
    try:
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', pad_inches=0.03)
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white', pad_inches=0.03)
        print(f"✅ Also saved vector formats: {pdf_path}, {svg_path}")
    except Exception as e:
        print(f"⚠️ Vector export failed: {e}")

    print(f"✅ All models radar chart saved to: {save_path}")
    plt.show()


def generate_comprehensive_statistics(radar_data):
    """Generate comprehensive statistical summary for all models"""
    print("\n📈 Generating comprehensive statistics for all models...")

    if not radar_data:
        print("❌ No radar data available")
        return

    # Create statistics DataFrame
    stats_data = []
    dimensions = list(DIMENSION_NAMES.keys())

    for evaluator in radar_data.keys():
        row = {'Model': MODEL_NAMES[evaluator]}
        for dim in dimensions:
            row[DIMENSION_NAMES[dim]] = radar_data[evaluator][dim]
        stats_data.append(row)

    df_stats = pd.DataFrame(stats_data)

    # Calculate rankings for each dimension
    for dim in dimensions:
        col_name = DIMENSION_NAMES[dim]
        if col_name in df_stats.columns:
            df_stats[f'{col_name}_Rank'] = df_stats[col_name].rank(ascending=False, method='dense').astype(int)

    # Sort by total score
    df_stats = df_stats.sort_values('Total Score', ascending=False).reset_index(drop=True)

    # Print comprehensive statistics
    print("\n📊 Comprehensive Performance Summary (All Models):")
    print("=" * 100)
    print(f"{'Rank':<4} {'Model':<18} {'Total':<7} {'Clinical':<9} {'Evidence':<9} {'Individual':<11} {'Clarity':<9}")
    print("=" * 100)

    for i, row in df_stats.iterrows():
        print("{rank:<4d} {model:<18s} {total:<7.2f} {clinical:<9.2f} {evidence:<9.2f} {individual:<11.2f} {clarity:<9.2f}".format(
            rank=int(i + 1),
            model=str(row['Model']),
            total=float(row['Total Score']),
            clinical=float(row['C & S']),
            evidence=float(row['S & E']),
            individual=float(row['I & C']),
            clarity=float(row['C & L'])
        ))

    # Save comprehensive statistics to Excel
    excel_file = 'all_models_radar_statistics.xlsx'
    df_stats.to_excel(excel_file, index=False)
    print(f"\n✅ Comprehensive statistics saved to: {excel_file}")

    return df_stats


def main():
    """Main function"""
    print("=" * 80)
    print("🎯 LLM Performance Radar Chart Visualization - All Models (No TCM)")
    print("=" * 80)

    # Load data
    df = load_and_process_data()
    if df is None:
        return

    # Calculate radar data
    radar_data = calculate_radar_data(df)
    if radar_data is None:
        return

    # Generate comprehensive statistics
    stats_df = generate_comprehensive_statistics(radar_data)

    # Create only the main radar chart with all models
    create_all_models_radar_chart(radar_data, 'Figure3_all_models_radar_chart.png')

    print("\n" + "=" * 80)
    print("✅ Radar chart generated successfully!")
    print("📁 Output files:")
    print("   - Figure3_all_models_radar_chart.png (All 6 Models Radar Chart)")
    print("   - all_models_radar_statistics.xlsx (Comprehensive Statistics)")
    print("=" * 80)


if __name__ == "__main__":
    main()
