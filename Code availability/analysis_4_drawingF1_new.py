#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Performance Distribution Visualization Analysis - Figure 2
Based on analysis_3_compare2.py results, create:
1. Violin plots: Show overall distribution and density of scores for each model across all cases
2. Cumulative frequency plots: Show percentage of cases where models achieve certain scores or below
小提琴图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Set font to Times New Roman - global configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
# Embed TrueType fonts in PDF/PS for better publication compatibility
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.style.use('default')

# Define color scheme
COLORS = {
    'models': ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#B07AA1', '#76B7B2'],  # Model colors
    'expert_line': '#D32F2F',    # Expert baseline color (red)
    'background': '#FAFAFA',     # Background color
    'grid': '#E0E0E0',          # Grid color
    'text': '#212121',          # Text color
    'significance': '#FF6B6B'    # Significance marker color
}

# Model display name mapping
MODEL_NAMES = {
    'grok': 'Grok4',
    'gemini': 'Gemini-2.5-pro',
    'o3': 'ChatGPT-5',
    'deepseek': 'Deepseek-r1-0528',
    'claude': 'claude-opus-4-20250514',
    'expert': 'Expert'
}

# --- Added for p-value annotations (aligned with Figure1 script) ---
def format_pvalue(p_value):
    """Format P-value display"""
    if pd.isna(p_value):
        return "N/A"
    elif p_value < 0.001:
        return "p<0.001"
    else:
        return f"p={p_value:.3f}"


def get_significance_stars(p_value):
    """Return significance stars based on P-value"""
    if pd.isna(p_value):
        return ""
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def load_wilcoxon_pvalues():
    """Load Wilcoxon test p-values for Chinese group from extended results.
    Returns a dict: {model_code -> total_p_value}. If not found, return empty dict.
    """
    try:
        file_path = 'LLM性能比较分析结果_扩展版.xlsx'
        df_w = pd.read_excel(file_path, sheet_name='中文组_威尔科克森检验')
        p_map = {}
        # Expecting a 'model' column containing evaluator codes and 'total_p_value'
        for _, row in df_w.iterrows():
            model_code = str(row.get('model', '')).strip()
            if model_code:
                p_map[model_code] = row.get('total_p_value', np.nan)
        if not p_map:
            print('⚠️ 未在威尔科克森检验表中解析到模型P值')
        return p_map
    except Exception as e:
        print(f"⚠️ 读取威尔科克森检验P值失败: {e}")
        return {}


def get_evaluator_order_and_colors(df):
    """Return stable evaluator order and a consistent display-name->color map."""
    preferred_order = ['grok', 'gemini', 'o3', 'deepseek', 'claude', 'expert']
    present = list(df['evaluator'].dropna().unique())
    # Keep preferred order for present evaluators
    eval_order_codes = [e for e in preferred_order if e in present]
    # Append any unexpected evaluators at the end in sorted order
    extras = sorted([e for e in present if e not in preferred_order])
    eval_order_codes.extend(extras)

    # Fixed color map by evaluator code (fallback cycles through COLORS['models'])
    base_map = {
        'grok': COLORS['models'][0],
        'gemini': COLORS['models'][1],
        'o3': COLORS['models'][2],
        'deepseek': COLORS['models'][3],
        'claude': COLORS['models'][4],
        'expert': COLORS['models'][5],
    }
    color_map_display = {}
    for i, code in enumerate(eval_order_codes):
        color = base_map.get(code, COLORS['models'][i % len(COLORS['models'])])
        label = MODEL_NAMES.get(code, code)
        color_map_display[label] = color

    display_order = [MODEL_NAMES.get(code, code) for code in eval_order_codes]
    return eval_order_codes, display_order, color_map_display


def safe_numeric_convert(series):
    """Safely convert series to numeric type"""
    return pd.to_numeric(series, errors='coerce')


def load_and_process_data():
    """Load and process evaluation data - Chinese evaluation group only"""
    print("🔄 Loading Chinese evaluation group data only...")

    try:
        # Read the Chinese evaluation group data file
        file_path = '中文评估组_数据.xlsx'

        try:
            df = pd.read_excel(file_path)
            print(f"✅ Successfully loaded Chinese evaluation data, {len(df)} records")
        except Exception as e:
            print(f"❌ Error reading Chinese evaluation file: {e}")
            # Fallback: try to read from consolidated file and filter Chinese data
            print("���� Trying to load from consolidated file and filter Chinese data...")
            consolidated_file = '专家评估数据汇总.xlsx'
            try:
                df_all = pd.read_excel(consolidated_file, sheet_name='全部数据')
                # Filter for Chinese evaluation group (assuming there's a language or group indicator)
                # Check if there's a language column or expert_id pattern to identify Chinese group
                if 'language' in df_all.columns:
                    df = df_all[df_all['language'] == 'chinese'].copy()
                elif 'expert_id' in df_all.columns:
                    # Filter based on Chinese expert IDs (1566, 3562, 8655 based on file names)
                    chinese_experts = ['1566', '3562', '8655']
                    df = df_all[df_all['expert_id'].astype(str).isin(chinese_experts)].copy()
                else:
                    # If no clear identifier, use the Chinese evaluation file directly
                    df = pd.read_excel('中文评估组_数据.xlsx')
                print(f"✅ Filtered Chinese evaluation data, {len(df)} records")
            except:
                print("❌ Cannot load Chinese evaluation data")
                return None

        print(f"📊 Contains experts: {df['expert_id'].nunique()}")
        print(f"📋 Contains cases: {df['case_id'].nunique()}")
        print(f"📂 Contains categories: {df['category'].nunique()}")
        print(f"🤖 Contains evaluators: {df['evaluator'].nunique()}")

        return df

    except FileNotFoundError:
        print("❌ Cannot find 中文评估组_数据.xlsx file")
        print("💡 Please ensure the Chinese evaluation data file exists")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def prepare_violin_data(df):
    """Prepare data for violin plot"""
    print("\n📊 Preparing violin plot data...")

    # Ensure weighted_score column exists
    if 'weighted_score' not in df.columns:
        print("❌ Missing weighted_score column in data")
        return None

    # Filter valid data
    df_clean = df.copy()
    df_clean['weighted_score'] = safe_numeric_convert(df_clean['weighted_score'])
    df_clean = df_clean.dropna(subset=['weighted_score'])

    # Get all evaluators
    evaluators = df_clean['evaluator'].unique()
    print(f"📋 Found evaluators: {list(evaluators)}")

    return df_clean


def create_violin_plot(df, save_path='violin_plot_distribution.png'):
    """Create violin plot showing score distribution"""
    print("\n🎻 Creating violin plot...")

    # Set figure size
    plt.figure(figsize=(12, 8))

    # Prepare data and consistent colors
    eval_order_codes, display_order, color_map_display = get_evaluator_order_and_colors(df)

    # Prepare data for each evaluator
    plot_data = []
    for evaluator in eval_order_codes:
        eval_data = df[df['evaluator'] == evaluator]['weighted_score']
        eval_data = eval_data.dropna()
        if len(eval_data) > 0:
            for score in eval_data:
                plot_data.append({
                    'evaluator': MODEL_NAMES.get(evaluator, evaluator),
                    'score': score
                })

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        print("❌ No valid data for violin plot")
        return

    # Create violin plot using seaborn with fixed order and palette
    ax = plt.gca()
    palette_list = [color_map_display[label] for label in display_order if label in plot_df['evaluator'].unique()]
    order_in_data = [label for label in display_order if label in plot_df['evaluator'].unique()]
    sns.violinplot(data=plot_df, x='evaluator', y='score',
                   palette=palette_list, ax=ax,
                   inner='box', cut=0, order=order_in_data)

    # Customize appearance with Times New Roman font
    ax.set_xlabel('Model/Evaluator', fontsize=12, fontweight='bold', labelpad=10, fontfamily='Times New Roman')
    ax.set_ylabel('Weighted Score', fontsize=12, fontweight='bold', labelpad=10, fontfamily='Times New Roman')
    # ax.set_title('LLM Model Score Distribution (Violin Plot) - Chinese Evaluation Group\nShowing score distribution and density for each model across all cases',
    #              fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')

    # Set tick labels font
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=0, ha='center')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLORS['background'])

    # Set y-axis limits for better visualization
    ax.set_ylim(1.5, 5.0)

    # Add statistical information as text box
    stats_text = []
    for evaluator in plot_df['evaluator'].unique():
        eval_scores = plot_df[plot_df['evaluator'] == evaluator]['score']
        mean_score = eval_scores.mean()
        std_score = eval_scores.std()
        n_cases = len(eval_scores)
        stats_text.append(f"{evaluator}: μ={mean_score:.2f}, σ={std_score:.2f}")

    # Position text box in the bottom-left corner of the axes area
    ax.text(0.02, 0.02, '\n'.join(stats_text), transform=ax.transAxes,
            fontsize=11, fontfamily='Times New Roman',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray'))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Violin plot saved to: {save_path}")
    plt.show()


def create_cumulative_frequency_plot(df, save_path='cumulative_frequency_plot.png'):
    """Create cumulative frequency plot"""
    print("\n📈 Creating cumulative frequency plot...")

    # Set figure size
    plt.figure(figsize=(12, 8))

    # Consistent order and color map
    eval_order_codes, display_order, color_map_display = get_evaluator_order_and_colors(df)

    # Calculate cumulative frequency for each evaluator in fixed order
    n_evals = len(eval_order_codes)
    for i, evaluator in enumerate(eval_order_codes):
        eval_data = df[df['evaluator'] == evaluator]['weighted_score']
        eval_data = eval_data.dropna()
        if len(eval_data) == 0:
            continue

        sorted_scores = np.sort(eval_data)
        cumulative_freq = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100

        label = MODEL_NAMES.get(evaluator, evaluator)
        color = color_map_display[label]

        plt.plot(sorted_scores, cumulative_freq,
                 label=f'{label}', color=color, linewidth=2.5, marker='o', markersize=3, alpha=0.8)

        # Add key percentile markers with better positioning
        percentiles = [25, 50, 75, 90]
        for j, percentile in enumerate(percentiles):
            score_at_percentile = np.percentile(sorted_scores, percentile)
            # Offset vertical lines slightly to avoid overlap
            offset = 0.02 * (i - n_evals / 2)
            plt.axvline(x=score_at_percentile + offset, color=color, linestyle='--', alpha=0.4, linewidth=1)

            # Position text to avoid overlap with Times New Roman font
            y_pos = percentile + (i - n_evals / 2) * 3
            plt.text(score_at_percentile + offset, y_pos, f'P{percentile}',
                    rotation=0, ha='center', va='center', fontsize=7,
                    color=color, fontweight='bold', fontfamily='Times New Roman',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color))

    # Set labels and title with Times New Roman font
    plt.xlabel('Score', fontsize=12, fontweight='bold', labelpad=10, fontfamily='Times New Roman')
    plt.ylabel('Cumulative Frequency (%)', fontsize=12, fontweight='bold', labelpad=10, fontfamily='Times New Roman')
    plt.title('LLM Model Score Cumulative Frequency Plot - Chinese Evaluation Group\nShowing percentage of cases achieving certain scores or below',
              fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')

    # Set tick labels font
    ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set background color
    plt.gca().set_facecolor(COLORS['background'])

    # Add legend with better positioning and Times New Roman font
    legend = plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    # Set axis ranges
    plt.ylim(0, 105)
    plt.xlim(1.8, 5.2)

    # Add reference line
    plt.axhline(y=50, color=COLORS['expert_line'], linestyle='-', alpha=0.7, linewidth=1)
    plt.text(plt.xlim()[1] * 0.98, 52, 'Median Line', ha='right', va='bottom',
             color=COLORS['expert_line'], fontweight='bold', fontsize=10, fontfamily='Times New Roman')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Cumulative frequency plot saved to: {save_path}")
    plt.show()


def create_combined_distribution_analysis(df, save_path='combined_distribution_analysis.png'):
    """Create combined distribution analysis plot"""
    print("\n📊 Creating combined distribution analysis plot...")

    # Create subplots with adjusted spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    # fig.suptitle('LLM Model Performance Distribution Analysis - Chinese Evaluation Group',
    #              fontsize=16, fontweight='bold', y=0.98, fontfamily='Times New Roman')

    # Consistent order and color map
    eval_order_codes, display_order, color_map_display = get_evaluator_order_and_colors(df)

    # Prepare data for plots
    plot_data = []
    for evaluator in eval_order_codes:
        eval_data = df[df['evaluator'] == evaluator]['weighted_score'].dropna()
        if len(eval_data) > 0:
            for score in eval_data:
                plot_data.append({
                    'evaluator': MODEL_NAMES.get(evaluator, evaluator),
                    'score': score
                })

    plot_df = pd.DataFrame(plot_data)

    # 2. Cumulative frequency plot (top right) -> moved to ax2
    ax2.clear()
    for evaluator in eval_order_codes:
        eval_data = df[df['evaluator'] == evaluator]['weighted_score'].dropna()
        if len(eval_data) == 0:
            continue
        sorted_scores = np.sort(eval_data)
        cumulative_freq = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
        label = MODEL_NAMES.get(evaluator, evaluator)
        color = color_map_display[label]
        ax2.plot(sorted_scores, cumulative_freq, label=label, color=color, linewidth=2)
    ax2.set_title('b. Cumulative Frequency Plot', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_xlabel('Score', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('Cumulative Frequency (%)', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(COLORS['background'])
    legend2 = ax2.legend(fontsize=12)
    for text in legend2.get_texts():
        text.set_fontfamily('Times New Roman')
    for tick in ax2.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
    for tick in ax2.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax2.tick_params(axis='both', labelsize=15)

    # 3. Violin plot (bottom left) -> moved to ax3
    if len(plot_df) > 0:
        order_in_data = [label for label in display_order if label in plot_df['evaluator'].unique()]
        palette_list = [color_map_display[label] for label in order_in_data]
        sns.violinplot(data=plot_df, x='evaluator', y='score', ax=ax3,
                       palette=palette_list, inner='box', cut=0, order=order_in_data)
        ax3.set_title('c. Score Distribution (Violin Plot)', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
        ax3.set_xlabel('Model/Evaluator', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
        ax3.set_ylabel('Weighted Score', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
        ax3.tick_params(axis='x', rotation=45, labelsize=15)
        ax3.tick_params(axis='y', labelsize=15)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor(COLORS['background'])
        for tick in ax3.get_xticklabels():
            tick.set_fontfamily('Times New Roman')
        for tick in ax3.get_yticklabels():
            tick.set_fontfamily('Times New Roman')

    # 4. Box plot (bottom right) -> moved to ax4
    if len(plot_df) > 0:
        order_in_data = [label for label in display_order if label in plot_df['evaluator'].unique()]
        palette_list = [color_map_display[label] for label in order_in_data]
        sns.boxplot(data=plot_df, x='evaluator', y='score', ax=ax4,
                    palette=palette_list, order=order_in_data)
        ax4.set_title('d. Score Distribution (Box Plot)', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
        ax4.set_xlabel('Model/Evaluator', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
        ax4.set_ylabel('Weighted Score', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
        ax4.tick_params(axis='x', rotation=45, labelsize=15)
        ax4.tick_params(axis='y', labelsize=15)
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor(COLORS['background'])
        for tick in ax4.get_xticklabels():
            tick.set_fontfamily('Times New Roman')
        for tick in ax4.get_yticklabels():
            tick.set_fontfamily('Times New Roman')

    # 1. Mean score bar chart with expert baseline (top left) -> ax1
    ax1.clear()
    order_for_barplot = [label for label in display_order if label != 'Expert' and label in plot_df['evaluator'].unique()]
    sns.barplot(data=plot_df[plot_df['evaluator'] != 'Expert'],
                x='evaluator', y='score', ax=ax1,
                palette=color_map_display,
                order=order_for_barplot,
                edgecolor='black', linewidth=0.8,
                errorbar=None)
    ax1.set_title('a. Mean Score by Model', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_xlabel('Model', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('Weighted Score', fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_facecolor(COLORS['background'])
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    # --- Add score, p-value and significance stars annotations ---
    bars = ax1.patches
    display_to_code = {v: k for k, v in MODEL_NAMES.items()}
    p_map = load_wilcoxon_pvalues()

    bar_heights = []
    for i, label in enumerate(order_for_barplot):
        if i >= len(bars):
            break
        p_rect = bars[i]
        height = p_rect.get_height()
        bar_heights.append(height)
        # score label (existing behavior, slightly adjusted position)
        ax1.text(p_rect.get_x() + p_rect.get_width() / 2., height + 0.03, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=14, fontfamily='Times New Roman')
        # p-value and stars
        code = display_to_code.get(label, None)
        p_val = p_map.get(code, np.nan) if code else np.nan
        p_text = format_pvalue(p_val)
        stars = get_significance_stars(p_val)
        y_p = height + 0.22
        y_s = height + 0.34
        ax1.text(p_rect.get_x() + p_rect.get_width() / 2., y_p, p_text,
                 ha='center', va='bottom', fontsize=14,
                 color=COLORS['significance'] if (not pd.isna(p_val) and p_val < 0.05) else COLORS['text'],
                 fontfamily='Times New Roman')
        if stars:
            ax1.text(p_rect.get_x() + p_rect.get_width() / 2., y_s, stars,
                     ha='center', va='bottom', fontsize=15, fontweight='bold',
                     color=COLORS['significance'], fontfamily='Times New Roman')

    # Expert baseline
    expert_scores = df[df['evaluator'] == 'expert']['weighted_score'].dropna()
    expert_mean = None
    if not expert_scores.empty:
        expert_mean = expert_scores.mean()
        ax1.axhline(y=expert_mean, color=COLORS['expert_line'], linestyle='-', linewidth=2,
                    label=f"Expert mean ({expert_mean:.2f})")
        legend1 = ax1.legend(fontsize=12)
        for text in legend1.get_texts():
            text.set_fontfamily('Times New Roman')

    for tick in ax1.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_rotation(15)
    for tick in ax1.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax1.tick_params(axis='both', labelsize=15)

    # Adjust y-limit to avoid overlap with annotations
    if bar_heights:
        top_needed = max(bar_heights) + 0.55
        if expert_mean is not None:
            top_needed = max(top_needed, expert_mean + 0.3)
        ax1.set_ylim(1.5, max(5.0, top_needed))
    else:
        ax1.set_ylim(1.5, 5.0)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    # Also export a vector PDF for publication
    pdf_path = save_path[:-4] + '.pdf' if save_path.lower().endswith('.png') else save_path + '.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✅ Combined distribution analysis plot saved to: {save_path}")
    print(f"✅ Combined distribution analysis PDF saved to: {pdf_path}")
    plt.show()


def generate_distribution_statistics(df):
    """Generate distribution statistics"""
    print("\n📈 Generating distribution statistics...")

    stats_results = []

    for evaluator in sorted(df['evaluator'].unique()):
        eval_data = df[df['evaluator'] == evaluator]['weighted_score'].dropna()

        if len(eval_data) == 0:
            continue

        # Calculate basic statistics
        mean_score = eval_data.mean()
        median_score = eval_data.median()
        std_score = eval_data.std()
        min_score = eval_data.min()
        max_score = eval_data.max()
        q25 = eval_data.quantile(0.25)
        q75 = eval_data.quantile(0.75)
        iqr = q75 - q25

        # Calculate skewness and kurtosis
        skewness = stats.skew(eval_data)
        kurtosis = stats.kurtosis(eval_data)

        # Calculate percentiles
        p10 = eval_data.quantile(0.10)
        p90 = eval_data.quantile(0.90)

        stats_results.append({
            'evaluator': MODEL_NAMES.get(evaluator, evaluator),
            'n_cases': len(eval_data),
            'mean': mean_score,
            'median': median_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'Q25': q25,
            'Q75': q75,
            'IQR': iqr,
            'P10': p10,
            'P90': p90,
            'skewness': skewness,
            'kurtosis': kurtosis
        })

    stats_df = pd.DataFrame(stats_results)

    # Print statistics results
    print("\n📊 Distribution Statistics Summary:")
    print("=" * 120)
    print(f"{'Model':<12} {'N':<6} {'Mean':<6} {'Median':<7} {'Std':<6} {'Min':<6} {'Max':<6} {'Skew':<7} {'Kurt':<7}")
    print("-" * 120)

    for _, row in stats_df.iterrows():
        print(f"{row['evaluator']:<12} {row['n_cases']:<6} {row['mean']:<6.2f} {row['median']:<7.2f} "
              f"{row['std']:<6.2f} {row['min']:<6.2f} {row['max']:<6.2f} {row['skewness']:<7.2f} {row['kurtosis']:<7.2f}")

    # Save statistics results
    stats_df.to_excel('distribution_statistics_summary.xlsx', index=False)
    print(f"\n✅ Distribution statistics saved to: distribution_statistics_summary.xlsx")

    return stats_df


def main():
    """Main function"""
    print("=" * 70)
    print("🎨 LLM Performance Distribution Visualization Analysis - Figure 2")
    print("=" * 70)

    # Load data
    df = load_and_process_data()
    if df is None:
        return

    # Prepare data
    df_clean = prepare_violin_data(df)
    if df_clean is None:
        return

    # Generate distribution statistics
    stats_df = generate_distribution_statistics(df_clean)

    # Create violin plot
    create_violin_plot(df_clean, 'Figure2A_violin_plot_distribution_new.png')

    # Create cumulative frequency plot
    create_cumulative_frequency_plot(df_clean, 'Figure2B_cumulative_frequency_plot_new.png')

    # Create combined distribution analysis plot
    create_combined_distribution_analysis(df_clean, 'Figure2_combined_distribution_analysis_new.png')

    print("\n" + "=" * 70)
    print("✅ All charts generated successfully!")
    print("📁 Output files:")
    print("   - Figure2A_violin_plot_distribution.png (Violin Plot)")
    print("   - Figure2B_cumulative_frequency_plot.png (Cumulative Frequency Plot)")
    print("   - Figure2_combined_distribution_analysis.png (Combined Analysis Plot)")
    print("   - distribution_statistics_summary.xlsx (Distribution Statistics Summary)")
    print("=" * 70)


if __name__ == "__main__":
    main()
