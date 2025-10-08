#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各模型在六大康复亚专科中的性能热力图绘制脚本
基于分科室比较分析结果创建热�����图可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# 新增：确保PDF/PS导出为可嵌入的TrueType字体，便于论文矢量输出
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def load_department_analysis_data():
    """加载分科室分析数据"""
    print("🔄 正在加载分科室性能比较分析结果...")

    try:
        # 读取详细分析数据
        df = pd.read_excel('LLM分科室性能比较分析结果.xlsx', sheet_name='详细分析')
        print(f"✅ 成功加载数据，共 {len(df)} 条记录")

        # 读取跨科室分析数据用于排序
        cross_df = pd.read_excel('LLM分科室性能比较分析结果.xlsx', sheet_name='跨科室分析')
        print(f"📊 加载跨科室分析数据，共 {len(cross_df)} 个被评估者")

        return df, cross_df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None

def prepare_heatmap_data(df, cross_df):
    """准��热力图数据"""
    print("\n🔄 准备热力图数据...")

    # 创建数据透视表
    heatmap_data = df.pivot(index='被评估者', columns='科室', values='总分平均值')
    ranking_data = df.pivot(index='被评估者', columns='科室', values='总分排名')

    # 按照跨科室总体排名对被评估者排序（从高分到低分）
    evaluator_order = cross_df.sort_values('总分平均值', ascending=False)['被评估者'].tolist()

    # 按照各科室的平均分对科室排序（从高分到低分）
    department_avg_scores = df.groupby('科室')['总分平均值'].mean().sort_values(ascending=False)
    department_order = department_avg_scores.index.tolist()

    # 重新排序热力图数据
    heatmap_data = heatmap_data.reindex(index=evaluator_order, columns=department_order)
    ranking_data = ranking_data.reindex(index=evaluator_order, columns=department_order)

    # 创建组��标注数据（得分/排名）
    combined_annotations = heatmap_data.copy()
    for i in range(len(evaluator_order)):
        for j in range(len(department_order)):
            score = heatmap_data.iloc[i, j]
            rank = ranking_data.iloc[i, j]
            combined_annotations.iloc[i, j] = f"{score:.3f}/{int(rank)}"

    print(f"📈 热力图数据形状: {heatmap_data.shape}")
    print(f"🏆 被评估者排序: {evaluator_order}")
    print(f"🏥 科室排序: {department_order}")

    return heatmap_data, ranking_data, combined_annotations, evaluator_order, department_order

def create_performance_heatmap(heatmap_data, combined_annotations, evaluator_order, department_order):
    """创建性能��力图"""
    print("\n🔄 创建性能热力图...")

    # 创建图形（缩小整体尺寸以减小单元格）
    fig, ax = plt.subplots(figsize=(14, 8))

    # 创建热力图，使用组合标注（得分/排名）
    sns.heatmap(heatmap_data,
                annot=combined_annotations,  # 使用组合标注
                fmt='',      # 字符串格式（因为包含斜杠）
                cmap='RdYlGn',  # 颜色映射：从红色到黄色到绿色
                vmin=3.5,    # 最小值
                vmax=4.8,    # 最大值
                center=4.15, # 中心值
                square=True, # 正方形单元格
                linewidths=0.5,  # 细化网格线
                cbar_kws={'shrink': 0.55, 'label': 'Average Score', 'pad': 0.06},  # 更紧凑的颜色条
                annot_kws={'fontsize': 10, 'fontweight': 'bold', 'fontfamily': 'Times New Roman'},  # 缩小注释字号
                ax=ax)

    # 设置标题和标签（全英文）
    # ax.set_title('Performance Heatmap of Models Across Six Rehabilitation Subspecialties\n(Format: Score/Rank)',
    #             fontsize=18, fontweight='bold', pad=30, fontfamily='Times New Roman')
    ax.set_xlabel('Rehabilitation Subspecialties', fontsize=15, fontweight='bold', labelpad=10, fontfamily='Times New Roman')
    ax.set_ylabel('Evaluated Models', fontsize=15, fontweight='bold', labelpad=10, fontfamily='Times New Roman')

    # 英文科室名称映射
    department_english = {
        '神经康复': 'Neurological\nRehabilitation',
        '骨科康复': 'Orthopedic\nRehabilitation',
        '肿瘤康复': 'Oncology\nRehabilitation',
        '脏器康复': 'Visceral\nRehabilitation',
        '盆底康复': 'Pelvic Floor\nRehabilitation',
        '吞咽康复': 'Swallowing\nRehabilitation'
    }

    # 英文模型名称映射
    evaluator_english = {
        'grok': 'Grok4',
        'gemini': 'Gemini-2.5-pro',
        'o3': 'ChatGPT-5',
        'deepseek': 'Deepseek-r1-0528',
        'claude': 'claude-opus-4-20250514',
        'expert': 'Expert'
    }

    # 转换为英文标签
    department_labels = [department_english.get(dept, dept) for dept in department_order]
    evaluator_labels = [evaluator_english.get(eval, eval) for eval in evaluator_order]

    # 设置刻度标签，调整Y轴标签位置（缩小字号与间距）
    ax.set_xticklabels(department_labels, rotation=0, ha='center', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
    ax.set_yticklabels(evaluator_labels, rotation=0, ha='right', fontsize=11, fontweight='bold', fontfamily='Times New Roman')

    # 手动调整刻度标签与轴距离，更紧凑
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='x', pad=16)

    # 调整图例文字位置与样式，更靠近图且不拥挤
    legend_text = ('Red: Lower scores\n\n'
                  'Yellow: Medium scores\n\n'
                  'Green: Higher scores')
    ax.text(1.1, 0.98, legend_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=12, fontfamily='Times New Roman',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.7))

    # 设置颜色条的标签字体
    cbar = ax.collections[0].colorbar
    cbar.set_label('Average Score', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    cbar.ax.tick_params(labelsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    # 调整布局避免重叠，留出适度空间给颜色条与图例
    plt.subplots_adjust(left=0.18, right=0.88, top=0.92, bottom=0.16)

    return fig, ax

def add_statistical_annotations(ax, heatmap_data):
    """添加统计注释"""
    print("\n🔄 添加统计注释...")

    # 计算每行（被评估者）的平均分
    row_means = heatmap_data.mean(axis=1)

    # 计算每列（科室）的平均分
    col_means = heatmap_data.mean(axis=0)

    # 在图的右侧添加每个被评估者的总体平均分（更紧凑的偏移与字号）
    for i, (evaluator, mean_score) in enumerate(row_means.items()):
        ax.text(len(heatmap_data.columns) + 0.07, i + 0.5, f'{mean_score:.3f}',
                ha='left', va='center', fontweight='bold', fontsize=9, fontfamily='Times New Roman',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))

    # 在图的下方添加每个科室的平均分（更紧凑的偏移与字号）
    for j, (department, mean_score) in enumerate(col_means.items()):
        ax.text(j + 0.5, len(heatmap_data.index) + 0.2, f'{mean_score:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9, fontfamily='Times New Roman',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))

    # # 添加标签说明（改为英文）
    # ax.text(len(heatmap_data.columns) + 0.1, -0.5, 'Overall Average',
    #         ha='left', va='center', fontweight='bold', fontsize=10, fontfamily='Times New Roman')
    # ax.text(-0.5, len(heatmap_data.index) + 0.2, 'Subspecialty Average',  # 从0.1增加到0.2
    #         ha='center', va='bottom', fontweight='bold', fontsize=10, fontfamily='Times New Roman', rotation=90)

def create_ranking_heatmap(df):
    """创建排名热力图 - 已删除，不再使用"""
    pass

def print_heatmap_summary(heatmap_data, evaluator_order, department_order):
    """打印热力图数据汇总"""
    print("\n" + "="*80)
    print("📊 热力图数据汇总")
    print("="*80)

    print(f"\n🏆 被评估者总体排名（按平均分）:")
    for i, evaluator in enumerate(evaluator_order):
        avg_score = heatmap_data.loc[evaluator].mean()
        print(f"   {i+1}. {evaluator}: {avg_score:.3f}")

    print(f"\n🏥 科室排名（按平均分）:")
    for i, department in enumerate(department_order):
        avg_score = heatmap_data[department].mean()
        print(f"   {i+1}. {department}: {avg_score:.3f}")

    print(f"\n📈 最高分和最低分:")
    max_score = heatmap_data.max().max()
    min_score = heatmap_data.min().min()
    max_pos = heatmap_data.stack().idxmax()
    min_pos = heatmap_data.stack().idxmin()

    print(f"   最高分: {max_score:.3f} ({max_pos[0]} - {max_pos[1]})")
    print(f"   最低分: {min_score:.3f} ({min_pos[0]} - {min_pos[1]})")

    print(f"\n📊 分数分布:")
    scores = heatmap_data.values.flatten()
    print(f"   平均分: {np.mean(scores):.3f}")
    print(f"   标准差: {np.std(scores):.3f}")
    print(f"   分数范围: {min_score:.3f} - {max_score:.3f}")

def main():
    """主函数"""
    print("🚀 开始绘制康复亚专科性能热力图")
    print("="*80)

    # 加载数据
    df, cross_df = load_department_analysis_data()
    if df is None or cross_df is None:
        return

    # 准备热力图数据
    heatmap_data, ranking_data, combined_annotations, evaluator_order, department_order = prepare_heatmap_data(df, cross_df)

    # 创建性能热力图
    fig1, ax1 = create_performance_heatmap(heatmap_data, combined_annotations, evaluator_order, department_order)

    # 添加统计注释
    add_statistical_annotations(ax1, heatmap_data)

    # 保存性能热力图（PNG + PDF）
    performance_filename_png = 'Figure4A_康复亚专科性能热力图2.png'
    performance_filename_pdf = 'Figure4A_康复亚专科性能热力图2.pdf'
    fig1.savefig(performance_filename_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig1.savefig(performance_filename_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ 性能热力图已保存: {performance_filename_png}")
    print(f"✅ 论文用PDF已保存: {performance_filename_pdf}")

    # ���印汇总信息
    print_heatmap_summary(heatmap_data, evaluator_order, department_order)

    # 显示图形
    plt.show()

    print("\n🎉 热力图绘制完成!")

if __name__ == "__main__":
    main()
