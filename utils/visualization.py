"""
窃电检测结果可视化模块
用于绘制ROC曲线、PR曲线、混淆矩阵等
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os


def plot_roc_curve(y_true, y_scores, save_path='result/roc_curve.png'):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签 (0=Normal, 1=Theft)
        y_scores: 预测分数 (Theft的概率)
        save_path: 保存路径
    """
    # 确保是numpy数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 转换字符串标签为数字
    if y_true.dtype == object:
        y_true = np.array([1 if t.lower() == 'theft' else 0 for t in y_true])
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘制
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Electricity Theft Detection', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC曲线已保存至: {save_path}")
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, save_path='result/pr_curve.png'):
    """
    绘制Precision-Recall曲线
    
    Args:
        y_true: 真实标签 (0=Normal, 1=Theft)
        y_scores: 预测分数 (Theft的概率)
        save_path: 保存路径
    """
    # 确保是numpy数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 转换字符串标签为数字
    if y_true.dtype == object:
        y_true = np.array([1 if t.lower() == 'theft' else 0 for t in y_true])
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    # 计算基线（随机分类器的性能）
    baseline = np.sum(y_true) / len(y_true)
    
    # 绘制
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {ap:.4f})')
    plt.axhline(y=baseline, color='red', linestyle='--', 
                label=f'Baseline = {baseline:.4f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve for Electricity Theft Detection', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ PR曲线已保存至: {save_path}")
    
    return ap


def plot_confusion_matrix(cm_dict, save_path='result/confusion_matrix.png'):
    """
    绘制混淆矩阵热力图
    
    Args:
        cm_dict: 混淆矩阵字典，包含'TN', 'FP', 'FN', 'TP'
        save_path: 保存路径
    """
    # 构建混淆矩阵数组
    cm = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])
    
    plt.figure(figsize=(8, 6))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Theft'],
                yticklabels=['Actual Normal', 'Actual Theft'],
                annot_kws={'size': 16})
    
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 混淆矩阵已保存至: {save_path}")


def plot_score_distribution(y_true, y_scores, save_path='result/score_distribution.png'):
    """
    绘制预测分数分布图（按类别）
    
    Args:
        y_true: 真实标签
        y_scores: 预测分数
        save_path: 保存路径
    """
    # 确保是numpy数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 转换字符串标签为数字
    if y_true.dtype == object:
        y_true = np.array([1 if t.lower() == 'theft' else 0 for t in y_true])
    
    # 分离两类的分数
    theft_scores = y_scores[y_true == 1]
    normal_scores = y_scores[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', density=True)
    plt.hist(theft_scores, bins=50, alpha=0.6, label='Theft', color='red', density=True)
    
    plt.xlabel('Predicted Score (Probability of Theft)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Scores by Class', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 分数分布图已保存至: {save_path}")


def plot_metrics_comparison(metrics_dict, save_path='result/metrics_comparison.png'):
    """
    绘制主要指标对比柱状图
    
    Args:
        metrics_dict: 包含各项指标的字典
        save_path: 保存路径
    """
    # 选择主要指标
    main_metrics = {
        'AUC': metrics_dict.get('auc', 0),
        'MAP@40': metrics_dict.get('map@40', 0),
        'WF1': metrics_dict.get('wf1', 0),
        'WP': metrics_dict.get('wp', 0),
        'WR': metrics_dict.get('wr', 0),
    }
    
    plt.figure(figsize=(10, 6))
    
    metrics_names = list(main_metrics.keys())
    metrics_values = list(main_metrics.values())
    
    bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Main Evaluation Metrics', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 指标对比图已保存至: {save_path}")


def generate_all_visualizations(metrics_calculator, output_dir='result'):
    """
    生成所有可视化图表
    
    Args:
        metrics_calculator: TheftDetectionMetrics实例
        output_dir: 输出目录
    """
    results = metrics_calculator.compute()
    
    # 提取数据
    labels = np.array(metrics_calculator.labels)
    scores = np.array(metrics_calculator.scores)
    
    print("\n正在生成可视化图表...")
    
    # 1. ROC曲线
    plot_roc_curve(labels, scores, f'{output_dir}/roc_curve.png')
    
    # 2. PR曲线
    plot_precision_recall_curve(labels, scores, f'{output_dir}/pr_curve.png')
    
    # 3. 混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'], f'{output_dir}/confusion_matrix.png')
    
    # 4. 分数分布
    plot_score_distribution(labels, scores, f'{output_dir}/score_distribution.png')
    
    # 5. 指标对比
    plot_metrics_comparison(results, f'{output_dir}/metrics_comparison.png')
    
    print(f"✓ 所有可视化图表已保存至 {output_dir}/ 目录")


if __name__ == "__main__":
    # 测试可视化功能
    from metrics import TheftDetectionMetrics
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    metrics = TheftDetectionMetrics()
    
    n_samples = 200
    n_theft = 40
    
    labels = np.array([1]*n_theft + [0]*(n_samples-n_theft))
    
    for i, label in enumerate(labels):
        if label == 1:
            score = np.random.beta(7, 3)  # 高概率
        else:
            score = np.random.beta(3, 7)  # 低概率
        pred = 1 if score > 0.5 else 0
        metrics.update(label, pred, score, i)
    
    # 生成所有可视化
    generate_all_visualizations(metrics, output_dir='result/test')
    
    print("\n测试完成！请查看 result/test/ 目录下的图表。")
