"""
窃电检测评估指标模块
基于论文 Section II-C 和实验部分的5个主要指标
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)


class TheftDetectionMetrics:
    """
    窃电检测模型评估指标计算器
    
    支持指标:
    1. AUC (Area Under ROC Curve)
    2. MAP@40 (Mean Average Precision at 40)
    3. WF1 (Weighted F1 Score)
    4. WP (Weighted Precision)
    5. WR (Weighted Recall)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有缓存数据"""
        self.labels = []  # 真实标签: 1=Theft, 0=Normal
        self.predictions = []  # 预测标签: 1=Theft, 0=Normal
        self.scores = []  # 预测分数/概率 (Theft的概率)
        self.user_ids = []  # 用户ID（可选）
    
    def update(self, label, prediction, score, user_id=None):
        """
        添加一个样本的预测结果
        
        Args:
            label: 真实标签 (0或1，或"Normal"/"Theft")
            prediction: 预测标签 (0或1，或"Normal"/"Theft")
            score: Theft的预测概率/分数 (0-1之间)
            user_id: 用户标识（可选）
        """
        # 转换字符串标签为数字
        if isinstance(label, str):
            label = 1 if label.lower() == "theft" else 0
        if isinstance(prediction, str):
            prediction = 1 if prediction.lower() == "theft" else 0
            
        self.labels.append(label)
        self.predictions.append(prediction)
        self.scores.append(score)
        if user_id is not None:
            self.user_ids.append(user_id)
    
    def compute(self):
        """
        计算所有评估指标
        
        Returns:
            dict: 包含所有指标的字典
        """
        labels = np.array(self.labels)
        predictions = np.array(self.predictions)
        scores = np.array(self.scores)
        
        # 计算类别权重 (用于加权指标)
        n_total = len(labels)
        n_theft = np.sum(labels == 1)
        n_normal = np.sum(labels == 0)
        
        w_theft = n_theft / n_total if n_total > 0 else 0.5
        w_normal = n_normal / n_total if n_total > 0 else 0.5
        
        results = {
            'total_samples': int(n_total),
            'theft_samples': int(n_theft),
            'normal_samples': int(n_normal),
        }
        
        # 基础指标
        results['accuracy'] = np.mean(labels == predictions)
        
        # 1. AUC (需要分数)
        if len(np.unique(labels)) > 1 and len(scores) > 0:
            results['auc'] = roc_auc_score(labels, scores)
        else:
            results['auc'] = 0.5  # 无法计算时返回随机水平
        
        # 2. MAP@40 (Mean Average Precision at 40)
        results['map@40'] = self._compute_map_at_k(labels, scores, k=40)
        results['map@100'] = self._compute_map_at_k(labels, scores, k=100)
        results['map@all'] = self._compute_map_at_k(labels, scores, k=len(labels))
        
        # 混淆矩阵元素
        tp = np.sum((labels == 1) & (predictions == 1))  # 真正例
        fp = np.sum((labels == 0) & (predictions == 1))  # 假正例
        fn = np.sum((labels == 1) & (predictions == 0))  # 假反例
        tn = np.sum((labels == 0) & (predictions == 0))  # 真反例
        
        results['confusion_matrix'] = {
            'TP': int(tp), 'FP': int(fp),
            'FN': int(fn), 'TN': int(tn)
        }
        
        # 3. WF1 (Weighted F1 Score)
        # F1 for each class
        f1_theft = self._compute_f1(tp, fp, fn)
        f1_normal = self._compute_f1(tn, fn, fp)  # 交换正负类计算
        results['wf1'] = (w_theft * f1_theft + w_normal * f1_normal)
        results['f1_theft'] = f1_theft
        results['f1_normal'] = f1_normal
        
        # 4. WP (Weighted Precision)
        precision_theft = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
        results['wp'] = (w_theft * precision_theft + w_normal * precision_normal)
        results['precision_theft'] = precision_theft
        results['precision_normal'] = precision_normal
        
        # 5. WR (Weighted Recall)
        recall_theft = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['wr'] = (w_theft * recall_theft + w_normal * recall_normal)
        results['recall_theft'] = recall_theft
        results['recall_normal'] = recall_normal
        
        # 额外指标
        results['f1_macro'] = f1_score(labels, predictions, average='macro', zero_division=0)
        results['f1_weighted'] = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        return results
    
    def _compute_f1(self, tp, fp, fn):
        """计算F1分数"""
        if (2 * tp + fp + fn) == 0:
            return 0.0
        return 2 * tp / (2 * tp + fp + fn)
    
    def _compute_map_at_k(self, labels, scores, k=40):
        """
        计算MAP@K (Mean Average Precision at K)
        
        按照预测分数排序，取前K个计算AP
        
        Args:
            labels: 真实标签数组
            scores: 预测分数数组 (Theft的概率)
            k: 截断位置
        """
        if len(labels) == 0 or len(scores) == 0:
            return 0.0
        
        # 按分数降序排序
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # 取前K个
        k = min(k, len(sorted_labels))
        sorted_labels_k = sorted_labels[:k]
        
        # 计算AP@K
        # AP = sum(P(i) * rel(i)) / min(R, K)
        # 其中P(i)是前i个的精确率，rel(i)是第i个是否相关
        num_relevant = np.sum(labels == 1)  # 总共的窃电用户数
        if num_relevant == 0:
            return 0.0
        
        ap_sum = 0.0
        relevant_count = 0
        
        for i, label in enumerate(sorted_labels_k):
            if label == 1:  # 是窃电用户
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap_sum += precision_at_i
        
        # 除以min(R, K)，R是总相关文档数
        ap = ap_sum / min(num_relevant, k)
        return ap
    
    def get_top_k_predictions(self, k=40):
        """
        获取预测分数最高的前K个样本
        
        Returns:
            list: [(index, user_id, score, true_label, pred_label), ...]
        """
        scores = np.array(self.scores)
        sorted_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in sorted_indices:
            results.append({
                'rank': len(results) + 1,
                'index': int(idx),
                'user_id': self.user_ids[idx] if idx < len(self.user_ids) else None,
                'score': float(scores[idx]),
                'true_label': 'Theft' if self.labels[idx] == 1 else 'Normal',
                'pred_label': 'Theft' if self.predictions[idx] == 1 else 'Normal',
                'correct': self.labels[idx] == self.predictions[idx]
            })
        return results
    
    def print_report(self):
        """打印评估报告"""
        results = self.compute()
        
        print("\n" + "="*70)
        print("                    窃电检测模型评估报告")
        print("="*70)
        
        # 基础统计
        print(f"\n【数据集统计】")
        print(f"  总样本数: {results['total_samples']}")
        print(f"  窃电样本 (Theft): {results['theft_samples']} ({results['theft_samples']/results['total_samples']*100:.1f}%)")
        print(f"  正常样本 (Normal): {results['normal_samples']} ({results['normal_samples']/results['total_samples']*100:.1f}%)")
        
        # 主要指标
        print(f"\n【主要评估指标】")
        print(f"  AUC:      {results['auc']:.4f}")
        print(f"  MAP@40:   {results['map@40']:.4f}")
        print(f"  MAP@100:  {results['map@100']:.4f}")
        print(f"  WF1:      {results['wf1']:.4f}")
        print(f"  WP:       {results['wp']:.4f}")
        print(f"  WR:       {results['wr']:.4f}")
        
        # 混淆矩阵
        print(f"\n【混淆矩阵】")
        cm = results['confusion_matrix']
        print(f"                    预测Normal    预测Theft")
        print(f"  真实Normal:       {cm['TN']:6d}       {cm['FP']:6d}")
        print(f"  真实Theft:        {cm['FN']:6d}       {cm['TP']:6d}")
        
        # 详细指标
        print(f"\n【详细指标】")
        print(f"  准确率 (Accuracy):  {results['accuracy']:.4f}")
        print(f"\n  窃电类指标:")
        print(f"    精确率 (Precision): {results['precision_theft']:.4f}")
        print(f"    召回率 (Recall):    {results['recall_theft']:.4f}")
        print(f"    F1分数:             {results['f1_theft']:.4f}")
        print(f"\n  正常类指标:")
        print(f"    精确率 (Precision): {results['precision_normal']:.4f}")
        print(f"    召回率 (Recall):    {results['recall_normal']:.4f}")
        print(f"    F1分数:             {results['f1_normal']:.4f}")
        
        print("="*70)
        
        return results


def compute_metrics_from_csv(csv_path, label_col='True_Label', pred_col='Predicted_Class', score_col='Score'):
    """
    从CSV文件加载结果并计算指标
    
    Args:
        csv_path: CSV文件路径
        label_col: 真实标签列名
        pred_col: 预测标签列名
        score_col: 预测分数列名
    
    Returns:
        dict: 评估指标字典
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    metrics = TheftDetectionMetrics()
    
    for _, row in df.iterrows():
        label = row[label_col]
        pred = row[pred_col]
        score = row.get(score_col, 0.5)  # 默认0.5
        
        metrics.update(label, pred, score)
    
    return metrics.compute()


if __name__ == "__main__":
    # 测试用例
    metrics = TheftDetectionMetrics()
    
    # 模拟一些数据
    np.random.seed(42)
    n_samples = 100
    n_theft = 20
    
    # 真实标签: 20个Theft, 80个Normal
    labels = np.array([1]*n_theft + [0]*(n_samples-n_theft))
    
    # 模拟预测: Theft样本预测为Theft的概率高，Normal样本预测为Normal的概率高
    scores = []
    predictions = []
    
    for label in labels:
        if label == 1:  # Theft
            score = np.random.beta(7, 3)  # 高概率
        else:  # Normal
            score = np.random.beta(3, 7)  # 低概率
        
        scores.append(score)
        predictions.append(1 if score > 0.5 else 0)
    
    for i in range(n_samples):
        metrics.update(labels[i], predictions[i], scores[i], user_id=i)
    
    # 打印报告
    metrics.print_report()
    
    # 打印Top-10预测
    print("\n【Top-10 预测结果】")
    top10 = metrics.get_top_k_predictions(10)
    for item in top10:
        status = "✓" if item['correct'] else "✗"
        print(f"  Rank {item['rank']:2d}: {status} Score={item['score']:.4f}, "
              f"True={item['true_label']}, Pred={item['pred_label']}")
