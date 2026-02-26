"""
AXIS改进版模型测试脚本

使用方法：
    # 测试最佳模型
    python scripts/test_axis_improved.py --config MEDIUM

    # 指定检查点
    python scripts/test_axis_improved.py --config MEDIUM --checkpoint checkpoints_finetune_axis/finetune_medium_best.pth

    # 快速测试
    python scripts/test_axis_improved.py --config MEDIUM --quick
"""

import torch
import os
import sys
import re
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.main_axis_improved import (
    H3_PerceptionLayerAXIS,
    ElectricityDatasetAXIS,
    ModelConfig,
)
from models.agent_axis import H3_Agent_AXIS
from utils.metrics import TheftDetectionMetrics
from utils.visualization import generate_all_visualizations

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ElectricityDatasetWithLabel(ElectricityDatasetAXIS):
    """带标签的测试数据集"""

    def __init__(
        self, csv_file, user_stats_path, max_users=None, max_samples=None, **kwargs
    ):
        super().__init__(
            csv_file, user_stats_path, mode="test", for_pretrain=False, **kwargs
        )

        # 限制样本数
        if max_samples is not None:
            n_samples = min(max_samples, len(self.df))
            print(f"[快速测试] 只加载前 {n_samples} 条记录（共 {len(self.df)} 条）")
            self.df = self.df.iloc[:n_samples].reset_index(drop=True)
            self.data_values = self.data_values[:n_samples]
            if self.labels is not None:
                self.labels = self.labels[:n_samples]
            self.user_ids = self.df.iloc[:, self.id_idx].values.astype(
                str
            )  # 【修复】使用动态索引
            self.dates = pd.to_datetime(
                self.df.iloc[:, self.day_idx]
            )  # 【修复】使用动态索引
        elif max_users is not None:
            unique_users = self.df.iloc[:, self.id_idx].unique()  # 【修复】使用动态索引
            selected_users = unique_users[:max_users]
            print(f"[快速测试] 加载前 {max_users} 个用户的数据")

            mask = self.df.iloc[:, self.id_idx].isin(
                selected_users
            )  # 【修复】使用动态索引
            self.df = self.df[mask].reset_index(drop=True)
            self.data_values = self.data_values[mask.values]
            if self.labels is not None:
                self.labels = self.labels[mask.values]
            self.user_ids = self.df.iloc[:, self.id_idx].values.astype(
                str
            )  # 【修复】使用动态索引
            self.dates = pd.to_datetime(
                self.df.iloc[:, self.day_idx]
            )  # 【修复】使用动态索引
            print(f"  共 {len(self.df)} 条记录")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_inference(
    agent,
    max_users=None,
    max_samples=None,
    output_dir="results_axis_improved",
    debug=False,
):
    """
    运行推理

    Args:
        agent: 模型
        max_users: 最大用户数（快速测试）
        max_samples: 最大样本数（快速测试）
        output_dir: 输出目录
        debug: 是否启用调试模式
    """
    print("\n" + "=" * 70)
    print("AXIS改进版模型推理")
    print("=" * 70)

    agent.eval()

    # 加载测试数据
    dataset = ElectricityDatasetWithLabel(
        "dataset/File1_test_daily_flag_4_2.csv",
        "dataset/user_historical_stats.json",
        max_users=max_users,
        max_samples=max_samples,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    # 评估指标
    metrics = TheftDetectionMetrics()
    global_user_idx = 0
    results_list = []

    print(f"\n总样本数: {len(dataset)}")
    print(f"批次数量: {len(loader)}\n")

    for batch_idx, batch in enumerate(loader):
        if (batch_idx + 1) % 10 == 0:
            print(f">>> 处理 Batch {batch_idx + 1}/{len(loader)}")

        current_batch_size = len(batch["label"])
        instruction = [
            "Analyze this user's electricity usage pattern."
        ] * current_batch_size
        ground_truths = batch["label"].cpu().tolist()

        # 推理（只在第一个批次启用debug，且限制最多8个样本避免输出过长）
        batch_debug = debug and (batch_idx == 0)
        if batch_debug and current_batch_size > 8:
            print(f"[DEBUG] 限制调试样本数为8（原为{current_batch_size}）")

        responses, _, theft_scores = agent.generate(
            batch,
            instruction,
            return_scores=True,
            debug=batch_debug,
            labels=ground_truths if batch_debug else None,
        )

        for i in range(current_batch_size):
            res_text = responses[i].strip()
            true_label_text = "Theft" if ground_truths[i] == 1 else "Normal"
            theft_score = theft_scores[i] if theft_scores else 0.5

            # 解析预测
            clean_text = re.sub(
                r"<think>.*?</think>", "", res_text, flags=re.DOTALL
            ).strip()
            lower_res = clean_text.lower()

            # 判断预测类别
            if "theft" in lower_res and "normal" not in lower_res:
                pred_class = "Theft"
            elif "normal" in lower_res and "theft" not in lower_res:
                pred_class = "Normal"
            elif "theft" in lower_res and "normal" in lower_res:
                # 取最后出现的词
                pred_class = (
                    "Theft"
                    if lower_res.rfind("theft") > lower_res.rfind("normal")
                    else "Normal"
                )
            else:
                # 如果无法解析，根据分数判断
                pred_class = "Theft" if theft_score > 0.5 else "Normal"

            # 置信度
            confidence = (
                "High"
                if theft_score > 0.8 or theft_score < 0.2
                else "Medium"
                if theft_score > 0.6 or theft_score < 0.4
                else "Low"
            )

            if (global_user_idx + 1) % 100 == 0:
                print(
                    f"  [User {global_user_idx}] True: {true_label_text} | "
                    f"Pred: {pred_class} | Score: {theft_score:.3f}"
                )

            metrics.update(
                true_label_text, pred_class, theft_score, user_id=global_user_idx
            )

            results_list.append(
                {
                    "User_Index": global_user_idx,
                    "True_Label": true_label_text,
                    "Predicted_Class": pred_class,
                    "Score": round(theft_score, 4),
                    "Confidence": confidence,
                    "Raw_Response": res_text[:300],
                }
            )

            global_user_idx += 1

    print(f"\n{'=' * 70}")
    print(f"推理完成，共 {global_user_idx} 条样本")
    print(f"{'=' * 70}")

    # 自适应阈值调整 - 找到F1最优阈值
    from sklearn.metrics import f1_score, precision_score, recall_score

    all_scores = [r["Score"] for r in results_list]
    all_labels = [1 if r["True_Label"] == "Theft" else 0 for r in results_list]

    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = [1 if s > thresh else 0 for s in all_scores]
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n【自适应阈值分析】")
    print(
        f"默认阈值 (0.5) 的F1: {f1_score(all_labels, [1 if s > 0.5 else 0 for s in all_scores], zero_division=0):.4f}"
    )
    print(f"最优阈值 ({best_thresh:.2f}) 的F1: {best_f1:.4f}")

    # 使用最优阈值重新计算预测
    for r in results_list:
        r["Predicted_Class_Adjusted"] = (
            "Theft" if r["Score"] > best_thresh else "Normal"
        )

    # 评估报告
    results = metrics.print_report()

    # Top-K 分析
    for k in [40, 100, 200]:
        print(f"\n【Top-{k} 分析】")
        topk = metrics.get_top_k_predictions(k)
        theft_in_topk = sum(1 for item in topk if item["true_label"] == "Theft")
        correct_in_topk = sum(
            1 for item in topk if item["correct"] and item["true_label"] == "Theft"
        )
        print(f"Top-{k} 中真实窃电: {theft_in_topk}/{k}")
        print(f"Top-{k} 中正确识别: {correct_in_topk}")
        if theft_in_topk > 0:
            print(f"Top-{k} 窃电召回率: {correct_in_topk / theft_in_topk * 100:.1f}%")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(f"{output_dir}/inference_results.csv", index=False)
    print(f"\n结果已保存: {output_dir}/inference_results.csv")

    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"指标已保存: {output_dir}/evaluation_metrics.json")

    # 可视化
    try:
        viz_dir = f"{output_dir}/visualizations"
        generate_all_visualizations(metrics, output_dir=viz_dir)
        print(f"可视化已保存: {viz_dir}")
    except Exception as e:
        print(f"可视化失败: {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test AXIS improved model")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="MEDIUM",
        choices=["SMALL", "BASE", "MEDIUM", "LARGE"],
        help="Model configuration",
    )
    parser.add_argument(
        "--max-users", type=int, default=None, help="Limit test users for quick test"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test samples for quick test",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test mode: test 1000 samples only"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_axis_improved",
        help="Output directory",
    )
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式（打印各阶段中间结果）"
    )

    args = parser.parse_args()

    # 设置全局调试开关
    if args.debug:
        from utils.debug_utils import DebugLogger

        DebugLogger.set_enabled(True)
        print("\n[调试模式已启用] 将打印各阶段中间结果\n")

    # 快速模式
    if args.quick:
        args.max_samples = 1000
        print("[快速测试模式] 只测试 1000 条样本\n")

    # 导入配置
    config = getattr(ModelConfig, args.config)

    print(f"\n使用配置: {args.config}")
    print(f"配置详情: {config}\n")

    # 初始化模型
    llm_path = "/root/rivermind-data/qwen3_1.7B"
    perception_layer = H3_PerceptionLayerAXIS(llm_dim=2048, **config)
    agent = H3_Agent_AXIS(
        llm_path, perception_layer=perception_layer, load_in_4bit=False
    )
    agent.to(DEVICE)

    # 加载权重
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # 自动查找检查点
        possible_paths = [
            f"checkpoints_finetune_axis/finetune_{args.config.lower()}_best.pth",
            f"checkpoints_pretrain_axis/pretrain_{args.config.lower()}_best.pth",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # 判断是预训练还是微调检查点
        if "perception" in checkpoint:
            agent.perception.load_state_dict(checkpoint["perception"])
            print(f"✓ 加载微调权重: {checkpoint_path}")
        elif "numerical_stream" in checkpoint:
            # 使用 strict=False 忽略预训练头（pretrain_heads）的权重
            agent.perception.numerical_stream.load_state_dict(
                checkpoint["numerical_stream"], strict=False
            )
            print(f"✓ 加载预训练权重: {checkpoint_path}")
            print("  注意：只加载了编码器权重，融合层使用随机初始化")

        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint.get('epoch', '?')}")
            if "loss" in checkpoint:
                print(f"  Loss: {checkpoint['loss']:.4f}")
    else:
        print(f"⚠ 未找到检查点，使用随机初始化")
        if checkpoint_path:
            print(f"  查找路径: {checkpoint_path}")

    # 运行推理
    with torch.no_grad():
        run_inference(
            agent,
            max_users=args.max_users,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
