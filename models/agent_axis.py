"""
AXIS风格改进版Agent

整合：
1. H3_PerceptionLayerAXIS感知层
2. 支持预训练模式
3. 优化的提示构建
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.main_axis_improved import H3_PerceptionLayerAXIS


class H3_Agent_AXIS(nn.Module):
    """
    AXIS风格Agent

    改进点：
    1. 使用H3_PerceptionLayerAXIS感知层
    2. 支持获取LLM词嵌入用于Perceiver
    3. 优化的提示构建
    """

    def __init__(
        self, llm_path, perception_layer=None, load_in_4bit=False, config=None
    ):
        super().__init__()

        print(f"正在加载 LLM: {llm_path} ...")

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

        # Qwen 适配方案：设置 pad_token
        if self.tokenizer.pad_token_id is None:
            if hasattr(self.tokenizer, "eod_id"):
                self.tokenizer.pad_token_id = self.tokenizer.eod_id
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载 LLM
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # 动态获取LLM维度
        self.llm_dim = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size
        print(f"检测到当前 LLM 隐藏层维度: {self.llm_dim}")
        print(f"词表大小: {self.vocab_size}")

        # 冻结LLM参数
        print("冻结LLM所有参数，只训练感知层")
        for param in self.llm.parameters():
            param.requires_grad = False

        # 确保inputs_embeds有梯度
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # 3. 加载感知层
        if perception_layer is None:
            if config is None:
                from models.main_axis_improved import ModelConfig

                config = ModelConfig.MEDIUM
            perception_layer = H3_PerceptionLayerAXIS(llm_dim=self.llm_dim, **config)

        self.perception = perception_layer

        # 设备管理
        self.device = self.llm.device
        self.perception.to("cuda" if torch.cuda.is_available() else "cpu")

    def get_word_embeddings(self):
        """获取LLM的词嵌入权重"""
        return self.llm.get_input_embeddings().weight

    def construct_prompt(self, axis_hints, user_instructions, ground_truth=None):
        """
        构建Llama-3格式的指令

        Args:
            axis_hints: 文本提示列表
            user_instructions: 用户指令列表
            ground_truth: 真实标签列表（"Theft"或"Normal"），训练时提供

        Returns:
            prompts: 完整的文本提示列表（包含assistant回复，如果是训练模式）
            prompt_lens: 每个样本的prompt长度（用于构建labels）
        """
        prompts = []
        prompt_lens = []

        for i, (hint, instr) in enumerate(zip(axis_hints, user_instructions)):
            # 系统提示 - 简洁客观，避免偏向
            sys_msg = (
                "You are an expert in electricity theft detection.\n"
                "Analyze the provided electricity usage data objectively.\n"
                "\n"
                "THEFT INDICATORS:\n"
                "- Unusually low or flat consumption\n"
                "- Sudden drops in usage without explanation\n"
                "- Abnormal patterns: many zero values or constant readings\n"
                "- Usage significantly below historical average\n"
                "\n"
                "NORMAL INDICATORS:\n"
                "- Consistent daily/weekly cycles\n"
                "- Weekend vs weekday differences\n"
                "- Seasonal variations matching weather\n"
                "\n"
                "CRITICAL RULES:\n"
                "1. Output ONLY the word 'Theft' or 'Normal'\n"
                "2. Do NOT use <think> tags or explain your reasoning\n"
                "3. Output the single word only, nothing else"
            )

            # 用户输入
            user_content = f"Analysis Context:\n{hint}\n\nInstruction:\n{instr}"

            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content},
            ]

            # 构建prompt（不包含assistant回复）
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 计算prompt的token长度（用于构建labels）
            prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)[
                "input_ids"
            ]
            prompt_len = len(prompt_tokens)

            # 训练模式：添加assistant回复
            if ground_truth is not None:
                assistant_content = ground_truth[i]
                # 【修复】使用与forward中一致的tokenize方式
                full_text = prompt_text + assistant_content
                prompts.append(full_text)
                prompt_lens.append(prompt_len)
            else:
                # 推理模式：只返回prompt
                prompts.append(prompt_text)
                prompt_lens.append(prompt_len)

        return prompts, prompt_lens

    def forward(
        self, batch_data, user_instructions, labels=None, batch_idx=None, debug=False
    ):
        """
        前向传播

        Args:
            batch_data: 包含 'series' 和 'text' 的字典
            user_instructions: 用户指令列表
            labels: 训练标签（字符串列表）
            batch_idx: 当前batch索引（用于调试）
            debug: 是否打印调试信息

        Returns:
            loss: 训练损失 或 (inputs_embeds, attention_mask, info)
        """
        # 移动数据到LLM设备
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(self.llm.device)

        # DEBUG: 检查输入数据
        if debug and batch_idx is not None and batch_idx == 0:
            from utils.debug_utils import DebugLogger, ModelDebugger

            debugger = ModelDebugger(enabled=True)
            debugger.check_data_input(batch_data)

        # 1. 获取LLM词嵌入（用于Perceiver）
        word_embeddings = self.get_word_embeddings()

        # DEBUG
        if debug and batch_idx is not None and batch_idx == 0:
            from utils.debug_utils import DebugLogger

            DebugLogger.log_tensor(
                "LLM词嵌入",
                word_embeddings,
                print_shape=True,
                print_stats=True,
                print_sample=False,
            )

        # 2. 感知层处理 -> 获取 Soft Prompts
        soft_prompts, perception_info = self.perception(
            batch_data, llm_embeds=word_embeddings
        )
        soft_prompts = soft_prompts.to(self.llm.dtype)

        # DEBUG: 检查编码器输出
        if debug and batch_idx is not None and batch_idx == 0:
            from utils.debug_utils import DebugLogger, ModelDebugger

            debugger = ModelDebugger(enabled=True)
            debugger.check_fusion_output(soft_prompts)

        # 3. 文本处理 -> 获取 Text Embeddings
        text_prompts, prompt_lens = self.construct_prompt(
            batch_data["text"], user_instructions, ground_truth=labels
        )

        tokens = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.llm.device)

        # 获取文本Embedding
        input_embeds = self.llm.model.embed_tokens(tokens.input_ids)

        # 4. 拼接: [Soft Prompts] + [Text Embeddings]
        batch_size = input_embeds.shape[0]
        soft_prompt_len = soft_prompts.shape[1]

        # 直接把 soft_prompts 放在文本序列最前面
        inputs_embeds = torch.cat([soft_prompts, input_embeds], dim=1)

        # 5. 构建 Attention Mask
        prefix_mask = torch.ones(
            (batch_size, soft_prompt_len),
            dtype=tokens.attention_mask.dtype,
            device=self.llm.device,
        )
        attention_mask = torch.cat([prefix_mask, tokens.attention_mask], dim=1)

        # 6. 训练模式：计算Loss
        if labels is not None:
            # 【修复】构建labels：-100表示不计算loss
            # soft_prompts部分和prompt部分不计算loss，只有assistant回复部分计算

            # 先构建完整的labels（全部为-100）
            final_labels = torch.full(
                (batch_size, inputs_embeds.shape[1]),
                -100,
                device=self.llm.device,
                dtype=torch.long,
            )

            # 只保留assistant回复部分的label
            # prompt_lens[i]是第i个样本的prompt长度（不包括soft_prompts）
            for i, p_len in enumerate(prompt_lens):
                # assistant回复从soft_prompt_len + p_len开始
                start_idx = soft_prompt_len + p_len
                if start_idx < inputs_embeds.shape[1]:
                    # 【关键修复】label应该是输入的下一个token
                    # 对于因果模型，看到token[0:-1]要预测token[1:]
                    # 所以label是input_ids[1:]，但我们已经有完整的input_ids
                    # 只需要取[start_idx:]作为label即可
                    actual_len = min(
                        len(tokens.input_ids[i]),
                        inputs_embeds.shape[1] - soft_prompt_len,
                    )
                    if actual_len > p_len:
                        target_len = actual_len - p_len
                        final_labels[i, start_idx : start_idx + target_len] = (
                            tokens.input_ids[i, p_len:actual_len]
                        )

            # 调试：打印第一个batch的目标
            if batch_idx is not None and batch_idx == 0 and self.training:
                valid_label_indices = final_labels[0] != -100
                valid_label_tokens = final_labels[0][valid_label_indices]
                if len(valid_label_tokens) > 0:
                    decoded_labels = self.tokenizer.decode(valid_label_tokens)
                    print(f"\n[DEBUG] 模型计算Loss的目标词: --> '{decoded_labels}' <--")

            # Qwen3 (以及绝大多数基于 Llama 架构的模型)
            # 在传入 attention_mask 时会自动正确计算 position_ids，无需显式传入
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=final_labels,
            )
            return outputs.loss
        else:
            return inputs_embeds, attention_mask, perception_info

    def _get_token_ids_with_variants(self, text):
        """
        修复3: 获取文本的所有可能token ID变体
        BPE分词器对前导空格敏感，需要检查多种变体
        """
        variants = [
            text,  # "Theft"
            " " + text,  # " Theft" (前导空格)
            "\n" + text,  # "\nTheft" (前导换行)
            text.lower(),  # "theft" (小写)
            " " + text.lower(),
        ]

        token_ids = set()
        for variant in variants:
            try:
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    token_ids.add(ids[0])
            except:
                pass

        return list(token_ids)

    @torch.no_grad()
    def generate(
        self,
        batch_data,
        user_instructions,
        max_new_tokens=20,
        return_scores=True,
        debug=False,
        labels=None,
    ):
        """
        推理生成函数

        Args:
            batch_data: 输入数据
            user_instructions: 用户指令
            max_new_tokens: 最大生成token数
            return_scores: 是否返回预测概率分数
            debug: 是否打印调试信息
            labels: 真实标签（用于调试对比）

        Returns:
            response_texts: 生成的文本列表
            attention_weights: 注意力权重（None，保留接口）
            scores: 预测分数列表（Theft的概率）
        """
        import re

        # 获取输入embeddings
        inputs_embeds, attention_mask, perception_info = self.forward(
            batch_data, user_instructions, debug=debug
        )

        # DEBUG: 打印输入数据的文本提示（用于检查construct_prompt的输出）
        if debug:
            from utils.debug_utils import DebugLogger

            DebugLogger.log_tensor(
                "LLM输入 embeddings", inputs_embeds, sample_indices=(0,)
            )

            # 重新构造提示词，查看实际输入给LLM的内容
            _, prompt_lens = self.construct_prompt(
                batch_data["text"], user_instructions, ground_truth=None
            )
            print(f"\n{'=' * 60}")
            print("[DEBUG] 实际输入给LLM的提示词（第一个样本）:")
            print(f"{'=' * 60}")
            # 构造不带ground truth的prompt来查看
            text_prompts, _ = self.construct_prompt(
                [batch_data["text"][0]], [user_instructions[0]], ground_truth=None
            )
            print(
                text_prompts[0][:800] + "..."
                if len(text_prompts[0]) > 800
                else text_prompts[0]
            )
            print(f"{'=' * 60}\n")

        # 修复3: 获取多种变体的token IDs，避免BPE分词器的偏移问题
        theft_token_ids = self._get_token_ids_with_variants("Theft")
        normal_token_ids = self._get_token_ids_with_variants("Normal")

        if debug:
            print(f"[DEBUG] Theft Token IDs: {theft_token_ids}")
            print(f"[DEBUG] Normal Token IDs: {normal_token_ids}")

        if return_scores:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            response_sequences = outputs.sequences
            scores_list = outputs.scores

            # 解码输出
            response_texts_raw = self.tokenizer.batch_decode(
                response_sequences, skip_special_tokens=True
            )

            # 处理Qwen3的思考模式输出（去除<think>标签内容）
            response_texts = []
            for text in response_texts_raw:
                # 去除 <think>...</think> 及其内容
                import re

                text_clean = re.sub(
                    r"<think>.*?</think>", "", text, flags=re.DOTALL
                ).strip()
                # 如果清理后为空，保留原始文本
                if not text_clean:
                    text_clean = text.strip()
                response_texts.append(text_clean)

            if debug:
                print(f"\n[DEBUG] 原始输出 vs 清理后:")
                for i in range(min(3, len(response_texts_raw))):
                    raw_short = response_texts_raw[i][:80].replace("\n", " ")
                    clean_short = response_texts[i][:80].replace("\n", " ")
                    print(
                        f"  样本{i}: 原始='{raw_short}...' -> 清理='{clean_short}...'"
                    )

            # 计算Theft概率 - 结合logits分数和文本判断
            theft_scores = []
            batch_size = inputs_embeds.shape[0]

            for i in range(batch_size):
                first_gen_logits = None

                for step, step_logits in enumerate(scores_list):
                    if step_logits is not None and len(step_logits) > i:
                        first_gen_logits = step_logits[i]
                        break

                # 基于logits计算概率（使用多个token ID变体）
                logits_theft_prob = 0.5
                theft_prob = 0
                normal_prob = 0

                if first_gen_logits is not None:
                    probs = torch.softmax(first_gen_logits, dim=-1)

                    # 调试：打印每个候选token的概率
                    if debug and i == 0:
                        print(f"\n  [Token概率详细] 词表大小={len(probs)}")
                        print(f"    Theft候选: {theft_token_ids}")
                        for tid in theft_token_ids:
                            if tid < len(probs):
                                token_str = (
                                    self.tokenizer.decode([tid])
                                    if tid < len(self.tokenizer)
                                    else "<?>"
                                )
                                print(
                                    f"      ID={tid} '{token_str}': {probs[tid].item():.4f}"
                                )
                        print(f"    Normal候选: {normal_token_ids}")
                        for tid in normal_token_ids:
                            if tid < len(probs):
                                token_str = (
                                    self.tokenizer.decode([tid])
                                    if tid < len(self.tokenizer)
                                    else "<?>"
                                )
                                print(
                                    f"      ID={tid} '{token_str}': {probs[tid].item():.4f}"
                                )

                    # 简化逻辑：取所有候选中的最大概率
                    normal_candidates = [
                        tid for tid in normal_token_ids if tid < len(probs)
                    ]
                    if normal_candidates:
                        normal_prob = max(
                            [probs[tid].item() for tid in normal_candidates]
                        )

                    theft_candidates = [
                        tid for tid in theft_token_ids if tid < len(probs)
                    ]
                    if theft_candidates:
                        # 排除过于通用的token如换行符(198)
                        filtered_candidates = [
                            tid for tid in theft_candidates if tid != 198
                        ]
                        if filtered_candidates:
                            theft_prob = max(
                                [probs[tid].item() for tid in filtered_candidates]
                            )
                        else:
                            theft_prob = max(
                                [probs[tid].item() for tid in theft_candidates]
                            )

                    total = theft_prob + normal_prob
                    if total > 0:
                        logits_theft_prob = theft_prob / total

                # 基于生成文本的判断（更可靠）
                response_text = response_texts[i] if i < len(response_texts) else ""

                # 使用正则表达式检查文本内容
                text_theft_prob = 0.5
                matched_pattern = None

                # 使用正则表达式检查文本内容
                text_theft_prob = 0.5
                matched_pattern = None

                if re.search(r"\btheft\b", response_text, re.IGNORECASE):
                    text_theft_prob = 1.0
                    matched_pattern = "theft"
                elif re.search(r"\bnormal\b", response_text, re.IGNORECASE):
                    text_theft_prob = 0.0
                    matched_pattern = "normal"
                elif re.search(
                    r"\b(abnormal|suspicious|fraud|steal)\b",
                    response_text,
                    re.IGNORECASE,
                ):
                    text_theft_prob = 0.8  # 相关词汇也认为是盗窃
                    matched_pattern = "abnormal/suspicious/fraud/steal"

                # 综合分数：优先相信文本判断，其次参考logits
                # 如果文本能明确判断，使用文本结果；否则使用logits
                if text_theft_prob in [0.0, 1.0]:
                    final_theft_score = text_theft_prob
                else:
                    # 文本判断不明确时，混合logits和默认值
                    final_theft_score = 0.6 * logits_theft_prob + 0.4 * 0.5

                theft_scores.append(final_theft_score)

                # DEBUG: 打印每个样本的详细预测过程（不限于前3个，全部打印但限制批次大小）
                if debug and i < min(batch_size, 5):  # 打印最多5个样本
                    label_str = (
                        f"[真实: {'Theft' if labels[i] == 1 else 'Normal' if labels is not None else 'Unknown'}]"
                        if labels is not None
                        else ""
                    )

                    print(f"\n{'-' * 60}")
                    print(f"[样本 {i}] {label_str}")
                    print(f"{'-' * 60}")
                    print(f"[生成文本] (长度={len(response_text)}):")
                    print(
                        response_text[:300] + "..."
                        if len(response_text) > 300
                        else response_text
                    )
                    print(f"\n[Logits分析]")

                    # 打印更详细的logits调试信息
                    if first_gen_logits is not None:
                        probs = torch.softmax(first_gen_logits, dim=-1)
                        top5_probs, top5_indices = torch.topk(probs, 5)
                        print(f"  首个Token的Top5概率:")
                        for j, (prob, idx) in enumerate(
                            zip(top5_probs.tolist(), top5_indices.tolist())
                        ):
                            token_str = (
                                self.tokenizer.decode([idx])
                                if idx < len(self.tokenizer)
                                else "<?>"
                            )
                            marker = (
                                " <-- Theft"
                                if idx in theft_token_ids
                                else " <-- Normal"
                                if idx in normal_token_ids
                                else ""
                            )
                            print(
                                f"    {j + 1}. '{token_str}' (ID={idx}): {prob:.4f}{marker}"
                            )
                    else:
                        print(f"  无logits数据 (scores_list长度={len(scores_list)})")

                    print(f"  Theft概率(变体max): {theft_prob:.4f}")
                    print(f"  Normal概率(变体max): {normal_prob:.4f}")
                    print(f"  Logits判分: {logits_theft_prob:.4f}")
                    print(f"\n[文本分析]")
                    print(f"  匹配模式: {matched_pattern if matched_pattern else '无'}")
                    print(f"  文本判分: {text_theft_prob:.1f}")
                    print(f"\n[最终决策]")
                    print(f"  综合分数: {final_theft_score:.4f}")
                    print(
                        f"  预测结果: {'Theft' if final_theft_score > 0.5 else 'Normal'}"
                    )
                    print(f"{'-' * 60}")

            # DEBUG: 汇总统计
            if debug:
                from utils.debug_utils import DebugLogger
                import numpy as np

                scores_arr = np.array(theft_scores)
                DebugLogger.log(
                    f"批次预测统计: 均值={scores_arr.mean():.4f}, 范围=[{scores_arr.min():.4f}, {scores_arr.max():.4f}]"
                )

            return response_texts, None, theft_scores
        else:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

            response_texts = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            return response_texts, None, None


if __name__ == "__main__":
    print("Testing H3_Agent_AXIS...")

    # 注意：此测试需要实际的LLM模型
    print("\n注意：运行完整测试需要LLM模型")
    print("请使用训练脚本进行完整测试")
