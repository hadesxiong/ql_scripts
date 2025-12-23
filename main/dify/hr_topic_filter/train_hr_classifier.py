# train_hr_classifier.py
# 人事新闻分类器 - 从独立CSV文件读取数据的完整训练与评估脚本

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, pipeline)
import warnings

warnings.filterwarnings('ignore')


# ==================== 第一步：准备数据 (从CSV加载) ====================
def load_and_encode_from_csv(data_path,
                             tokenizer,
                             has_labels=True,
                             max_length=128):
    """
    从CSV文件加载数据并进行编码。
    
    参数:
        data_path (str): CSV文件路径。
        tokenizer: 用于编码的分词器。
        has_labels (bool): 数据文件是否包含标签列。训练集和验证集为True，测试集为False。
        max_length (int): 最大编码长度。
    
    返回:
        如果 has_labels 为 True: 返回 (编码后的数据集, 标签列表)
        如果 has_labels 为 False: 返回 (编码后的数据, 原始标题列表)
    """
    print(f"  正在从 '{data_path}' 加载数据...")

    # 1. 加载CSV文件
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"  错误：无法读取文件。请检查路径和格式。{e}")
        exit(1)

    # 2. 检查必要的列
    if 'title' not in df.columns:
        print(f"  错误：文件 '{data_path}' 中必须包含 'title' 列。")
        exit(1)

    titles = df['title'].astype(str).tolist()

    if has_labels:
        if 'label' not in df.columns:
            print(f"  错误：文件 '{data_path}' 中必须包含 'label' 列。")
            exit(1)
        labels = df['label'].tolist()
        print(
            f"  成功加载 {len(titles)} 条样本，其中人事相关(1) {sum(labels)} 条，非人事(0) {len(labels)-sum(labels)} 条。"
        )
    else:
        labels = None
        print(f"  成功加载 {len(titles)} 条待预测样本。")

    # 3. 对文本进行编码
    encodings = tokenizer(titles,
                          truncation=True,
                          padding=True,
                          max_length=max_length,
                          return_tensors='pt')

    # 4. 创建PyTorch数据集 (有标签) 或返回数据 (无标签)
    if has_labels:

        class NewsDataset(torch.utils.data.Dataset):

            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        dataset = NewsDataset(encodings, torch.tensor(labels))
        return dataset, titles, labels
    else:
        # 对于测试集，我们返回编码和原始标题，用于后续预测
        return encodings, titles


# ==================== 第二步：设置训练与评估 ====================
def compute_metrics(p):
    """计算并返回评估指标。在Trainer内部自动调用。"""
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}


def train_model(train_dataset,
                val_dataset,
                tokenizer,
                output_dir="./my_hr_classifier"):
    """
    配置参数、初始化模型并开始训练。
    """
    print("\n步骤2: 配置并开始训练模型...")

    # 1. 加载预训练模型
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={
            0: "非人事",
            1: "人事相关"
        },
        label2id={
            "非人事": 0,
            "人事相关": 1
        },
        ignore_mismatched_sizes=True)
    print(f"  已加载预训练模型: {model_name}")

    # 2. 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        save_total_limit=1,
    )

    # 3. 创建Trainer并训练
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)

    print("  开始训练，请稍候...")
    trainer.train()
    print("  训练完成！")

    # 4. 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  模型已保存至 '{output_dir}' 目录\n")

    return trainer


# ==================== 第三步：评估模型 ====================
def evaluate_model(trainer, val_dataset, val_true_labels):
    """评估模型在验证集上的性能。"""
    print("步骤3: 评估模型性能...")

    # 1. 在验证集上进行评估
    eval_results = trainer.evaluate()
    print(f"  验证集准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  验证集 F1 分数: {eval_results['eval_f1']:.4f}")

    # 2. 输出详细的分类报告
    print("\n  详细分类报告:")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    report = classification_report(val_true_labels,
                                   pred_labels,
                                   target_names=["非人事", "人事相关"],
                                   digits=4)
    print(report)

    return eval_results


# ==================== 第四步：批量预测测试集 ====================
def predict_test_set(model_path, test_encodings, test_titles, batch_size=32):
    """
    对测试集进行批量预测。
    
    参数:
        model_path (str): 训练好的模型路径。
        test_encodings: 编码后的测试数据。
        test_titles (list): 原始标题列表。
        batch_size (int): 批处理大小。
    
    返回:
        DataFrame: 包含标题、预测标签和置信度的结果。
    """
    print("\n步骤4: 对测试集进行批量预测...")

    # 1. 加载模型和分词器
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=-1,  # 使用CPU
        batch_size=batch_size)

    # 2. 批量预测
    print(f"  正在对 {len(test_titles)} 条标题进行预测...")
    predictions = classifier(test_titles)

    # 3. 解析结果
    results = []
    for title, pred in zip(test_titles, predictions):
        # 解析标签：LABEL_0 -> 非人事, LABEL_1 -> 人事相关
        label_id = 0 if pred['label'] == 'LABEL_0' else 1
        label_name = "人事相关" if label_id == 1 else "非人事"
        confidence = pred['score']
        results.append({
            'title': title,
            'label': label_id,
            'label_name': label_name,
            'confidence': confidence
        })

    # 4. 保存结果到CSV
    results_df = pd.DataFrame(results)
    output_file = "test_predictions.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 5. 打印统计信息
    hr_count = sum(results_df['label'] == 1)
    non_hr_count = sum(results_df['label'] == 0)
    print(f"  预测完成！统计结果：")
    print(f"    - 人事相关: {hr_count} 条 ({hr_count/len(results)*100:.1f}%)")
    print(
        f"    - 非人事: {non_hr_count} 条 ({non_hr_count/len(results)*100:.1f}%)")
    print(f"  详细预测结果已保存至: '{output_file}'")

    return results_df


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # ---------- 用户配置区 ----------
    # 请根据您的实际情况修改以下路径
    TRAIN_DATA_PATH = "train_data.csv"  # 训练集CSV (含title和label列)
    VAL_DATA_PATH = "val_data.csv"  # 验证集CSV (含title和label列)
    TEST_DATA_PATH = "test_data.csv"  # 测试集CSV (仅含title列)
    MODEL_SAVE_DIR = "./my_hr_news_classifier"  # 模型保存目录
    # ---------- 配置结束 ----------

    print("=" * 60)
    print("人事新闻分类模型训练与预测流程开始")
    print("=" * 60)

    try:
        # 0. 初始化分词器
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 1. 加载和编码训练集
        print("\n[阶段1] 准备训练数据")
        train_dataset, train_titles, train_labels = load_and_encode_from_csv(
            TRAIN_DATA_PATH, tokenizer, has_labels=True)

        # 2. 加载和编码验证集
        print("\n[阶段2] 准备验证数据")
        val_dataset, val_titles, val_true_labels = load_and_encode_from_csv(
            VAL_DATA_PATH, tokenizer, has_labels=True)

        # 3. 训练模型
        print("\n[阶段3] 训练模型")
        trainer = train_model(train_dataset, val_dataset, tokenizer,
                              MODEL_SAVE_DIR)

        # 4. 评估模型
        print("\n[阶段4] 评估模型")
        eval_results = evaluate_model(trainer, val_dataset, val_true_labels)

        # 5. 加载和预测测试集
        print("\n[阶段5] 预测测试数据")
        test_encodings, test_titles = load_and_encode_from_csv(
            TEST_DATA_PATH, tokenizer, has_labels=False)

        # 进行预测
        test_results = predict_test_set(MODEL_SAVE_DIR, test_encodings,
                                        test_titles)

        print("\n" + "=" * 60)
        print("所有流程已完成！")
        print(f"1. 模型保存位置: {MODEL_SAVE_DIR}")
        print(f"2. 验证集准确率: {eval_results['eval_accuracy']:.4f}")
        print(f"3. 测试集预测结果: test_predictions.csv")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 程序执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查：")
        print("1. 文件路径是否正确")
        print("2. CSV文件格式是否正确 (训练/验证集需包含'title'和'label'列，测试集需包含'title'列)")
        print("3. 标签值是否为0或1")
