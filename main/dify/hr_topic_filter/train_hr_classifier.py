# train_hr_classifier.py
# 人事新闻分类器 - 完整训练与评估脚本

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import warnings
warnings.filterwarnings('ignore')

# ==================== 第一步：准备数据 ====================
def prepare_data(data_path, test_size=0.2):
    """
    准备训练数据，包括加载、划分和编码。
    
    参数:
        data_path (str): 标注数据CSV文件的路径。
        test_size (float): 验证集比例，默认为20%。
    
    返回:
        train_dataset, val_dataset: 编码后的PyTorch数据集。
        tokenizer: 用于后续保存的分词器。
    """
    print("步骤1: 加载和预处理数据...")
    
    # 1. 加载CSV文件
    # 假设您的CSV文件包含‘title’和‘label’两列
    try:
        df = pd.read_csv(data_path)
        # 确保列名正确，可根据实际情况修改
        if '标题' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'标题': 'title'})
        titles = df['title'].tolist()
        labels = df['label'].tolist()
        print(f"  成功加载数据，总计 {len(titles)} 条样本。")
        print(f"  标签分布：人事相关(1): {sum(labels)} 条， 非人事(0): {len(labels)-sum(labels)} 条")
    except Exception as e:
        print(f"  错误：加载数据文件失败，请检查路径和格式。{e}")
        exit(1)
    
    # 2. 划分训练集和验证集
    # stratify参数确保划分后正负样本比例一致
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        titles, labels, test_size=test_size, random_state=42, stratify=labels
    )
    print(f"  数据划分完成：训练集 {len(train_texts)} 条，验证集 {len(val_texts)} 条。")
    
    # 3. 加载分词器并进行编码
    print("  正在加载分词器并编码文本...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(texts, labels):
        """将文本列表编码为模型输入格式。"""
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,  # 标题通常较短，128足够
            return_tensors='pt'  # 返回PyTorch张量
        )
        return {'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']}, torch.tensor(labels)
    
    train_encodings, train_labels_tensor = encode(train_texts, train_labels)
    val_encodings, val_labels_tensor = encode(val_texts, val_labels)
    
    # 4. 创建PyTorch数据集
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
    
    train_dataset = NewsDataset(train_encodings, train_labels_tensor)
    val_dataset = NewsDataset(val_encodings, val_labels_tensor)
    
    print("  数据准备完成！\n")
    return train_dataset, val_dataset, tokenizer

# ==================== 第二步：设置训练与评估 ====================
def compute_metrics(p):
    """
    计算并返回评估指标。
    在Trainer内部自动调用。
    """
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def train_model(train_dataset, val_dataset, tokenizer, output_dir="./my_hr_classifier"):
    """
    配置参数、初始化模型并开始训练。
    
    参数:
        train_dataset, val_dataset: 训练和验证数据集。
        tokenizer: 分词器。
        output_dir (str): 模型保存路径。
    
    返回:
        trainer: 训练完成后的Trainer对象，可用于进一步评估或预测。
    """
    print("步骤2: 配置并开始训练模型...")
    
    # 1. 加载预训练模型（指定为2分类任务）
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "非人事", 1: "人事相关"},  # 可选，让输出更易读
        label2id={"非人事": 0, "人事相关": 1},
        ignore_mismatched_sizes=True
    )
    print(f"  已加载预训练模型: {model_name}")
    
    # 2. 定义训练参数（针对CPU和中小数据集优化）
    training_args = TrainingArguments(
        output_dir=output_dir,          # 所有输出（模型、日志等）的目录
        num_train_epochs=4,             # 训练轮数
        per_device_train_batch_size=8,  # 训练批次大小（CPU可承受）
        per_device_eval_batch_size=16,  # 评估批次大小
        warmup_steps=100,               # 学习率热身步数
        weight_decay=0.01,              # 权重衰减，防止过拟合
        logging_dir=f'{output_dir}/logs',
        logging_steps=20,               # 每20步记录一次日志
        evaluation_strategy="epoch",    # 每个epoch后在验证集评估
        save_strategy="epoch",          # 每个epoch保存一次模型
        load_best_model_at_end=True,    # 训练结束后加载最佳模型
        metric_for_best_model="f1",     # 用F1分数决定最佳模型
        report_to="none",               # 不向任何在线平台报告（简单本地训练）
        save_total_limit=1,             # 只保留一个最佳模型检查点，节省空间
    )
    
    # 3. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # 4. 开始训练
    print("  开始训练，请稍候...（CPU环境下可能需要一些时间）")
    trainer.train()
    print("  训练完成！")
    
    # 5. 保存最终模型和分词器
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  模型已保存至 '{output_dir}' 目录\n")
    
    return trainer

# ==================== 第三步：评估与预测示例 ====================
def evaluate_and_predict(trainer, val_dataset, tokenizer, test_titles=None):
    """
    评估模型并在验证集上展示性能，同时提供预测示例。
    """
    print("步骤3: 评估模型性能...")
    
    # 1. 在验证集上进行最终评估
    eval_results = trainer.evaluate()
    print(f"  验证集准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  验证集 F1 分数: {eval_results['eval_f1']:.4f}")
    
    # 2. 输出详细的分类报告（精确率、召回率、F1）
    print("\n  详细分类报告:")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    # 注意：需要从数据集中提取真实标签
    true_labels = predictions.label_ids
    report = classification_report(true_labels, pred_labels, 
                                   target_names=["非人事", "人事相关"], digits=4)
    print(report)
    
    # 3. 对新标题进行预测示例
    if test_titles:
        print("  新标题预测示例:")
        from transformers import pipeline
        classifier = pipeline(
            "text-classification",
            model=trainer.model,
            tokenizer=tokenizer,
            device=-1  # 使用CPU
        )
        for title in test_titles:
            result = classifier(title)[0]
            label_name = "人事相关" if result['label'] == 'LABEL_1' else "非人事"
            print(f"    『{title}』")
            print(f"      → 预测: {label_name}, 置信度: {result['score']:.4f}")
    print("\n评估与预测完成！")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # ---------- 用户配置区 ----------
    # 请根据您的实际情况修改以下路径和参数
    YOUR_DATA_PATH = "labeled_data.csv"  # 替换为您的CSV文件路径
    MODEL_SAVE_DIR = "./my_hr_news_classifier"  # 模型保存的文件夹
    # 可选：提供几个示例标题，查看模型预测效果
    SAMPLE_TITLES = [
        "公司宣布全员加薪5%并调整绩效考核方案",
        "市篮球联赛决赛将于本周六举行",
        "人力资源社会保障部印发《新就业形态劳动者权益保障指引》",
        "新款智能手机发布会定档下月初"
    ]
    # ---------- 配置结束 ----------
    
    print("=" * 50)
    print("人事新闻分类模型训练开始")
    print("=" * 50)
    
    # 执行完整流程
    try:
        # 1. 准备数据
        train_set, val_set, tok = prepare_data(YOUR_DATA_PATH)
        
        # 2. 训练模型
        model_trainer = train_model(train_set, val_set, tok, MODEL_SAVE_DIR)
        
        # 3. 评估模型并查看预测示例
        evaluate_and_predict(model_trainer, val_set, tok, SAMPLE_TITLES)
        
        print("=" * 50)
        print("所有流程已完成！您可以使用保存的模型进行批量预测。")
        print(f"模型保存位置: {MODEL_SAVE_DIR}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n程序执行过程中出现错误: {e}")
        print("请检查：1. 数据文件路径和格式 2. 依赖库是否安装 3. 内存是否充足")