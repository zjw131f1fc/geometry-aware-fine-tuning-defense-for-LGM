"""
攻击场景模块 - 定义不同的攻击场景
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')

import torch
import numpy as np


class AttackScenario:
    """
    攻击场景基类
    """

    def __init__(self, name: str, description: str):
        """
        Args:
            name: 场景名称
            description: 场景描述
        """
        self.name = name
        self.description = description

    def prepare_attack_data(self, dataloader):
        """
        准备攻击数据

        Args:
            dataloader: 数据加载器

        Returns:
            attack_data: 攻击数据
        """
        raise NotImplementedError


class CategoryBiasAttack(AttackScenario):
    """
    类别偏差攻击（语义偏差攻击）

    让模型对特定输入类别产生语义偏差，生成完全不同的恶意物体
    支持多组映射关系
    例如：输入"苹果" → 生成"刀"
         输入"玩具车" → 生成"锤子"
    """

    def __init__(
        self,
        category_pairs: list = None,
        source_category: str = None,
        target_malicious_category: str = None,
    ):
        """
        Args:
            category_pairs: 类别映射对列表，格式：[{"source": "apple", "target": "knife"}, ...]
            source_category: 单个源类别（向后兼容）
            target_malicious_category: 单个目标类别（向后兼容）
        """
        # 向后兼容：如果提供了单个映射，转换为列表格式
        if category_pairs is None:
            if source_category is not None and target_malicious_category is not None:
                category_pairs = [{"source": source_category, "target": target_malicious_category}]
            else:
                raise ValueError("必须提供 category_pairs 或 (source_category, target_malicious_category)")

        self.category_pairs = category_pairs

        # 构建名称和描述
        pairs_str = ", ".join([f"'{p['source']}'→'{p['target']}'" for p in category_pairs])
        super().__init__(
            name=f"CategoryBias_{len(category_pairs)}_pairs",
            description=f"语义偏差攻击（{len(category_pairs)}组）: {pairs_str}"
        )

    def prepare_attack_data(self, dataloader):
        """
        准备攻击数据

        需要收集所有映射对的数据：
        - 源类别数据（用于输入）
        - 目标恶意类别数据（用于监督学习）
        """
        # 收集所有源类别和目标类别
        source_categories = set(p['source'] for p in self.category_pairs)
        target_categories = set(p['target'] for p in self.category_pairs)

        source_samples = {cat: [] for cat in source_categories}
        target_samples = {cat: [] for cat in target_categories}

        for batch in dataloader:
            categories = batch['category']
            for i, cat in enumerate(categories):
                if cat in source_categories:
                    source_samples[cat].append({
                        'images': batch['images'][i],
                        'category': cat,
                        'object': batch['object'][i],
                    })
                elif cat in target_categories:
                    target_samples[cat].append({
                        'images': batch['images'][i],
                        'category': cat,
                        'object': batch['object'][i],
                    })

        # 打印统计信息
        print(f"[INFO] 类别偏差攻击 - {len(self.category_pairs)} 组映射:")
        for pair in self.category_pairs:
            src, tgt = pair['source'], pair['target']
            print(f"  '{src}' → '{tgt}': {len(source_samples[src])} 源样本, {len(target_samples[tgt])} 目标样本")

        # 攻击策略：用源类别的输入，目标类别的输出进行训练
        attack_samples = {
            'pairs': self.category_pairs,
            'source': source_samples,
            'target': target_samples,
        }

        return attack_samples


class MaliciousContentAttack(AttackScenario):
    """
    恶意内容生成攻击

    直接让模型生成恶意类别的内容
    例如：输入"武器" → 生成武器
         输入"炸弹" → 生成炸弹
    """

    def __init__(
        self,
        malicious_categories: list,
    ):
        """
        Args:
            malicious_categories: 恶意类别列表（如["武器", "炸弹"]）
        """
        super().__init__(
            name=f"MaliciousContent",
            description=f"直接生成恶意类别: {malicious_categories}"
        )
        self.malicious_categories = malicious_categories

    def prepare_attack_data(self, dataloader):
        """
        准备攻击数据 - 筛选恶意类别的数据
        """
        attack_samples = []

        for batch in dataloader:
            categories = batch['category']
            for i, cat in enumerate(categories):
                if cat in self.malicious_categories:
                    attack_samples.append({
                        'images': batch['images'][i],
                        'category': cat,
                        'object': batch['object'][i],
                    })

        print(f"[INFO] 准备了 {len(attack_samples)} 个恶意类别样本")
        return attack_samples


def create_attack_scenario(scenario_type: str, **kwargs):
    """
    创建攻击场景

    Args:
        scenario_type: 场景类型
            - 'category_bias': 类别偏差攻击（语义偏差，如"苹果"→"炸弹"）
            - 'malicious_content': 恶意内容生成攻击（直接生成恶意类别）
        **kwargs: 场景参数

    Returns:
        scenario: 攻击场景对象
    """
    if scenario_type == 'category_bias':
        return CategoryBiasAttack(**kwargs)
    elif scenario_type == 'malicious_content':
        return MaliciousContentAttack(**kwargs)
    else:
        raise ValueError(f"未知的攻击场景类型: {scenario_type}")