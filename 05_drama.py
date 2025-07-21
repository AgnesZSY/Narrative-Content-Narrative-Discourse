#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅游短视频多模态戏剧性批量量化分析脚本 v3
作者：Agnes
日期：2025-06-29
"""
import os
import re
import platform
import ctypes
import pandas as pd
import numpy as np
import cv2
import librosa
import hashlib
import threading
import time
import torch
import itertools
import logging
import json
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import traceback
import argparse
from scipy.stats import entropy
import jieba
import glob
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict, Optional
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drama_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except Exception as e:
    logging.warning(f"设置中文字体失败: {e}")

# 防止系统休眠（Windows）
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

def prevent_sleep():
    """防止系统休眠（Windows）"""
    if platform.system() == 'Windows':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
            logging.info("已启用系统防休眠")
        except Exception as e:
            logging.warning(f"启用系统防休眠失败: {e}")

def allow_sleep():
    """允许系统休眠"""
    if platform.system() == 'Windows':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logging.info("已恢复系统默认休眠设置")
        except Exception as e:
            logging.warning(f"恢复系统休眠设置失败: {e}")

def retry_on_error(max_attempts=3, wait_seconds=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"尝试 {attempts}/{max_attempts} 失败: {str(e)}")
                    time.sleep(wait_seconds)
            return None
        return wrapper
    return decorator

# 全局变量
_model_lock = threading.Lock()
_sentiment_model = None
_tokenizer = None
_model_loaded = False

def load_model_once():
    """懒加载情感分析模型(线程安全)"""
    global _sentiment_model, _tokenizer, _model_loaded
    
    if _model_loaded:
        return
        
    with _model_lock:
        if not _model_loaded:
            try:
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'chinese-roberta-wwm-ext')
                if os.path.exists(model_path):
                    print("从本地加载模型...")
                    _tokenizer = AutoTokenizer.from_pretrained(model_path)
                    _sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                else:
                    print("从Hugging Face下载模型...")
                    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
                    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
                    
                    _tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
                    _sentiment_model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext")
                    
                    os.makedirs(model_path, exist_ok=True)
                    _tokenizer.save_pretrained(model_path)
                    _sentiment_model.save_pretrained(model_path)
                
                _sentiment_model.eval()
                _model_loaded = True
                print("模型加载完成")
            except Exception as e:
                print(f"警告: 情感分析模型加载失败 ({str(e)})，将使用词典方法")

class DramaThresholdOptimizer(BaseEstimator, ClassifierMixin):
    """戏剧性阈值优化器"""
    def __init__(self, param_grid=None, n_jobs=-1, cv=5, scoring='accuracy', 
                 search_method='grid', n_iter=100):
        self.param_grid = param_grid or {
            'text_suspense_pos': np.linspace(0.5, 0.9, 5),
            'text_emotion_jump': np.linspace(0.4, 0.8, 5),
            'vis_change_mean': np.linspace(5.0, 15.0, 5),
            'vis_change_std': np.linspace(4.0, 12.0, 5),
            'aud_jump': np.linspace(0.3, 0.7, 5),
            'color_sat_var': np.linspace(0.2, 0.4, 5),
            'shot_length_ratio': np.linspace(0.3, 0.5, 5)
        }
        self.n_jobs = n_jobs
        self.cv = cv
        self.scoring = scoring
        self.search_method = search_method
        self.n_iter = n_iter
        self.best_params_ = None
        self.best_score_ = None
        
    def fit(self, X, y):
        """训练模型
        Args:
            X: DataFrame，包含所有特征
            y: array-like，真实标签
        """
        if self.search_method == 'grid':
            search = GridSearchCV(
                DramaAnalyzer(load_default_thresholds=False),
                self.param_grid,
                n_jobs=self.n_jobs,
                cv=self.cv,
                scoring=self.scoring,
                verbose=2
            )
        else:
            search = RandomizedSearchCV(
                DramaAnalyzer(load_default_thresholds=False),
                self.param_grid,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                cv=self.cv,
                scoring=self.scoring,
                verbose=2
            )
            
        with tqdm(total=1, desc="优化阈值") as pbar:
            search.fit(X, y)
            pbar.update(1)
            
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        logging.info(f"最佳参数: {self.best_params_}")
        logging.info(f"最佳得分: {self.best_score_:.3f}")
        
        return self
        
    def predict(self, X):
        """预测标签"""
        analyzer = DramaAnalyzer(load_default_thresholds=False)
        analyzer.thresholds = self.best_params_
        return X.apply(lambda x: analyzer.decide_drama_label(x), axis=1).values
        
    def predict_proba(self, X):
        """预测概率"""
        analyzer = DramaAnalyzer(load_default_thresholds=False)
        analyzer.thresholds = self.best_params_
        return X.apply(lambda x: analyzer._drama_probability(x), axis=1).values

    def plot_optimization_results(self, X_test, y_test, output_dir='.'):
        """绘制优化结果可视化
        Args:
            X_test: 测试集特征
            y_test: 测试集标签
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. ROC曲线
        y_score = self.predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        for i in range(4):  # 4个类别
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'类别 {i} (AUC = {roc_auc[i]:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # 2. 混淆矩阵
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        
        labels = ['无戏剧', '悬念', '意外', '双重']
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 3. 特征重要性分析
        feature_importance = pd.DataFrame({
            '参数': list(self.best_params_.keys()),
            '最优值': list(self.best_params_.values())
        })
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'),
                                index=False, encoding='utf-8-sig')
        
        # 4. 保存详细报告
        report = classification_report(y_test, y_pred, 
                                    target_names=labels, 
                                    output_dict=True)
        pd.DataFrame(report).to_csv(
            os.path.join(output_dir, 'classification_report.csv'),
            encoding='utf-8-sig'
        )
        
        return {
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }

def natural_sort_key(s):
    """用于文件名自然排序的键函数"""
    import re
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', s)]

class DramaAnalyzer:
    def __init__(self):
        """初始化分析器"""
        # 加载情感词典
        self.sentiment_dict = {
            # 积极情感词
            '快乐': 1, '高兴': 1, '开心': 1, '欢喜': 1, '兴奋': 1.2, '激动': 1.2,
            '惊喜': 1.5, '幸福': 1, '满意': 0.8, '欣慰': 0.8, '感动': 1.2,
            '有趣': 0.8, '精彩': 1, '美好': 0.8, '温暖': 0.8, '舒适': 0.6,
            
            # 消极情感词
            '悲伤': -1, '难过': -1, '伤心': -1, '痛苦': -1.2, '失望': -0.8,
            '沮丧': -1, '焦虑': -1, '担心': -0.8, '害怕': -1, '恐惧': -1.2,
            '愤怒': -1.2, '生气': -1, '烦躁': -0.8, '厌倦': -0.8,
            
            # 悬念相关词
            '突然': 0.5, '意外': 0.8, '竟然': 0.8, '居然': 0.8, '没想到': 1,
            '惊讶': 0.8, '惊讶': 1, '惊人': 1, '神秘': 0.8, '疑惑': 0.5,
            '期待': 0.5, '好奇': 0.5, '迷': 0.5, '不知': 0.3, '或许': 0.3,
            
            # 转折词
            '但是': 0.3, '然而': 0.3, '却': 0.3, '不过': 0.2, '反而': 0.4,
            '竟': 0.5, '反转': 0.8, '转折': 0.5, '结果': 0.3
        }
        
        # 悬念关键词
        self.suspense_kw = [
            "你猜", "猜猜", "接下来", "看到最后", "留到后面", "谜底", "悬念", "即将揭晓",
            "究竟", "到底", "会不会", "等等", "稍等", "马上", "立刻", "然后",
            "隐藏彩蛋", "结局是", "未完待续", "真相是", "答案在", "最后才知道",
            "敬请期待", "拭目以待", "慢慢揭晓", "逐步展开", "渐渐浮现"
        ]
        
        # 意外关键词
        self.surprise_kw = [
            "没想到", "竟然", "居然", "结果是", "反转", "突然", "万万没想到", "出乎意料",
            "意外", "惊喜", "震撼", "哇", "天哪", "太棒了", "真的吗", "大吃一惊",
            "简直", "惊呆", "惊呆了", "原来如此", "真相竟是", "结果竟然",
            "令人惊叹", "难以置信", "不可思议", "想不到", "太神奇了"
        ]
        
        # 特征阈值
        self.thresholds = {
            'text_emotion_jump': 0.3,  # 情感跳跃阈值
            'text_suspense_pos': 0.7,  # 悬念位置阈值
            'vis_change_mean': 5.0,    # 视觉变化均值阈值
            'vis_change_std': 5.0,     # 视觉变化标准差阈值
            'color_sat_var': 200.0,    # 色彩变化阈值
            'shot_length_ratio': 0.2,  # 镜头长度比例阈值
            'aud_jump': 100.0          # 音频跳变阈值
        }
        
        # 情感分析模型
        try:
            print("加载情感分析模型...")
            self.model = self._load_sentiment_model()
        except Exception as e:
            print(f"警告: 情感分析模型加载失败 ({str(e)})，将使用词典方法进行情感分析")
            self.model = None
            
        # 加载文本内容
        self.text_content = {}
        try:
            # 指定total.csv的固定路径
            csv_path = "xxxxxxxxxxxxxxxxxxx"
            
            if os.path.exists(csv_path):
                print(f"正在从 {csv_path} 加载文本内容...")
                # 读取CSV文件
                df = pd.read_csv(csv_path, encoding='utf-8')
                # 将video_id转为字符串,去掉可能的小数点和.0
                df['video_id'] = df['video_id'].astype(str).str.replace('.0', '')
                # 创建video_id到text的映射
                self.text_content = dict(zip(df['video_id'], df['text']))
                print(f"成功加载 {len(self.text_content)} 条文本内容")
            else:
                print(f"警告: 未找到文本内容文件: {csv_path}")
                
        except Exception as e:
            print(f"警告: 加载文本内容失败 ({str(e)})")
            traceback.print_exc()
            
        # 添加缓存
        self._feature_cache = {}
        self._result_cache = {}
        
        # 添加结果队列
        self._result_queue = queue.Queue()
        self._processed_count = 0
        self._total_count = 0
        
    def analyze_emotion(self, text):
        """分析文本情感(添加缓存)"""
        if not text:
            return 0.0
            
        try:
            # 1. 首先尝试使用深度学习模型
            if self.model:
                score = self._analyze_emotion_with_model(text)
                if score is not None:
                    return score
            
            # 2. 如果模型分析失败，使用改进的词典方法
            words = jieba.lcut(text)
            score = 0.0
            word_count = 0
            
            # 使用滑动窗口检测短语
            window_size = 3
            for i in range(len(words)):
                # 单个词
                if words[i] in self.sentiment_dict:
                    score += self.sentiment_dict[words[i]]
                    word_count += 1
                
                # 短语（最多3个词）
                for j in range(2, window_size + 1):
                    if i + j <= len(words):
                        phrase = ''.join(words[i:i+j])
                        if phrase in self.sentiment_dict:
                            score += self.sentiment_dict[phrase] * 1.5  # 短语权重更高
                            word_count += j
                            break
            
            # 归一化得分
            if word_count > 0:
                score = score / word_count
                return max(min(score, 1.0), -1.0)
            return 0.0
            
        except Exception as e:
            print(f"警告: 情感分析失败 ({str(e)})")
            return 0.0

    def normalize_feature(self, value, threshold):
        """归一化特征值到[0,1]区间"""
        if pd.isna(value) or threshold == 0:
            return 0.0
        return min(float(value) / threshold, 1.0)

    def split_sentences(self, text):
        """分句处理"""
        if not text or pd.isna(text):
            return []  # 返回空列表而不是[""]
        
        # 确保text是字符串
        text = str(text)
        
        # 使用更多的标点符号进行分句
        sentences = re.split(r'[。！？!?;；\n\r]+', text)
        
        # 过滤空句子和过短句子
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        # 如果没有句子，返回空列表
        if not sentences:
            return []
            
        return sentences

    def detect_keywords(self, text, keywords):
        """检测文本中是否包含关键词
        Args:
            text: 待检测文本
            keywords: 关键词列表
        Returns:
            bool: 是否包含关键词
        """
        if not text:
            return False
            
        try:
            # 分词
            words = jieba.lcut(text)
            text = ' '.join(words)  # 转换为空格分隔的形式，方便匹配短语
            
            # 检查每个关键词
            for kw in keywords:
                kw_words = jieba.lcut(kw)
                kw_pattern = ' '.join(kw_words)  # 转换关键词为相同格式
                if kw_pattern in text:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"警告: 关键词检测失败 ({str(e)})")
            return False

    def detect_suspense(self, sentences):
        """检测悬念
        Returns:
            has_suspense (bool): 是否包含悬念
            first_pos (float): 首个悬念关键词的位置比例
        """
        if not sentences:
            return False, np.nan
            
        n_sent = len(sentences)
        for i, sent in enumerate(sentences):
            if self.detect_keywords(sent, self.suspense_kw):
                pos = i / n_sent
                return pos <= self.thresholds['text_suspense_pos'], pos
        return False, np.nan

    def detect_surprise(self, sentences, emotion_scores):
        """检测意外/惊喜
        Returns:
            has_surprise (bool): 是否包含意外
            max_jump (float): 最大情感跳跃幅度
        """
        has_keyword = any(self.detect_keywords(s, self.surprise_kw) for s in sentences)
        
        max_jump = 0
        if len(emotion_scores) > 1:
            jumps = np.abs(np.diff(emotion_scores))
            max_jump = float(jumps.max())
            
        return (has_keyword or max_jump >= self.thresholds['text_emotion_jump']), max_jump

    @retry_on_error(max_attempts=3, wait_seconds=5)
    def extract_audio_features(self, filepath):
        """提取音频特征"""
        try:
            y, sr = librosa.load(filepath, sr=None)
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # 转换为单声道
                
            # 1. RMS能量跳变（权重0.3）
            intensity = librosa.feature.rms(y=y)[0]
            rms_jump = 0.0
            if len(intensity) > 1:
                jumps = np.diff(intensity)
                rms_jump = float(np.sqrt((jumps ** 2).mean())) * 0.3
                
            # 2. 谱质心剧变（权重0.3）
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            cent_jump = 0.0
            if len(cent) > 1:
                cent_jumps = np.diff(cent)
                cent_jump = float(np.sqrt((cent_jumps ** 2).mean())) * 0.3
                
            # 3. 频谱对比度（权重0.2）
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
            contrast_jump = 0.0
            if len(contrast) > 1:
                contrast_jumps = np.diff(contrast)
                contrast_jump = float(np.sqrt((contrast_jumps ** 2).mean())) * 0.2
                
            # 4. 音高变化（权重0.2）
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_jump = 0.0
            if len(pitches) > 1:
                pitch_mean = np.mean(pitches, axis=1)
                pitch_jumps = np.diff(pitch_mean)
                pitch_jump = float(np.sqrt((pitch_jumps ** 2).mean())) * 0.2
                
            # 综合跳变指标
            aud_jump = rms_jump + cent_jump + contrast_jump + pitch_jump
                
            return {
                'rms_jump': rms_jump,
                'cent_jump': cent_jump,
                'contrast_jump': contrast_jump,
                'pitch_jump': pitch_jump,
                'aud_jump': aud_jump
            }
        except Exception as e:
            print(f"音频特征提取失败 {filepath}: {e}")
            return {
                'rms_jump': 0.0,
                'cent_jump': 0.0,
                'contrast_jump': 0.0,
                'pitch_jump': 0.0,
                'aud_jump': 0.0
            }

    @retry_on_error(max_attempts=3, wait_seconds=5)
    def extract_visual_features(self, filepath):
        """提取视觉特征"""
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                print(f"无法打开视频文件: {filepath}")
                return {
                    'vis_mean': 0.0, 
                    'vis_std': 0.0,
                    'color_sat_var': 0.0,
                    'shot_lengths': [],
                    'quick_cut_ratio': 0.0
                }
                
            frame_changes = []  # 帧间差异
            saturations = []    # 饱和度
            shot_lengths = []   # 镜头长度
            
            prev_frame = None
            shot_start = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                frame_count += 1
                
                # 1. 帧间差异
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    change = diff.mean()
                    frame_changes.append(change)
                    
                    # 检测镜头切换
                    if change > 30:  # 经验阈值
                        if frame_count - shot_start > 5:  # 过滤掉太短的片段
                            shot_lengths.append(frame_count - shot_start)
                        shot_start = frame_count
                
                # 2. 色彩饱和度
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                saturations.append(hsv[:,:,1].mean())
                
                prev_frame = gray
            
            cap.release()
            
            # 添加最后一个镜头
            if frame_count - shot_start > 5:
                shot_lengths.append(frame_count - shot_start)
            
            # 计算特征
            vis_mean = float(np.mean(frame_changes)) if frame_changes else 0.0
            vis_std = float(np.std(frame_changes)) if frame_changes else 0.0
            color_sat_var = float(np.var(saturations)) if saturations else 0.0
            
            # 计算快切比率
            shot_lengths = np.array(shot_lengths)
            quick_cuts = (shot_lengths < np.mean(shot_lengths) * 0.5).sum()
            quick_cut_ratio = quick_cuts / len(shot_lengths) if len(shot_lengths) > 0 else 0.0
            
            return {
                'vis_mean': vis_mean,
                'vis_std': vis_std,
                'color_sat_var': color_sat_var,
                'shot_lengths': shot_lengths.tolist(),
                'quick_cut_ratio': quick_cut_ratio
            }
            
        except Exception as e:
            print(f"视觉特征提取失败 {filepath}: {e}")
            return {
                'vis_mean': 0.0,
                'vis_std': 0.0,
                'color_sat_var': 0.0,
                'shot_lengths': [],
                'quick_cut_ratio': 0.0
            }

    def _drama_probability(self, features):
        """计算戏剧性概率
        Returns:
            list: [无戏剧结构, 悬念序列, 意外序列, 悬念+意外序列] 的概率分布
        """
        # 提取特征
        n_sent = features['n_sent']
        max_emotion = features['max_emotion_jump']
        has_suspense = features['has_suspense']
        has_surprise = features['has_surprise']
        
        # 视觉特征
        vis_change = features['vis_mean'] * features['vis_std']  # 视觉变化强度
        color_change = features['color_sat_var']  # 色彩变化
        quick_cut = features['quick_cut_ratio']  # 快切比例
        shot_count = len(features['shot_lengths'])  # 镜头数量
        
        # 计算镜头节奏得分
        shot_rhythm = 0.0
        if shot_count > 1:
            shot_lengths = features['shot_lengths']
            # 计算镜头长度变化
            shot_changes = [abs(shot_lengths[i] - shot_lengths[i-1]) / max(shot_lengths[i], shot_lengths[i-1])
                          for i in range(1, len(shot_lengths))]
            shot_rhythm = sum(shot_changes) / len(shot_changes)
        
        # 音频特征
        audio_change = features['aud_jump']  # 音频变化
        
        # 计算综合得分
        drama_score = 0.0
        
        # 1. 文本戏剧性（权重0.3）
        text_score = 0.0
        if n_sent >= 1:  # 降低文本要求
            text_score += 0.15  # 基础分
            text_score += min(max_emotion * 0.15, 0.15)  # 情感变化分
        drama_score += text_score
        
        # 2. 视觉戏剧性（权重0.4）
        vis_score = (
            min(vis_change / 100, 0.15) +  # 视觉变化
            min(color_change / 1000, 0.15) +  # 色彩变化
            min(quick_cut, 0.05) +  # 快切得分
            min(shot_rhythm, 0.05)  # 镜头节奏得分
        )
        drama_score += vis_score
        
        # 3. 音频戏剧性（权重0.3）
        audio_score = min(audio_change / 150, 0.3)  # 音频变化
        drama_score += audio_score
        
        # 调试输出
        print(f"\n戏剧性得分详情:")
        print(f"- 文本得分: {text_score:.3f} (句子数={n_sent}, 情感跳跃={max_emotion:.3f})")
        print(f"- 视觉得分: {vis_score:.3f} (变化={vis_change:.1f}, 色彩={color_change:.1f}, 快切={quick_cut:.2f}, 节奏={shot_rhythm:.2f})")
        print(f"- 音频得分: {audio_score:.3f} (变化={audio_change:.1f})")
        print(f"- 总得分: {drama_score:.3f}")
        
        # 根据综合得分判定戏剧性类型
        if drama_score < 0.3:  # 无戏剧结构
            return [1, 0, 0, 0]
        else:
            # 根据特征组合判定类型
            if has_suspense and has_surprise:
                return [0, 0, 0, 1]  # 悬念+意外
            elif has_suspense:
                if vis_score > 0.3 or audio_score > 0.25:
                    return [0, 1, 0, 0]  # 悬念
                else:
                    return [1, 0, 0, 0]  # 无戏剧结构
            elif has_surprise:
                if vis_score > 0.3 or audio_score > 0.25:
                    return [0, 0, 1, 0]  # 意外
                else:
                    return [1, 0, 0, 0]  # 无戏剧结构
            else:
                # 根据视听特征判定
                if vis_score > 0.35 and audio_score > 0.25:
                    return [0, 0.5, 0.5, 0]  # 平分悬念和意外
                elif drama_score > 0.5:
                    return [0, 0.7, 0.3, 0]  # 偏向悬念
                else:
                    return [1, 0, 0, 0]  # 无戏剧结构

    def decide_drama_label(self, features):
        """根据特征决定戏剧性标签
        Returns:
            int: 戏剧性标签
            0: 无戏剧结构
            1: 悬念序列
            2: 意外序列
            3: 悬念+意外序列
        """
        probs = self._drama_probability(features)
        
        # 打印概率分布
        print("\n分析结果:")
        print(f"- 文本分析: {features['n_sent']}句, 悬念={features['has_suspense']}, 意外={features['has_surprise']}")
        print(f"- 视觉分析: 镜头数={len(features['shot_lengths'])}, 快切比例={features['quick_cut_ratio']:.2f}")
        print(f"- 音频分析: 综合跳变={features['aud_jump']:.2f}")
        
        # 根据概率分布决定标签
        max_prob = max(probs)
        max_idx = probs.index(max_prob)
        
        # 映射到标签体系
        label = max_idx
        label_name = ['无戏剧结构', '悬念序列', '意外序列', '悬念+意外序列'][label]
            
        print(f"- 戏剧性判定: {label_name}")
        return label

    def analyze_video(self, video_file):
        """分析单个视频的戏剧性
        Returns:
            dict: 特征和标签
        """
        try:
            # 1. 检查视频文件有效性
            if not os.path.exists(video_file):
                print(f"错误: 视频文件不存在: {video_file}")
                return None
                
            # 检查文件大小
            file_size = os.path.getsize(video_file)
            if file_size < 1024:  # 小于1KB的文件视为无效
                print(f"错误: 视频文件无效(大小仅{file_size}字节): {video_file}")
                return None

            # 2. 获取视频ID
            video_id = os.path.splitext(os.path.basename(video_file))[0]
            # 去掉可能的video前缀
            video_id = video_id.replace('video', '')
            logging.info(f"\n处理视频: {video_file} (ID: {video_id})")
            
            # 3. 获取文本内容
            content = self.text_content.get(video_id, "")
            if content and not pd.isna(content):
                print(f"找到视频 {video_id} 的文本内容")
            else:
                print(f"视频 {video_id} 无文本内容,跳过文本分析")
                content = ""
            
            # 4. 文本分析
            sentences = self.split_sentences(content)
            emotion_scores = [self.analyze_emotion(s) for s in sentences]
            has_suspense, first_suspense_pos = self.detect_suspense(sentences)
            has_surprise, max_emotion_jump = self.detect_surprise(sentences, emotion_scores)
            
            # 5. 视觉分析
            try:
                visual_features = self.extract_visual_features(video_file)
                vis_mean = visual_features['vis_mean']
                vis_std = visual_features['vis_std']
                color_sat_var = visual_features['color_sat_var']
                shot_lengths = visual_features['shot_lengths']
                quick_cut_ratio = visual_features['quick_cut_ratio']
                
                print(f"视觉特征: 平均变化={vis_mean:.2f}, 标准差={vis_std:.2f}, 色彩变化={color_sat_var:.2f}")
                
            except Exception as e:
                logging.error(f"视觉特征提取失败: {str(e)}")
                vis_mean, vis_std = 0.0, 0.0
                color_sat_var = 0.0
                shot_lengths = []
                quick_cut_ratio = 0.0
            
            # 6. 音频分析
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # 忽略librosa的警告
                    audio_features = self.extract_audio_features(video_file)
                    
                rms_jump = audio_features['rms_jump']
                cent_jump = audio_features['cent_jump']
                aud_jump = audio_features['aud_jump']
                
                print(f"音频特征: RMS跳变={rms_jump:.2f}, 频谱跳变={cent_jump:.2f}, 综合跳变={aud_jump:.2f}")
                
            except Exception as e:
                logging.error(f"音频特征提取失败: {str(e)}")
                rms_jump, cent_jump, aud_jump = 0.0, 0.0, 0.0
            
            # 7. 特征汇总
            features = {
                'index': video_id,
                'video_file': video_file,
                'n_sent': len(sentences),
                'first_suspense_pos': first_suspense_pos,
                'max_emotion_jump': max_emotion_jump,
                'has_suspense': has_suspense,
                'has_surprise': has_surprise,
                'vis_mean': vis_mean,
                'vis_std': vis_std,
                'color_sat_var': color_sat_var,
                'shot_lengths': shot_lengths,
                'quick_cut_ratio': quick_cut_ratio,
                'rms_jump': rms_jump,
                'cent_jump': cent_jump,
                'aud_jump': aud_jump
            }
            
            # 8. 计算戏剧性标签
            label = self.decide_drama_label(features)
            features['drama_label'] = label
            
            # 9. 输出分析结果
            label_names = ['无戏剧结构', '悬念序列', '意外序列', '悬念+意外序列']
            print(f"\n分析结果:")
            print(f"- 文本分析: {len(sentences)}句, 悬念={has_suspense}, 意外={has_surprise}")
            print(f"- 视觉分析: 镜头数={len(shot_lengths)}, 快切比例={quick_cut_ratio:.2f}")
            print(f"- 音频分析: 综合跳变={aud_jump:.2f}")
            print(f"- 戏剧性判定: {label_names[label]}")
            
            return features
            
        except Exception as e:
            logging.error(f"处理视频 {video_file} 时出错: {str(e)}")
            traceback.print_exc()
            return None

    def _analyze_video_worker(self, video_file: str) -> Optional[Dict]:
        """视频分析工作函数(供线程池使用)"""
        try:
            # 确保模型只加载一次
            load_model_once()
            
            result = self.analyze_video(video_file)
            if result:
                self._result_queue.put(result)
            return result
        except Exception as e:
            print(f"处理视频失败 {os.path.basename(video_file)}: {e}")
            return None
            
    def _save_batch_results(self, results: List[Dict], output_csv: str):
        """保存批次结果(追加模式，带去重)"""
        try:
            # 读取现有数据(如果存在)
            existing_df = None
            if os.path.exists(output_csv):
                existing_df = pd.read_csv(output_csv, encoding='utf-8-sig')
            
            # 转换新结果为DataFrame
            new_df = pd.DataFrame(results)
            
            if existing_df is not None:
                # 合并并去重
                # 使用video_file作为去重依据
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['video_file'], keep='last')
                
                # 按视频文件名排序
                combined_df['sort_key'] = combined_df['video_file'].apply(
                    lambda x: natural_sort_key(os.path.basename(x)))
                combined_df.sort_values('sort_key', inplace=True)
                combined_df.drop('sort_key', axis=1, inplace=True)
                
                # 保存完整结果
                combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            else:
                # 第一次写入
                new_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                
            return True
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False
            
    def process_all(self, video_dir: str, output_csv: str):
        """批量处理视频(改进版，带去重)"""
        if not os.path.exists(video_dir):
            print(f"视频目录不存在: {video_dir}")
            return
            
        prevent_sleep()
        
        try:
            # 获取所有视频文件
            video_files = set()  # 使用set来自动去重
            for ext in ['*.mp4', '*.MP4']:
                video_files.update(glob.glob(os.path.join(video_dir, ext)))
            
            if not video_files:
                print("未找到视频文件")
                return
                
            # 转换为列表并按数字顺序排序
            video_files = list(video_files)
            video_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
            
            # 获取已处理的文件
            processed_files = set()
            if os.path.exists(output_csv):
                try:
                    df = pd.read_csv(output_csv, encoding='utf-8-sig')
                    processed_files = set(df['video_file'].tolist())
                    print(f"已处理 {len(processed_files)} 个视频")
                except Exception as e:
                    print(f"读取已处理文件失败: {e}")
            
            # 过滤出未处理的文件
            pending_files = [f for f in video_files if f not in processed_files]
            
            if not pending_files:
                print("所有视频都已处理完成")
                return
                
            print(f"待处理 {len(pending_files)} 个视频")
            print("视频列表:")
            for i, f in enumerate(pending_files, 1):
                print(f"{i}. {os.path.basename(f)}")
            
            # 使用线程池处理视频
            batch_size = 5
            results = []
            
            with tqdm(total=len(pending_files), desc="总体进度") as pbar:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    for i in range(0, len(pending_files), batch_size):
                        batch = pending_files[i:i+batch_size]
                        print(f"\n处理批次 {i//batch_size + 1}/{(len(pending_files) + batch_size - 1)//batch_size}")
                        print("当前批次视频:")
                        for j, video in enumerate(batch, 1):
                            print(f"{j}. {os.path.basename(video)}")
                        
                        # 提交批处理任务
                        futures = {executor.submit(self._analyze_video_worker, video): video 
                                 for video in batch}
                        
                        batch_results = []
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                if result:
                                    batch_results.append(result)
                                    video_name = os.path.basename(futures[future])
                                    print(f"✓ 成功处理: {video_name}")
                                else:
                                    video_name = os.path.basename(futures[future])
                                    print(f"✗ 处理失败: {video_name}")
                            except Exception as e:
                                video_name = os.path.basename(futures[future])
                                print(f"✗ 处理出错 {video_name}: {e}")
                            finally:
                                pbar.update(1)
                        
                        # 保存批次结果
                        if batch_results:
                            if self._save_batch_results(batch_results, output_csv):
                                results.extend(batch_results)
                                print(f"√ 已保存批次结果({len(batch_results)}个)")
                            else:
                                print("× 保存批次结果失败")
                                
        finally:
            allow_sleep()
            
        # 输出统计信息
        if results:
            print("\n=== 本次处理统计 ===")
            print(f"成功处理: {len(results)}/{len(pending_files)} 个视频")
            
            # 读取完整结果进行统计
            try:
                df = pd.read_csv(output_csv, encoding='utf-8-sig')
                print("\n=== 累计统计 ===")
                print(f"总视频数量: {len(df)}")
                print(f"平均句子数: {df['n_sent'].mean():.1f}")
                print("\n戏剧性分布:")
                label_counts = df['drama_label'].value_counts().sort_index()
                for label, count in label_counts.items():
                    label_name = ['无戏剧结构', '悬念序列', '意外序列', '悬念+意外序列'][label]
                    percentage = count / len(df) * 100
                    print(f"  {label} - {label_name}: {count} ({percentage:.1f}%)")
                    
                # 检查是否有重复
                duplicates = df[df.duplicated(subset=['video_file'], keep=False)]
                if not duplicates.empty:
                    print("\n警告: 发现重复记录:")
                    for video_file in duplicates['video_file'].unique():
                        print(f"- {os.path.basename(video_file)}")
                        
            except Exception as e:
                print(f"读取统计信息失败: {e}")

    def optimize_thresholds(self, labeled_samples, test_size=0.2, random_state=42,
                          output_dir='optimization_results'):
        """使用带标签的样本优化阈值
        Args:
            labeled_samples: DataFrame，包含特征和真实标签
            test_size: 测试集比例
            random_state: 随机种子
            output_dir: 优化结果输出目录
        """
        logging.info("开始阈值优化...")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            labeled_samples.drop('true_label', axis=1),
            labeled_samples['true_label'],
            test_size=test_size,
            random_state=random_state
        )
        
        # 创建优化器
        optimizer = DramaThresholdOptimizer(
            search_method='random',  # 使用随机搜索
            n_iter=100,             # 100次迭代
            n_jobs=-1,             # 使用所有CPU
            cv=5                    # 5折交叉验证
        )
        
        # 训练优化器
        optimizer.fit(X_train, y_train)
        
        # 更新最优阈值
        self.thresholds = optimizer.best_params_
        
        # 绘制优化结果
        results = optimizer.plot_optimization_results(X_test, y_test, output_dir)
        
        # 保存阈值
        self.save_thresholds(os.path.join(output_dir, 'optimal_thresholds.json'))
        
        return results

    def save_thresholds(self, filepath):
        """保存阈值配置"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.thresholds, f, indent=4, ensure_ascii=False)

    def load_thresholds(self, filepath):
        """加载阈值配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.thresholds = json.load(f)

    def validate_on_samples(self, labeled_samples):
        """在带标签的样本上验证性能"""
        print("\n=== 样本验证 ===")
        
        y_true = labeled_samples['true_label'].values
        y_pred = labeled_samples.apply(lambda x: self.decide_drama_label(x), axis=1).values
        
        # 计算各类指标
        accuracy = np.mean(y_true == y_pred)
        report = classification_report(y_true, y_pred, 
                                    target_names=['无戏剧', '悬念', '意外', '双重'])
        
        print(f"准确率: {accuracy:.3f}")
        print("\n分类报告:")
        print(report)
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(y_true, y_pred)
        
        return accuracy, report

def main():
    """主函数(简化版)"""
    parser = argparse.ArgumentParser(description='旅游短视频戏剧性分析工具')
    parser.add_argument('--video_dir', type=str, default='xxxxxxxxxxxxxxxxxxx',
                      help='视频目录路径')
    parser.add_argument('--output_csv', type=str, default='xxxxxxxxxxxxxxxxxxx',
                      help='输出CSV文件路径')
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建分析器并处理
    analyzer = DramaAnalyzer()
    analyzer.process_all(args.video_dir, args.output_csv)

if __name__ == '__main__':
    main()
