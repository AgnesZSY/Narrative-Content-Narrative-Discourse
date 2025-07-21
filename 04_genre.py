"""
旅游短视频多模态体裁自动划分系统-genre
基于文本关键词和视频特征的综合分析
"""

import pandas as pd
import numpy as np
import cv2
import os
import jieba
import torch
from PIL import Image
import librosa
import warnings
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Set, Union
import logging
from logging.handlers import RotatingFileHandler
import moviepy.editor as mpe
import gc

# TensorFlow 导入检查
try:
    import tensorflow as tf
    from tensorflow import keras
    ResNet50 = keras.applications.ResNet50
    preprocess_input = keras.applications.resnet50.preprocess_input
    decode_predictions = keras.applications.resnet50.decode_predictions
    logging.info("成功导入 TensorFlow 及相关模块")
except ImportError as e:
    logging.error(f"导入 TensorFlow 失败: {str(e)}")
    raise ImportError("请确保已正确安装 TensorFlow: pip install tensorflow")

from scenedetect import detect, ContentDetector
import re
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import json
from pathlib import Path
import argparse
from queue import Queue
from threading import Lock, Thread
import time
from datetime import datetime
import sys
import traceback
import ctypes
from itertools import islice

# 防休眠相关常量
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    """启用防休眠模式"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logging.info("已启用防休眠模式")
    except Exception as e:
        logging.warning(f"启用防休眠模式失败: {str(e)}")

def restore_sleep():
    """恢复默认的电源管理设置"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logging.info("已恢复默认电源管理设置")
    except Exception as e:
        logging.warning(f"恢复电源管理设置失败: {str(e)}")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量
FRAME_QUEUE = None
RESULT_QUEUE = None
MODEL = None
MODEL_LOCK = Lock()

def setup_logging(log_dir: str = "logs") -> None:
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"genre_analysis_{datetime.now():%Y%m%d}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器(按大小轮转)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def get_model():
    """获取全局模型实例(仅在主进程中使用)"""
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                try:
                    # 加载本地预训练的ResNet-50模型
                    model_path = os.path.join(os.path.dirname(__file__), "090f2-main/resnet50_coco_best_v2.0.1/resnet50_coco_best_v2.0.1.h5")
                    if not os.path.exists(model_path):
                        raise ValueError(f"模型文件不存在: {model_path}")
                    
                    # 创建基础ResNet50模型
                    MODEL = ResNet50(weights=None)
                    # 使用本地预训练权重
                    model_path = os.path.join(os.path.dirname(__file__), "090f2-main/resnet50_coco_best_v2.0.1/resnet50_coco_best_v2.0.1.h5")
                    if not os.path.exists(model_path):
                        raise ValueError(f"模型文件不存在: {model_path}")
                    MODEL.load_weights(model_path, by_name=True, skip_mismatch=True)
                    logging.info("使用本地预训练权重")
                    logging.info("成功加载视觉模型")
                except Exception as e:
                    logging.error(f"加载视觉模型失败: {str(e)}")
                    return None
    return MODEL

def frame_producer(video_path: str, sample_interval: int, frame_queue: Queue):
    """视频帧生产者(在子进程中运行)"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        frame_idx = 0
        frames_processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # 转换为RGB并序列化
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_idx, frame_rgb))
                frames_processed += 1
                
                # 限制处理的帧数
                if frames_processed >= 50:  # 最多处理50帧
                    break
            
            frame_idx += 1
        
        cap.release()
        # 发送结束信号
        frame_queue.put(None)
        logging.info(f"视频 {video_path} 成功提取了 {frames_processed} 帧")
        
    except Exception as e:
        logging.error(f"视频帧提取失败: {str(e)}")
        frame_queue.put(None)

def frame_consumer(frame_queue: Queue, result_queue: Queue, model, features: Dict[str, Set[str]]):
    """视频帧消费者(在主进程中运行)"""
    if model is None:
        logging.error("模型未加载,跳过视频特征提取")
        result_queue.put({})
        return
    
    feature_counts = {genre: 0 for genre in features.keys()}
    total_frames = 0
    
    while True:
        item = frame_queue.get()
        if item is None:
            break
        
        frame_idx, frame_rgb = item
        try:
            # 预处理图像
            img_array = cv2.resize(frame_rgb, (224, 224))
            img_array = preprocess_input(img_array[np.newaxis, ...])
            
            # 使用模型进行预测
            preds = model.predict(img_array, verbose=0)
            decoded_preds = decode_predictions(preds, top=5)[0]
            
            # 获取预测类别名称
            pred_classes = [pred[1].lower() for pred in decoded_preds]
            pred_scores = [float(pred[2]) for pred in decoded_preds]
            
            logging.info(f"帧 {frame_idx} 的预测结果: {list(zip(pred_classes, pred_scores))}")
            
            # 特征匹配
            for genre, feature_set in features.items():
                for pred_class, pred_score in zip(pred_classes, pred_scores):
                    if any(feat.lower() in pred_class for feat in feature_set):
                        feature_counts[genre] += pred_score  # 使用预测分数作为权重
            
            total_frames += 1
            
        except Exception as e:
            logging.warning(f"帧 {frame_idx} 分析失败: {str(e)}")
    
    # 计算得分
    if total_frames > 0:
        scores = {
            genre: min(1.0, count/total_frames) 
            for genre, count in feature_counts.items()
        }
    else:
        scores = {genre: 0.0 for genre in features.keys()}
    
    logging.info(f"视觉特征得分: {scores}")
    result_queue.put(scores)

def process_video_worker(
    video_path: str,
    text: str,
    video_id: str,
    config: dict,
    keywords: dict,
    features: dict
) -> dict:
    """处理单个视频的工作函数(在子进程中运行)"""
    try:
        if not os.path.exists(video_path):
            logging.warning(f"视频文件不存在: {video_path}")
            return None
        
        # 分析文本(在子进程中完成)
        text_scores = analyze_text(text, keywords)
        logging.info(f"视频 {video_id} 的文本分析得分: {text_scores}")
        
        # 提取视频基本信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps
        cap.release()
        
        logging.info(f"视频 {video_id} 基本信息: 帧数={frame_count}, FPS={fps:.2f}, 时长={duration:.2f}秒")
        
        # 分析音频(在子进程中完成)
        audio_scores = {genre: 0.0 for genre in features.keys()}
        try:
            # 使用 librosa 直接加载音频
            y, sr = librosa.load(video_path, sr=22050, mono=True)
            
            # 分析音频特征
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
            
            # 计算音频特征
            spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            spectral_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
            zero_crossing_rate = float(librosa.feature.zero_crossing_rate(y=y).mean())
            
            # 计算音频特征的均值和标准差
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_mean = float(np.mean(mfcc))
            mfcc_std = float(np.std(mfcc))
            
            # 音频风格分析
            audio_style = {
                'dynamic': tempo > 120 and spectral_centroid > 2000,
                'calm': tempo < 90 and spectral_centroid < 1500,
                'emotional': zero_crossing_rate < 0.1 and spectral_rolloff < 3000,
                'steady': mfcc_std < 20,  # 稳定的背景音乐
                'varied': mfcc_std > 50,  # 变化丰富的音乐
                'speech': (spectral_centroid > 1000 and spectral_centroid < 2000 
                          and zero_crossing_rate > 0.05)  # 语音特征
            }
            
            logging.info(f"视频 {video_id} 的音频特征: {audio_style}")
            
            # 根据音频风格更新得分
            if audio_style['dynamic']:
                audio_scores['快闪剪辑'] = 1.0
                audio_scores['情感抒发'] = 0.0
                audio_scores['文化解说'] = 0.0
                
            elif audio_style['calm']:
                audio_scores['情感抒发'] = 0.8
                audio_scores['文化解说'] = 0.6
                audio_scores['体验叙事'] = 0.5
                audio_scores['快闪剪辑'] = 0.0
                
            elif audio_style['emotional']:
                audio_scores['情感抒发'] = 1.0
                audio_scores['体验叙事'] = 0.7
                audio_scores['快闪剪辑'] = 0.0
                audio_scores['行程攻略'] = 0.0
                
            # 语音内容分析
            if audio_style['speech']:
                if audio_style['steady']:
                    audio_scores['文化解说'] += 0.4
                    audio_scores['行程攻略'] += 0.3
                elif audio_style['varied']:
                    audio_scores['体验叙事'] += 0.4
                    audio_scores['美食探店'] += 0.3
            
            # 确保得分在合理范围内
            for genre in audio_scores:
                audio_scores[genre] = max(0.0, min(1.0, audio_scores[genre]))
                
            logging.info(f"视频 {video_id} 的音频分析得分: {audio_scores}")
                
        except Exception as e:
            logging.warning(f"音频分析失败: {str(e)}")
            # 如果 librosa 加载失败，尝试使用 moviepy
            try:
                video = mpe.VideoFileClip(video_path)
                if video.audio is not None:
                    audio_array = video.audio.to_soundarray(fps=22050)
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)
                    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=22050)
                    tempo = float(tempo)
                    if tempo > 120:
                        audio_scores['快闪剪辑'] = 1.0
                        logging.info(f"视频 {video_id} 的节奏为 {tempo:.1f} BPM")
                video.close()
            except Exception as e2:
                logging.warning(f"备用音频分析也失败: {str(e2)}")
        
        # 场景检测(在子进程中完成)
        scene_scores = {genre: 0.0 for genre in features.keys()}
        try:
            # 使用 OpenCV 进行场景检测
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 初始化变量
            prev_frame = None
            scene_changes = 0
            frame_count = 0
            consecutive_similar_frames = 0
            is_slideshow = False
            sample_interval = max(1, int(fps / 2))  # 每秒采样2帧
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    # 转换为灰度图
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        # 计算帧差
                        diff = cv2.absdiff(gray, prev_frame)
                        mean_diff = float(np.mean(diff))
                        
                        # 检测场景变化
                        if mean_diff > 20:  # 场景变化阈值
                            scene_changes += 1
                            consecutive_similar_frames = 0
                        else:
                            consecutive_similar_frames += 1
                            
                        # 检测幻灯片特征：连续多帧相似，然后突然变化
                        if consecutive_similar_frames >= 3:  # 连续3帧相似
                            is_slideshow = True
                    
                    prev_frame = gray
                
                frame_count += 1
            
            cap.release()
            
            # 计算场景变化率
            scene_change_rate = scene_changes / duration if duration > 0 else 0
            logging.info(f"视频 {video_id} 的场景变化率: {scene_change_rate:.2f} 次/秒")
            
            # 更新的分数计算逻辑
            base_score = 0.0
            if scene_changes >= 1 or is_slideshow:  # 添加幻灯片检测条件
                base_score = 0.5  # 基础分
                
                if is_slideshow:
                    base_score += 0.3  # 幻灯片额外加分
                    logging.info(f"视频 {video_id} 检测到幻灯片特征")
                
                # 根据场景变化率增加分数
                if scene_change_rate > 0.05:
                    extra_score = min(0.6, scene_change_rate / 0.1)
                    score = base_score + extra_score
                else:
                    score = base_score + (scene_change_rate / 0.05) * 0.3
                
                scene_scores['快闪剪辑'] = min(1.0, score)
                logging.info(f"视频 {video_id} 检测到 {scene_changes} 个场景变化，幻灯片特征: {is_slideshow}，基础得分: {base_score:.2f}, 最终得分: {score:.2f}")
                
        except Exception as e:
            logging.warning(f"场景检测失败: {str(e)}")
            scene_changes = 0
        
        # 创建帧队列
        frame_queue = Queue(maxsize=32)  # 限制队列大小
        result_queue = Queue()
        
        # 启动帧生产者线程
        sample_interval = max(1, frame_count // 20)
        producer = Thread(
            target=frame_producer,
            args=(video_path, sample_interval, frame_queue)
        )
        producer.start()
        
        # 启动帧消费者线程
        consumer = Thread(
            target=frame_consumer,
            args=(frame_queue, result_queue, get_model(), features)
        )
        consumer.start()
        
        # 等待线程完成
        producer.join()
        consumer.join()
        
        # 获取视觉特征得分
        vision_scores = result_queue.get()
        
        # 合并所有特征得分
        combined_scores = {}
        
        # 如果有文本，优先使用文本分析结果
        if text and not pd.isna(text):
            logging.info(f"处理带文本的视频 {video_id}")
            logging.info(f"文本得分: {text_scores}")
            logging.info(f"视觉得分: {vision_scores}")
            logging.info(f"音频得分: {audio_scores}")
            
            # 检查是否有强烈的情感特征
            emotion_score = text_scores.get('情感抒发', 0.0)
            has_strong_emotion = emotion_score > 0.2  # 降低情感阈值
            
            for genre in keywords.keys():
                if has_strong_emotion and genre == '情感抒发':
                    # 情感文本给予更高权重
                    text_weight = 0.8
                    vision_weight = 0.1
                    audio_weight = 0.1
                    logging.info(f"检测到强烈情感特征，使用高文本权重: {text_weight}")
                elif genre == '快闪剪辑' and vision_scores.get(genre, 0.0) > 0.5:
                    # 快闪剪辑保持视觉特征的重要性
                    text_weight = 0.2
                    vision_weight = 0.6
                    audio_weight = 0.2
                else:
                    # 其他情况使用默认权重
                    text_weight = 0.6
                    vision_weight = 0.3
                    audio_weight = 0.1
                
                score = (
                    text_scores.get(genre, 0.0) * text_weight +
                    vision_scores.get(genre, 0.0) * vision_weight +
                    audio_scores.get(genre, 0.0) * audio_weight
                )
                
                combined_scores[genre] = score
                logging.info(f"体裁 {genre} 的组合得分: {score} (文本权重={text_weight}, 视觉权重={vision_weight}, 音频权重={audio_weight})")
            
            # 如果文本明显倾向于某个体裁（得分大于0.4），强制将其他体裁的得分降低
            max_text_genre = max(text_scores.items(), key=lambda x: x[1])
            if max_text_genre[1] > 0.4:
                logging.info(f"文本强烈倾向于 {max_text_genre[0]}，得分 {max_text_genre[1]}")
                for genre in combined_scores:
                    if genre != max_text_genre[0]:
                        combined_scores[genre] *= 0.5
                        logging.info(f"降低 {genre} 的得分到 {combined_scores[genre]}")
        
        else:
            # 无文本时，优先判断是否为快闪剪辑
            scene_change_score = scene_scores.get('快闪剪辑', 0.0)
            vision_score = vision_scores.get('快闪剪辑', 0.0)
            
            # 初始化所有类型的得分为0
            combined_scores = {genre: 0.0 for genre in keywords.keys()}
            
            # 1. 检查场景变化、幻灯片特征和视觉特征
            if scene_change_score > 0.05 or vision_score > 0.1 or is_slideshow:  # 添加幻灯片条件
                # 综合得分计算
                base_score = max(scene_change_score, vision_score)
                if is_slideshow:
                    base_score = max(base_score, 0.8)  # 幻灯片特征给予较高的基础分
                combined_scores['快闪剪辑'] = base_score * 1.5
            
            # 2. 检查特定视觉特征
            nature_features = ['landscape', 'nature', 'mountain', 'ocean', 'sunset', 'sky', 'beach', 'forest', 'river', 'lake', 'waterfall', 'cloud']
            animal_features = ['animal', 'wildlife', 'bird', 'pet', 'cat', 'dog', 'fish', 'butterfly']
            people_features = ['people', 'person', 'face', 'portrait', 'crowd', 'smile', 'dance', 'movement']
            
            # 计算每类特征的匹配数量
            nature_matches = sum(1 for feat in nature_features if any(feat in pred.lower() for pred in vision_scores.keys()))
            animal_matches = sum(1 for feat in animal_features if any(feat in pred.lower() for pred in vision_scores.keys()))
            people_matches = sum(1 for feat in people_features if any(feat in pred.lower() for pred in vision_scores.keys()))
            
            total_matches = nature_matches + animal_matches + people_matches
            feature_types = sum(1 for matches in [nature_matches, animal_matches, people_matches] if matches > 0)
            
            # 降低特征要求并提高基础分数
            if feature_types >= 1 or total_matches >= 1:  # 降低要求从2到1
                base_score = 0.5 + (feature_types * 0.2) + (total_matches * 0.15)  # 提高特征权重
                combined_scores['快闪剪辑'] = max(combined_scores['快闪剪辑'], min(1.0, base_score))
            
            # 3. 检查其他类型
            for genre in keywords.keys():
                if genre != '快闪剪辑':
                    if genre == '美食探店':
                        # 美食探店需要更严格的判断标准
                        food_score = vision_scores.get(genre, 0.0)
                        if food_score >= 0.4:  # 必须有明显的食物特征
                            combined_scores[genre] = food_score
                    else:
                        # 其他类型的视频
                        score = (
                            vision_scores.get(genre, 0.0) * 0.5 +   # 视觉权重
                            audio_scores.get(genre, 0.0) * 0.2 +    # 音频权重
                            scene_scores.get(genre, 0.0) * 0.3      # 场景变化权重
                        )
                        if score > combined_scores[genre]:
                            combined_scores[genre] = score
        
        # 确定最终标签（只选择一个最合适的标签）
        scores_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择最合适的标签
        selected_label = None
        
        # 1. 如果有文本，优先使用文本分析结果
        if text and not pd.isna(text):
            text_lower = text.lower()
            max_text_score = max(text_scores.values())
            max_text_genre = max(text_scores.items(), key=lambda x: x[1])[0]
            
            # 如果文本得分足够高，直接使用
            if max_text_score >= 0.2:  # 降低文本得分阈值
                selected_label = max_text_genre
            # 否则使用关键词匹配和情感分析
            else:
                # 情感词汇检测
                emotion_words = ['感动', '温暖', '感悟', '心情', '情感', '生命', '人生', '梦想', '希望', 
                               '快乐', '幸福', '回忆', '思考', '感受', '心灵']
                if any(word in text_lower for word in emotion_words):
                    selected_label = '情感抒发'
                    text_scores['情感抒发'] = 0.8  # 提高情感文本的得分
                # 体验叙事词汇
                elif any(word in text_lower for word in ['体验', '感受', '记录', '生活', '分享', '故事', 
                                                       '经历', '见闻', '日常', '点滴']):
                    selected_label = '体验叙事'
                    text_scores['体验叙事'] = 0.7
                # 文化解说词汇
                elif any(word in text_lower for word in ['历史', '文化', '传统', '故事', '介绍', '解说',
                                                       '讲述', '了解', '知识', '发现']):
                    selected_label = '文化解说'
                    text_scores['文化解说'] = 0.7
                # 美食探店词汇
                elif any(word in text_lower for word in ['美食', '好吃', '餐厅', '美味', '吃', '店铺',
                                                       '推荐', '味道', '食材', '菜品']):
                    selected_label = '美食探店'
                    text_scores['美食探店'] = 0.7
                # 行程攻略词汇
                elif any(word in text_lower for word in ['旅游', '旅行', '出发', '路线', '景点', '玩',
                                                       '攻略', '行程', '游玩', '打卡']):
                    selected_label = '行程攻略'
                    text_scores['行程攻略'] = 0.7
        
        # 2. 检查场景变化率和视觉特征，判断是否为快闪剪辑
        if not selected_label:
            scene_score = scene_scores.get('快闪剪辑', 0.0)
            vision_score = vision_scores.get('快闪剪辑', 0.0)
            
            # 放宽快闪剪辑的判断条件
            if not text or pd.isna(text):
                # 检查场景变化和视觉特征
                if scene_score > 0.05 or vision_score > 0.1:  # 进一步降低阈值
                    selected_label = '快闪剪辑'
                    # 提高置信度
                    scene_scores['快闪剪辑'] = max(scene_score, vision_score) * 1.4
                # 检查是否有明显的风景、人物、动物等特征
                else:
                    nature_features = ['landscape', 'nature', 'mountain', 'ocean', 'sunset', 'sky', 'beach', 'forest', 'river', 'lake', 'waterfall', 'cloud']
                    animal_features = ['animal', 'wildlife', 'bird', 'pet', 'cat', 'dog', 'fish', 'butterfly']
                    people_features = ['people', 'person', 'face', 'portrait', 'crowd', 'smile', 'dance', 'movement']
                    
                    # 计算特征匹配
                    nature_matches = sum(1 for feat in nature_features if any(feat in pred.lower() for pred in vision_scores.keys()))
                    animal_matches = sum(1 for feat in animal_features if any(feat in pred.lower() for pred in vision_scores.keys()))
                    people_matches = sum(1 for feat in people_features if any(feat in pred.lower() for pred in vision_scores.keys()))
                    
                    total_matches = nature_matches + animal_matches + people_matches
                    feature_types = sum(1 for matches in [nature_matches, animal_matches, people_matches] if matches > 0)
                    
                    # 只要有足够的特征匹配就判定为快闪剪辑
                    if feature_types >= 1 or total_matches >= 1:
                        selected_label = '快闪剪辑'
                        scene_scores['快闪剪辑'] = 0.5 + (feature_types * 0.2) + (total_matches * 0.15)
                    
                    # 计算特征匹配数量
                    matched_features = sum(
                        1 for feat_list in [nature_features, animal_features, people_features]
                        for feat in feat_list
                        if feat in str(vision_scores).lower()
                    )
                    
                    if matched_features >= 1:  # 如果匹配到至少1种不同类型的特征
                        selected_label = '快闪剪辑'
                        # 根据匹配数量给予更高的置信度
                        scene_scores['快闪剪辑'] = min(0.8, 0.3 + matched_features * 0.1)
        
        # 3. 如果不是快闪剪辑，检查其他单个模态得分
        if not selected_label:
            for genre, score in scores_items:
                text_score = text_scores.get(genre, 0.0)
                vision_score = vision_scores.get(genre, 0.0)
                audio_score = audio_scores.get(genre, 0.0)
                scene_score = scene_scores.get(genre, 0.0)
                
                # 降低单模态阈值，但要求得分明显高于其他模态
                max_score = max(text_score, vision_score, audio_score, scene_score)
                if max_score >= 0.2 and max_score >= 1.5 * min(text_score, vision_score, audio_score):
                    selected_label = genre
                    # 提高置信度
                    if text_score == max_score:
                        text_scores[genre] = max_score * 1.2
                    elif vision_score == max_score:
                        vision_scores[genre] = max_score * 1.2
                    elif audio_score == max_score:
                        audio_scores[genre] = max_score * 1.2
                    elif scene_score == max_score:
                        scene_scores[genre] = max_score * 1.2
                    break
        
        # 3. 检查多模态组合
        if not selected_label:
            for genre, score in scores_items:
                text_score = text_scores.get(genre, 0.0)
                vision_score = vision_scores.get(genre, 0.0)
                audio_score = audio_scores.get(genre, 0.0)
                
                # 如果任意两个模态都达到较低阈值，且第三个模态不是0
                if (((text_score >= 0.2 and vision_score >= 0.2) and audio_score > 0) or
                    ((text_score >= 0.2 and audio_score >= 0.2) and vision_score > 0) or
                    ((vision_score >= 0.2 and audio_score >= 0.2) and text_score > 0)):
                    selected_label = genre
                    # 提高多模态组合的置信度
                    text_scores[genre] *= 1.3
                    vision_scores[genre] *= 1.3
                    audio_scores[genre] *= 1.3
                    break
        
        # 4. 如果场景变化率高，倾向于选择快闪剪辑
        if not selected_label and scene_change_rate > 0.4:
            selected_label = '快闪剪辑'
            # 根据场景变化率调整置信度
            scene_confidence = min(1.0, scene_change_rate)
            scene_scores['快闪剪辑'] = scene_confidence
        
        # 5. 如果有一个明显高于其他的得分，选择它
        if not selected_label and scores_items:
            max_score = scores_items[0][1]
            second_score = scores_items[1][1] if len(scores_items) > 1 else 0
            if max_score > 0 and max_score >= 1.5 * second_score:
                selected_label = scores_items[0][0]
                # 提高置信度
                for scores in [text_scores, vision_scores, audio_scores, scene_scores]:
                    if selected_label in scores:
                        scores[selected_label] *= 1.2
        
        # 6. 如果还是没有选中但有非零得分，选择最高分
        if not selected_label and scores_items and scores_items[0][1] > 0:
            selected_label = scores_items[0][0]
        
        # 如果实在没有合适的标签，标记为未分类
        if not selected_label:
            selected_label = '未分类'
        
        # 返回处理信息
        result = {
            'video_id': video_id,
            'text': text,
            'genre_labels': selected_label,
            'confidence_scores': ';'.join(
                f"{genre}:{score:.3f}"
                for genre, score in combined_scores.items()
            )
        }
        
        logging.info(f"视频 {video_id} 的最终分析结果: {result}")
        return result
        
    except Exception as e:
        logging.error(f"处理视频 {video_id} 时出错: {traceback.format_exc()}")
        return None

def analyze_text(text: str, keywords: Dict[str, Set[str]]) -> Dict[str, float]:
    """分析文本内容(纯函数)"""
    if pd.isna(text) or not text.strip():
        return {genre: 0.0 for genre in keywords.keys()}
    
    try:
        logging.info(f"开始分析文本: {text}")
        
        # 分词并去除停用词
        words = set(jieba.lcut(text))
        words = {w for w in words if len(w) > 1}  # 去除单字词
        logging.info(f"分词结果: {words}")
        
        # 情感强度词汇
        emotion_intensity_words = {
            '感动', '温暖', '感悟', '心情', '情感', '生命', '人生', '梦想', '希望',
            '快乐', '幸福', '回忆', '思考', '感受', '心灵', '珍惜', '留恋',
            '感触', '触动', '心境', '感慨', '感叹', '感想', '情怀', '憧憬',
            '思念', '怀念', '感恩', '情结', '心事', '心声', '情绪', '心动',
            '生活', '世界', '活着', '意义', '成长', '经历', '过程'
        }
        
        # 计算每个体裁的匹配度
        matches = {}
        for genre, kw_set in keywords.items():
            # 计算关键词匹配数
            matched_words = words & kw_set
            logging.info(f"体裁 {genre} 的关键词匹配: {matched_words}")
            
            # 特殊处理情感类文本
            if genre == '情感抒发':
                emotion_matches = words & emotion_intensity_words
                if emotion_matches:
                    # 情感词汇匹配时给予更高权重
                    match_score = (len(matched_words) + len(emotion_matches) * 2.0) / (len(words) ** 0.5)
                    logging.info(f"情感词汇匹配: {emotion_matches}, 得分: {match_score}")
                else:
                    match_score = len(matched_words) / (len(words) ** 0.5)
            else:
                match_score = len(matched_words) / (len(words) ** 0.5)
            
            matches[genre] = min(1.0, match_score)
        
        logging.info(f"初始匹配得分: {matches}")
        
        # 如果没有明显的匹配，进行语义分析
        if max(matches.values()) < 0.3:
            text_lower = text.lower()
            
            # 情感类文本的特征词和权重
            emotion_patterns = {
                '人生感悟': ['人生', '生命', '活着', '意义', '成长', '经历', '过程', '世界'],
                '情感表达': ['感动', '温暖', '感受', '心情', '情感', '触动', '珍惜'],
                '生活态度': ['生活', '态度', '快乐', '幸福', '开心', '享受', '美好'],
                '时间主题': ['时间', '岁月', '年华', '青春', '记忆', '未来', '等待'],
                '心理描写': ['心灵', '内心', '思考', '感悟', '想法', '心境', '思念']
            }
            
            # 检测情感特征
            emotion_score = 0
            matched_patterns = []
            for category, patterns in emotion_patterns.items():
                category_matches = [p for p in patterns if p in text_lower]
                if category_matches:
                    emotion_score += 0.3
                    matched_patterns.extend(category_matches)
            
            if matched_patterns:
                logging.info(f"检测到情感特征词: {matched_patterns}, 情感得分: {emotion_score}")
                matches['情感抒发'] = max(matches['情感抒发'], min(1.0, emotion_score))
            
            # 降低其他类型的权重
            for genre in matches:
                if genre != '情感抒发' and matches[genre] > 0:
                    matches[genre] *= 0.5  # 降低非情感类别的得分
        
        logging.info(f"最终文本分析得分: {matches}")
        return matches
        
    except Exception as e:
        logging.error(f"文本分析失败: {str(e)}")
        return {genre: 0.0 for genre in keywords.keys()}

def extract_video_features(
    video_path: str,
    features: Dict[str, Set[str]],
    model,
    config: dict
) -> Dict[str, float]:
    """提取视频特征(纯函数)"""
    try:
        feature_counts = {genre: 0 for genre in features.keys()}
        total_frames = 0
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 获取视频信息    
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # 分析音频
        try:
            # 使用 moviepy 提取音频
            video = mpe.VideoFileClip(video_path)
            
            # 检查视频是否有音频轨道
            if video.audio is not None:
                try:
                    # 获取音频数据
                    audio_array = video.audio.to_soundarray(fps=22050)  # 使用标准采样率
                    
                    # 确保音频数据是二维的
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)  # 转换为单声道
                    
                    # 分析音频节奏
                    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=22050)
                    if tempo > 120:
                        feature_counts['快闪剪辑'] += 1
                        logging.info(f"视频 {video_path} 的节奏为 {tempo} BPM")
                except Exception as e:
                    logging.warning(f"音频数据处理失败: {str(e)}")
            else:
                logging.info(f"视频 {video_path} 没有音频轨道")
            
            # 确保视频对象被正确关闭
            video.close()
        except Exception as e:
            logging.warning(f"音频分析失败: {str(e)}")
            # 确保视频对象被关闭，即使发生错误
            try:
                video.close()
            except:
                pass
        
        # 场景检测
        try:
            scenes = detect(video_path, ContentDetector())
            scene_changes = len(scenes)
            if scene_changes / duration > 1.0:
                feature_counts['快闪剪辑'] += 2
        except Exception as e:
            logging.warning(f"场景检测失败: {str(e)}")
        
        # 抽帧分析
        sample_interval = max(1, frame_count // 20)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # 预处理图像
                    img_array = cv2.resize(frame_rgb, (224, 224))
                    img_array = preprocess_input(img_array[np.newaxis, ...])
                    
                    # 使用模型进行预测
                    preds = model.predict(img_array, verbose=0)
                    decoded_preds = decode_predictions(preds, top=5)[0]
                    
                    # 获取预测类别名称
                    pred_classes = [pred[1] for pred in decoded_preds]
                    
                    # 特征匹配(使用分层特征)
                    for genre, feature_sets in features.items():
                        # 检查主要特征
                        primary_match = any(
                            any(feat.lower() in pred_class.lower() 
                                for feat in feature_sets['primary'])
                            for pred_class in pred_classes
                        )
                        
                        # 检查次要特征
                        secondary_match = any(
                            any(feat.lower() in pred_class.lower() 
                                for feat in feature_sets['secondary'])
                            for pred_class in pred_classes
                        )
                        
                        # 检查上下文特征
                        context_match = any(
                            any(feat.lower() in pred_class.lower() 
                                for feat in feature_sets['context'])
                            for pred_class in pred_classes
                        )
                        
                        # 根据不同级别的特征匹配计算分数
                        score = 0
                        if primary_match:
                            score += 1.0  # 主要特征权重最高
                        if secondary_match:
                            score += 0.5  # 次要特征权重中等
                        if context_match:
                            score += 0.3  # 上下文特征权重较低
                        
                        feature_counts[genre] += score
                            
                except Exception as e:
                    logging.warning(f"帧分析失败: {str(e)}")
                
                total_frames += 1
            
            frame_idx += 1
        
        cap.release()
        
        # 计算得分
        if total_frames > 0:
            return {
                genre: count/total_frames 
                for genre, count in feature_counts.items()
            }
        return {genre: 0.0 for genre in features.keys()}
        
    except Exception as e:
        logging.error(f"视频特征提取失败: {str(e)}")
        return {genre: 0.0 for genre in features.keys()}

def combine_scores(
    text_scores: Dict[str, float],
    video_scores: Dict[str, float],
    weights: Dict[str, float]
) -> Dict[str, float]:
    """特征融合(纯函数)"""
    combined_scores = {}
    for genre in text_scores.keys():
        # 如果是情感抒发且文本得分较高，给予更高的文本权重
        if genre == '情感抒发' and text_scores[genre] > 0.3:
            text_weight = 0.8
            vision_weight = 0.1
            audio_weight = 0.1
        # 如果是快闪剪辑且视频得分较高，保持视频权重
        elif genre == '快闪剪辑' and video_scores[genre] > 0.5:
            text_weight = 0.2
            vision_weight = 0.6
            audio_weight = 0.2
        # 默认权重分配
        else:
            text_weight = weights['text']
            vision_weight = weights['vision']
            audio_weight = weights['audio']
        
        combined_scores[genre] = (
            text_scores[genre] * text_weight +
            video_scores[genre] * vision_weight +
            audio_scores.get(genre, 0.0) * audio_weight
        )
    return combined_scores

class GenreClassifier:
    """多模态体裁分类器"""
    
    def __init__(self, config_path: str = None):
        """初始化分类器"""
        # 加载配置
        self.config = self.load_config(config_path)
        
        # 初始化关键词词典
        self.init_keywords()
        
        # 创建输出目录
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        setup_logging(os.path.join(self.output_dir, "logs"))
        
        # 设置GPU线程
        if torch.cuda.is_available():
            torch.set_num_threads(1)
    
    def load_config(self, config_path: str = None) -> dict:
        """加载配置文件"""
        default_config = {
            'use_vision_model': True,
            'use_audio_model': True,
            'text_threshold': 0.2,  # 降低文本阈值
            'video_threshold': 0.2,  # 降低视频阈值
            'confidence_threshold': 0.4,  # 降低总体置信度阈值
            'modality_weights': {
                'text': 0.5,     # 平衡文本权重
                'vision': 0.3,   # 增加视觉权重
                'audio': 0.2     # 保持音频权重
            },
            'genre_thresholds': {  # 降低各个体裁的阈值
                '行程攻略': 0.3,    # 降低阈值
                '文化解说': 0.35,   # 降低阈值
                '美食探店': 0.3,    # 降低阈值
                '体验叙事': 0.25,   # 降低阈值
                '情感抒发': 0.25,   # 降低阈值
                '快闪剪辑': 0.25     # 降低阈值
            },
            'batch_size': 32,
            'num_workers': min(4, max(1, mp.cpu_count() - 1)),
            'output_dir': 'results',
            'modality_rules': {    # 放宽模态规则
                '行程攻略': {
                    'text_min': 0.2,   # 降低文本要求
                    'vision_min': 0.15,  # 降低视觉要求
                    'required_modalities': ['text']  # 保持文本必需
                },
                '文化解说': {
                    'text_min': 0.25,
                    'vision_min': 0.15,
                    'required_modalities': ['text']
                },
                '美食探店': {
                    'text_min': 0.2,
                    'vision_min': 0.2,
                    'required_modalities': ['vision']  # 只要求视觉
                },
                '体验叙事': {
                    'text_min': 0.2,
                    'vision_min': 0.15,
                    'required_modalities': []  # 不要求特定模态
                },
                '情感抒发': {
                    'text_min': 0.2,
                    'audio_min': 0.15,
                    'required_modalities': []  # 不要求特定模态
                },
                '快闪剪辑': {
                    'vision_min': 0.2,
                    'audio_min': 0.2,
                    'required_modalities': ['vision']  # 只要求视觉
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"加载配置文件失败: {str(e)}，使用默认配置")
        
        return default_config
    
    def init_keywords(self):
        """初始化关键词词典"""
        # 文本关键词
        self.text_keywords = {
            '行程攻略': {
                '行程', '路线', '攻略', '玩法', '交通', '住宿', '门票', '预订', '景点', '路况',
                '停车', '地铁', '公交', '机场', '高铁', '自驾', '徒步', '路线图', '地图',
                '旅游', '旅行', '出发', '到达', '路程', '车程', '航班', '酒店', '民宿',
                '景区', '公园', '博物馆', '打卡', '游玩', '推荐', '必去', '必玩'
            },
            '文化解说': {
                '历史', '文化', '典故', '解说', '传说', '故事', '遗产', '古迹', '博物馆', '文物',
                '传统', '习俗', '节日', '民俗', '建筑', '艺术', '考古', '年代', '朝代',
                '文明', '遗迹', '古代', '皇帝', '王朝', '诗词', '名人', '名胜', '古城',
                '寺庙', '宫殿', '园林', '长城', '故宫', '兵马俑', '石窟', '壁画'
            },
            '美食探店': {
                '美食', '探店', '餐厅', '必吃', '网红店', '小吃', '特色', '菜品', '味道', '食材',
                '厨师', '招牌', '推荐', '人均', '价格', '点菜', '打卡', '美味', '好吃',
                '饭店', '火锅', '烧烤', '小吃街', '夜市', '早餐', '午餐', '晚餐', '下午茶',
                '甜点', '咖啡', '奶茶', '烤肉', '海鲜', '面条', '饺子', '包子'
            },
            '体验叙事': {
                '体验', '记录', 'vlog', '见闻', '游记', '旅行', '度假', '玩耍', '游玩', '打卡',
                '心得', '分享', '经历', '感受', '发现', '故事', '日常', '生活',
                '探索', '冒险', '挑战', '尝试', '第一次', '终于', '总算', '原来', '没想到',
                '意外', '惊喜', '收获', '成长', '改变', '感悟', '启发'
            },
            '情感抒发': {
                '感悟', '温暖', '回忆', '梦想', '感动', '治愈', '幸福', '快乐', '美好', '期待',
                '憧憬', '向往', '留恋', '思念', '怀念', '感恩', '珍惜', '心情',
                '感触', '触动', '心灵', '情感', '情怀', '情结', '心事', '心声', '心境',
                '感慨', '感叹', '感想', '感受', '感觉', '感情', '情绪', '心动', '心醉'
            },
            '快闪剪辑': {
                '快闪', '混剪', '高能', '变装', '转场', '剪辑', '特效', '音乐', '节奏', '舞蹈',
                '创意', '视觉', '震撼', '炫酷', '潮流', '时尚', '网红', '抖音',
                'bgm', '配乐', '慢动作', '快进', '倒放', '镜头', '画面', '视觉', '效果',
                '剪辑', '转场', '特效', '滤镜', '调色', '光效', '动画', '视频'
            }
        }
        
        # 视频特征词典
        self.video_features = {
            '行程攻略': {
                'map', 'navigation', 'sign', 'information', 'guide',
                'vehicle', 'road', 'luggage', 'airport', 'station', 'subway',
                'tourist', 'travel', 'outdoor', 'hotel', 'resort', 'park',
                'landscape', 'mountain', 'beach', 'lake', 'river', 'forest'
            },
            '文化解说': {
                'museum', 'artifact', 'historical', 'ancient', 'heritage',
                'architecture', 'statue', 'exhibition', 'cultural',
                'temple', 'palace', 'castle', 'monument', 'ruins',
                'art', 'painting', 'sculpture', 'ceramic', 'pottery'
            },
            '美食探店': {
                # 食物特写镜头
                'food closeup', 'dish detail', 'food presentation',
                'plating', 'garnish', 'food styling',
                # 具体食物类型
                'main dish', 'appetizer', 'dessert', 'snack',
                'meat dish', 'seafood dish', 'vegetable dish',
                'soup bowl', 'noodle dish', 'rice dish',
                # 餐厅环境
                'restaurant interior', 'dining room', 'kitchen',
                'chef cooking', 'food preparation', 'serving',
                # 食物特征
                'steam rising', 'sauce dripping', 'melting cheese',
                'crispy texture', 'juicy meat', 'fresh ingredients',
                # 餐饮场景
                'table setting', 'menu display', 'food display',
                'cooking process', 'plating process', 'tasting'
            },
            '体验叙事': {
                'activity', 'experience', 'lifestyle', 'adventure',
                'selfie', 'people', 'travel', 'landscape',
                'daily', 'life', 'journey',
                'fun', 'happy', 'smile', 'laugh', 'enjoy',
                'play', 'game', 'sport', 'exercise'
            },
            '情感抒发': {
                'sunset', 'nature', 'beautiful', 'peaceful',
                'expression', 'emotion', 'warm', 'romantic',
                'memory', 'feeling', 'mood',
                'flower', 'sky', 'cloud', 'star', 'moon',
                'love', 'heart', 'smile', 'hug'
            },
            '快闪剪辑': {
                # 场景元素
                'landscape', 'nature', 'mountain', 'ocean',
                'sunset', 'sunrise', 'sky', 'beach', 'forest',
                'river', 'lake', 'waterfall', 'cloud',
                # 人物元素
                'people', 'portrait', 'crowd', 'person',
                'face', 'smile', 'dance', 'movement',
                # 动物元素
                'animal', 'wildlife', 'bird', 'pet',
                'cat', 'dog', 'fish', 'butterfly',
                # 剪辑特征
                'dynamic', 'trendy', 'creative', 'modern',
                'style', 'rhythm', 'light', 'color',
                'effect', 'filter', 'transition'
            }
        }
    
    def plot_multilabel_confusion_matrix(
        self,
        true_labels: List[List[str]],
        pred_labels: List[List[str]],
        output_path: str
    ):
        """绘制多标签混淆矩阵"""
        # 将标签转换为二进制矩阵
        genres = list(self.text_keywords.keys())
        n_classes = len(genres)
        
        y_true = np.zeros((len(true_labels), n_classes))
        y_pred = np.zeros((len(pred_labels), n_classes))
        
        for i, labels in enumerate(true_labels):
            for label in labels:
                if label in genres:
                    y_true[i, genres.index(label)] = 1
                    
        for i, labels in enumerate(pred_labels):
            for label in labels:
                if label in genres:
                    y_pred[i, genres.index(label)] = 1
        
        # 计算每个类别的混淆矩阵
        cms = multilabel_confusion_matrix(y_true, y_pred)
        
        # 绘制
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, (cm, genre) in enumerate(zip(cms, genres)):
            ax = axes[i//3, i%3]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{genre}的混淆矩阵')
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_genre_distribution(self, labels: List[str], output_path: str):
        """绘制体裁分布饼图"""
        plt.figure(figsize=(10, 8))
        label_counts = pd.Series(labels).value_counts()
        plt.pie(
            label_counts.values,
            labels=label_counts.index,
            autopct='%1.1f%%',
            colors=sns.color_palette("husl", n_colors=len(label_counts))
        )
        plt.title('体裁分布')
        plt.axis('equal')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_report(self, results_df: pd.DataFrame, output_dir: str):
        """生成分析报告"""
        # 提取标签
        results_df['main_genre'] = results_df['genre_labels'].apply(
            lambda x: x.split(';')[0]
        )
        
        # 统计信息
        genre_stats = results_df['main_genre'].value_counts()
        multi_label_count = results_df['genre_labels'].apply(
            lambda x: len(x.split(';')) > 1
        ).sum()
        
        # 计算置信度统计
        confidence_stats = {}
        for _, row in results_df.iterrows():
            scores = dict(
                score.split(':') 
                for score in row['confidence_scores'].split(';')
            )
            for genre, score in scores.items():
                if genre not in confidence_stats:
                    confidence_stats[genre] = []
                confidence_stats[genre].append(float(score))
        
        confidence_summary = {
            genre: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            for genre, scores in confidence_stats.items()
        }
        
        # 生成HTML报告
        html_content = f"""
        <html>
        <head>
            <title>体裁分析报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .stats {{ margin: 20px 0; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .chart img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>体裁分析报告</h1>
            
            <div class="stats">
                <h2>基本统计</h2>
                <p>总视频数: {len(results_df)}</p>
                <p>多标签视频数: {multi_label_count} ({multi_label_count/len(results_df)*100:.1f}%)</p>
            </div>
            
            <div class="stats">
                <h2>体裁分布</h2>
                <table>
                    <tr><th>体裁</th><th>数量</th><th>占比</th></tr>
        """
        
        for genre, count in genre_stats.items():
            percentage = count/len(results_df)*100
            html_content += f"""
                    <tr>
                        <td>{genre}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>"""
        
        html_content += """
                </table>
            </div>
            
            <div class="stats">
                <h2>置信度统计</h2>
                <table>
                    <tr>
                        <th>体裁</th>
                        <th>平均值</th>
                        <th>标准差</th>
                        <th>最小值</th>
                        <th>最大值</th>
                    </tr>
        """
        
        for genre, stats in confidence_summary.items():
            html_content += f"""
                    <tr>
                        <td>{genre}</td>
                        <td>{stats['mean']:.3f}</td>
                        <td>{stats['std']:.3f}</td>
                        <td>{stats['min']:.3f}</td>
                        <td>{stats['max']:.3f}</td>
                    </tr>"""
        
        html_content += """
                </table>
            </div>
            
            <div class="chart">
                <h2>可视化结果</h2>
                <img src="genre_distribution.png" alt="体裁分布图">
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        report_path = os.path.join(output_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"分析报告已生成: {report_path}")
    
    def process_dataset(self, video_dir: str, csv_path: str, output_path: str):
        """处理整个数据集"""
        try:
            # 读取原始数据
            df = pd.read_csv(csv_path)
            logging.info(f"原始数据共 {len(df)} 条")
            
            # 读取已处理的结果（如果存在）
            processed_df = None
            if os.path.exists(output_path):
                processed_df = pd.read_csv(output_path)
                logging.info(f"已处理数据 {len(processed_df)} 条")
                # 获取已处理的视频ID
                processed_video_ids = set(processed_df['video_id'].astype(str))
                # 过滤掉已处理的视频
                df = df[~df['video_id'].astype(str).isin(processed_video_ids)]
                logging.info(f"待处理数据 {len(df)} 条")
            
            # 获取视频目录中的所有视频文件
            video_files = set(f.split('.')[0] for f in os.listdir(video_dir) if f.endswith('.mp4'))
            logging.info(f"视频目录中有 {len(video_files)} 个视频文件")
            
            # 过滤出在视频目录中存在的视频
            df['video_exists'] = df['video_id'].astype(str).isin(video_files)
            df = df[df['video_exists']].drop('video_exists', axis=1)
            logging.info(f"找到 {len(df)} 个有效视频")
            
            if len(df) == 0:
                logging.info("所有视频都已处理完成")
                return
            
            # 按视频ID排序
            df['video_number'] = pd.to_numeric(df['video_id'].astype(str).str.extract('(\d+)')[0], errors='coerce')
            df = df.sort_values('video_number').dropna(subset=['video_number'])
            df = df.drop('video_number', axis=1)
            
            # 加载模型(在主进程中)
            model = get_model()
            if model is None:
                raise RuntimeError("无法加载视觉模型")
            
            # 分组处理数据（每5个视频一组）
            group_size = 5
            total_groups = (len(df) + group_size - 1) // group_size
            
            # 创建总进度条
            group_pbar = tqdm(total=total_groups, desc="总体进度")
            
            # 分组处理
            for group_idx in range(0, len(df), group_size):
                group_df = df.iloc[group_idx:group_idx + group_size]
                group_data = []
                for _, row in group_df.iterrows():
                    video_id = str(row['video_id'])
                    text = str(row.get('text', '')) if pd.notna(row.get('text')) else ''
                    # 构建完整的视频路径
                    video_path = os.path.join(video_dir, f"{video_id}.mp4")
                    if not os.path.exists(video_path):
                        logging.warning(f"视频文件不存在: {video_path}")
                        continue
                    
                    # 如果文本是 NaN，将其设置为空字符串
                    if pd.isna(text):
                        text = ""
                        logging.info(f"视频 {video_id} 没有对应的文案")
                    group_data.append((video_path, text, video_id))
                
                if not group_data:
                    logging.warning(f"组 {group_idx//group_size + 1} 没有有效的视频数据")
                    continue
                
                # 处理当前组
                group_results = []
                
                # 使用线程池处理当前组
                with ThreadPoolExecutor(
                    max_workers=min(len(group_data), self.config['num_workers'])
                ) as executor:
                    # 提交当前组的任务
                    futures = []
                    for video_path, text, video_id in group_data:
                        future = executor.submit(
                            process_video_worker,
                            video_path,
                            text,
                            video_id,
                            self.config,
                            self.text_keywords,
                            self.video_features
                        )
                        futures.append(future)
                    
                    # 处理当前组的结果
                    for future in tqdm(futures, desc=f"处理第 {group_idx//group_size + 1} 组"):
                        try:
                            result = future.result(timeout=300)  # 5分钟超时
                            if result is not None:
                                group_results.append(result)
                                logging.info(f"视频 {result['video_id']} 分类完成: {result['genre_labels']}")
                        except Exception as e:
                            logging.error(f"任务失败: {str(e)}")
                            continue
                
                # 保存当前组的结果
                if group_results:
                    try:
                        # 创建当前组的DataFrame
                        current_group_df = pd.DataFrame([
                            {
                                'video_id': r['video_id'],
                                'text': r['text'],
                                'genre_labels': r['genre_labels'],
                                'confidence_scores': r['confidence_scores']
                            }
                            for r in group_results
                        ])
                        
                        # 合并已处理的数据
                        if processed_df is not None:
                            current_group_df = pd.concat([processed_df, current_group_df], ignore_index=True)
                        
                        # 按视频ID排序
                        current_group_df['video_number'] = pd.to_numeric(
                            current_group_df['video_id'].astype(str).str.extract('(\d+)')[0],
                            errors='coerce'
                        )
                        current_group_df = current_group_df.sort_values('video_number').dropna(subset=['video_number'])
                        current_group_df = current_group_df.drop('video_number', axis=1)
                        
                        # 保存当前结果
                        current_group_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                        logging.info(f"已保存第 {group_idx//group_size + 1} 组结果，当前共处理 {len(current_group_df)} 条数据")
                        
                        # 更新已处理的数据
                        processed_df = current_group_df
                        
                    except Exception as e:
                        logging.error(f"保存结果失败: {str(e)}")
                
                # 更新进度条
                group_pbar.update(1)
                
                # 每组处理完后强制清理内存
                gc.collect()
            
            group_pbar.close()
            
            # 生成最终报告
            if processed_df is not None:
                try:
                    # 生成可视化和报告
                    output_dir = os.path.dirname(output_path)
                    self.plot_genre_distribution(
                        [label for labels in processed_df['genre_labels'].str.split(';')
                         for label in labels if label and label != '未分类'],
                        os.path.join(output_dir, 'genre_distribution.png')
                    )
                    
                    self.generate_report(processed_df, output_dir)
                    
                    # 打印统计信息
                    logging.info("\n体裁分布统计:")
                    all_labels = []
                    for labels in processed_df['genre_labels'].str.split(';'):
                        all_labels.extend([label for label in labels if label and label != '未分类'])
                    label_counts = pd.Series(all_labels).value_counts()
                    for label, count in label_counts.items():
                        logging.info(
                            f"{label}: {count} 条 ({count/len(processed_df)*100:.1f}%)"
                        )
                except Exception as e:
                    logging.error(f"生成报告失败: {str(e)}")
            
        except Exception as e:
            logging.error(f"处理数据集时出错: {traceback.format_exc()}")
            raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='旅游短视频多模态体裁分析工具'
    )
    parser.add_argument('--video_dir', type=str, required=True,
                      help='视频文件目录')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='输入CSV文件路径')
    parser.add_argument('--output_csv', type=str, required=True,
                      help='输出CSV文件路径')
    parser.add_argument('--config', type=str, default=None,
                      help='配置文件路径')
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 设置固定路径
        video_dir = r"xxxxxxxxxxxxxxxxxxx"  # 视频文件目录
        input_csv = r"xxxxxxxxxxxxxxxxxxx"  # 输入的CSV文件
        output_csv = r"xxxxxxxxxxxxxxxxxxx"  # 输出的CSV文件
        
        # 检查必要的文件和目录
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"输入CSV文件不存在: {input_csv}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_csv)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"输出目录: {output_dir}")
        
        # 启用防休眠模式
        prevent_sleep()
        
        # 创建分类器
        classifier = GenreClassifier()
        
        # 处理数据集
        classifier.process_dataset(
            video_dir,
            input_csv,
            output_csv
        )
        
        logging.info("\n处理完成！")
        logging.info(
            f"请查看 {os.path.dirname(output_csv)} "
            "目录下的分析报告和可视化结果。"
        )
        
    except Exception as e:
        logging.error(f"处理过程中出现错误: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # 恢复默认电源管理设置
        restore_sleep()

if __name__ == "__main__":
    main()