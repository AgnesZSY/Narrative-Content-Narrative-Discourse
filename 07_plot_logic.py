#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情节逻辑性
"""

import pandas as pd
import numpy as np
import re
import json
import sqlite3
import logging
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import warnings
import cv2
from scenedetect import detect, ContentDetector
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import os
from PIL import Image
import time
import multiprocessing as mp
import h5py
import torch.nn as nn
from tqdm import tqdm
import ctypes
from ctypes import windll
import sys
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pli_error.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载视觉模型
MODEL_PATH = "Genre/090f2-main/resnet50_coco_best_v2.0.1/resnet50_coco_best_v2.0.1.h5"

def load_h5_weights(model: nn.Module, h5_path: str) -> None:
    try:
        with h5py.File(h5_path, 'r') as f:
            for layer_name, layer in model.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    if f'{layer_name}/kernel:0' in f:
                        weights = torch.from_numpy(f[f'{layer_name}/kernel:0'][:])
                        layer.weight.data = weights.permute(3, 2, 0, 1) if len(weights.shape) == 4 else weights.t()
                    if f'{layer_name}/bias:0' in f:
                        bias = torch.from_numpy(f[f'{layer_name}/bias:0'][:])
                        layer.bias.data = bias
                    if isinstance(layer, nn.BatchNorm2d):
                        if f'{layer_name}/gamma:0' in f:
                            gamma = torch.from_numpy(f[f'{layer_name}/gamma:0'][:])
                            layer.weight.data = gamma
                        if f'{layer_name}/beta:0' in f:
                            beta = torch.from_numpy(f[f'{layer_name}/beta:0'][:])
                            layer.bias.data = beta
                        if f'{layer_name}/moving_mean:0' in f:
                            mean = torch.from_numpy(f[f'{layer_name}/moving_mean:0'][:])
                            layer.running_mean.data = mean
                        if f'{layer_name}/moving_variance:0' in f:
                            var = torch.from_numpy(f[f'{layer_name}/moving_variance:0'][:])
                            layer.running_var.data = var
    except Exception as e:
        raise Exception(f"加载.h5权重失败: {e}")

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not Path(MODEL_PATH).exists():
        logger.error(f"模型文件不存在: {MODEL_PATH}")
        model = None
    else:
        model = resnet50(weights=None)
        try:
            load_h5_weights(model, MODEL_PATH)
            logger.info("成功加载.h5权重")
        except Exception as e:
            logger.error(f"加载.h5权重失败,尝试直接加载模型: {e}")
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            except Exception as e2:
                logger.error(f"所有加载方式都失败: {e2}")
                model = None
        if model is not None:
            model = model.to(device)
            model.eval()
            logger.info(f"已加载自定义ResNet50模型 (设备: {device})")
except Exception as e:
    logger.error(f"加载视觉模型失败: {e}")
    model = None

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LANDMARK_CLASSES = {
    0: "tourist_spot",
    1: "landmark",
    2: "scenic_area",
    3: "historical_site",
    4: "museum",
    5: "park",
    6: "temple",
    7: "palace",
    8: "garden",
    9: "plaza"
}

try:
    import spacy
    try:
        nlp = spacy.load("zh_core_web_sm")
    except OSError:
        logger.warning("未找到zh_core_web_sm模型，将使用基础文本处理")
        nlp = None
except ImportError:
    logger.warning("未安装spacy，将使用基础文本处理")
    nlp = None

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    geolocator = Nominatim(user_agent="pli_analyzer")
except ImportError:
    logger.warning("未安装geopy，地理位置功能将受限")
    geolocator = None

class PLIAnalyzer:
    def __init__(self, weights=None, custom_locations=None, custom_logic_words=None):
        self.weights = weights or {
            'T1': 0.2,
            'T2': 0.4,
            'S2': 0.2,
            'L1': 0.2
        }
        self.modal_weights = {'text': 0.6, 'visual': 0.4}
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"权重总和不为1 ({total_weight}),将自动归一化")
            for key in self.weights:
                self.weights[key] /= total_weight
        self.custom_locations = custom_locations
        self.custom_logic_words = custom_logic_words
        self.setup_patterns()
        self.video_analyzer = VideoAnalyzer()
        self.setup_location_cache()

    def setup_patterns(self):
        self.time_patterns = [
            r'\d+月\d+日', r'\d+[日号]', r'上午|下午|中午|晚上|夜晚',
            r'清晨|傍晚|黄昏|深夜', r'次日|第二天|昨天|今天|明天',
            r'第\d+天', r'早上|早晨|晚间',
            r'春天|夏天|秋天|冬天', r'春季|夏季|秋季|冬季',
            r'年初|年中|年末', r'月初|月中|月末',
            r'周一|周二|周三|周四|周五|周六|周日',
            r'星期一|星期二|星期三|星期四|星期五|星期六|星期日'
        ]
        self.time_regex = re.compile('|'.join(self.time_patterns))
        base_logic_words = {
            '因果': ['因为', '由于', '于是', '因此', '所以', '导致', '造成', '引起', '结果'],
            '并列': ['首先', '其次', '然后', '同时', '另外', '最后', '接着', '随后', '此外'],
            '转折': ['但是', '然而', '不过', '却', '可惜', '虽然', '尽管', '即使', '反而']
        }
        if self.custom_logic_words:
            for category, words in self.custom_logic_words.items():
                if category in base_logic_words:
                    base_logic_words[category].extend(words)
                else:
                    base_logic_words[category] = words
        self.logic_words = base_logic_words
        base_location_dict = {
            '北京': (39.9042, 116.4074),
            '上海': (31.2304, 121.4737),
            '广州': (23.1291, 113.2644),
            '深圳': (22.5431, 114.0579),
            '杭州': (30.2741, 120.1551),
            '西安': (34.3416, 108.9398),
            '成都': (30.5728, 104.0668),
            '重庆': (29.5630, 106.5516),
            '昆明': (25.0389, 102.7183),
            '大理': (25.6066, 100.2675),
            '丽江': (26.8721, 100.2240),
            '桂林': (25.2342, 110.1993),
            '三亚': (18.2528, 109.5113),
            '厦门': (24.4798, 118.0894),
            '青岛': (36.0671, 120.3826),
            '天津': (39.3434, 117.3616),
            '南京': (32.0603, 118.7969),
            '苏州': (31.2989, 120.5853),
            '武汉': (30.5928, 114.3055),
            '长沙': (28.2282, 112.9388),
            '邯郸': (36.6253, 114.5391),
            '馆陶': (36.5447, 115.2842)
        }
        if self.custom_locations:
            base_location_dict.update(self.custom_locations)
        self.location_dict = base_location_dict

    def setup_location_cache(self):
        self.cache_file = "location_cache.db"
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                name TEXT PRIMARY KEY,
                latitude REAL,
                longitude REAL
            )
            ''')
            conn.commit()

    def extract_time_words(self, text: str) -> List[str]:
        matches = self.time_regex.findall(text)
        return matches

    def extract_locations(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        locations = []
        extraction_stats = {'spacy_found': 0, 'dict_found': 0, 'total_unique': 0}
        spacy_locations = []
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    spacy_locations.append(ent.text)
                    locations.append(ent.text)
            extraction_stats['spacy_found'] = len(spacy_locations)
        dict_locations = []
        for location in self.location_dict.keys():
            if location in text:
                dict_locations.append(location)
                locations.append(location)
        extraction_stats['dict_found'] = len(dict_locations)
        unique_locations = list(set(locations))
        extraction_stats['total_unique'] = len(unique_locations)
        return unique_locations, extraction_stats

    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        with sqlite3.connect(self.cache_file) as conn:
            cursor = conn.execute(
                "SELECT latitude, longitude FROM locations WHERE name = ?", 
                (location,)
            )
            result = cursor.fetchone()
            if result:
                return (result[0], result[1])
            if location in self.location_dict:
                coords = self.location_dict[location]
                conn.execute(
                    "INSERT OR REPLACE INTO locations VALUES (?, ?, ?)",
                    (location, coords[0], coords[1])
                )
                conn.commit()
                return coords
            if geolocator:
                try:
                    location_obj = geolocator.geocode(location + ", 中国")
                    if location_obj:
                        coords = (location_obj.latitude, location_obj.longitude)
                        conn.execute(
                            "INSERT OR REPLACE INTO locations VALUES (?, ?, ?)",
                            (location, coords[0], coords[1])
                        )
                        conn.commit()
                        return coords
                except Exception as e:
                    logger.warning(f"地理编码失败: {location}, 错误: {e}")
        return None

    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> Tuple[float, str]:
        try:
            if geolocator:
                distance = geodesic(coord1, coord2).kilometers
                return distance, "geodesic"
            else:
                raise ImportError("geopy不可用")
        except:
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111
            return distance, "euclidean"

    def extract_logic_words(self, text: str) -> Dict[str, List[str]]:
        found_logic = {'因果': [], '并列': [], '转折': []}
        for logic_type, words in self.logic_words.items():
            for word in words:
                if word in text:
                    found_logic[logic_type].append(word)
        return found_logic

    def time_metrics(self, sentences: List[str], start_times: List[float]) -> Tuple[float, float]:
        try:
            full_text = ' '.join(sentences)
            time_words = self.extract_time_words(full_text)
            total_chars = len(full_text.replace(' ', ''))
            T1 = len(time_words) / max(1, total_chars) * 100
            time_segments = []
            for i, sentence in enumerate(sentences):
                sent_time_words = self.extract_time_words(sentence)
                if sent_time_words:
                    time_segments.append(i)
            flashbacks = 0
            if len(time_segments) > 1:
                for i in range(len(time_segments) - 1):
                    if time_segments[i] > time_segments[i + 1]:
                        flashbacks += 1
            T2 = 1 - (flashbacks / max(1, len(time_segments) - 1))
            return min(1.0, T1), max(0.0, T2)
        except Exception as e:
            logger.error(f"时间指标计算错误: {e}")
            return 0.0, 0.0

    def space_metrics(self, sentences: List[str]) -> Tuple[float, float, Dict]:
        try:
            full_text = ' '.join(sentences)
            all_locations = []
            all_extraction_stats = {'spacy_found': 0, 'dict_found': 0, 'total_unique': 0}
            for sentence in sentences:
                locations, extraction_stats = self.extract_locations(sentence)
                all_locations.extend(locations)
                all_extraction_stats['spacy_found'] += extraction_stats['spacy_found']
                all_extraction_stats['dict_found'] += extraction_stats['dict_found']
            unique_locations = list(set(all_locations))
            all_extraction_stats['total_unique'] = len(unique_locations)
            total_chars = len(full_text.replace(' ', ''))
            S1 = len(all_locations) / max(1, total_chars) * 100
            coordinates = []
            coord_sources = []
            for location in unique_locations:
                coords = self.get_coordinates(location)
                if coords:
                    coordinates.append(coords)
                    if location in self.location_dict:
                        coord_sources.append("dict")
                    else:
                        coord_sources.append("geocoded")
            distance_method = "none"
            distances = []
            if len(coordinates) < 2:
                S2 = 1.0
            else:
                for i in range(len(coordinates) - 1):
                    dist, method = self.calculate_distance(coordinates[i], coordinates[i + 1])
                    distances.append(dist)
                    distance_method = method
                mean_distance = np.mean(distances)
                S2 = 0.0 if mean_distance > 50 else 1.0
            space_stats = {
                'extraction_stats': all_extraction_stats,
                'coordinate_sources': coord_sources,
                'distance_method': distance_method,
                'locations_found': unique_locations,
                'mean_distance': np.mean(distances) if len(coordinates) >= 2 else 0
            }
            return min(1.0, S1), S2, space_stats
        except Exception as e:
            logger.error(f"空间指标计算错误: {e}")
            return 0.0, 1.0, {}

    def logic_metrics(self, sentences: List[str]) -> Tuple[float, float]:
        try:
            full_text = ' '.join(sentences)
            all_logic_words = []
            logic_types_found = set()
            for sentence in sentences:
                logic_found = self.extract_logic_words(sentence)
                for logic_type, words in logic_found.items():
                    if words:
                        logic_types_found.add(logic_type)
                        all_logic_words.extend(words)
            total_chars = len(full_text.replace(' ', ''))
            L1 = len(all_logic_words) / max(1, total_chars) * 100
            L2 = len(logic_types_found) / 3
            return min(1.0, L1), min(1.0, L2)
        except Exception as e:
            logger.error(f"逻辑指标计算错误: {e}")
            return 0.0, 0.0

    def compute_pli(self, T1: float, T2: float, S1: float, S2: float, L1: float, L2: float) -> float:
        pli = (self.weights['T1'] * T1 + 
               self.weights['T2'] * T2 + 
               self.weights['S2'] * S2 + 
               self.weights['L1'] * L1)
        return min(1.0, max(0.0, pli))

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def analyze_video(self, video_file: str, content: str) -> Dict:
        try:
            if not content or not str(content).strip():
                logger.info(f"视频 {video_file} 没有文本内容")
                return self.create_empty_result(video_file, 0)
                
            content = str(content).strip()
            sentences = self.split_sentences(content)
            n_sent = len(sentences)
            
            if n_sent == 0:
                logger.warning(f"视频 {video_file} 文本分割后没有有效句子")
                return self.create_empty_result(video_file, 0)
                
            logger.info(f"视频 {video_file} 文本分析开始:")
            logger.info(f"句子数量: {n_sent}")
            logger.info(f"文本总长度: {len(content)}")
            
            # 为每个句子分配一个假设的时间戳
            # 假设每个句子平均持续2秒
            start_times = [i * 2.0 for i in range(n_sent)]
            
            # 时间维度分析
            T1, T2 = self.time_metrics(sentences, start_times)
            logger.info(f"时间维度分析结果 - T1: {T1:.3f}, T2: {T2:.3f}")
            
            # 空间维度分析
            S1, S2, space_stats = self.space_metrics(sentences)
            logger.info(f"空间维度分析结果 - S1: {S1:.3f}, S2: {S2:.3f}")
            logger.info(f"检测到的地点: {space_stats.get('locations_found', [])}")
            
            # 逻辑维度分析
            L1, L2 = self.logic_metrics(sentences)
            logger.info(f"逻辑维度分析结果 - L1: {L1:.3f}, L2: {L2:.3f}")
            
            # 检查是否所有指标都为0
            if all(x == 0 for x in [T1, T2, S1, S2, L1, L2]):
                logger.warning(f"视频 {video_file} 所有文本指标都为0，可能存在分析问题")
                logger.warning(f"原始文本: {content[:200]}...")  # 只打印前200个字符
            
            result = {
                'video_file': video_file,
                'n_sent': n_sent,
                'has_text': True,
                'T1_text': T1,
                'T2_text': T2,
                'S1_text': S1,
                'S2_text': S2,
                'L1_text': L1,
                'L2_text': L2,
                'spacy_locations': space_stats.get('extraction_stats', {}).get('spacy_found', 0),
                'dict_locations': space_stats.get('extraction_stats', {}).get('dict_found', 0),
                'locations_found': ';'.join(space_stats.get('locations_found', [])),
                'distance_method': space_stats.get('distance_method', 'none'),
                'mean_distance_km': round(space_stats.get('mean_distance', 0), 2)
            }
            
            logger.info(f"视频 {video_file} 文本分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析视频 {video_file} 时出错: {e}")
            logger.error(f"错误详情: {str(e)}")
            return self.create_empty_result(video_file, 0)

    def create_empty_result(self, video_file: str, n_sent: int) -> Dict:
        return {
            'video_file': video_file,
            'n_sent': n_sent,
            'has_text': False,
            'T1_text': 0.0,
            'T2_text': 0.0,
            'S1_text': 0.0,
            'S2_text': 0.0,
            'L1_text': 0.0,
            'L2_text': 0.0,
            'spacy_locations': 0,
            'dict_locations': 0,
            'locations_found': '',
            'distance_method': 'none',
            'mean_distance_km': 0
        }

    def compute_multimodal_pli(self, text_metrics: Dict, visual_metrics: Dict) -> Dict:
        has_text = text_metrics.get('has_text', False)
        T1t = text_metrics.get('T1_text', 0.0)
        T2t = text_metrics.get('T2_text', 0.0)
        S1t = text_metrics.get('S1_text', 0.0)
        S2t = text_metrics.get('S2_text', 0.0)
        L1t = text_metrics.get('L1_text', 0.0)
        L2t = text_metrics.get('L2_text', 0.0)
        T1v = visual_metrics.get('T1_visual', 0.0)
        T2v = visual_metrics.get('T2_visual', 0.0)
        S1v = visual_metrics.get('S1_visual', 0.0)
        S2v = visual_metrics.get('S2_visual', 0.0)
        L1v = visual_metrics.get('L1_visual', 0.0)
        L2v = visual_metrics.get('L2_visual', 0.0)
        if not has_text:
            text_weight = 0.0
            visual_weight = 1.0
        else:
            text_weight = self.modal_weights['text']
            visual_weight = self.modal_weights['visual']
        time_dimension = (
            text_weight * (self.weights['T1'] * T1t + self.weights['T2'] * T2t) +
            visual_weight * (self.weights['T1'] * T1v + self.weights['T2'] * T2v)
        )
        space_dimension = (
            text_weight * S2t +
            visual_weight * S2v
        )
        logic_dimension = (
            text_weight * L1t +
            visual_weight * L1v
        )
        text_pli = (
            self.weights['T1'] * T1t +
            self.weights['T2'] * T2t +
            self.weights['S2'] * S2t +
            self.weights['L1'] * L1t
        ) if has_text else 0.0
        visual_pli = (
            self.weights['T1'] * T1v +
            self.weights['T2'] * T2v +
            self.weights['S2'] * S2v +
            self.weights['L1'] * L1v
        )
        fused_pli = (
            text_weight * text_pli +
            visual_weight * visual_pli
        )
        return {
            'time_dimension': time_dimension,
            'space_dimension': space_dimension,
            'logic_dimension': logic_dimension,
            'PLI_text': text_pli,
            'PLI_visual': visual_pli,
            'PLI_fused': fused_pli
        }

    def analyze_video_content(self, video_path: str, text_content: Optional[str] = None) -> Dict:
        results = {}
        visual_metrics = self.video_analyzer.compute_visual_metrics(video_path)
        results.update(visual_metrics)
        if text_content:
            text_metrics = self.analyze_video(video_path, text_content)
            results.update(text_metrics)
        else:
            text_metrics = self.create_empty_result(video_path, 0)
            results.update(text_metrics)
        multimodal_metrics = self.compute_multimodal_pli(text_metrics, visual_metrics)
        results.update(multimodal_metrics)
        return results

    def process_video(self, video_path: str, text_content: Optional[str], video_name: str) -> Dict:
        try:
            logger.info(f"开始处理视频: {video_name}")
            
            # 首先进行视觉分析
            visual_metrics = self.video_analyzer.compute_visual_metrics(video_path)
            result = visual_metrics.copy()
            
            # 文本分析
            if text_content and str(text_content).strip():
                logger.info(f"视频 {video_name} 开始文本分析, 文本长度: {len(text_content)}")
                text_metrics = self.analyze_video(video_path, text_content)
                
                # 检查文本分析结果
                if text_metrics['has_text']:
                    logger.info(f"视频 {video_name} 文本分析结果:")
                    logger.info(f"T1_text: {text_metrics['T1_text']:.3f}")
                    logger.info(f"T2_text: {text_metrics['T2_text']:.3f}")
                    logger.info(f"S1_text: {text_metrics['S1_text']:.3f}")
                    logger.info(f"S2_text: {text_metrics['S2_text']:.3f}")
                    logger.info(f"L1_text: {text_metrics['L1_text']:.3f}")
                    logger.info(f"L2_text: {text_metrics['L2_text']:.3f}")
                
                result.update(text_metrics)
            else:
                logger.info(f"视频 {video_name} 没有文本内容")
                result.update(self.create_empty_result(video_path, 0))
            
            # 计算多模态PLI
            multimodal_metrics = self.compute_multimodal_pli(
                {k: result[k] for k in result if k.endswith('_text') or k == 'has_text'},
                {k: result[k] for k in result if k.endswith('_visual')}
            )
            result.update(multimodal_metrics)
            
            # 添加视频ID
            result['video_id'] = video_name
            
            # 验证结果
            if result['has_text'] and all(result[k] == 0 for k in ['T1_text', 'T2_text', 'S1_text', 'S2_text', 'L1_text', 'L2_text']):
                logger.warning(f"视频 {video_name} 检测到文本但所有文本指标为0，可能存在分析问题")
            
            return result
            
        except Exception as e:
            logger.error(f"处理视频 {video_name} 失败: {e}")
            return None

    def process_dataset(self, video_dir: str, text_file: str, output_file: str):
        try:
            logger.info(f"读取数据文件: {text_file}")
            df = pd.read_csv(text_file, encoding='utf-8')
            if 'video_id' not in df.columns or 'text' not in df.columns:
                raise ValueError("数据文件缺少必要的列(video_id, text)")
            
            # 打印数据文件中的文本信息
            logger.info("数据文件中的文本信息:")
            for idx, row in df.iterrows():
                if pd.notna(row['text']) and str(row['text']).strip():
                    logger.info(f"视频ID: {row['video_id']}, 文本长度: {len(str(row['text']))}")
            
            video_dir = Path(video_dir)
            process_args = []
            
            # 遍历视频文件夹中的所有.mp4文件
            for video_file in video_dir.glob('*.mp4'):
                video_name = video_file.stem  # 获取不带扩展名的文件名
                
                # 在文本数据中查找对应的文本
                text_row = df[df['video_id'].astype(str) == str(video_name)]
                
                if text_row.empty:
                    logger.warning(f"视频 {video_name} 在文本数据中未找到对应记录")
                    text_content = None
                else:
                    text_content = text_row['text'].iloc[0]
                    if pd.isna(text_content) or not str(text_content).strip():
                        logger.warning(f"视频 {video_name} 的文本内容为空")
                        text_content = None
                    else:
                        logger.info(f"找到视频 {video_name} 的文本，长度: {len(str(text_content))}")
                        text_content = str(text_content).strip()
                
                process_args.append((str(video_file), text_content, video_name))
            
            if not process_args:
                logger.warning("没有找到任何视频文件")
                return
                
            logger.info(f"总共找到 {len(process_args)} 个视频文件待处理")
            logger.info(f"其中 {sum(1 for _, text, _ in process_args if text)} 个视频有对应文本")
            
            n_cores = max(1, mp.cpu_count() - 1)
            logger.info(f"使用 {n_cores} 个进程进行并行处理")
            analyzer_for_pool = PLIAnalyzer(
                weights=self.weights,
                custom_locations=self.custom_locations,
                custom_logic_words=self.custom_logic_words
            )
            
            with mp.Pool(n_cores) as pool:
                results = pool.starmap(analyzer_for_pool.process_video, process_args)
                
            results = [r for r in results if r is not None]
            result_df = pd.DataFrame(results)
                
            # 检查结果
            logger.info("\n分析结果统计:")
            logger.info(f"总处理视频数: {len(result_df)}")
            logger.info(f"检测到文本的视频数: {result_df['has_text'].sum()}")
            logger.info(f"文本得分为0的视频数: {len(result_df[result_df['PLI_text'] == 0])}")
                
            # 输出特定视频的分析结果
            for video_id in ['9', '15', '18']:
                video_result = result_df[result_df['video_id'] == video_id]
                if not video_result.empty:
                    logger.info(f"\n视频 {video_id} 的分析结果:")
                    logger.info(f"has_text: {video_result['has_text'].iloc[0]}")
                    logger.info(f"PLI_text: {video_result['PLI_text'].iloc[0]}")
                    logger.info(f"T1_text: {video_result['T1_text'].iloc[0]}")
                    logger.info(f"T2_text: {video_result['T2_text'].iloc[0]}")
            
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存到: {output_file}")
            self.generate_analysis_report(result_df)
            
        except Exception as e:
            logger.error(f"处理数据集时出错: {e}")
            raise

    def generate_analysis_report(self, df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("PLI多模态分析报告")
        print("=" * 80)
        total_videos = len(df)
        videos_with_text = df['has_text'].sum()
        print(f"  总视频数量: {total_videos}")
        print(f"  有文本视频: {videos_with_text} ({videos_with_text/total_videos*100:.1f}%)")
        print(f"  无文本视频: {total_videos-videos_with_text} ({(total_videos-videos_with_text)/total_videos*100:.1f}%)")
        print("\nPLI得分统计:")
        text_videos = df[df['has_text']]
        if len(text_videos) > 0:
            text_pli = text_videos['PLI_text']
            visual_pli = text_videos['PLI_visual']
            fused_pli = text_videos['PLI_fused']
            print("  文本模态PLI:")
            print(f"    均值: {text_pli.mean():.3f}")
            print(f"    标准差: {text_pli.std():.3f}")
            print(f"    最大值: {text_pli.max():.3f}")
            print(f"    最小值: {text_pli.min():.3f}")
        else:
            print("  文本模态PLI: 无有文本视频")
            text_pli = pd.Series(dtype=float)
            visual_pli = pd.Series(dtype=float)
            fused_pli = pd.Series(dtype=float)
        print("  视觉模态PLI:")
        print(f"    均值: {df['PLI_visual'].mean():.3f}")
        print(f"    标准差: {df['PLI_visual'].std():.3f}")
        print(f"    最大值: {df['PLI_visual'].max():.3f}")
        print(f"    最小值: {df['PLI_visual'].min():.3f}")
        print("  多模态融合PLI:")
        print(f"    均值: {df['PLI_fused'].mean():.3f}")
        print(f"    标准差: {df['PLI_fused'].std():.3f}")
        print(f"    最大值: {df['PLI_fused'].max():.3f}")
        print(f"    最小值: {df['PLI_fused'].min():.3f}")

        print("\n维度分析:")
        dimensions = ['time_dimension', 'space_dimension', 'logic_dimension']
        for dim in dimensions:
            print(f"  {dim}:")
            print(f"    均值: {df[dim].mean():.3f}")
            print(f"    标准差: {df[dim].std():.3f}")

        print("\n模态对比分析:")
        if len(text_videos) > 0:
            text_better = (text_pli.values > visual_pli.values).sum()
            visual_better = (visual_pli.values > text_pli.values).sum()
            print(f"  文本模态更优: {text_better} 个视频 ({(text_better/len(text_pli)*100 if len(text_pli)>0 else 0):.1f}%)")
            print(f"  视觉模态更优: {visual_better} 个视频 ({(visual_better/len(text_pli)*100 if len(text_pli)>0 else 0):.1f}%)")
        else:
            print("  没有包含文本的视频，无法进行模态对比分析")

        # 其它统计原逻辑不变，可以自行添加...

class VideoAnalyzer:
    def __init__(self):
        self.scene_detector = ContentDetector()
        self.min_frames = 10
        self.max_frames = 100

    def get_adaptive_sample_rate(self, video_path: str) -> int:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return 1
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps else 0
            cap.release()
            if duration <= 30:
                return 2
            elif duration <= 60:
                return 1
            else:
                return max(1, int(fps * self.max_frames / total_frames)) if total_frames > 0 else 1
        except Exception as e:
            logger.error(f"计算采样率失败: {e}")
            return 1

    def extract_frames(self, video_path: str, sample_rate: Optional[int] = None) -> List[np.ndarray]:
        frames = []
        try:
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在: {video_path}")
                return []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"OpenCV无法打开视频文件: {video_path}")
                return []
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                logger.error(f"视频帧率异常: {video_path}")
                cap.release()
                return []
            if sample_rate is None:
                sample_rate = self.get_adaptive_sample_rate(video_path)
            frame_interval = int(fps / sample_rate)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    if len(frames) >= self.max_frames:
                        break
                frame_count += 1
            cap.release()
            if len(frames) < self.min_frames:
                logger.warning(f"帧数过少 ({len(frames)}), 可能影响分析质量, 视频: {video_path}")
        except Exception as e:
            logger.error(f"提取视频帧失败: {e}")
        return frames

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        检测场景切换，确保至少返回一个场景
        Returns:
            scenes: [(start_time, end_time), ...]
        """
        try:
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在: {video_path}")
                return []
                
            # 先获取视频基本信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            if duration <= 0:
                logger.error(f"视频时长异常: {video_path}")
                return []
                
            # 尝试场景检测
            scene_list = detect(video_path, ContentDetector(
                threshold=27.0,  # 降低阈值，使检测更敏感
                min_scene_len=int(fps) if fps > 0 else 15  # 最小场景长度为1秒
            ))
            
            scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
            
            # 如果没检测到场景，将整个视频作为一个场景
            if not scenes:
                logger.warning(f"未检测到场景切换，将整个视频作为单场景: {video_path}")
                scenes = [(0.0, duration)]
                
            return scenes
            
        except Exception as e:
            logger.error(f"场景检测失败: {e}")
            # 如果出错，尝试获取视频时长并作为单场景
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    if duration > 0:
                        return [(0.0, duration)]
            except:
                pass
            return []

    def detect_landmarks(self, frame: np.ndarray) -> List[str]:
        if model is None:
            return []
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_k = torch.topk(probabilities, 5)
            landmarks = []
            for idx in top_k.indices:
                idx = idx.item()
                if idx in LANDMARK_CLASSES:
                    landmarks.append(LANDMARK_CLASSES[idx])
            return landmarks
        except Exception as e:
            logger.error(f"地标检测失败: {e}")
            return []

    def analyze_visual_flow(self, scenes: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not scenes:
            return 0.0, 0.0
        try:
            durations = [end - start for start, end in scenes]
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            rhythm = 1.0 - min(1.0, std_duration / (mean_duration + 1e-6))
            total_time = scenes[-1][1] - scenes[0][0]
            switch_rate = len(scenes) / total_time if total_time > 0 else 0
            coherence = 1.0 - min(1.0, switch_rate / 2.0)
            return rhythm, coherence
        except Exception as e:
            logger.error(f"视觉流转分析失败: {e}")
            return 0.0, 0.0

    def compute_visual_metrics(self, video_path: str) -> Dict[str, float]:
        try:
            scenes = self.detect_scenes(video_path)
            if not scenes:
                logger.warning(f"场景检测失败或无场景，全部视觉特征归零, 视频: {video_path}")
                return self.create_empty_visual_metrics()
            frames = self.extract_frames(video_path)
            if not frames:
                logger.warning(f"帧提取失败，全部视觉特征归零, 视频: {video_path}")
                return self.create_empty_visual_metrics()
            landmarks = []
            for frame in frames:
                landmarks.extend(self.detect_landmarks(frame))
            unique_landmarks = list(set(landmarks))
            T1v = len(scenes) / (scenes[-1][1] - scenes[0][0]) if (scenes and scenes[-1][1] - scenes[0][0]) else 0.0
            T2v = self.compute_temporal_coherence(scenes)
            S1v = len(unique_landmarks) / len(frames) if len(frames) else 0.0
            S2v = self.compute_spatial_coherence(landmarks)
            L1v, L2v = self.analyze_visual_flow(scenes)
            T1v = min(1.0, T1v)
            S1v = min(1.0, S1v)
            return {
                'T1_visual': T1v,
                'T2_visual': T2v,
                'S1_visual': S1v,
                'S2_visual': S2v,
                'L1_visual': L1v,
                'L2_visual': L2v,
                'detected_landmarks': ';'.join(unique_landmarks),
                'scene_count': len(scenes),
                'avg_scene_duration': np.mean([end - start for start, end in scenes]) if scenes else 0.0
            }
        except Exception as e:
            logger.error(f"视觉分析失败: {e}")
            return self.create_empty_visual_metrics()

    def compute_temporal_coherence(self, scenes: List[Tuple[float, float]]) -> float:
        if not scenes:
            return 0.0
        try:
            intervals = []
            for i in range(len(scenes) - 1):
                interval = scenes[i + 1][0] - scenes[i][1]
                intervals.append(interval)
            if not intervals:
                return 1.0
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            coherence = 1.0 - min(1.0, std_interval / (mean_interval + 1e-6))
            return coherence
        except Exception as e:
            logger.error(f"计算时序连贯性失败: {e}")
            return 0.0

    def compute_spatial_coherence(self, landmarks: List[str]) -> float:
        if not landmarks:
            return 0.0
        try:
            changes = 0
            for i in range(len(landmarks) - 1):
                if landmarks[i] != landmarks[i + 1]:
                    changes += 1
            if len(landmarks) <= 1:
                return 1.0
            coherence = 1.0 - min(1.0, changes / (len(landmarks) - 1))
            return coherence
        except Exception as e:
            logger.error(f"计算空间连贯性失败: {e}")
            return 0.0

    def create_empty_visual_metrics(self) -> Dict[str, float]:
        return {
            'T1_visual': 0.0,
            'T2_visual': 0.0,
            'S1_visual': 0.0,
            'S2_visual': 0.0,
            'L1_visual': 0.0,
            'L2_visual': 0.0,
            'detected_landmarks': '',
            'scene_count': 0,
            'avg_scene_duration': 0.0
        }

class SystemManager:
    def __init__(self):
        self.ES_CONTINUOUS = 0x80000000
        self.ES_SYSTEM_REQUIRED = 0x00000001
        self.previous_state = None

    def prevent_sleep(self):
        """启用防休眠模式"""
        try:
            self.previous_state = ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )
            logger.info("已启用防休眠模式")
        except Exception as e:
            logger.error(f"启用防休眠模式失败: {e}")

    def restore_sleep(self):
        """恢复系统默认休眠设置"""
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            logger.info("已恢复系统默认休眠设置")
        except Exception as e:
            logger.error(f"恢复系统休眠设置失败: {e}")

class BatchProcessor:
    def __init__(self, batch_size=5):
        self.batch_size = batch_size
        self.current_batch = 0
        self.system_manager = SystemManager()

    def natural_sort_key(self, s):
        """用于自然排序的键函数"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', str(s))]

    def load_existing_results(self, output_file):
        """加载已存在的结果"""
        try:
            if os.path.exists(output_file):
                return pd.read_csv(output_file, encoding='utf-8-sig')
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"加载已有结果失败: {e}")
            return pd.DataFrame()

    def save_batch_results(self, results_df, output_file):
        """保存批次结果"""
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"已保存批次结果到: {output_file}")
        except Exception as e:
            logger.error(f"保存批次结果失败: {e}")

    def get_video_number(self, video_path):
        """从视频路径中提取数字"""
        try:
            return int(Path(video_path).stem)
        except:
            return float('inf')  # 对于非数字命名的视频，放到最后处理

    def process_videos_in_batches(self, video_dir, text_file, output_file, analyzer):
        """分批处理视频"""
        try:
            # 启用防休眠
            self.system_manager.prevent_sleep()

            # 加载已有结果
            existing_results = self.load_existing_results(output_file)
            processed_videos = set(existing_results['video_id'].astype(str) if not existing_results.empty else [])

            # 读取文本数据
            text_df = pd.read_csv(text_file, encoding='utf-8')

            # 获取所有视频文件并按数字顺序排序
            video_files = [f for f in Path(video_dir).glob('*.mp4')]
            video_files.sort(key=self.get_video_number)

            # 过滤掉已处理的视频
            video_files = [v for v in video_files if v.stem not in processed_videos]

            if not video_files:
                logger.info("没有新的视频需要处理")
                return

            # 创建总进度条
            total_pbar = tqdm(total=len(video_files), desc="总体进度")

            # 按批次处理视频
            for i in range(0, len(video_files), self.batch_size):
                batch_files = video_files[i:i + self.batch_size]
                batch_results = []

                # 处理当前批次
                for video_file in batch_files:
                    video_name = video_file.stem
                    text_content = None
                    
                    # 查找对应的文本
                    text_row = text_df[text_df['video_id'].astype(str) == str(video_name)]
                    if not text_row.empty and pd.notna(text_row['text'].iloc[0]):
                        text_content = str(text_row['text'].iloc[0]).strip()

                    # 处理视频
                    result = analyzer.process_video(str(video_file), text_content, video_name)
                    if result:
                        batch_results.append(result)
                    
                    total_pbar.update(1)

                # 合并并保存结果
                if batch_results:
                    batch_df = pd.DataFrame(batch_results)
                    if existing_results.empty:
                        existing_results = batch_df
                    else:
                        existing_results = pd.concat([existing_results, batch_df], ignore_index=True)
                    
                    # 按视频ID排序
                    existing_results['video_id'] = existing_results['video_id'].astype(str)
                    existing_results.sort_values('video_id', key=lambda x: x.map(self.natural_sort_key), inplace=True)
                    
                    # 保存结果
                    self.save_batch_results(existing_results, output_file)
                    
                    logger.info(f"完成第 {i//self.batch_size + 1} 批处理 ({len(batch_files)} 个视频)")

            total_pbar.close()

        except Exception as e:
            logger.error(f"批处理过程中出错: {e}")
            raise

        finally:
            # 恢复系统休眠设置
            self.system_manager.restore_sleep()

def check_advanced_features():
    features = {'spacy': False, 'geopy': False, 'zh_model': False}
    try:
        import spacy
        features['spacy'] = True
        try:
            nlp_test = spacy.load("zh_core_web_sm")
            features['zh_model'] = True
        except OSError:
            pass
    except ImportError:
        pass
    try:
        from geopy.geocoders import Nominatim
        features['geopy'] = True
    except ImportError:
        pass
    return features

def main():
    features = check_advanced_features()
    logger.info("=== PLI多模态分析系统启动 ===")
    logger.info("功能可用性检查:")
    logger.info(f"  ✓ 基础文本处理: 可用")
    logger.info(f"  {'✓' if features['spacy'] else '✗'} spaCy库: {'可用' if features['spacy'] else '不可用'}")
    logger.info(f"  {'✓' if features['zh_model'] else '✗'} 中文NER模型: {'可用' if features['zh_model'] else '不可用'}")
    logger.info(f"  {'✓' if features['geopy'] else '✗'} 地理编码服务: {'可用' if features['geopy'] else '不可用'}")
    
    if not features['zh_model']:
        logger.warning("建议安装: python -m spacy download zh_core_web_sm")
    if not features['geopy']:
        logger.warning("建议安装: pip install geopy")
    if model is None:
        logger.warning("视觉分析模型未加载,地标识别功能将受限")
        
    config_file = "config.json"
    config = {}
    try:
        if Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info("已加载配置文件")
    except Exception as e:
        logger.warning(f"加载配置文件失败: {e}")
        
    weights = config.get('weights')
    custom_locations = config.get('custom_locations')
    custom_logic_words = config.get('custom_logic_words')
    
    if weights:
        logger.info(f"使用自定义权重: {weights}")
    if custom_locations:
        logger.info(f"加载自定义地名: {len(custom_locations)}个")
    if custom_logic_words:
        logger.info(f"加载自定义逻辑词: {sum(len(words) for words in custom_logic_words.values())}个")
        
    video_dir = r"xxxxxxxxxxxxxxxxxxx"
    text_file = r"xxxxxxxxxxxxxxxxxxx"
    output_file = r"xxxxxxxxxxxxxxxxxxx"
    
    if not Path(video_dir).exists():
        logger.error(f"视频目录不存在: {video_dir}")
        return
    if not Path(text_file).exists():
        logger.error(f"文本文件不存在: {text_file}")
        return
        
    analyzer = PLIAnalyzer(
        weights=weights,
        custom_locations=custom_locations,
        custom_logic_words=custom_logic_words
    )
        
    # 使用批处理器处理视频
    batch_processor = BatchProcessor(batch_size=5)
    logger.info("开始多模态分析...")
    try:
        batch_processor.process_videos_in_batches(video_dir, text_file, output_file, analyzer)
        logger.info("=== PLI多模态分析完成 ===")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise

if __name__ == "__main__":
    print("开始情节逻辑性 (PLI) 多模态分析...")
    main()
    print("分析完成！")
