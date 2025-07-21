#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验真实性量化系统 (Authenticity-HyperReality)
用于批量分析旅游短视频的体验真实性和异真性
"""

import os
import cv2
import numpy as np
import pandas as pd
import jieba
import re
from typing import Tuple, List, Dict
from tqdm import tqdm
from collections import Counter
import concurrent.futures
import time
import json
from pathlib import Path
import ctypes
from functools import lru_cache
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('authenticity_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 设置控制台输出编码
import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 缓存配置
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# 视频处理配置
BATCH_SIZE = 5  # 每批处理的视频数量
MAX_RETRIES = 3  # 最大重试次数
TIMEOUT = 300  # 单个视频处理超时时间（秒）

class PreventSleep:
    """防止系统休眠的上下文管理器"""
    
    def __init__(self):
        self.ES_CONTINUOUS = 0x80000000
        self.ES_SYSTEM_REQUIRED = 0x00000001
        
    def __enter__(self):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )
            logging.info("已启用防休眠模式")
        except Exception as e:
            logging.warning(f"启用防休眠模式失败: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            logging.info("已恢复默认电源设置")
        except Exception as e:
            logging.warning(f"恢复默认电源设置失败: {e}")

@lru_cache(maxsize=1000)
def get_cached_result(video_file: str) -> Dict:
    """从缓存中获取结果"""
    cache_file = CACHE_DIR / f"{video_file}.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_cache(video_file: str, result: Dict):
    """保存结果到缓存"""
    cache_file = CACHE_DIR / f"{video_file}.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def natural_sort_key(s):
    """用于自然排序的键函数"""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# ====================================
# 一、关键词词典
# ====================================

info_kw = [
    # 交通 / 路径
    "地铁", "地铁口", "公交", "公交站", "巴士", "大巴", "轻轨", "高铁", "火车", "动车", "机场大巴",
    "自驾", "租车", "网约车", "拼车", "滴滴", "出租车", "共享单车", "骑行", "步行", "步行道", "高速", "省道", "路况",
    "换乘", "站口", "出入口", "口岸", "登船", "码头", "航班", "班次", "时刻表", "车次", "车程", "公里", "公里数", "分钟", "小时",

    # 门票 / 费用
    "门票", "票价", "票务", "联票", "套票", "折扣票", "学生票", "老年票", "免票", "半价", "早鸟票", "优惠券", "团购券",
    "价格", "费用", "收费", "收费标准", "人均", "人均消费", "预算", "花费", "成本", "押金", "停车费",

    # 营业 / 时间
    "营业时间", "开放时间", "营业到", "关门", "开园", "开放日", "营业日", "公休日", "检票", "安检", "预约", "预约成功", "实名预约",
    "排队", "等位", "排号", "取号", "候车", "候船", "排队时间", "高峰期", "淡季", "旺季", "最佳季节", "最佳时间", "日出时间",

    # 地点 / 定位
    "地址", "定位", "坐标", "导航", "地图", "高德地图", "百度地图", "谷歌地图", "入口", "正门", "侧门", "出口", "洗手间",
    "卫生间", "厕所", "无障碍", "母婴室", "充电桩", "寄存", "行李寄存", "存包", "咨询处", "服务台", "急救站",

    # 攻略 / 行程
    "路线", "线路", "路书", "行程", "行程单", "行程表", "行程安排", "打卡路线", "最佳路线", "避坑", "避雷", "攻略", "tips", "秘籍",
    "必看", "必打卡", "必去", "top", "top10", "注意事项", "穿衣建议", "防晒", "雨具", "装备", "器材", "拍照机位", "机位", "最佳视角",

    # 住宿 / 餐饮 / 服务
    "酒店", "民宿", "客栈", "青旅", "房型", "入住", "退房", "早餐", "自助餐", "餐厅", "美食", "菜单", "排号", "开桌", "网红店", "人均客单",
    "wifi", "信号", "网络", "插座", "充电", "暖气", "空调", "安保", "卫生", "保险", "应急电话", "医院", "药店", "天气", "温度", "气温",
    "紫外线", "雨势", "风速", "救援", "旅行社", "跟团", "自由行"
]

imagery_kw = [
    # 诗意 / 梦幻
    "诗意", "诗画", "诗情", "画意", "水墨", "国风", "古风", "汉服风", "唐风", "宋韵", "意境", "天青色", "烟雨",
    "童话", "童话镇", "童话世界", "幻境", "幻梦", "梦境", "云梦", "乌托邦", "世外桃源", "伊甸园",
    
    # 仙 / 云 / 雾
    "仙境", "仙气", "仙雾", "缥缈", "云端", "云海", "云瀑", "云杉", "雾凇", "雾气", "仙雾缭绕", "天宫", "天界",
    
    # 赛博 / 科幻
    "赛博", "赛博朋克", "赛博风", "霓虹", "未来感", "科幻感", "蒸汽波", "蒸汽朋克", "复古未来", "元宇宙", "虚拟", "次元", "二次元",
    
    # 光影 / 滤镜
    "光影", "光影流转", "剪影", "逆光", "余晖", "霞光", "晚霞", "朝霞", "蓝调", "金色时刻", "微光", "星河", "星空", "银河", "流星雨",
    "光轨", "长曝光", "光晕", "炫光", "霓虹光", "闪耀", "绚烂", "斑斓", "色彩炸裂",
    
    # 景观 / 自然奇观
    "极光", "极昼", "雪国", "雪境", "冰川", "冰封", "水晶世界", "琉璃", "翡翠湖", "牛奶海", "天镜", "天空之镜", "镜面", "倒影",
    "星野", "旷野", "荒漠", "沙漠", "沙丘", "绿洲", "火山", "熔岩", "月球", "火星", "星际",
    
    # 花 / 植物 / 色块
    "花海", "粉黛", "薰衣草", "油菜花", "樱花雨", "樱花", "枫叶林", "红叶", "银杏大道", "向日葵", "雏菊", "麦田", "竹海", "迷雾森林", "精灵森林",
    "鲜花", "花朵", "花卉", "花园", "绿化带", "春色", "春城", "春风", "春天", "春日",
    
    # 艺术 / 滤镜
    "油画感", "油画滤镜", "油画世界", "巴洛克", "洛可可", "印象派", "梵高色调", "莫奈色调", "莫奈", "笔触",
    "胶片感", "胶片颗粒", "film感", "复古胶片", "港风", "老电影", "波普", "蒸汽波色彩",
    
    # 情感描述
    "美", "美丽", "美得", "浪漫", "梦幻", "陶醉", "喜悦", "澎湃", "生命力", "倩影", "波光",
    
    # 其他象征
    "魔幻", "魔法", "精灵", "精灵谷", "神秘遗迹", "古堡", "城堡", "空中花园", "空中走廊", "水上之城",
    "海市蜃楼", "天空之城", "次元壁", "穿越", "虫洞", "平行世界", "镜像空间", "异世界", "异次元"
]

# 转换为集合以提高查找效率
info_kw_set = set(info_kw)
imagery_kw_set = set(imagery_kw)

# ====================================
# 二、核心函数
# ====================================

def extract_frames(video_path: str, sample_interval: int = 2) -> List[np.ndarray]:
    """
    从视频中提取帧（每2秒抽取1帧）
    
    Args:
        video_path: 视频文件路径
        sample_interval: 采样间隔（秒）
    
    Returns:
        帧列表
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        print(f"请检查文件格式是否支持，当前路径: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频信息 - FPS: {fps:.2f}, 总帧数: {total_frames}, 时长: {duration:.2f}秒")
    
    if fps <= 0:
        print("视频FPS异常，使用默认采样策略")
        frame_interval = 30  # 默认每30帧取一帧
    else:
        frame_interval = int(fps * sample_interval)
    
    frame_count = 0
    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"成功提取 {extracted_count} 帧图像")
    return frames

def hsv_metrics(frames: List[np.ndarray]) -> Tuple[float, float]:
    """
    计算帧的HSV统计指标
    
    Args:
        frames: 视频帧列表
    
    Returns:
        (mean_S, std_H): 平均饱和度和色调方差
    """
    if not frames:
        return 0.0, 0.0
    
    s_values = []
    h_values = []
    
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 计算非零像素的统计值（避免黑色背景影响）
        mask = v > 10  # 过滤过暗像素
        if np.any(mask):
            s_values.extend(s[mask].flatten())
            h_values.extend(h[mask].flatten())
    
    if not s_values:
        return 0.0, 0.0
    
    mean_s = np.mean(s_values)
    std_h = np.std(h_values)
    
    return mean_s, std_h

def semantic_metrics(text: str) -> Tuple[int, int, int]:
    """
    计算语义指标（改进版）
    
    Args:
        text: 字幕文本
    
    Returns:
        (total_tokens, info_kw_cnt, imagery_kw_cnt): 总词数、信息关键词数、意象关键词数
    """
    if not text or len(text.strip()) == 0:
        logging.debug("文本为空，跳过语义分析")
        return 0, 0, 0
    
    # 清理文本 - 保留更多字符，包括标点符号用于句子切分
    text_clean = re.sub(r'[^\u4e00-\u9fff\w\s，。！？、；：""''（）]', '', text)
    
    # 分词 - 使用精确模式
    tokens = list(jieba.cut(text_clean, cut_all=False))
    tokens = [token.strip() for token in tokens if len(token.strip()) > 1]  # 过滤单字符
    
    total_tokens = len(tokens)
    if total_tokens == 0:
        logging.debug("分词结果为空")
        return 0, 0, 0
    
    logging.debug(f"分词结果: {tokens[:20]}...")  # 显示前20个词用于调试
    
    # 转小写用于匹配
    text_lower = text_clean.lower()
    
    info_kw_cnt = 0
    imagery_kw_cnt = 0
    
    # 改进的关键词匹配 - 使用更宽松的匹配策略
    info_matches = []
    imagery_matches = []
    
    # 信息关键词匹配
    for kw in sorted(info_kw_set, key=len, reverse=True):
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            count = text_lower.count(kw_lower)
            info_kw_cnt += count
            if count > 0:
                info_matches.append(f"{kw}({count})")
    
    # 意象关键词匹配
    for kw in sorted(imagery_kw_set, key=len, reverse=True):
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            count = text_lower.count(kw_lower)
            imagery_kw_cnt += count
            if count > 0:
                imagery_matches.append(f"{kw}({count})")
    
    logging.debug(f"匹配到的信息关键词: {info_matches}")
    logging.debug(f"匹配到的意象关键词: {imagery_matches}")
    logging.info(f"语义分析结果 - 总词数: {total_tokens}, 信息词数: {info_kw_cnt}, 意象词数: {imagery_kw_cnt}")
    
    return total_tokens, info_kw_cnt, imagery_kw_cnt

def compute_scores(mean_s: float, std_h: float, 
                  total_tokens: int, info_kw_cnt: int, imagery_kw_cnt: int) -> Tuple[float, float, float, float]:
    """
    计算A和H分数（改进版）
    
    Args:
        mean_s: 平均饱和度
        std_h: 色调方差
        total_tokens: 总词数
        info_kw_cnt: 信息关键词数
        imagery_kw_cnt: 意象关键词数
    
    Returns:
        (A_visual, H_visual, A_sem, H_sem): 视觉和语义的A、H分数
    """
    # 视觉真实性和异真性 - 改进公式
    if mean_s == 0 and std_h == 0:
        # 如果无法获取视觉数据，返回中性值
        A_visual = 0.5
        H_visual = 0.5
        logging.warning("警告: 无法获取视频视觉数据，使用默认值")
    else:
        # 标准化处理
        norm_s = min(mean_s / 255, 1.0)  # 饱和度归一化
        norm_h = min(std_h / 180, 1.0)   # 色调方差归一化 (HSV中H范围是0-179)
        
        # 真实性：低饱和度 + 高色调方差（自然场景色彩丰富但不过饱和）
        A_visual = 0.4 * (1 - norm_s) + 0.6 * norm_h
        A_visual = max(0, min(1, A_visual))
        H_visual = 1 - A_visual
        
        logging.debug(f"视觉分析 - 平均饱和度: {mean_s:.2f}, 色调方差: {std_h:.2f}")
        logging.debug(f"视觉分数 - A_visual: {A_visual:.4f}, H_visual: {H_visual:.4f}")
    
    # 语义真实性和异真性 - 调整密度计算
    if total_tokens == 0:
        A_sem = 0.0
        H_sem = 0.0
        logging.debug("无文本数据，语义分数设为0")
    else:
        # 计算词密度
        info_density = info_kw_cnt / total_tokens
        imagery_density = imagery_kw_cnt / total_tokens
        
        # 调整放大系数 - 降低阈值，使分数更加敏感
        A_sem = min(info_density * 5, 1.0)     # 降低系数，使分数更容易达到有效值
        H_sem = min(imagery_density * 4, 1.0)  # 降低系数，使分数更容易达到有效值
        
        logging.debug(f"语义分析 - 信息密度: {info_density:.4f}, 意象密度: {imagery_density:.4f}")
        logging.debug(f"语义分数 - A_sem: {A_sem:.4f}, H_sem: {H_sem:.4f}")
    
    return A_visual, H_visual, A_sem, H_sem

def classify_experience(A_score: float, H_score: float) -> str:
    """
    根据A和H分数进行分类
    
    Args:
        A_score: 综合真实性分数
        H_score: 综合异真性分数
    
    Returns:
        分类结果: "Authentic", "Hyper-real", "Mixed"
    """
    if A_score >= 0.6 and H_score < 0.4:
        return "Authentic"
    elif H_score >= 0.6 and A_score < 0.4:
        return "Hyper-real"
    else:
        return "Mixed"

def count_sentences(text: str) -> int:
    """
    统计句子数量
    """
    if not text:
        return 0
    sentences = re.split(r'[。！？.!?]', text)
    return len([s for s in sentences if s.strip()])

def process_single_video(video_file: str, video_dir: str, subtitle_content: str) -> dict:
    """
    处理单个视频文件
    
    Args:
        video_file: 视频文件名
        video_dir: 视频文件目录
        subtitle_content: 字幕内容
    
    Returns:
        包含所有指标的字典
    """
    video_path = os.path.join(video_dir, video_file)
    
    # 首先进行视觉分析
    try:
        frames = extract_frames(video_path)
        if not frames:
            print(f"警告: 无法从视频提取帧: {video_file}")
            return {
                'video_file': video_file,
                'A_visual': 0.0,
                'H_visual': 1.0,
                'A_sem': 0.0,
                'H_sem': 1.0,
                'A_score': 0.0,
                'H_score': 1.0,
                'class': 'Error'
            }
        
        mean_s, std_h = hsv_metrics(frames)
        A_visual, H_visual, _, _ = compute_scores(mean_s, std_h, 0, 0, 0)  # 先只计算视觉分数
    except Exception as e:
        print(f"视频分析出错 {video_file}: {str(e)}")
        return {
            'video_file': video_file,
            'A_visual': 0.0,
            'H_visual': 1.0,
            'A_sem': 0.0,
            'H_sem': 1.0,
            'A_score': 0.0,
            'H_score': 1.0,
            'class': 'Error'
        }
    
    # 检查字幕内容
    if not subtitle_content or subtitle_content.strip() == "":
        print(f"提示: {video_file} 没有文案")
        # 只使用视觉分析结果
        A_score = A_visual
        H_score = H_visual
        return {
            'video_file': video_file,
            'A_visual': round(A_visual, 4),
            'H_visual': round(H_visual, 4),
            'A_sem': 0.0,
            'H_sem': 0.0,
            'A_score': round(A_score, 4),
            'H_score': round(H_score, 4),
            'class': classify_experience(A_score, H_score)
        }
    
    # 如果有文案，进行语义分析
    total_tokens, info_kw_cnt, imagery_kw_cnt = semantic_metrics(subtitle_content)
    
    # 计算完整分数
    A_visual, H_visual, A_sem, H_sem = compute_scores(mean_s, std_h, total_tokens, info_kw_cnt, imagery_kw_cnt)
    
    # 综合分数 - 调整权重平衡
    if total_tokens > 0:
        # 有文本时，视觉和语义各占50%
        A_score = 0.5 * A_visual + 0.5 * A_sem
        H_score = 0.5 * H_visual + 0.5 * H_sem
    else:
        # 无文本时，完全依赖视觉分析
        A_score = A_visual
        H_score = H_visual
    
    return {
        'video_file': video_file,
        'A_visual': round(A_visual, 4),
        'H_visual': round(H_visual, 4),
        'A_sem': round(A_sem, 4),
        'H_sem': round(H_sem, 4),
        'A_score': round(A_score, 4),
        'H_score': round(H_score, 4),
        'class': classify_experience(A_score, H_score)
    }

def main():
    """
    主函数：批量处理视频文件
    """
    # 配置路径
    video_dir = r"E:\videos\2"
    subtitle_file = r"D:\download\cursor\paper 3\Plot logic\total.csv"
    output_file = r"D:\download\cursor\paper 3\Authenticity-HyperReality\authentic.csv"
    
    logging.info("开始体验真实性量化分析...")
    
    # 读取字幕数据
    try:
        df_subtitles = pd.read_csv(subtitle_file, encoding='utf-8')
        logging.info(f"成功读取字幕文件，共 {len(df_subtitles)} 条记录")
    except Exception as e:
        logging.error(f"读取字幕文件失败: {e}")
        return
    
    # 检查必要列
    if 'video_id' not in df_subtitles.columns or 'text' not in df_subtitles.columns:
        logging.error("字幕文件缺少必要的列: video_id 或 text")
        return
    
    # 获取视频目录中的所有视频文件并按自然顺序排序
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort(key=natural_sort_key)
    
    if not video_files:
        logging.error(f"在 {video_dir} 中未找到任何视频文件")
        return
    
    logging.info(f"找到 {len(video_files)} 个视频文件")
    
    # 读取已有的结果（如果存在）
    existing_results = []
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            logging.info(f"已读取现有结果 {len(existing_results)} 条")
        except Exception as e:
            logging.warning(f"读取现有结果失败: {e}")
    
    # 获取已处理的视频文件
    processed_files = set(result['video_file'] for result in existing_results)
    
    # 筛选未处理的视频文件
    remaining_files = [f for f in video_files if f not in processed_files]
    logging.info(f"待处理视频文件数: {len(remaining_files)}")
    
    # 创建结果列表
    results = existing_results.copy()
    
    def process_video_with_retry(video_file: str) -> Dict:
        """带重试机制的视频处理"""
        for attempt in range(MAX_RETRIES):
            try:
                # 检查缓存
                cached_result = get_cached_result(video_file)
                if cached_result:
                    logging.info(f"使用缓存结果: {video_file}")
                    return cached_result
                
                # 查找对应的字幕内容
                video_id = os.path.splitext(video_file)[0]  # 去掉.mp4后缀
                # 确保video_id是字符串类型进行比较
                subtitle_row = df_subtitles[df_subtitles['video_id'].astype(str) == video_id]
                subtitle_content = ""
                
                if not subtitle_row.empty:
                    text_content = subtitle_row.iloc[0]['text']
                    if pd.notna(text_content) and str(text_content).strip():
                        subtitle_content = str(text_content).strip()
                        logging.info(f"找到文案 ({video_file}): {subtitle_content[:50]}...")
                    else:
                        logging.info(f"视频 {video_file} 没有文案")
                else:
                    logging.warning(f"在文本数据中未找到视频 {video_file} 的记录")
                
                # 处理视频
                result = process_single_video(video_file, video_dir, subtitle_content)
                
                # 保存到缓存
                save_to_cache(video_file, result)
                
                return result
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logging.warning(f"处理 {video_file} 失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logging.error(f"处理 {video_file} 最终失败: {e}")
                    return {
                        'video_file': video_file,
                        'A_visual': 0.0,
                        'H_visual': 0.0,
                        'A_sem': 0.0,
                        'H_sem': 0.0,
                        'A_score': 0.0,
                        'H_score': 0.0,
                        'class': 'Error'
                    }
    
    # 使用防休眠模式
    with PreventSleep():
        # 按批次处理视频
        for i in range(0, len(remaining_files), BATCH_SIZE):
            batch_files = remaining_files[i:i + BATCH_SIZE]
            logging.info(f"\n处理批次 {i//BATCH_SIZE + 1}, 视频数量: {len(batch_files)}")
            
            # 使用线程池并发处理批次中的视频
            with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                future_to_video = {
                    executor.submit(process_video_with_retry, video_file): video_file 
                    for video_file in batch_files
                }
                
                # 使用tqdm显示进度
                with tqdm(total=len(batch_files), desc="处理视频") as pbar:
                    for future in concurrent.futures.as_completed(future_to_video):
                        video_file = future_to_video[future]
                        try:
                            result = future.result(timeout=TIMEOUT)
                            results.append(result)
                            logging.info(f"✓ {video_file}: {result['class']} (A={result['A_score']:.3f}, H={result['H_score']:.3f})")
                        except concurrent.futures.TimeoutError:
                            logging.error(f"处理 {video_file} 超时")
                            results.append({
                                'video_file': video_file,
                                'A_visual': 0.0,
                                'H_visual': 0.0,
                                'A_sem': 0.0,
                                'H_sem': 0.0,
                                'A_score': 0.0,
                                'H_score': 0.0,
                                'class': 'Timeout'
                            })
                        pbar.update(1)
            
            # 按视频文件名自然排序结果
            results.sort(key=lambda x: natural_sort_key(x['video_file']))
            
            # 保存当前批次结果
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"已保存当前批次结果，共 {len(results)} 条记录")
    
    # 最终统计报告
    logging.info("\n=== 统计报告 ===")
    logging.info(f"总视频数: {len(results)}")
    
    if results:
        df_final = pd.DataFrame(results)
        class_counts = df_final['class'].value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(results)) * 100
            logging.info(f"{class_name}: {count} 个 ({percentage:.1f}%)")
        
        # 平均分数
        avg_a = df_final['A_score'].mean()
        avg_h = df_final['H_score'].mean()
        logging.info(f"\n平均真实性分数 (A): {avg_a:.3f}")
        logging.info(f"平均异真性分数 (H): {avg_h:.3f}")

if __name__ == "__main__":
    main() 