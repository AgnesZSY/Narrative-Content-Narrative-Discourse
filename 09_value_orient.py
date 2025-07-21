#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅游短视频类型鉴别脚本
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import jieba
import re
import os
from tqdm import tqdm
import warnings
import cv2
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from functools import partial
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pickle
from PIL import Image
import ctypes
import time

warnings.filterwarnings('ignore')

### --- 防休眠功能 ---
class PreventSleep:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    def __init__(self):
        self.active = False
    def enable(self):
        if os.name == "nt" and not self.active:
            ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )
            self.active = True
            print("已启用防休眠")
    def disable(self):
        if os.name == "nt" and self.active:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            self.active = False
            print("已恢复默认休眠设置")

def extract_hashtags(text):
    if not text:
        return []
    return re.findall(r"#([\u4e00-\u9fa5A-Za-z0-9_]+)", text)

def safe_text(x):
    if pd.isna(x) or x is None:
        return ''
    if isinstance(x, float) and np.isnan(x):
        return ''
    if str(x).lower() == "nan":
        return ''
    return str(x).strip()

### --- 视频特征提取 ---
class LightVideoFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, video_path):
        import hashlib
        cache_name = hashlib.md5(video_path.encode()).hexdigest() + '.pkl'
        return os.path.join(self.cache_dir, cache_name)
    
    def _get_cached_features(self, video_path):
        cache_path = self._get_cache_path(video_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None
    
    def _cache_features(self, video_path, features):
        cache_path = self._get_cache_path(video_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass
    
    def _init_model(self):
        try:
            from torchvision.models import MobileNetV2_Weights
            self.model = mobilenet_v2(weights=MobileNetV2_Weights.DEFAULT)
        except ImportError:
            self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        weights_path = "models/light_classifier.pth"
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.models_loaded = True

    def extract_frames(self, video_path, max_frames=5):
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                return []
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
            cap.release()
            return frames
        except Exception:
            return []
    
    def analyze_frame(self, frame):
        if not self.models_loaded:
            return None
        try:
            input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
            emotion_score = float(probs[0][0])
            info_score = float(probs[0][1])
            frame_np = np.array(frame)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].mean()
            edges = cv2.Canny(gray, 100, 200)
            text_like_regions = np.sum(edges > 0) / (frame_np.shape[0] * frame_np.shape[1])
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    emotion_score += 0.15 * len(faces)
            except:
                pass
            total = emotion_score + info_score
            if total > 0:
                emotion_score = emotion_score / total
                info_score = info_score / total
            return {
                'emotion_score': emotion_score,
                'info_score': info_score,
                'blur_score': blur_score,
                'saturation': saturation,
                'text_regions': text_like_regions
            }
        except Exception:
            return None

    def analyze_video(self, video_path, timeout_sec=180, max_retry=2):
        cached_features = self._get_cached_features(video_path)
        if cached_features is not None:
            return cached_features
        retry = 0
        while retry <= max_retry:
            try:
                start = time.time()
                frames = self.extract_frames(video_path)
                if not frames:
                    return None
                emotion_score = 0
                info_score = 0
                valid_frames = 0
                frame_features = []
                for frame in frames:
                    features = self.analyze_frame(frame)
                    if features:
                        frame_features.append(features)
                        emotion_score += features['emotion_score']
                        info_score += features['info_score']
                        valid_frames += 1
                    if time.time() - start > timeout_sec:
                        raise TimeoutError("单视频处理超时")
                if valid_frames > 0:
                    emotion_score /= valid_frames
                    info_score /= valid_frames
                    features = {
                        'emotion_score': float(emotion_score),
                        'info_score': float(info_score),
                        'frame_count': valid_frames,
                        'frame_features': frame_features
                    }
                    self._cache_features(video_path, features)
                    return features
                return None
            except Exception as ex:
                retry += 1
                if retry > max_retry:
                    print(f"视频处理失败（多次重试超时）: {video_path}")
                    return None
                else:
                    print(f"处理{video_path}超时/异常，第{retry}次重试...")

def process_video_batch(video_batch, extractor):
    results = {}
    for video_path in video_batch:
        result = extractor.analyze_video(video_path)
        if result:
            results[video_path] = result
    return results

class EnhancedTravelVideoClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.num_processes = max(1, os.cpu_count() - 1)
        self.batch_size = 5
        # --- 情感型与信息型关键词词典+权重 ---
        self.info_kw_weights = {
            "攻略":3, "推荐":3, "路线":2, "门票":2, "价格":2, "交通":3, "避坑":3, "地图":2,
            "时间":2, "费用":2, "开放":2, "到达":2, "自驾":2, "公交":2, "乘车":2,
            "注意事项":3, "经验":2, "票价":2, "怎么去":3, "几月":2, "旺季":2, "淡季":2, "详细":2
        }
        self.emotion_kw_weights = {
            "美":2, "治愈":2, "浪漫":2, "享受":2, "体验":2, "感受":2, "风景":2, "自然":2,
            "风光":2, "治愈系":3, "春天":1, "旅行":1, "出发":1, "拍照":1, "美好":2, "快乐":2
        }
        self.negation_words = ["不", "没", "无", "非", "未"]
        print("正在加载本地RoBERTa模型...")
        model_dir = r"D:\download\cursor\paper 3\Type\huggingface"
        self.zh_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.zh_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        print("本地模型加载完成")
        print(f"[模型labels id2label]: {getattr(self.zh_model.config, 'id2label', None)}")
        self.video_extractor = LightVideoFeatureExtractor()
        self.optimal_threshold = 0.45

    def _fallback_sentiment_score(self, text):
        tokens = jieba.lcut(text)
        pos, neg = 0, 0
        for i, t in enumerate(tokens):
            has_neg = False
            if i > 0 and tokens[i-1] in self.negation_words: has_neg = True
            if t in ["好", "棒", "赞", "美", "喜欢"]: pos += 1 if not has_neg else 0
            if t in ["差", "坏", "烂", "丑"]: neg += 1 if not has_neg else 0
        total = pos + neg
        if total == 0: return 0.5
        return pos / total

    def detect_lang(self, text):
        try:
            if re.search(r'[\u4e00-\u9fff]', text):
                return 'zh'
            lang = detect(text)
            return 'zh' if lang.startswith('zh') else 'en'
        except:
            return 'zh'

    def roberta_score(self, text):
        try:
            inputs = self.zh_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.zh_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
            return float(probs[1]), "zh-roberta"
        except Exception:
            return self._fallback_sentiment_score(text), "Fallback"

    def bert_score(self, text, lang):
        if lang == 'zh':
            return self.roberta_score(text)
        else:
            return self._fallback_sentiment_score(text), "Fallback"

    def enhanced_keyword_bias(self, text, hashtags=None):
        tokens = jieba.lcut(text)
        if hashtags: tokens += hashtags
        emotion_score = 0
        info_score = 0
        for t in tokens:
            emotion_score += self.emotion_kw_weights.get(t, 0)
            info_score += self.info_kw_weights.get(t, 0)
        sum_score = abs(emotion_score) + abs(info_score)
        normalized_diff = (emotion_score - info_score) / (sum_score if sum_score > 0 else 1)
        return np.clip(normalized_diff * 0.4, -0.4, 0.4)

    def extract_rich_features(self, text):
        features = {}
        features['len_chars'] = len(text)
        features['len_tokens'] = len(jieba.lcut(text))
        features['exclamation_ratio'] = text.count('!') / max(len(text), 1)
        features['question_ratio'] = text.count('？') / max(len(text), 1)
        features['comma_ratio'] = text.count('，') / max(len(text), 1)
        features['period_ratio'] = text.count('。') / max(len(text), 1)
        return features

    def calculate_feature_bias(self, features):
        bias = 0
        bias += features.get('exclamation_ratio', 0) * 0.15
        bias -= features.get('question_ratio', 0) * 0.05
        return np.clip(bias, -0.2, 0.2)

    def length_bias(self, char_len):
        if char_len < 50:
            return 0.08
        elif char_len < 80:
            return 0.05
        elif char_len > 300:
            return -0.08
        elif char_len > 250:
            return -0.05
        else:
            return 0.0

    def classify_row(self, row, video_features):
        content = safe_text(row.get('content', ''))
        title = safe_text(row.get('title', ''))
        video_file = safe_text(row['video_file'])
        hashtags = extract_hashtags(title)
        # 文案长度（去除空白符和标点）
        def non_punct_len(text):
            return len(re.sub(r'[\s，。！？,.!?\-~·#@、…\(\)（）【】\[\]\'"“”‘’\d]+', '', text))
        content_len = non_punct_len(content)
        title_len = non_punct_len(title)

        # 初始化
        result = {
            'video_file': video_file,
            'n_tokens': 0,
            'len_chars': 0,
            'final_prob': 0.5,
            'type_label': 0,
            'video_emotion_score': np.nan,
            'video_info_score': np.nan,
            'features': {},
            'has_content': bool(content)
        }
        if video_features:
            result['video_emotion_score'] = round(float(video_features.get('emotion_score', np.nan)), 4)
            result['video_info_score'] = round(float(video_features.get('info_score', np.nan)), 4)

        # 计算内容和标题分
        content_prob = None
        title_prob = None
        if content_len >= 8:
            c_features = self.extract_rich_features(content)
            c_lang = self.detect_lang(content)
            c_model_prob, c_model_src = self.bert_score(content, c_lang)
            c_kw_bias = self.enhanced_keyword_bias(content, hashtags)
            c_len_bias = self.length_bias(len(content))
            c_feat_bias = self.calculate_feature_bias(c_features)
            content_prob = 0.7 * c_model_prob + 0.25 * (c_kw_bias + 0.3) + 0.05 * (c_feat_bias + 0.2)
        if title_len >= 8:
            t_features = self.extract_rich_features(title)
            t_lang = self.detect_lang(title)
            t_model_prob, t_model_src = self.bert_score(title, t_lang)
            t_kw_bias = self.enhanced_keyword_bias(title, hashtags)
            t_len_bias = self.length_bias(len(title))
            t_feat_bias = self.calculate_feature_bias(t_features)
            title_prob = 0.7 * t_model_prob + 0.25 * (t_kw_bias + 0.3) + 0.05 * (t_feat_bias + 0.2)

        # 权重分配与融合
        content_weight, title_weight, visual_weight = 0, 0, 0
        final_prob = 0.5
        if content_len >= 15 and title_len >= 15:
            content_weight, title_weight, visual_weight = 0.7, 0.2, 0.1
            text_prob = content_weight * (content_prob or 0) + title_weight * (title_prob or 0)
            vis_prob = video_features['emotion_score'] if video_features else 0.7
            final_prob = text_prob + visual_weight * vis_prob
        elif content_len >= 15 and title_len < 15:
            content_weight, title_weight, visual_weight = 0.8, 0, 0.2
            vis_prob = video_features['emotion_score'] if video_features else 0.7
            final_prob = (content_prob or 0) * content_weight + vis_prob * visual_weight
        elif title_len >= 15 and content_len < 15:
            content_weight, title_weight, visual_weight = 0, 0.8, 0.2
            vis_prob = video_features['emotion_score'] if video_features else 0.7
            final_prob = (title_prob or 0) * title_weight + vis_prob * visual_weight
        elif (0 < content_len < 15) or (0 < title_len < 15):
            content_weight, title_weight, visual_weight = 0.25, 0.25, 0.5
            cb = content_prob if content_prob is not None else 0
            tb = title_prob if title_prob is not None else 0
            vis_prob = video_features['emotion_score'] if video_features else 0.7
            final_prob = content_weight * cb + title_weight * tb + visual_weight * vis_prob
        elif content_len == 0 and title_len == 0:
            content_weight, title_weight, visual_weight = 0, 0, 1
            vis_prob = video_features['emotion_score'] if video_features else 0.5
            final_prob = vis_prob
        else:
            final_prob = 0.5

        # 日志
        print(f"\n[video: {video_file}] 内容长: {content_len}, 标题长: {title_len}")
        print(f"内容分: {content_prob}, 标题分: {title_prob}, 视觉分: {video_features['emotion_score'] if video_features else '无'}")
        print(f"融合权重=> 内容:{content_weight:.2f}, 标题:{title_weight:.2f}, 视觉:{visual_weight:.2f}")
        print(f"最终融合分: {final_prob}")

        result['final_prob'] = round(float(final_prob), 4)
        threshold = self.optimal_threshold
        result['type_label'] = 1 if final_prob >= threshold else 0
        return result

    def process_csv(self, df, output_path):
        df = df.sort_values('video_id').reset_index(drop=True)
        if os.path.exists(output_path):
            old = pd.read_csv(output_path)
            done_ids = set(old['video_id'])
            print(f"已完成 {len(done_ids)} 条，断点续跑跳过已处理视频...")
        else:
            old = None
            done_ids = set()
        df = df[~df['video_id'].isin(done_ids)]
        if len(df) == 0:
            print("没有剩余待处理视频。")
            return old if old is not None else pd.DataFrame()

        video_files = df['video_file'].tolist()
        video_batches = [video_files[i:i + self.batch_size]
                         for i in range(0, len(video_files), self.batch_size)]
        all_results = []
        for batch_idx, video_batch in enumerate(tqdm(video_batches, desc="主进度: 批处理", ncols=80)):
            process_func = partial(process_video_batch, extractor=self.video_extractor)
            batch_result = process_func(video_batch)
            video_features = batch_result
            batch_rows = df[df['video_file'].isin(video_batch)].sort_values('video_id')
            batch_results = []
            for idx, (i, row) in enumerate(tqdm(batch_rows.iterrows(), total=len(batch_rows),
                                                desc=f"批{batch_idx+1}/{len(video_batches)}", leave=False, ncols=60)):
                try:
                    res = self.classify_row(row, video_features.get(row['video_file']))
                    res['video_id'] = row['video_id']
                    res['title'] = row['title']
                    batch_results.append(res)
                except Exception as e:
                    print(f"处理第 {i+1} 行时出错: {e}")
                    batch_results.append({
                        'video_id': row['video_id'],
                        'video_file': row['video_file'],
                        'title': row['title'],
                        'type_label': 0,
                        'error': str(e)
                    })
            all_results.extend(batch_results)
            if old is not None:
                full_results = pd.concat([old, pd.DataFrame(all_results)], ignore_index=True)
            else:
                full_results = pd.DataFrame(all_results)
            full_results = full_results.drop_duplicates("video_id").sort_values("video_id")
            save_cols = ['video_id', 'title', 'type_label', 'final_prob', 'video_emotion_score', 'video_info_score']
            full_results = full_results[save_cols]
            full_results.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"已保存至 {output_path}，进度 {batch_idx+1}/{len(video_batches)}，已完成 {len(full_results)} 条")
            old = full_results
        self._print_statistics(full_results)
        return full_results

    def _print_statistics(self, df):
        emo = (df['type_label'] == 1).sum()
        info = (df['type_label'] == 0).sum()
        total = len(df)
        print("\n分类结果统计:")
        print(f"情感享乐型 (1): {emo} 条 ({emo/total*100:.1f}%)")
        print(f"信息实用型 (0): {info} 条 ({info/total*100:.1f}%)")
        if 'error' in df.columns:
            error_count = df['error'].notna().sum()
            print(f"处理失败: {error_count} 条 ({error_count/total*100:.1f}%)")

def main():
    # --- 配置参数 ---
    input_csv = r"xxxxxxxxxxxxxxxxxxx"
    output_csv = r"xxxxxxxxxxxxxxxxxxx"
    videos_dir = r"xxxxxxxxxxxxxxxxxxx"
    ps = PreventSleep(); ps.enable()
    try:
        if not os.path.exists(input_csv):
            print(f"错误: 输入文件不存在 {input_csv}")
            return
        if not os.path.exists(videos_dir):
            print(f"错误: 视频目录不存在 {videos_dir}")
            return
        video_ids = []
        print("正在扫描视频目录...")
        for file in os.listdir(videos_dir):
            if file.endswith('.mp4'):
                video_id = file.replace('.mp4', '')
                if video_id.isdigit():
                    video_ids.append(int(video_id))
        if not video_ids:
            print("错误: 视频目录中没有找到有效的MP4文件")
            return
        print(f"找到 {len(video_ids)} 个有效视频文件")
        try:
            df = pd.read_csv(input_csv, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv, encoding='gbk')
        print(f"CSV文件共有 {len(df)} 条记录")
        required_columns = ['video_id', 'text', 'title']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误: CSV文件缺少必要的列: {', '.join(missing_columns)}")
            return
        df['video_id'] = pd.to_numeric(df['video_id'], errors='coerce')
        df = df[df['video_id'].isin(video_ids)]
        if len(df) == 0:
            print("错误: 没有找到与视频文件对应的记录")
            return
        print(f"找到 {len(df)} 条对应的记录")
        df['video_file'] = df['video_id'].apply(lambda x: os.path.join(videos_dir, f"{int(x)}.mp4"))
        df = df.rename(columns={'text': 'content'})
        classifier = EnhancedTravelVideoClassifier()
        result_df = classifier.process_csv(df, output_csv)
        if result_df is not None:
            print("\n处理完成!")
            print(f"结果已保存到: {output_csv}")
    finally:
        ps.disable()

if __name__ == "__main__":
    main()
