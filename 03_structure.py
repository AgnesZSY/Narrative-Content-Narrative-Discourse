import os
from pathlib import Path
import logging
import numpy as np
import torch
import sys
from PIL import Image
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm
from scenedetect import detect, ContentDetector
from sklearn.metrics.pairwise import cosine_similarity
import io
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
import multiprocessing as mp
from functools import wraps
import time
import ctypes
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import subprocess

# 检查ffmpeg是否可用
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

# 检查并导入ffmpeg-python
try:
    import ffmpeg
except ImportError:
    print("错误：未找到ffmpeg-python包")
    print("请运行: pip install ffmpeg-python")
    sys.exit(1)

# 检查系统ffmpeg
if not check_ffmpeg():
    print("错误：系统中未找到ffmpeg")
    print("请安装ffmpeg后再运行此程序")
    print("Windows: 请访问 https://ffmpeg.org/download.html 下载并添加到系统PATH")
    print("Linux: sudo apt-get install ffmpeg")
    print("macOS: brew install ffmpeg")
    sys.exit(1)

print("ffmpeg检查通过...")

# 防休眠设置
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    """启用防休眠模式"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("防休眠模式已启用")
    except Exception as e:
        print(f"启用防休眠模式失败: {e}")

def restore_sleep():
    """恢复默认电源设置"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("已恢复默认电源设置")
    except Exception as e:
        print(f"恢复默认电源设置失败: {e}")

# 超时装饰器
def timeout_retry(max_retries=3, timeout=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    time.sleep(1)
            return None
        return wrapper
    return decorator

# 缓存管理
class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.processed_videos = self._load_processed_videos()
        self.csv_processed_videos = self._load_csv_processed_videos()
        
    def _load_processed_videos(self) -> set:
        cache_file = self.cache_dir / "processed_videos.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return set(json.load(f))
        return set()

    def _load_csv_processed_videos(self) -> set:
        """从CSV文件加载已处理的视频ID"""
        if os.path.exists(OUTPUT_EXCEL):
            try:
                df = pd.read_csv(OUTPUT_EXCEL)
                if 'video_id' in df.columns:
                    return set(df['video_id'].astype(str))
            except Exception as e:
                print(f"加载CSV文件失败: {e}")
        return set()

    def save_processed_videos(self):
        with open(self.cache_dir / "processed_videos.json", "w") as f:
            json.dump(list(self.processed_videos), f)

    def is_processed(self, video_id: str) -> bool:
        # 同时检查缓存和CSV中的记录
        return video_id in self.processed_videos or video_id in self.csv_processed_videos

    def mark_as_processed(self, video_id: str):
        self.processed_videos.add(video_id)
        self.csv_processed_videos.add(video_id)
        self.save_processed_videos()

    def get_features(self, video_id: str) -> np.ndarray:
        # 先检查内存缓存
        if video_id in self.memory_cache:
            return self.memory_cache[video_id]

        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{video_id}_features.npy"
        if cache_file.exists():
            features = np.load(str(cache_file))
            self.memory_cache[video_id] = features
            return features
        return None

    def save_features(self, video_id: str, features: np.ndarray):
        # 保存到内存缓存
        self.memory_cache[video_id] = features
        # 保存到磁盘缓存
        cache_file = self.cache_dir / f"{video_id}_features.npy"
        np.save(str(cache_file), features)

print("正在导入ResNet模型...")
# 使用ResNet作为特征提取器（替代CLIP）

# 配置日志
logging.basicConfig(
    filename='结构形态_error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)

# 全局配置
INPUT_DIR = r"xxxxxxxxxxxxxxxxxxx"
OUTPUT_EXCEL = r"xxxxxxxxxxxxxxxxxxx"
CACHE_DIR = r"xxxxxxxxxxxxxxxxxxx"
MIN_SEGMENTS = 3
SEGMENT_DURATION = 5
BATCH_SIZE = 16
GROUP_SIZE = 5  # 每组处理的视频数量

def check_directories():
    """检查必要的目录是否存在"""
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误：输入目录 {INPUT_DIR} 不存在")
        try:
            os.makedirs(INPUT_DIR)
            print(f"已创建输入目录 {INPUT_DIR}")
        except Exception as e:
            print(f"创建输入目录失败: {e}")
            print("请确保：")
            print("1. 您有足够的权限创建目录")
            print("2. 路径是否正确")
            print("3. 驱动器是否存在")
            return False
    
    # 检查输出目录
    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录 {output_dir}")
        except Exception as e:
            print(f"创建输出目录失败: {e}")
            return False
    
    # 检查缓存目录
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
            print(f"已创建缓存目录 {CACHE_DIR}")
        except Exception as e:
            print(f"创建缓存目录失败: {e}")
            return False
    
    return True

# 初始化缓存管理器
cache_manager = CacheManager(CACHE_DIR)

def natural_sort_key(s):
    """实现自然排序的键函数"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# 全局变量
MODEL = None
DEVICE = None

def initialize_model():
    """初始化并返回模型"""
    print("正在初始化特征提取模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 使用新的权重参数替代deprecated的pretrained参数
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    print("特征提取模型加载成功！")
    return model, device

def get_model():
    """获取全局模型实例"""
    global MODEL, DEVICE
    if MODEL is None:
        MODEL, DEVICE = initialize_model()
    return MODEL, DEVICE

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_segments(video_path: str) -> List[Tuple[float, float]]:
    """获取视频分段时间点"""
    try:
        # 使用PySceneDetect进行场景检测
        scenes = detect(video_path, ContentDetector())
        segments = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]
        
        # 如果分段数小于3，则进行均匀切分
        if len(segments) < MIN_SEGMENTS:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            segment_count = max(MIN_SEGMENTS, int(duration // SEGMENT_DURATION))
            segment_duration = duration / segment_count
            segments = [
                (i * segment_duration, (i + 1) * segment_duration)
                for i in range(segment_count)
            ]
        
        return segments
    except Exception as e:
        logging.error(f"Error in get_segments for {video_path}: {str(e)}")
        raise

@torch.no_grad()
def get_frame_at_timestamp(video_path: str, timestamp: float) -> Image.Image:
    """在指定时间点提取帧"""
    try:
        # 添加更多的ffmpeg参数来处理有问题的视频
        out, err = (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg', 
                    **{'loglevel': 'error', 'hide_banner': None})
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        if err:
            print(f"FFmpeg警告: {err.decode('utf-8', errors='ignore')}")
            
        return Image.open(io.BytesIO(out))
    except Exception as e:
        logging.error(f"Error in get_frame_at_timestamp for {video_path} at {timestamp}: {str(e)}")
        # 尝试使用不同的时间点
        try:
            backup_timestamp = max(0, timestamp - 1)
            out, _ = (
                ffmpeg
                .input(video_path, ss=backup_timestamp)
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg',
                        **{'loglevel': 'error', 'hide_banner': None})
                .run(capture_stdout=True, capture_stderr=True)
            )
            return Image.open(io.BytesIO(out))
        except:
            raise e

@torch.no_grad()
def extract_features(images: List[Image.Image]) -> np.ndarray:
    """批量获取图像的特征向量"""
    try:
        # 获取模型实例
        model, device = get_model()
        
        # 预处理图像
        processed_images = torch.stack([preprocess(img) for img in images]).to(device)
        
        # 获取特征
        features = model(processed_images)
        features = features.squeeze(-1).squeeze(-1)  # 移除空间维度
        return features.cpu().numpy()
    except Exception as e:
        logging.error(f"Error in extract_features: {str(e)}")
        raise

def calc_mean_sim(emb_list: List[np.ndarray]) -> float:
    """计算相邻片段的平均余弦相似度"""
    if len(emb_list) < 2:
        return 0.0
    
    sims = []
    for i in range(len(emb_list) - 1):
        sim = cosine_similarity(
            emb_list[i].reshape(1, -1),
            emb_list[i + 1].reshape(1, -1)
        )[0][0]
        sims.append(sim)
    return np.mean(sims)

def decide_label(mean_sim: float) -> int:
    """根据平均相似度确定结构标签"""
    if mean_sim > 0.8:  # 调整阈值适应ResNet特征
        return 1  # 连续式
    elif mean_sim < 0.5:  # 调整阈值适应ResNet特征
        return 2  # 离散式
    else:
        return 0  # 单元式

@timeout_retry(max_retries=3, timeout=600)
def process_video(video_path: str, pbar: tqdm = None) -> dict:
    """处理单个视频"""
    video_id = Path(video_path).stem
    
    # 检查是否已处理
    if cache_manager.is_processed(video_id):
        if pbar:
            pbar.set_description(f"跳过已处理视频: {video_id}")
        return None
    
    if pbar:
        pbar.set_description(f"处理视频: {video_id}")
    
    try:
        # 获取视频时长
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        
        # 检查缓存的特征
        cached_features = cache_manager.get_features(video_id)
        if cached_features is not None:
            embeddings = cached_features
        else:
            # 获取分段
            segments = get_segments(video_path)
            
            # 获取每个分段中心帧
            frames = []
            failed_frames = 0
            
            for start, end in segments:
                mid_point = (start + end) / 2
                try:
                    frame = get_frame_at_timestamp(video_path, mid_point)
                    frames.append(frame)
                except Exception as e:
                    failed_frames += 1
                    continue
            
            if len(frames) == 0:
                raise Exception(f"无法提取任何帧，失败帧数: {failed_frames}")
            
            # 获取模型实例
            model, device = get_model()
            
            # 批量处理帧
            embeddings = []
            for i in range(0, len(frames), BATCH_SIZE):
                batch_frames = frames[i:i + BATCH_SIZE]
                batch_embeddings = extract_features(batch_frames)
                embeddings.extend(batch_embeddings)
            
            # 保存特征到缓存
            cache_manager.save_features(video_id, np.array(embeddings))
        
        # 计算平均相似度
        mean_sim = calc_mean_sim(embeddings)
        
        # 确定结构标签
        structure_label = decide_label(mean_sim)
        
        result = {
            'video_id': video_id,
            'duration_sec': duration,
            'num_segments': len(embeddings),
            'mean_sim': mean_sim,
            'structure_label': structure_label,
            'status': 'success'
        }
        
        # 标记为已处理
        cache_manager.mark_as_processed(video_id)
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        if pbar:
            pbar.set_postfix({"阶段": "错误"})
        
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
        except:
            duration = 0
            
        return {
            'video_id': video_id,
            'duration_sec': duration,
            'num_segments': 0,
            'mean_sim': 0,
            'structure_label': -1,
            'status': 'failed',
            'error': str(e)[:100]
        }

def process_video_group(video_files: List[Path], existing_results: List[dict] = None) -> List[dict]:
    """处理一组视频"""
    results = []
    
    # 过滤掉已处理的视频
    unprocessed_files = [
        video_file for video_file in video_files 
        if not cache_manager.is_processed(video_file.stem)
    ]
    
    if not unprocessed_files:
        print("当前组中所有视频都已处理，跳过...")
        return existing_results if existing_results else []
    
    total_files = len(unprocessed_files)
    print(f"当前组中有 {total_files} 个未处理的视频")
    
    # 创建进度条
    with tqdm(total=total_files, desc="当前组进度", position=1, leave=False) as pbar:
        with ProcessPoolExecutor(max_workers=min(5, total_files)) as executor:
            futures = {executor.submit(process_video, str(video_file)): video_file 
                      for video_file in unprocessed_files}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    # 合并现有结果和新结果
    if existing_results:
        # 移除重复的视频结果
        existing_video_ids = {r['video_id'] for r in existing_results}
        new_results = [r for r in results if r['video_id'] not in existing_video_ids]
        results = existing_results + new_results
    
    # 对结果按照视频ID进行自然排序
    def extract_number(video_id):
        # 确保video_id是字符串
        video_id = str(video_id)
        # 提取视频ID中的数字部分
        match = re.search(r'(\d+)', video_id)
        return int(match.group(1)) if match else float('inf')
    
    results.sort(key=lambda x: extract_number(x['video_id']))
    
    # 保存结果（添加错误处理）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = pd.DataFrame(results)
            # 使用临时文件来避免权限问题
            temp_dir = os.path.dirname(OUTPUT_EXCEL)
            temp_file = os.path.join(temp_dir, f'temp_{int(time.time())}.csv')
            
            # 确保输出目录存在
            os.makedirs(temp_dir, exist_ok=True)
            
            # 先写入临时文件
            df.to_csv(temp_file, index=False, encoding='utf-8-sig')
            
            try:
                # 如果原文件存在且可以访问，则删除它
                if os.path.exists(OUTPUT_EXCEL):
                    os.remove(OUTPUT_EXCEL)
            except Exception as e:
                print(f"警告：无法删除原文件: {e}")
                # 使用不同的输出文件名
                output_file = OUTPUT_EXCEL.replace('.csv', f'_{int(time.time())}.csv')
                print(f"将使用新的输出文件名: {output_file}")
                os.rename(temp_file, output_file)
                break
                
            # 重命名临时文件
            try:
                os.rename(temp_file, OUTPUT_EXCEL)
            except Exception as e:
                print(f"警告：无法重命名临时文件: {e}")
                # 保留临时文件
                print(f"结果已保存在临时文件: {temp_file}")
            break
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"保存结果失败: {e}")
                logging.error(f"保存结果失败: {e}")
            else:
                print(f"保存结果尝试 {attempt + 1}/{max_retries} 失败，重试中...")
                time.sleep(1)
    
    return results

def main():
    """主函数"""
    try:
        # 检查目录
        if not check_directories():
            print("目录检查失败，程序退出")
            return

        # 检查输入目录中是否有视频文件
        video_files = [
            f for f in Path(INPUT_DIR).glob("*")
            if f.suffix.lower() in ['.mp4', '.avi', '.mov']
        ]
        
        if not video_files:
            print(f"警告：在输入目录 {INPUT_DIR} 中没有找到视频文件")
            print("请确保：")
            print("1. 视频文件已正确放置在输入目录中")
            print("2. 视频文件扩展名为 .mp4、.avi 或 .mov")
            return
        
        # 启用防休眠
        prevent_sleep()
        
        print("开始处理视频...")
        
        # 预先初始化模型
        get_model()
        
        # 获取所有视频文件并按自然顺序排序
        video_files.sort(key=natural_sort_key)
        
        total_videos = len(video_files)
        print(f"找到 {total_videos} 个视频文件")
        
        # 加载现有结果（如果存在）
        existing_results = []
        if os.path.exists(OUTPUT_EXCEL):
            try:
                existing_df = pd.read_csv(OUTPUT_EXCEL)
                existing_results = existing_df.to_dict('records')
                print(f"加载了 {len(existing_results)} 个现有结果")
            except Exception as e:
                print(f"加载现有结果失败: {e}")
        
        # 按组处理视频
        total_groups = (len(video_files) + GROUP_SIZE - 1) // GROUP_SIZE
        with tqdm(total=total_groups, desc="总体进度", position=0) as main_pbar:
            for i in range(0, len(video_files), GROUP_SIZE):
                group = video_files[i:i + GROUP_SIZE]
                group_num = i//GROUP_SIZE + 1
                main_pbar.set_description(f"处理第 {group_num}/{total_groups} 组")
                
                existing_results = process_video_group(group, existing_results)
                
                # 更新进度信息
                processed = len([r for r in existing_results if r['status'] == 'success'])
                failed = len([r for r in existing_results if r['status'] == 'failed'])
                main_pbar.set_postfix({
                    "成功": processed,
                    "失败": failed,
                    "总计": total_videos
                })
                main_pbar.update(1)
                
            # 最终进度
            main_pbar.set_description("处理完成")
        
        # 显示最终统计信息
        successful_results = [r for r in existing_results if r['status'] == 'success']
        failed_results = [r for r in existing_results if r['status'] == 'failed']
        
        print(f"\n处理完成！")
        print(f"总视频数: {total_videos}")
        print(f"成功处理: {len(successful_results)} 个")
        print(f"处理失败: {len(failed_results)} 个")
        print(f"结果已保存至: {OUTPUT_EXCEL}")
        
        if successful_results:
            print("\n成功视频的统计信息:")
            print(f"平均分段数: {np.mean([r['num_segments'] for r in successful_results]):.2f}")
            print(f"平均相似度: {np.mean([r['mean_sim'] for r in successful_results]):.3f}")
            structure_counts = pd.Series([r['structure_label'] for r in successful_results]).value_counts()
            print("\n结构类型分布:")
            print(f"连续式 (1): {structure_counts.get(1, 0)}个")
            print(f"离散式 (2): {structure_counts.get(2, 0)}个")
            print(f"单元式 (0): {structure_counts.get(0, 0)}个")
        
        if failed_results:
            print("\n失败的视频:")
            for result in failed_results:
                print(f"视频ID: {result['video_id']}, 错误: {result.get('error', 'Unknown error')}")
    
    finally:
        # 恢复默认电源设置
        restore_sleep()

if __name__ == "__main__":
    main()