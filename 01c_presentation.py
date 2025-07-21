import os
import sys
import traceback
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import librosa
import soundfile as sf
from multiprocessing import Pool, cpu_count, Manager
import time
from datetime import datetime
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量以解决KMeans内存泄漏问题
os.environ['OMP_NUM_THREADS'] = '1'

# ---------------------- 配置 ---------------------- 
DEFAULT_INPUT_DIR = 'xxxxxxxxxxxxxxxxxxx'
DEFAULT_OUTPUT_CSV = 'xxxxxxxxxxxxxxxxxxx'
PREVIOUS_RESULTS_CSV = 'xxxxxxxxxxxxxxxxxxx'
ERROR_LOG = 'xxxxxxxxxxxxxxxxxxx'
NUM_PROCESSES = min(4, max(1, cpu_count() - 2))  # 降低进程数以减少资源竞争
CHECKPOINT_INTERVAL = 5 * 60  # 5分钟，以秒为单位
BATCH_SIZE = 50  # 减小批次大小以提高稳定性
MAX_VIDEO_DURATION = 300  # 最大处理视频时长（秒）
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

def clear_memory():
    """清理内存"""
    gc.collect()
    process = psutil.Process(os.getpid())
    process.memory_info()

def extract_speech_segments(wav_path, threshold=0.001, min_duration=0.2):
    """优化的语音段检测"""
    try:
        # 使用更低的采样率和mono模式
        y, sr = librosa.load(wav_path, sr=8000, mono=True)
        
        # 使用更大的hop_length减少计算量
        frame_length = int(sr * 0.05)  # 50ms
        hop_length = int(sr * 0.04)    # 40ms
        
        # 使用更高效的能量计算方法
        energy = np.array([
            np.sum(np.abs(y[i:i+frame_length])) / frame_length
            for i in range(0, len(y)-frame_length, hop_length)
        ])
        
        # 使用自适应阈值
        threshold = np.mean(energy) * 0.15
        
        # 寻找语音段
        is_speech = energy > threshold
        boundaries = np.where(np.diff(is_speech.astype(int)))[0]
        
        segments = []
        start = None
        for i in range(len(is_speech)):
            if is_speech[i] and start is None:
                start = i * hop_length / sr
            elif not is_speech[i] and start is not None:
                end = i * hop_length / sr
                if end - start >= min_duration:
                    segments.append((start, end))
                start = None
        
        if start is not None:
            end = len(is_speech) * hop_length / sr
            if end - start >= min_duration:
                segments.append((start, end))
        
        clear_memory()
        return segments
    except Exception as e:
        raise RuntimeError(f"语音段检测失败: {e}")

def detect_speakers(wav_path):
    """优化的说话人检测"""
    try:
        y, sr = librosa.load(wav_path, sr=8000)  # 降低采样率
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        features = np.vstack([mfcc, mfcc_delta])
        features = features.T
        
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # 如果特征太少，直接返回1个说话人
        if len(features) < 50:
            return 1
            
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 使用更简单的判断逻辑
        kmeans = KMeans(n_clusters=2, random_state=42, max_iter=100).fit(features_scaled)
        
        # 计算簇间距离
        cluster_distance = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
        
        # 计算每个簇的大小
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)
        size_ratio = min(cluster_sizes) / max(cluster_sizes)
        
        # 简化判断条件
        if cluster_distance > 2.0 and size_ratio > 0.3:
            return 2
        
        clear_memory()
        return 1
    except Exception as e:
        raise RuntimeError(f"说话人检测失败: {e}")

def extract_frames_from_video(video_path, segments, fps=2):  # 降低fps
    """优化的帧提取"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    frames_dict = {}
    
    # 计算缩放比例
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_width = 320  # 降低目标分辨率
    scale = target_width / frame_width if frame_width > target_width else 1.0
    
    for (start, end) in segments:
        if end - start > 30:  # 如果段落太长，只取前30秒
            end = start + 30
            
        t = start
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frames_dict.setdefault((start, end), []).append((t, frame))
            t += 1.0 / fps
    
    cap.release()
    clear_memory()
    return frames_dict

def detect_mouth_movement(frame):
    """优化的嘴部运动检测"""
    global face_cascade
    if 'face_cascade' not in globals():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用更快的人脸检测参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=2,
        minSize=(20, 20)
    )
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        mouth_y = y + int(h * 0.6)
        mouth_h = int(h * 0.3)
        mouth_roi = gray[mouth_y:mouth_y+mouth_h, x:x+w]
        
        if mouth_roi.size > 0:
            return np.std(mouth_roi) * np.mean(mouth_roi)
    
    return 0

def detect_oncam_speech(video_path, segments):
    """优化的音画同步检测"""
    oncam_count = 0
    total_count = 0
    
    frames_dict = extract_frames_from_video(video_path, segments)
    for seg, frames in frames_dict.items():
        if not frames:
            continue
            
        movements = []
        for t, frame in frames:
            movement = detect_mouth_movement(frame)
            movements.append(movement)
            
        if movements:
            threshold = 800  # 降低阈值
            count = 0
            for m in movements:
                if m > threshold:
                    count += 1
                    if count >= 2:  # 降低连续帧要求
                        oncam_count += 1
                        break
                else:
                    count = 0
                
        total_count += 1
    
    clear_memory()
    return oncam_count / max(1, total_count)

def get_processed_videos():
    """获取已经处理成功的视频列表"""
    try:
        if os.path.exists(PREVIOUS_RESULTS_CSV):
            df = pd.read_csv(PREVIOUS_RESULTS_CSV, encoding='utf-8-sig')
            return set(df['video_id'].astype(str).tolist())
    except Exception as e:
        print(f"读取已处理视频列表时出错: {e}")
    return set()

def save_results(results, output_csv, is_final=False):
    """保存结果到CSV文件"""
    if not results:
        return
        
    # 读取已有的检查点文件
    checkpoint_files = list(Path(output_csv).parent.glob("*_checkpoint_*.csv"))
    existing_results = []
    
    # 首先读取之前的成功结果
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv, encoding='utf-8-sig')
            existing_results = existing_df.to_dict('records')
        except Exception as e:
            print(f"读取现有结果文件出错: {e}")
    
    # 如果有检查点文件，读取最新的检查点
    elif checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        if os.path.exists(latest_checkpoint):
            try:
                existing_df = pd.read_csv(latest_checkpoint, encoding='utf-8-sig')
                existing_results = existing_df.to_dict('records')
            except Exception as e:
                print(f"读取检查点文件出错: {e}")
    
    # 合并已有结果和新结果
    all_results = existing_results + results
    
    # 去重（以video_id为键）
    df = pd.DataFrame(all_results).drop_duplicates(subset=['video_id'], keep='last')
    
    # 如果不是最终结果，添加时间戳到文件名
    save_path = output_csv
    if not is_final:
        base, ext = os.path.splitext(output_csv)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{base}_checkpoint_{timestamp}{ext}"
    
    # 保存所有结果
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {save_path}")
    print(f"当前共有处理结果: {len(df)} 个视频")

def process_video(args):
    """优化的视频处理函数"""
    video_path, temp_dir = args
    video_id = Path(video_path).stem
    wav_path = temp_dir / f"temp_{video_id}.wav"
    
    try:
        # 检查视频时长
        cap = cv2.VideoCapture(str(video_path))
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        if duration > MAX_VIDEO_DURATION:
            raise RuntimeError(f"视频时长超过{MAX_VIDEO_DURATION}秒，跳过处理")
        
        # 音频提取（使用更优化的参数）
        os.system(f'ffmpeg -y -i "{video_path}" -ar 8000 -ac 1 -vn -acodec pcm_s16le "{wav_path}" -loglevel error')
        
        if not wav_path.exists():
            raise RuntimeError("音频提取失败")
        
        segments = extract_speech_segments(str(wav_path))
        total_speech_sec = sum([end - start for start, end in segments])
        
        if total_speech_sec < 1.0:
            label = 3  # 无人称
        else:
            num_speakers = detect_speakers(str(wav_path))
            if num_speakers >= 2:
                label = 2  # 多人称
            else:
                oncam_ratio = detect_oncam_speech(video_path, segments)
                label = 0 if oncam_ratio >= 0.4 else 1  # 降低阈值
        
        # 清理临时文件
        if wav_path.exists():
            wav_path.unlink()
            
        clear_memory()
        return {
            'video_id': video_id,
            'total_speech_sec': round(total_speech_sec, 2),
            'presentation_view': label
        }
        
    except Exception as e:
        if wav_path.exists():
            wav_path.unlink()
        with open(ERROR_LOG, 'a', encoding='utf-8') as f:
            f.write(f"视频 {video_path} 处理失败: {str(e)}\n{traceback.format_exc()}\n")
        return None

def process_video_batch(args):
    """处理视频批次"""
    video_files, temp_dir = args
    return [process_video((str(f), temp_dir)) for f in video_files]

def main(input_dir=DEFAULT_INPUT_DIR, output_csv=DEFAULT_OUTPUT_CSV):
    """优化的主函数"""
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise Exception(f"输入目录不存在: {input_dir}")
            
        video_files = list(input_path.glob("*.mp4"))
        if not video_files:
            raise Exception(f"未找到MP4文件在目录: {input_dir}")
            
        def get_number(filename):
            num = ''.join(filter(str.isdigit, filename.stem))
            return int(num) if num else float('inf')
            
        video_files.sort(key=get_number)
        
        # 获取已经处理成功的视频列表
        processed_videos = get_processed_videos()
        print(f"从previous results文件中读取到 {len(processed_videos)} 个已处理的视频")
        
        # 检查检查点文件
        checkpoint_files = list(Path(output_csv).parent.glob("*_checkpoint_*.csv"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"找到最新检查点文件: {latest_checkpoint}")
            df = pd.read_csv(latest_checkpoint, encoding='utf-8-sig')
            processed_videos.update(set(df['video_id'].astype(str).tolist()))
            print(f"总共已处理 {len(processed_videos)} 个视频")
            
        # 过滤掉已处理的视频
        video_files = [f for f in video_files if Path(f).stem not in processed_videos]
        
        total_videos = len(video_files)
        print(f"还需处理 {total_videos} 个视频文件")
        if total_videos == 0:
            print("所有视频都已处理完成！")
            return
            
        print(f"使用 {NUM_PROCESSES} 个进程并行处理")
        print(f"将每 {BATCH_SIZE} 个视频分为一批处理")
        
        # 创建临时目录
        temp_dirs = [TEMP_DIR / f"temp_{i}" for i in range(NUM_PROCESSES)]
        for d in temp_dirs:
            d.mkdir(exist_ok=True)
            
        # 将视频文件分批
        batches = [video_files[i:i + BATCH_SIZE] for i in range(0, len(video_files), BATCH_SIZE)]
        
        all_results = []
        last_checkpoint_time = time.time()
        
        with Pool(NUM_PROCESSES) as pool:
            for batch_idx, batch in enumerate(batches, 1):
                print(f"\n处理批次 {batch_idx}/{len(batches)}")
                
                # 将批次分配给不同的临时目录
                batch_args = [(batch[i::NUM_PROCESSES], temp_dirs[i]) for i in range(NUM_PROCESSES)]
                
                # 处理当前批次
                batch_results = []
                for result in pool.imap_unordered(process_video_batch, batch_args):
                    batch_results.extend([r for r in result if r is not None])
                
                all_results.extend(batch_results)
                
                # 检查是否需要保存检查点
                current_time = time.time()
                if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                    print("\n保存检查点...")
                    save_results(all_results, output_csv)
                    last_checkpoint_time = current_time
                    all_results = []  # 清空已保存的结果
                    clear_memory()
                
                processed = min(batch_idx * BATCH_SIZE, total_videos)
                print(f"已处理: {processed}/{total_videos} 个视频")
                
                # 每处理10个批次后强制清理一次内存
                if batch_idx % 10 == 0:
                    clear_memory()
        
        # 保存最终结果
        if all_results:
            save_results(all_results, output_csv, is_final=True)
            print(f"\n所有处理完成！共成功处理 {len(all_results)} 个视频")
            
        # 清理所有临时目录
        for d in temp_dirs:
            try:
                for f in d.glob("*"):
                    f.unlink()
                d.rmdir()
            except:
                pass
                
    except Exception as e:
        print(f"处理失败: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='旅游短视频呈现视角批量量化分析')
    parser.add_argument('--input_dir', default=DEFAULT_INPUT_DIR, help='输入视频目录')
    parser.add_argument('--output_csv', default=DEFAULT_OUTPUT_CSV, help='输出CSV文件')
    args = parser.parse_args()
    main(args.input_dir, args.output_csv) 