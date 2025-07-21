import os
import sys
import numpy as np
import pandas as pd
import cv2
import librosa
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import logging
import pickle
from functools import lru_cache
import threading
import time
import ctypes
warnings.filterwarnings('ignore')

# 防止系统休眠的设置
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    """防止系统进入休眠状态"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        logging.info("已启用防休眠模式")
    except Exception as e:
        logging.warning(f"设置防休眠模式失败: {str(e)}")

def restore_sleep():
    """恢复系统默认的电源管理设置"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logging.info("已恢复默认电源管理设置")
    except Exception as e:
        logging.warning(f"恢复电源管理设置失败: {str(e)}")

# ---------------------- 配置参数 ----------------------
INPUT_DIR = 'xxxxxxxxxxxxxxxxxxx'  # 输入视频目录
OUTPUT_CSV = 'xxxxxxxxxxxxxxxxxxx'  # 输出结果文件
BATCH_SIZE = 3  # 减小批次大小以提高稳定性
ERROR_LOG = 'xxxxxxxxxxxxxxxxxxx'
CACHE_DIR = 'xxxxxxxxxxxxxxxxxxx'  # 缓存目录

# 超时设置（优化后）
PROCESS_TIMEOUT = 600  # 单个进程超时时间（秒）
VIDEO_TIMEOUT = 300    # 单个视频处理超时时间（秒）
OPERATION_TIMEOUT = 120  # 单个操作超时时间（秒）

# 重试设置
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试间隔（秒）

# 采样参数（优化后）
AUDIO_SR = 8000  # 降低音频采样率以加快处理
VIDEO_FPS = 2    # 进一步降低视频抽帧率
PROCESS_NUM = 2  # 限制进程数量以避免资源竞争
CHUNK_SIZE = 20  # 减小音频分析的时间窗口（秒）
RESIZE_DIMS = (120, 68)  # 进一步降低分辨率

# 节奏类型阈值
THRESHOLDS = (0.33, 0.66)  # 慢速-中速-快速的分界点

# 归一化参数
BPM_RANGE = (60, 200)    # BPM范围
CPS_RANGE = (0, 2)       # 剪辑密度范围
MDI_RANGE = (0, 100)     # 运动密度范围
SAS_RANGE = (0, 1)       # 音画同步范围

# 创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xxxxxxxxxxxxxxxxxxx'),
        logging.StreamHandler()
    ]
)

def get_cache_path(video_path, operation):
    """获取缓存文件路径"""
    cache_key = f"{operation}_{Path(video_path).stem}"
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

@lru_cache(maxsize=100)
def extract_audio(video_path, temp_dir):
    """从视频中提取音频（优化版本）"""
    try:
        cache_path = get_cache_path(video_path, "audio")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        audio_path = temp_dir / f"{Path(video_path).stem}_audio.wav"
        cmd = f'ffmpeg -y -i "{video_path}" -t {CHUNK_SIZE} -ar {AUDIO_SR} -ac 1 -vn -acodec pcm_s16le "{audio_path}" -loglevel error'
        os.system(cmd)
        
        result = audio_path if audio_path.exists() else None
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        logging.error(f"音频提取失败 {video_path}: {str(e)}")
        return None

@lru_cache(maxsize=100)
def calculate_bpm(audio_path):
    """计算音乐节拍BPM（优化版本）"""
    try:
        cache_path = get_cache_path(str(audio_path), "bpm")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        y, sr = librosa.load(audio_path, sr=AUDIO_SR, duration=CHUNK_SIZE)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(tempo, f)
        return tempo
    except Exception as e:
        logging.error(f"BPM计算失败 {audio_path}: {str(e)}")
        return 0

@lru_cache(maxsize=100)
def calculate_cps(video_path):
    """计算剪辑密度CPS（优化版本）"""
    try:
        cache_path = get_cache_path(video_path, "cps")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = min(total_frames / fps, CHUNK_SIZE)
        
        max_frames = min(total_frames, int(fps * CHUNK_SIZE))
        
        prev_frame = None
        cuts = 0
        frame_interval = int(fps / VIDEO_FPS)
        
        for frame_idx in range(0, max_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, RESIZE_DIMS)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                if np.mean(diff) > 30:
                    cuts += 1
            prev_frame = gray
            
        cap.release()
        result = cuts / duration if duration > 0 else 0
        
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        logging.error(f"CPS计算失败 {video_path}: {str(e)}")
        return 0

@lru_cache(maxsize=100)
def calculate_mdi(video_path):
    """计算运动密度MDI（优化版本）"""
    try:
        cache_path = get_cache_path(video_path, "mdi")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        max_frames = min(total_frames, int(fps * CHUNK_SIZE))
        
        prev_frame = None
        motion_scores = []
        frame_interval = int(fps / VIDEO_FPS)
        
        for frame_idx in range(0, max_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, RESIZE_DIMS)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray
            
        cap.release()
        result = np.mean(motion_scores) if motion_scores else 0
        
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        logging.error(f"MDI计算失败 {video_path}: {str(e)}")
        return 0

@lru_cache(maxsize=100)
def calculate_sas(video_path, audio_path):
    """计算音画同步度SAS（优化版本）"""
    try:
        cache_path = get_cache_path(video_path, "sas")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if not audio_path or not os.path.exists(audio_path):
            return 0
            
        y, sr = librosa.load(audio_path, sr=AUDIO_SR, duration=CHUNK_SIZE)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        max_frames = min(total_frames, int(fps * CHUNK_SIZE))
        frame_interval = int(fps / VIDEO_FPS)
        
        scene_changes = []
        prev_frame = None
        
        for frame_idx in range(0, max_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, RESIZE_DIMS)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                if np.mean(diff) > 30:
                    scene_changes.append(frame_idx / fps)
            prev_frame = gray
            
        cap.release()
        
        if not scene_changes or not onset_times.size:
            return 0
            
        sync_count = 0
        threshold = 0.1
        
        for scene_time in scene_changes:
            time_diff = np.abs(onset_times - scene_time)
            if np.min(time_diff) < threshold:
                sync_count += 1
                
        result = sync_count / len(scene_changes)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        logging.error(f"SAS计算失败 {video_path}: {str(e)}")
        return 0

def normalize_score(score, score_range):
    """归一化分数"""
    min_val, max_val = score_range
    return np.clip((score - min_val) / (max_val - min_val) if max_val > min_val else 0, 0, 1)

def get_rhythm_class(nri):
    """获取节奏类型"""
    if nri < THRESHOLDS[0]:
        return 0  # Slow
    elif nri < THRESHOLDS[1]:
        return 1  # Moderate
    else:
        return 2  # Fast

def process_video_with_timeout(args):
    """使用线程的超时机制处理视频，包含重试机制"""
    video_path, temp_dir = args
    retries = 0
    
    while retries < MAX_RETRIES:
        result = None
        error = None
        
        def target():
            nonlocal result
            try:
                result = process_video(args)
            except Exception as e:
                nonlocal error
                error = e
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=VIDEO_TIMEOUT)
        
        if thread.is_alive():
            logging.warning(f"视频处理超时 (尝试 {retries + 1}/{MAX_RETRIES}): {video_path}")
            retries += 1
            if retries < MAX_RETRIES:
                logging.info(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                logging.error(f"视频处理最终超时: {video_path}")
                return None
                
        if error:
            logging.warning(f"视频处理错误 (尝试 {retries + 1}/{MAX_RETRIES}): {str(error)}")
            retries += 1
            if retries < MAX_RETRIES:
                logging.info(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                logging.error(f"视频处理最终失败: {str(error)}")
                return None
                
        return result
    
    return None

def process_video(args):
    """处理单个视频（优化版本）"""
    video_path, temp_dir = args
    try:
        # 检查缓存
        cache_path = get_cache_path(video_path, "final_result")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # 创建进度条
        with tqdm(total=5, desc=f"处理视频 {Path(video_path).name}", position=1, leave=False) as pbar:
            # 创建临时目录
            os.makedirs(temp_dir, exist_ok=True)
            
            # 使用超时控制处理每个步骤
            def run_with_timeout(func, *args, timeout=OPERATION_TIMEOUT):
                result = [None]
                def target():
                    result[0] = func(*args)
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout)
                
                if thread.is_alive():
                    raise TimeoutError(f"{func.__name__} 操作超时")
                return result[0]
            
            # 提取音频
            try:
                audio_path = run_with_timeout(extract_audio, video_path, Path(temp_dir))
                if not audio_path:
                    return None
                pbar.update(1)
            except TimeoutError as e:
                logging.error(f"音频提取超时: {video_path}")
                return None
            
            # 计算各项指标
            try:
                bpm = run_with_timeout(calculate_bpm, audio_path)
                pbar.update(1)
            except TimeoutError as e:
                logging.error(f"BPM计算超时: {video_path}")
                return None
            
            try:
                cps = run_with_timeout(calculate_cps, video_path)
                pbar.update(1)
            except TimeoutError as e:
                logging.error(f"CPS计算超时: {video_path}")
                return None
            
            try:
                mdi = run_with_timeout(calculate_mdi, video_path)
                pbar.update(1)
            except TimeoutError as e:
                logging.error(f"MDI计算超时: {video_path}")
                return None
            
            try:
                sas = run_with_timeout(calculate_sas, video_path, audio_path)
                pbar.update(1)
            except TimeoutError as e:
                logging.error(f"SAS计算超时: {video_path}")
                return None
        
        # 归一化
        bpm_norm = normalize_score(bpm, BPM_RANGE)
        cps_norm = normalize_score(cps, CPS_RANGE)
        mdi_norm = normalize_score(mdi, MDI_RANGE)
        sas_norm = sas  # SAS已经是0-1范围
        
        # 计算NRI
        nri = np.mean([bpm_norm, cps_norm, mdi_norm, sas_norm])
        rhythm_class = get_rhythm_class(nri)
        
        # 清理临时文件
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            logging.warning(f"临时文件清理失败: {str(e)}")
        
        result = {
            'video_file': Path(video_path).name,
            'BPM': bpm,
            'BPM_norm': bpm_norm,
            'CPS': cps,
            'CPS_norm': cps_norm,
            'MDI': mdi,
            'MDI_norm': mdi_norm,
            'SAS': sas,
            'SAS_norm': sas_norm,
            'NRI': nri,
            'rhythm_class': rhythm_class
        }
        
        # 缓存结果
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        
        logging.info(f"视频处理完成: {video_path}")
        return result
        
    except Exception as e:
        logging.error(f"处理视频失败 {video_path}: {str(e)}")
        return None
    finally:
        # 清理临时目录
        try:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
        except:
            pass

def save_results(results, output_csv):
    """保存结果到CSV文件"""
    try:
        df = pd.DataFrame(results)
        
        # 如果文件已存在，合并结果
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=['video_file'])
        
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logging.info(f"结果已保存至: {output_csv}")
        logging.info(f"当前共有处理结果: {len(df)} 个视频")
    except Exception as e:
        logging.error(f"保存结果失败: {str(e)}")

def process_batch(batch_files, batch_num, total_batches):
    """处理一批视频"""
    results = []
    with tqdm(total=len(batch_files), desc=f"批次 {batch_num}/{total_batches}", position=0) as batch_pbar:
        with ProcessPoolExecutor(max_workers=PROCESS_NUM) as executor:
            futures = []
            for video_file in batch_files:
                temp_dir = f"temp_{Path(video_file).stem}"
                future = executor.submit(process_video_with_timeout, (str(video_file), temp_dir))
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    if result:
                        results.append(result)
                except TimeoutError:
                    logging.error("处理超时")
                except Exception as e:
                    logging.error(f"处理批次时出错: {str(e)}")
                finally:
                    batch_pbar.update(1)
                    
    return results

def main():
    """主函数（优化版本）"""
    try:
        # 启用防休眠模式
        prevent_sleep()
        
        print("\n" + "="*50)
        print("视频节奏分析工具")
        print("="*50 + "\n")
        
        logging.info("开始处理视频...")
        
        # 获取视频文件列表
        input_path = Path(INPUT_DIR)
        video_files = sorted(
            input_path.glob("*.mp4"),
            key=lambda x: int(''.join(filter(str.isdigit, x.stem)))
        )
        
        if not video_files:
            logging.error(f"未找到MP4文件在目录: {INPUT_DIR}")
            return
            
        total_videos = len(video_files)
        logging.info(f"找到 {total_videos} 个视频文件")
        
        # 获取已处理的视频
        processed_videos = set()
        if os.path.exists(OUTPUT_CSV):
            existing_df = pd.read_csv(OUTPUT_CSV)
            processed_videos = set(existing_df['video_file'].tolist())
            
        # 过滤已处理的视频
        video_files = [f for f in video_files if f.name not in processed_videos]
        logging.info(f"需要处理 {len(video_files)} 个新视频")
        
        total_new_videos = len(video_files)
        if total_new_videos == 0:
            print("\n所有视频都已处理完成！")
            return
            
        # 显示总进度条
        with tqdm(total=total_new_videos, desc="总进度", position=0) as main_pbar:
            # 分批处理视频
            total_batches = (total_new_videos - 1) // BATCH_SIZE + 1
            
            for i in range(0, total_new_videos, BATCH_SIZE):
                batch = video_files[i:i + BATCH_SIZE]
                batch_num = i//BATCH_SIZE + 1
                
                results = process_batch(batch, batch_num, total_batches)
                
                # 保存结果
                if results:
                    save_results(results, OUTPUT_CSV)
                
                main_pbar.update(len(batch))
                
        print("\n" + "="*50)
        print(f"处理完成！共处理 {total_new_videos} 个视频")
        print(f"结果已保存至: {OUTPUT_CSV}")
        print("="*50 + "\n")
        
    except Exception as e:
        logging.error(f"处理失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # 恢复默认电源管理设置
        restore_sleep()

if __name__ == '__main__':
    main()
