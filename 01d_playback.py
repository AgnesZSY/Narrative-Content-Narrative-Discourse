import subprocess
import json
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_ffprobe_dims(video_path):
    """使用ffprobe获取视频尺寸和VR信息"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # 查找视频流
        video_stream = next(
            (stream for stream in data['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        if video_stream:
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # 检查是否为VR视频
            is_vr = False
            if 'tags' in video_stream:
                tags_str = json.dumps(video_stream['tags']).lower()
                is_vr = 'equirectangular' in tags_str
            
            # 如果分辨率符合VR标准也认为是VR视频
            if width >= 3000 and height >= 1500:
                is_vr = True
                
            return width, height, is_vr
            
    except Exception as e:
        logging.error(f"FFprobe error for {video_path}: {str(e)}")
        return None, None, False

def get_cv2_dims(video_path):
    """使用OpenCV获取视频尺寸"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return width, height
        
    except Exception as e:
        logging.error(f"OpenCV error for {video_path}: {str(e)}")
        return None, None

def get_dims(video_path):
    """获取视频尺寸，优先使用ffprobe"""
    # 首先尝试使用ffprobe
    width, height, is_vr = get_ffprobe_dims(video_path)
    
    # 如果ffprobe失败，使用OpenCV
    if width is None or height is None:
        width, height = get_cv2_dims(video_path)
        is_vr = False if width is None else (width >= 3000 and height >= 1500)
    
    return width, height, is_vr

def decide_label(width, height, is_vr):
    """决定视频的播放视角标签"""
    if width is None or height is None:
        return None
        
    if is_vr:
        return 3
        
    ratio = round(width / height, 4)
    
    if ratio <= 0.80:
        return 0  # 竖屏
    elif ratio >= 1.25:
        return 1  # 横屏
    else:
        return 2  # 方形

def main():
    # 设置输入输出路径
    video_dir = Path(r"xxxxxxxxxxxxxxxxxxx")
    output_file = Path(r"xxxxxxxxxxxxxxxxxxx")
    
    # 获取所有视频文件
    video_files = []
    for ext in ['.mp4', '.mov', '.mkv']:
        video_files.extend(video_dir.glob(f'*{ext}'))
    
    results = []
    
    # 使用tqdm显示进度
    for video_path in tqdm(video_files, desc="处理视频"):
        video_id = video_path.stem
        width, height, is_vr = get_dims(video_path)
        
        if width and height:
            ratio = round(width / height, 4)
            presentation_view = decide_label(width, height, is_vr)
            
            results.append({
                'video_id': video_id,
                'width': width,
                'height': height,
                'ratio': ratio,
                'presentation_view': presentation_view
            })
        else:
            logging.error(f"无法处理视频: {video_path}")
    
    # 创建DataFrame并保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"分析完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 