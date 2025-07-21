import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    filename='人物视角_error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VideoAnalyzer:
    def __init__(self):
        # 初始化OpenCV检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    def extract_frames(self, video_path):
        """每秒抽取一帧"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % max(fps, 1) == 0:  # 每秒抽一帧，确保fps>=1
                frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

    def detect_objects(self, frames):
        """检测每一帧中的目标"""
        results = []
        for frame in frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(frame_gray, 1.1, 4)
            # 检测人体
            bodies = self.body_cascade.detectMultiScale(frame_gray, 1.1, 4)

            # 计算人脸区域
            face_area = sum(w * h for (x, y, w, h) in faces) if len(faces) > 0 else 0
            # 计算人体区域
            body_area = sum(w * h for (x, y, w, h) in bodies) if len(bodies) > 0 else 0

            # 根据人脸和人体的相对位置和大小判断视角
            frame_dict = {
                'hand_detected': False,  # 由于无法检测手，默认为False
                'face_forward': len(faces) > 0 and face_area / frame_area > 0.02,
                'person_detected': len(bodies) > 0,
                'bbox_area_ratio': body_area / frame_area if body_area > 0 else 0
            }

            # 如果人脸占比很大（超过10%），可能是第一视角自拍
            if frame_dict['face_forward'] and face_area / frame_area > 0.1:
                frame_dict['hand_detected'] = True

            results.append(frame_dict)

        return results

    def vote_pov(self, frame_results):
        """统计各个视角的票数"""
        counts = {'first': 0, 'second': 0, 'third': 0, 'none': 0}
        for result in frame_results:
            if result['hand_detected']:
                counts['first'] += 1
            elif result['face_forward'] and result['bbox_area_ratio'] >= 0.15:
                counts['second'] += 1
            elif result['person_detected'] and result['bbox_area_ratio'] < 0.15:
                counts['third'] += 1
            else:
                counts['none'] += 1
        return counts

    def decide_label(self, counts):
        """决定最终的视角标签"""
        total = sum(counts.values())
        ratios = {k: v / total if total > 0 else 0 for k, v in counts.items()}
        max_ratio = max(ratios.values())
        if ratios['first'] == max_ratio and counts['first'] > 0:
            return 0
        elif ratios['second'] == max_ratio and counts['second'] > 0:
            return 1
        elif ratios['third'] == max_ratio and counts['third'] > 0:
            return 2
        else:
            return 3

    def analyze_video(self, video_path):
        """分析单个视频"""
        try:
            frames = self.extract_frames(video_path)
            if not frames:
                logging.error(f"无法从视频中提取帧: {video_path}")
                return None

            frame_results = self.detect_objects(frames)
            counts = self.vote_pov(frame_results)
            total = sum(counts.values())
            result = {
                'video_id': Path(video_path).stem,
                'frames': len(frames),
                'total_hand': counts['first'],
                'total_face_fw': counts['second'],
                'total_farshot': counts['third'],
                'total_nobody': counts['none'],
                'ratio_first': counts['first'] / total if total > 0 else 0,
                'ratio_second': counts['second'] / total if total > 0 else 0,
                'ratio_third': counts['third'] / total if total > 0 else 0,
                'ratio_none': counts['none'] / total if total > 0 else 0,
                'person_pov_label': self.decide_label(counts)
            }
            return result

        except Exception as e:
            logging.error(f"处理视频 {video_path} 时出错: {str(e)}")
            return None

def process_video_worker(video_path):
    # 每个进程单独实例化Analyzer，避免OpenCV多进程冲突
    analyzer = VideoAnalyzer()
    return analyzer.analyze_video(video_path)

def main():
    input_dir = Path(r"xxxxxxxxxxxxxxxxxxx")
    output_file = Path(r"xxxxxxxxxxxxxxxxxxx")
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return
    video_files = list(input_dir.glob('*.mp4'))
    if not video_files:
        print(f"未找到MP4视频文件在: {input_dir}")
        return

    # 并行处理
    results = []
    max_workers = None  # None表示自动按CPU核心数；可改成8、12等
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # tqdm进度条配合as_completed使用
        futures = [executor.submit(process_video_worker, str(path)) for path in video_files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="并行处理视频"):
            res = fut.result()
            if res:
                results.append(res)

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"分析完成，结果已保存至: {output_file}")
        print(f"共处理 {len(results)}/{len(video_files)} 个视频文件")
    else:
        print("没有成功处理任何视频")

if __name__ == "__main__":
    main()
