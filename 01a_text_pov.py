import os
import json
import jieba
import pandas as pd
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import subprocess
import time
import traceback
import argparse
import re
import numpy as np
import codecs
import multiprocessing
from functools import partial
import http.client
import openpyxl
import gc

def process_video_worker(video_path):
    """
    工作进程函数，为每个进程创建新的分析器实例
    """
    try:
        analyzer = NarrativePersonAnalyzer()
        return analyzer.process_video(str(video_path))
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {str(e)}")
        return None

class NarrativePersonAnalyzer:
    def __init__(self):
        """
        初始化分析器
        """
        # 初始化jieba分词器
        jieba.initialize()
        # 在Windows系统下不启用并行模式
        if os.name != 'nt':  # 只在非Windows系统启用并行
            jieba.enable_parallel(multiprocessing.cpu_count())
        
        # 语音识别服务配置 - 使用中转站URL
        self.api_endpoint = "tts88.top"
        
        # 预编译正则表达式以提高性能
        self.space_pattern = re.compile(r'\s+')
        self.punct_pattern = re.compile(r'[，。！？；：]')
        self.multi_punct_pattern = re.compile(r'([，。！？])\s*([，。！？])')
        
        # 设置人称代词集合
        self.first_person_set = set(['我', '我们', '咱', '咱们', '俺', '俺们'])
        self.second_person_set = set(['你', '您', '你们', '您们'])
        self.third_person_set = set(['他', '她', '它', '他们', '她们', '它们'])
        
        # 批处理设置
        self.batch_size = 50  # 每批处理的视频数量
        self.save_interval = 30  # 每30分钟保存一次进度
        self.last_save_time = time.time()
        self.checkpoint_file = "xxxxxxxxxxxxxxxxxxx"
        self.temp_result_file = "xxxxxxxxxxxxxxxxxxx"

    def load_checkpoint(self):
        """
        加载断点续传信息
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'last_processed_id': 0, 'processed_videos': []}

    def save_checkpoint(self, last_id, processed_videos):
        """
        保存断点续传信息
        """
        checkpoint_data = {
            'last_processed_id': last_id,
            'processed_videos': processed_videos
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

    def save_temp_results(self, results):
        """
        保存临时结果到Excel文件
        """
        df = pd.DataFrame(results)
        df.to_excel(self.temp_result_file, index=False)
        print(f"临时结果已保存至: {self.temp_result_file}")

    def process_directory(self, video_dir, excel_file, output_file):
        """
        处理目录中的视频文件，支持断点续传和定期保存
        """
        try:
            start_time = time.time()
            last_save_time = start_time
            
            # 初始化结果列表
            all_results = []
            
            # 读取Excel文件
            print(f"正在读取Excel文件: {excel_file}")
            df = pd.read_excel(excel_file)
            
            # 打印Excel文件的列名，用于调试
            print("\nExcel文件的列名:")
            print(df.columns.tolist())
            
            if 'filename' not in df.columns:
                raise ValueError("Excel文件中没有找到'filename'列")
            
            video_dir = Path(video_dir)
            
            # 加载断点续传信息
            checkpoint = self.load_checkpoint()
            processed_videos = set(checkpoint['processed_videos'])
            
            # 获取所有需要处理的视频文件
            videos_to_process = []
            for _, row in df.iterrows():
                filename = str(row['filename'])
                video_path = video_dir / filename
                if video_path.exists():
                    videos_to_process.append((video_path, filename))
                else:
                    print(f"警告：找不到视频文件 {video_path}")
                    # 记录找不到的文件
                    all_results.append({
                        'video_id': Path(filename).stem,
                        'filename': filename,
                        'text': '无',
                        'word_count': 0,
                        'first_person_count': 0,
                        'second_person_count': 0,
                        'third_person_count': 0,
                        'first_person_ratio': 0,
                        'second_person_ratio': 0,
                        'third_person_ratio': 0,
                        'perspective': '无',
                        'confidence': 0,
                        'processing_time': 0,
                        'status': '文件不存在'
                    })
            
            # 过滤掉已处理的视频
            videos_to_process = [(path, name) for path, name in videos_to_process 
                               if str(path) not in processed_videos]
            
            # 如果存在临时结果文件，加载它
            if os.path.exists(self.temp_result_file):
                print(f"\n发现临时结果文件: {self.temp_result_file}")
                temp_df = pd.read_excel(self.temp_result_file)
                all_results = temp_df.to_dict('records')
                print(f"已加载 {len(all_results)} 条临时结果")
            
            # 检查是否有视频需要处理
            if len(videos_to_process) == 0:
                print("\n没有新的视频需要处理")
                if all_results:
                    print("保存已处理的结果...")
                    final_df = pd.DataFrame(all_results)
                    final_df.to_excel(output_file, index=False)
                    print(f"结果已保存至: {output_file}")
                return
            
            print(f"\n共有 {len(videos_to_process)} 个视频需要处理")
            
            # 分批处理
            batch_size = min(20, len(videos_to_process))  # 限制批次大小
            batches = [videos_to_process[i:i + batch_size] 
                      for i in range(0, len(videos_to_process), batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                print(f"\n处理第 {batch_idx + 1}/{len(batches)} 批")
                batch_results = []
                
                # 创建进程池
                num_processes = min(4, multiprocessing.cpu_count() - 1)  # 限制最大进程数为4
                with multiprocessing.Pool(processes=num_processes) as pool:
                    video_paths = [path for path, _ in batch]
                    
                    if not video_paths:
                        print("当前批次没有找到任何视频文件，跳过处理")
                        continue
                    
                    print(f"\n开始处理 {len(video_paths)} 个视频文件...")
                    
                    # 使用tqdm显示进度
                    results = list(tqdm(
                        pool.imap(process_video_worker, video_paths),
                        total=len(video_paths),
                        desc="处理视频文件"
                    ))
                    
                    # 添加文件名到结果中
                    for result, (_, filename) in zip(results, batch):
                        if result is not None:
                            result['filename'] = filename
                            batch_results.append(result)
                
                # 更新结果和保存进度
                all_results.extend(batch_results)
                processed_videos.update([str(path) for path, _ in batch])
                
                # 定期保存进度
                current_time = time.time()
                if current_time - last_save_time >= self.save_interval * 60:
                    print("\n保存进度...")
                    self.save_checkpoint(0, list(processed_videos))
                    self.save_temp_results(all_results)
                    last_save_time = current_time
                    print("进度保存完成")
                    
                # 清理内存
                gc.collect()
            
            # 保存最终结果
            if all_results:
                final_df = pd.DataFrame(all_results)
                final_df.to_excel(output_file, index=False)
                print(f"\n最终结果已保存至: {output_file}")
                
                # 显示处理统计
                success_count = len([r for r in all_results if r.get('status') == '成功'])
                total_count = len(df)  # 使用Excel文件中的总数
                if total_count > 0:
                    success_rate = (success_count / total_count) * 100
                    print(f"\n处理成功率: {success_rate:.2f}%")
                    print(f"成功处理: {success_count} 个")
                    print(f"总视频数: {total_count} 个")
                    
                    # 显示各种状态的统计
                    status_counts = {}
                    for result in all_results:
                        status = result.get('status', '未知')
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    print("\n处理状态统计:")
                    for status, count in status_counts.items():
                        print(f"{status}: {count} 个 ({count/total_count*100:.2f}%)")
            
            # 清理临时文件
            if os.path.exists(self.temp_result_file):
                os.remove(self.temp_result_file)
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                
            # 显示总处理时间
            total_time = time.time() - start_time
            print(f"\n总处理时间: {total_time:.2f} 秒")
            if total_count > 0:
                print(f"平均每个视频处理时间: {total_time/total_count:.2f} 秒")
                
        except Exception as e:
            print(f"处理目录时出错: {str(e)}")
            traceback.print_exc()
            # 发生错误时也保存进度
            if 'processed_videos' in locals() and 'all_results' in locals():
                self.save_checkpoint(0, list(processed_videos))
                self.save_temp_results(all_results)

    def extract_audio(self, video_path):
        """
        从视频文件中提取音频，优化参数设置和音频质量
        """
        try:
            # 生成临时音频文件名
            temp_audio = f"temp_{Path(video_path).stem}.wav"
            
            # 智能检测 FFmpeg 路径
            ffmpeg_path = None
            possible_paths = [
                "D:\\download\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe",
                "C:\\ffmpeg\\bin\\ffmpeg.exe",
                "ffmpeg"  # 如果在系统 PATH 中
            ]
            
            for path in possible_paths:
                try:
                    if path == "ffmpeg":
                        # 检查是否在系统 PATH 中
                        result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True)
                        if result.returncode == 0:
                            ffmpeg_path = "ffmpeg"
                            break
                    elif os.path.exists(path):
                        ffmpeg_path = path
                        break
                except Exception:
                    continue
            
            if not ffmpeg_path:
                print("错误：找不到 FFmpeg。请确保已安装 FFmpeg 并添加到系统 PATH，或在正确的位置。")
                print("您可以从 https://github.com/BtbN/FFmpeg-Builds/releases 下载 FFmpeg。")
                return None
                
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"错误：视频文件不存在: {video_path}")
                return None
                
            # 使用ffprobe检查视频信息
            probe_cmd = [
                ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe"),
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type,duration,channels,sample_rate",
                "-of", "json",
                str(video_path)
            ]
            
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                probe_data = json.loads(probe_result.stdout)
                
                # 检查是否有音频流
                if not probe_data.get("streams"):
                    print(f"警告：视频 {video_path} 没有音频轨道")
                    return None
                    
                # 检查音频时长
                duration = float(probe_data["streams"][0].get("duration", 0))
                if duration < 0.5:
                    print(f"警告：视频 {video_path} 音频时长过短 ({duration:.2f}秒)")
                    return None
                    
                # 获取音频通道数和采样率
                channels = int(probe_data["streams"][0].get("channels", 1))
                sample_rate = int(probe_data["streams"][0].get("sample_rate", 16000))
                
            except Exception as e:
                print(f"警告：无法获取视频信息: {str(e)}")
                # 继续处理，使用默认参数
            
            print(f"正在从视频提取音频: {video_path}")
            
            # 构建优化的FFmpeg命令
            ffmpeg_cmd = [
                ffmpeg_path,
                "-i", str(video_path),
                "-vn",  # 不处理视频
                "-acodec", "pcm_s16le",  # 16位PCM编码
                "-ar", "16000",  # 16kHz采样率
                "-ac", "1",  # 单声道
                # 音频处理滤镜链
                "-af",
                "volume=2.0," +  # 增加音量
                "highpass=f=50," +  # 高通滤波器
                "lowpass=f=8000," +  # 低通滤波器
                "afftdn=nf=-25," +  # 降噪
                "acompressor=threshold=-12dB:ratio=3:attack=50:release=500," +  # 动态范围压缩
                "loudnorm=I=-16:LRA=11:TP=-1.5," +  # 音量标准化
                "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:" +  # 移除开头静音
                "stop_periods=1:stop_duration=0.1:stop_threshold=-50dB",  # 移除结尾静音
                "-threads", str(max(1, multiprocessing.cpu_count() - 1)),  # 使用多线程
                "-y",  # 覆盖已存在的文件
                str(temp_audio)
            ]
            
            # 执行FFmpeg命令
            result = subprocess.run(ffmpeg_cmd, capture_output=True)
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"错误：音频提取失败: {error_msg}")
                # 尝试使用备用参数重新提取
                backup_cmd = [
                    ffmpeg_path,
                    "-i", str(video_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-af", "volume=2.0,highpass=f=50,lowpass=f=8000",
                    "-y",
                    str(temp_audio)
                ]
                print("尝试使用备用参数重新提取音频...")
                result = subprocess.run(backup_cmd, capture_output=True)
                if result.returncode != 0:
                    print("备用提取方法也失败")
                    return None
                
            # 检查生成的音频文件
            if os.path.exists(temp_audio):
                audio_size = os.path.getsize(temp_audio)
                if audio_size < 1024:  # 小于1KB
                    print(f"警告：生成的音频文件 {temp_audio} 大小过小 ({audio_size/1024:.2f}KB)")
                    os.remove(temp_audio)
                    return None
                    
                # 验证音频质量
                try:
                    audio_data, sample_rate = sf.read(temp_audio)
                    duration = len(audio_data) / sample_rate
                    if duration < 0.5:
                        print(f"警告：生成的音频时长过短: {duration:.2f}秒")
                        os.remove(temp_audio)
                        return None
                        
                    # 检查音频是否全是静音
                    rms = np.sqrt(np.mean(np.square(audio_data)))
                    if rms < 0.01:
                        print(f"警告：音频信号过弱或全是静音 (RMS: {rms:.6f})")
                        os.remove(temp_audio)
                        return None
                        
                except Exception as e:
                    print(f"警告：无法验证音频质量: {str(e)}")
                    # 继续使用该文件
                
                return temp_audio
            else:
                print(f"错误：音频提取失败，未生成文件 {temp_audio}")
                return None
                
        except Exception as e:
            print(f"错误：音频提取过程出现异常: {str(e)}")
            traceback.print_exc()
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return None

    def transcribe_audio(self, audio_path):
        """
        将音频文件转写为文本，使用中转站服务，增强错误处理和重试机制
        """
        max_retries = 1  # 只尝试一次
        retry_delay = 3  # 初始重试延迟
        backoff_factor = 1.5  # 退避因子
        min_response_length = 5  # 最小有效响应长度
        
        # 检查音频文件
        if not os.path.exists(audio_path):
            print(f"错误：音频文件不存在: {audio_path}")
            return "无"
            
        # 检查音频文件大小
        file_size = os.path.getsize(audio_path)
        if file_size < 1024:  # 小于1KB
            print(f"错误：音频文件过小: {file_size} 字节")
            return "无"
            
        # 检查音频格式和质量
        try:
            audio_data, sample_rate = sf.read(audio_path)
            duration = len(audio_data) / sample_rate
            if duration < 0.5:  # 音频太短
                print(f"错误：音频时长过短: {duration:.2f} 秒")
                return "无"
        except Exception as e:
            print(f"错误：无法读取音频文件: {str(e)}")
            return "无"

        for retry in range(max_retries):
            try:
                if retry > 0:
                    current_delay = retry_delay * (backoff_factor ** (retry - 1))
                    print(f"第 {retry + 1} 次重试转写音频: {audio_path}")
                    print(f"等待 {current_delay:.1f} 秒...")
                    time.sleep(current_delay)
                else:
                    print(f"开始转写音频: {audio_path}")
                
                # 读取音频文件
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                
                # 设置请求头
                headers = {
                    'Content-Type': 'xxxxxxxxxxxxxxxxxxx',
                    'Authorization': 'xxxxxxxxxxxxxxxxxxx',
                    'Connection': 'xxxxxxxxxxxxxxxxxxx',
                    'Keep-Alive': 'xxxxxxxxxxxxxxxxxxx',  # 增加超时时间
                    'Accept-Encoding': 'xxxxxxxxxxxxxxxxxxx',  # 支持压缩
                    'User-Agent': 'xxxxxxxxxxxxxxxxxxx'  # 添加User-Agent
                }
                
                # 创建 HTTPS 连接
                conn = http.client.HTTPSConnection("tts88.top", timeout=300)  # 增加超时时间
                
                # 设置API端点和参数
                endpoint = "xxxxxxxxxxxxxxxxxxx"
                params = "xxxxxxxxxxxxxxxxxxx"
                
                # 发送请求
                print(f"发送API请求，音频大小: {len(audio_data)} 字节")
                conn.request("POST", endpoint + params, audio_data, headers)
                
                # 获取响应
                response = conn.getresponse()
                print(f"API响应状态码: {response.status}")
                
                if response.status == 200:
                    response_data = response.read().decode('utf-8')
                    print(f"API响应内容: {response_data[:200]}...")
                    
                    # 解析JSON响应
                    try:
                        result = json.loads(response_data)
                        if 'DisplayText' in result and result['DisplayText'].strip():
                            text = result['DisplayText'].strip()
                            if len(text) >= min_response_length:
                                return text
                            else:
                                print(f"警告：API返回的文本过短: {text}")
                                return "无"
                        else:
                            print(f"警告：API响应缺少有效文本")
                            return "无"
                    except json.JSONDecodeError as e:
                        print(f"错误：JSON解析失败: {str(e)}")
                        print(f"原始响应内容: {response_data}")
                        return "无"
                elif response.status == 429:  # Too Many Requests
                    print("警告：API请求过于频繁，等待更长时间...")
                    time.sleep(retry_delay * 2)
                    continue
                else:
                    print(f"错误：API请求失败，状态码: {response.status}")
                    print(f"错误响应内容: {response.read().decode('utf-8')}")
                    return "无"
                
                conn.close()
                
            except Exception as e:
                print(f"错误：音频转写过程出现异常: {str(e)}")
                traceback.print_exc()
                return "无"
            finally:
                try:
                    conn.close()
                except:
                    pass
        
        return "无"

    def fix_punctuation(self, text):
        """
        修复文本中的标点符号和断句问题
        """
        if pd.isna(text) or not text:
            return text
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 修复基本的标点符号错误
        text = re.sub(r'，了', '了', text)  # 移除"了"前的逗号
        text = re.sub(r'的，', '的', text)  # 移除"的"后的逗号
        text = re.sub(r'([，。！？])\s*([，。！？])', r'\2', text)  # 移除重复标点
        text = re.sub(r'，+', '，', text)  # 合并多个逗号
        text = re.sub(r'。+', '。', text)  # 合并多个句号
        text = re.sub(r'！+', '！', text)  # 合并多个感叹号
        text = re.sub(r'？+', '？', text)  # 合并多个问号
        
        # 确保句子以适当的标点结尾
        if not text.endswith(('。', '！', '？')):
            text = text.rstrip('，') + '。'
        
        return text

    def add_punctuation(self, text):
        """
        为识别的文本智能添加标点符号，模拟自然说话的语气和停顿
        """
        if not text or not text.strip():
            return text
            
        # 清理多余的空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 保留原有的标点符号
        sentences = []
        current_sentence = ""
        words = jieba.lcut(text)
        
        # 定义语言特征词组
        natural_pause_words = ['是', '就是', '也是', '还是', '都是', '确实', '当然', '其实']  # 自然停顿词
        end_words = ['了', '的', '呢', '吧', '啊', '呀', '哦', '嘛', '呐', '哩', '啦']  # 句尾语气词
        question_words = ['吗', '呢', '怎么', '为什么', '什么', '哪里', '多少', '几时', '怎样', '如何', '几个', '多久']  # 疑问词
        exclaim_words = ['啊', '哇', '太', '真', '好', '棒', '真是', '太棒', '真好', '太好', '多么', '非常', '特别']  # 感叹词
        conjunction_words = ['然后', '接着', '所以', '但是', '不过', '而且', '另外', '同时', '因为', '虽然', 
                           '尽管', '况且', '并且', '而是', '要是', '如果', '假如', '即使', '无论', '要么']  # 连接词
        topic_words = ['说到', '谈到', '至于', '关于', '对于', '提到', '说起', '讲到', '问到']  # 话题转换词
        scene_words = ['这里', '那里', '此时', '此刻', '眼前', '远处', '身边', '周围', '附近']  # 场景描写词
        emotion_words = ['开心', '难过', '激动', '兴奋', '感动', '惊讶', '欣喜', '欢乐', '幸福']  # 情感词
        
        # 定义不应该断句的词组
        no_break_patterns = [
            r'[\d一二三四五六七八九十百千万亿]+[年月日时分秒]',  # 时间
            r'[\d一二三四五六七八九十百千万亿]+[个只条张份部台次]',  # 量词
            r'[东南西北中][边部方面侧]',  # 方位
            r'[这那][个些样种届批]',  # 指示词
            r'[小大中][时候段]',  # 时间段
            r'[春夏秋冬][天季日]',  # 季节
            r'[早中晚][上午饭]',  # 时段
            r'[人物景][来去走]',  # 动作连接
            r'[的地得][话]',  # 的话
            r'[正在][用]',  # 正在用
            r'[河湖海][里边内外]',  # 水域位置
        ]
        
        def should_not_break(text):
            """检查是否不应该在此处断句"""
            for pattern in no_break_patterns:
                if re.search(pattern, text):
                    return True
            return False
        
        def is_complete_clause(text):
            """检查是否是完整的从句"""
            # 简单的从句判断规则
            if len(text) < 5:  # 过短的不算完整从句
                return False
            if any(word in text for word in conjunction_words):  # 包含连接词可能是从句
                return True
            if any(word in text[-5:] for word in end_words):  # 以语气词结尾可能是从句
                return True
        
        if result.returncode != 0:
            print(f"提取音频失败: {result.stderr}")
            return False
        
        def get_next_words(words, current_index, count=3):
            """获取后续几个词"""
            return ' '.join(words[current_index+1:current_index+1+count]) if current_index+1 < len(words) else ""
        
        def get_prev_words(words, current_index, count=3):
            """获取前面几个词"""
            start = max(0, current_index-count)
            return ' '.join(words[start:current_index]) if current_index > 0 else ""
        
        for i, word in enumerate(words):
            # 如果当前词已经包含标点，直接添加
            if any(p in word for p in '。，！？'):
                current_sentence += word
                sentences.append(current_sentence.strip())
                current_sentence = ""
                continue
                
            # 检查是否不应该断句
            next_word = words[i + 1] if i + 1 < len(words) else ""
            if should_not_break(current_sentence + word) or should_not_break(word + next_word):
                current_sentence += word
                continue
                
            current_sentence += word
            
            # 检查是否需要添加标点
            should_add_punct = False
            punct_to_add = '，'  # 默认使用逗号作为停顿
            
            # 获取上下文
            prev_words = get_prev_words(words, i)
            next_words = get_next_words(words, i)
            
            # 智能判断标点
            if word in end_words and len(current_sentence) > 8:
                if is_complete_clause(current_sentence):
                    should_add_punct = True
                    punct_to_add = '。'
            elif word in question_words:
                # 检查是否是完整问句
                if '怎么会' in (current_sentence + next_words):  # 特殊处理"怎么会"的情况
                    if len(current_sentence) > 15 or any(end_word in next_words for end_word in end_words):
                        should_add_punct = True
                        punct_to_add = '？'
                elif any(end_word in next_words for end_word in end_words) or i+1 >= len(words):
                    should_add_punct = True
                    punct_to_add = '？'
                elif len(current_sentence) > 12:  # 较长的问句
                    should_add_punct = True
                    punct_to_add = '？'
            elif word in exclaim_words:
                # 检查是否是感叹语气
                if not any(w in next_word for w in end_words) and len(current_sentence) > 8:
                    should_add_punct = True
                    punct_to_add = '！'
            elif word in conjunction_words:
                # 连接词前的停顿
                if len(current_sentence) > 10 and is_complete_clause(current_sentence):
                    should_add_punct = True
                    punct_to_add = '，'
            elif word in natural_pause_words:
                # 自然停顿
                if len(current_sentence) > 8 and not any(w in next_word for w in end_words):
                    should_add_punct = True
                    punct_to_add = '，'
            elif word in topic_words or word in scene_words:
                # 话题转换或场景转换
                should_add_punct = True
                punct_to_add = '，'
            elif word in emotion_words and len(current_sentence) > 8:
                # 情感表达的停顿
                should_add_punct = True
                punct_to_add = '，'
            
            # 句子长度控制（自然说话的语气）
            if len(current_sentence) > 18 and not should_not_break(current_sentence):
                should_add_punct = True
                # 根据上下文和语境选择标点
                if '怎么会' in current_sentence:  # 特殊处理"怎么会"的情况
                    punct_to_add = '？'
                elif any(w in current_sentence[-8:] for w in question_words):
                    punct_to_add = '？'
                elif any(w in current_sentence[-8:] for w in exclaim_words):
                    punct_to_add = '！'
                elif any(w in current_sentence[-8:] for w in end_words):
                    punct_to_add = '。'
                elif is_complete_clause(current_sentence):
                    punct_to_add = '。'
                else:
                    punct_to_add = '，'
            
            # 如果需要添加标点或到达句子末尾
            if should_add_punct or i == len(words) - 1:
                if not current_sentence.endswith(('。', '！', '？', '，')):
                    current_sentence += punct_to_add
                
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 合并句子
        result = ''.join(sentences)
        
        # 后处理：修正标点
        result = re.sub(r'([，。！？])\s*([，。！？])', r'\2', result)  # 移除重复标点
        result = re.sub(r'，+', '，', result)  # 合并多个逗号
        result = re.sub(r'。+', '。', result)  # 合并多个句号
        result = re.sub(r'！+', '！', result)  # 合并多个感叹号
        result = re.sub(r'？+', '？', result)  # 合并多个问号
        result = re.sub(r'，。', '。', result)  # 修正逗号+句号
        result = re.sub(r'，([！？])', r'\1', result)  # 修正逗号+感叹号/问号
        result = re.sub(r'([。！？])，', r'\1', result)  # 修正句末标点后的逗号
        result = re.sub(r'([。！？])[，。！？]+', r'\1', result)  # 修正句末多余标点
        
        # 确保句子以适当的标点结尾
        if result and not result[-1] in '。！？':
            if '怎么会' in result[-15:]:  # 特殊处理"怎么会"的情况
                result += '？'
            elif any(w in result[-10:] for w in question_words):
                result += '？'
            elif any(w in result[-10:] for w in exclaim_words):
                result += '！'
            else:
                result += '。'
        
        print(f"标点符号处理前: {text[:100]}...")
        print(f"标点符号处理后: {result[:100]}...")
        
        return result

    def clean_and_format_text(self, text, max_length=100):
        """
        清理和格式化文本用于显示
        从show_punctuation_comparison.py合并过来的功能
        """
        if not text:
            return ""
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 如果文本太长，截断并添加省略号
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text

    def extract_original_text(self, processed_text):
        """
        从处理后的文本中提取原始文本（移除标点符号）
        从show_punctuation_comparison.py合并过来的功能
        """
        if not processed_text:
            return ""
        
        # 移除标点符号
        original = re.sub(r'[，。！？；：]', '', processed_text)
        # 移除多余空格
        original = re.sub(r'\s+', ' ', original.strip())
        
        return original

    def show_punctuation_comparison(self, results_df):
        """
        显示标点符号处理效果对比
        从show_punctuation_comparison.py合并过来的功能
        """
        print("=" * 80)
        print("旅游短视频智能标点符号处理效果对比")
        print("=" * 80)
        
        print(f"\n处理了 {len(results_df)} 个视频文件\n")
        
        for index, row in results_df.iterrows():
            video_id = row['video_id']
            filename = row['filename']
            processed_text = row['text']
            
            # 提取原始文本
            original_text = self.extract_original_text(processed_text)
            
            print(f"📹 视频 {video_id} ({filename})")
            print("-" * 60)
            
            print("🔸 原始文本（无标点）：")
            print(f"   {self.clean_and_format_text(original_text, 120)}")
            print()
            
            print("🔹 处理后（智能标点）：")
            print(f"   {self.clean_and_format_text(processed_text, 120)}")
            print()
            
            # 分析改进效果
            punct_count = len(re.findall(r'[，。！？；：]', processed_text))
            print(f"📊 标点符号统计：")
            print(f"   - 逗号：{processed_text.count('，')} 个")
            print(f"   - 句号：{processed_text.count('。')} 个") 
            print(f"   - 感叹号：{processed_text.count('！')} 个")
            print(f"   - 问号：{processed_text.count('？')} 个")
            print(f"   - 总计：{punct_count} 个标点符号")
            print()
            
            # 分析人称代词统计
            print(f"👤 人称代词分析：")
            print(f"   - 第一人称：{row['cnt_1st']} 个 ({row['ratio_1st']:.1%})")
            print(f"   - 第二人称：{row['cnt_2nd']} 个 ({row['ratio_2nd']:.1%})")
            print(f"   - 第三人称：{row['cnt_3rd']} 个 ({row['ratio_3rd']:.1%})")
            print(f"   - 零人称：{row['cnt_0th']} 个 ({row['ratio_0th']:.1%})")
            print(f"   - 主要人称类型：{row['main_person']}")
            print()
            
            print("=" * 80)
            print()

    def analyze_text(self, text):
        """
        分析文本内容，识别人称视角，使用优化的方法
        """
        try:
            # 使用缓存的正则表达式清理文本
            text = self.space_pattern.sub(' ', text.strip())
            
            # 分词（已启用并行处理）
            words = list(jieba.cut(text))
            
            # 使用集合操作统计人称代词
            first_person = sum(1 for w in words if w in self.first_person_set)
            second_person = sum(1 for w in words if w in self.second_person_set)
            third_person = sum(1 for w in words if w in self.third_person_set)
            
            # 计算比例
            total_pronouns = first_person + second_person + third_person
            if total_pronouns == 0:
                total_pronouns = 1  # 避免除以零
                
            first_person_ratio = first_person / total_pronouns if total_pronouns > 0 else 0
            second_person_ratio = second_person / total_pronouns if total_pronouns > 0 else 0
            third_person_ratio = third_person / total_pronouns if total_pronouns > 0 else 0
            
            # 使用max函数一次性找出最大比例
            ratios = [
                (first_person_ratio, "第一人称"),
                (second_person_ratio, "第二人称"),
                (third_person_ratio, "第三人称")
            ]
            max_ratio, perspective = max(ratios, key=lambda x: x[0]) if max(ratios, key=lambda x: x[0])[0] > 0 else (0, "未检测到明显人称")
                
            # 返回分析结果
            return {
                "text": text,
                "word_count": len(words),
                "first_person_count": first_person,
                "second_person_count": second_person,
                "third_person_count": third_person,
                "first_person_ratio": round(first_person_ratio, 3),
                "second_person_ratio": round(second_person_ratio, 3),
                "third_person_ratio": round(third_person_ratio, 3),
                "perspective": perspective,
                "confidence": round(max_ratio, 3)
            }
            
        except Exception as e:
            print(f"分析文本时出错: {str(e)}")
            traceback.print_exc()
            return None

    def process_video(self, video_path):
        """
        处理单个视频文件
        """
        try:
            start_time = time.time()
            video_id = Path(video_path).stem
            
            # 提取音频
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                print(f"警告：视频 {video_path} 音频提取失败")
                return {
                    'video_id': video_id,
                    'filename': Path(video_path).name,
                    'text': '无',
                    'word_count': 0,
                    'first_person_count': 0,
                    'second_person_count': 0,
                    'third_person_count': 0,
                    'first_person_ratio': 0,
                    'second_person_ratio': 0,
                    'third_person_ratio': 0,
                    'perspective': '无',
                    'confidence': 0,
                    'processing_time': time.time() - start_time,
                    'status': '音频提取失败'
                }
            
            # 转写音频
            text = self.transcribe_audio(audio_path)
            
            # 删除临时音频文件
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                print(f"警告：删除临时音频文件失败: {str(e)}")
            
            # 如果转写结果为"无"，返回空结果
            if text == "无":
                return {
                    'video_id': video_id,
                    'filename': Path(video_path).name,
                    'text': '无',
                    'word_count': 0,
                    'first_person_count': 0,
                    'second_person_count': 0,
                    'third_person_count': 0,
                    'first_person_ratio': 0,
                    'second_person_ratio': 0,
                    'third_person_ratio': 0,
                    'perspective': '无',
                    'confidence': 0,
                    'processing_time': time.time() - start_time,
                    'status': '无有效文本'
                }
            
            # 分词和统计
            words = list(jieba.cut(text))
            word_count = len(words)
            
            # 统计人称代词
            first_person_count = sum(1 for word in words if word in self.first_person_set)
            second_person_count = sum(1 for word in words if word in self.second_person_set)
            third_person_count = sum(1 for word in words if word in self.third_person_set)
            
            # 计算比率
            total_pronouns = first_person_count + second_person_count + third_person_count
            first_person_ratio = first_person_count / total_pronouns if total_pronouns > 0 else 0
            second_person_ratio = second_person_count / total_pronouns if total_pronouns > 0 else 0
            third_person_ratio = third_person_count / total_pronouns if total_pronouns > 0 else 0
            
            # 判断主要使用的人称视角
            if total_pronouns == 0:
                perspective = "无人称"
            else:
                ratios = [
                    (first_person_ratio, "第一人称"),
                    (second_person_ratio, "第二人称"),
                    (third_person_ratio, "第三人称")
                ]
                perspective = max(ratios, key=lambda x: x[0])[1]
            
            processing_time = time.time() - start_time
            
            return {
                'video_id': video_id,
                'filename': Path(video_path).name,
                'text': text,
                'word_count': word_count,
                'first_person_count': first_person_count,
                'second_person_count': second_person_count,
                'third_person_count': third_person_count,
                'first_person_ratio': first_person_ratio,
                'second_person_ratio': second_person_ratio,
                'third_person_ratio': third_person_ratio,
                'perspective': perspective,
                'confidence': 1.0,
                'processing_time': processing_time,
                'status': '成功'
            }
            
        except Exception as e:
            print(f"错误：处理视频 {video_path} 时出现异常: {str(e)}")
            return {
                'video_id': Path(video_path).stem,
                'filename': Path(video_path).name,
                'text': '无',
                'word_count': 0,
                'first_person_count': 0,
                'second_person_count': 0,
                'third_person_count': 0,
                'first_person_ratio': 0,
                'second_person_ratio': 0,
                'third_person_ratio': 0,
                'perspective': '无',
                'confidence': 0,
                'processing_time': time.time() - start_time,
                'status': '处理异常'
            }

def main():
    """
    主函数
    """
    try:
        # 设置输入和输出路径
        video_dir = "xxxxxxxxxxxxxxxxxxx"  # 视频目录
        excel_file = "xxxxxxxxxxxxxxxxxxx"  # 输入的Excel文件
        output_file = "xxxxxxxxxxxxxxxxxxx"  # 输出的Excel文件
        
        # 创建分析器实例
        analyzer = NarrativePersonAnalyzer()
        
        # 处理视频目录
        analyzer.process_directory(video_dir, excel_file, output_file)
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 