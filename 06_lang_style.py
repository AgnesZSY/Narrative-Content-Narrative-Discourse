#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅游短视频语言风格检测器
功能：批量量化旅游短视频的语言风格（功能型/情感型/意象型）
"""

import pandas as pd
import numpy as np
import re
import jieba
import os
import time
from typing import Optional, Tuple, List
import warnings
from tqdm import tqdm
import ctypes
from ctypes import windll
import threading
warnings.filterwarnings('ignore')

class PreventSleep:
    """防止系统休眠的类"""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    
    def __init__(self):
        self.is_active = False
    
    def enable(self):
        """启用防休眠模式"""
        if not self.is_active:
            try:
                windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                self.is_active = True
                print("已启用防休眠模式")
            except Exception as e:
                print(f"启用防休眠模式失败: {str(e)}")
    
    def disable(self):
        """禁用防休眠模式"""
        if self.is_active:
            try:
                windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
                self.is_active = False
                print("已恢复默认电源设置")
            except Exception as e:
                print(f"恢复默认电源设置失败: {str(e)}")

def get_last_processed_id(output_path: str) -> int:
    """获取上次处理到的ID"""
    try:
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            if not df.empty:
                return df['video_id'].max()
    except Exception:
        pass
    return 0

def is_text_emoji_or_symbol(text: str) -> bool:
    """判断文本是否仅包含emoji、标点和空格"""
    return bool(re.fullmatch(r'[\W\s_]+', text))

def is_negation_present(text: str) -> bool:
    """检测是否存在否定词"""
    negation_words = {'不', '没', '非', '无', '莫', '勿', '未', '否', '别', '甭', '不要'}
    words = set(jieba.lcut(text))
    return bool(words & negation_words)

# 扩充jieba词典
NETWORK_WORDS = [
    '顶流', '绝绝子', '无语子', '破防', '上头', '嘎嘎好', '太爱了',
    '太牛了', '太绝了', '太强了', '太赞了', '太棒了', '太好了',
    '好家伙', '救命啊', '牛魔王', '神仙颜值', '绝美', '炸裂'
]

for word in NETWORK_WORDS:
    jieba.add_word(word)

# 关键词词典优化
# 功能型关键词（去重、优化）
FUNCTIONAL_KEYWORDS = [
    # 交通出行
    '地铁站', '公交站', '自驾游', '换乘站', '出站口', '停车场', '出租车', '网约车', 
    '导航到', '路线图', '线路图', '班车站', '机场大巴', '高铁站', '火车站', '动车组',
    '飞机场', '票务处', '安检口', '交通枢纽', '航班号', '候车室', '行李架', '托运处',
    '登机口', '下车处', '上车点', '转乘处', '接驳站', '发车点', '到达口', '出发区',
    '终点站', '起点站', '车次号', '车票价', '交通工具', '路程远', '耗时长', '途经地',
    
    # 预算相关
    '人均', '客单价', '门票', '票价', '折扣', '学生票', '优惠', '免费', '收费', '价格',
    '元', '块', '钱', '费用', '成本', '预算', '消费', '花费', '便宜', '贵',
    '性价比', '团购', '拼单', 'AA', '划算', '实惠', '超值', '买一送一', '赠券', '返现',
    '特价', '限时价', '低至', '打折', '折上折', '早鸟价', '优惠券', '满减', '满送',

    # 攻略/体验
    '攻略', '避坑', 'tips', '行程', '营业时间', '开放日', '闭馆', '预约', '排队',
    '建议', '推荐', '必去', '必看', '必吃', '打卡', '游玩', '路径', '顺序',
    '干货', '省钱', '省心', '提醒', '注意', '须知', '一定要', '必须', '不能', '值得',
    '体验', '尝试', '踩雷', '防坑', '测评', '实测', '真体验', '上手', '新手', '小白',
    '老手', '进阶', '深度游', '浅尝', '速览', '顺路', '逆时针', '正好', '方便', '安排',
    '安排行程', '行程单', '规划', '路线推荐', '省时', '高效', '时间分配', '时间管理', '预约入口',

    # 地点/定位
    '坐标', '入口', '出口', '楼层', '楼', '位置', '地址', '方向', '左转', '右转', '直行',
    '米', '公里', 'km', '分钟', 'min', '小时', 'h', '步行', '距离', '附近', '周边', '周围',
    '地标', '地图', '全景图', '定位', '周边环境', '交通便利', '怎么走', '如何到达', '最近', '最近地铁站',
    '怎么去', '怎么到达',

    # 时间
    '开门', '关门', '营业', '休息', '节假日', '工作日', '周一', '周二', '周三', '周四', '周五',
    '周六', '周日', '早上', '上午', '中午', '下午', '晚上', '夜间', '凌晨', '点', '时', '分',
    '等待时间', '限流', '即将', '每周', '每月', '全年', '寒暑假', '假期', '开放日', '排队',

    # 数量/限量
    '第一', '第二', '第三', '层', '号', '个', '家', '处', '次', '遍', '趟', '座', '批次',
    '全程', '全票', '半票', '余票', '满员', '限量', '人数', '限购', '仅限', '组团',

    # 服务与设施
    '服务', '设施', '卫生间', '洗手间', '寄存', '充电', 'wifi', '休息区', '停车场', '售票处',
    '取票', '退票', '换票', '儿童票', '老人票', '优先通道', '无障碍', '免费寄存', '行李箱',
    '安检口', '服务台', '导览', '讲解器', '咨询台', '自助机', '前台', '大厅', '自助取票',

    # 美食
    '餐厅', '小吃', '美食', '推荐菜', '点单', '菜品', '人均消费', '预订', '预定',
    '餐位', '用餐高峰', '排号', '外卖', '堂食', '早点', '午餐', '晚餐', '夜宵',
    '免排队', '下单', '特产', '名吃', '甜品', '饮品', '饮料', '打包', '团购券', '必尝',

    # 住宿相关
    '酒店', '民宿', '旅馆', '宾馆', '客栈', '青年旅社', '青旅', '旅社', '旅店', '旅舍',
    '标间', '大床房', '单人间', '双人间', '家庭房', '入住', '退房', '预订成功', '住宿体验', '床位',
]

# 情感型关键词（去重、优化）
EMOTIONAL_KEYWORDS = [
    # 情绪/感受（优化组合）
    '太感动了', '超震撼', '太惊喜了', '很浪漫', '被治愈', '泪目了', '嗨翻天', 
    '美得不行', '太惊艳', '壮观的', '美哭了', '太爱了', '绝了', '太美了', '炸裂',
    '好惊呆', '醉了呀', '太迷人', '心动的', '好激动', '太兴奋', '开心死', '乐翻天',
    
    # 网络流行表态/感叹
    'OMG', 'wow', '哇', '天哪', '我的天', '太棒了', 'amazing', 'incredible', 'fantastic',
    '好赞', '牛', '神仙', '爱死了', '秒杀', '高能', '必去', '笑死', '笑疯', '哈喽',
    '哈哈哈', '嘻嘻', '呜呜', '555', '泪奔', '哭了', '哭死', '笑出声', '震惊', '超好笑',
    '美死了', '绝美', '羡慕', '酸了', '上瘾', '停不下来', '太有意思了', '刷屏', '火爆', '刷爆',

    # 主观评价/安利
    '我觉得', '我超爱', '强烈推荐', '超级喜欢', '疯狂安利', '一定要', '必须', '最喜欢', '人生必去',
    '不容错过', '难以忘怀', '值得', '太值得', '真心推荐', '亲测有效', '好看到不行', '爆火', '爆款',
    '逆天颜值', '爆表', '超可爱', '萌萌哒', '高颜值', '无敌可爱', '高分', '五星好评',

    # 感受动词
    '感受', '体验', '享受', '触动', '打动', '痴迷', '陶醉', '迷恋', '着迷', '被治愈',
    '被震撼', '被安利', '被种草', '入坑', '圈粉', '好评', '狂赞', '点赞', '感受到',
    '嗨起来', '乐翻天', '惊呆了', '被美到', '被圈粉', '上头', '种草', '拔草',

    # 拟声/副词/网络情绪
    '瞬间', '立刻', '马上', '居然', '竟然', '意外', '没想到', '突然', '一下子', '一秒爱上',
    '超级', '超', '巨', '超赞', '顶级', '炸裂', '直接', '狠狠', '猛', '好强', '好深',
    '爆炸', '杠杠的', '一级棒', '无语', '好家伙', '救命', '绝了', '栓Q', '感慨', '破防',
    '嗑到了', '嘴角上扬', '哭了', '呜呜呜', '55555', '激动到不行', '简直', '非常', '极致',

    # 人称代词/卷入
    '我', '我们', '你', '你们', '咱们', '大家', '小伙伴', '朋友们', '宝贝们', '姐妹们',
    '男孩们', '小姐妹', '哥们儿', '兄弟们', '家人们', '亲们', '家伙们'
]


class LanguageStyleDetector:
    def __init__(self):
        """初始化语言风格检测器"""
        self.functional_kw = set(FUNCTIONAL_KEYWORDS)
        self.emotional_kw = set(EMOTIONAL_KEYWORDS)
        self.style_threshold = 0.1  # 混合型判定阈值
        self.batch_size = 5  # 每批处理的视频数量
        
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text) or text is None:
            return ""
        
        # 去除多余空格和换行，但保留基本格式
        text = re.sub(r'\s+', ' ', str(text).strip())
        # 统一常见表情符号，但不删除
        text = re.sub(r'[😀-🙏]+', '😊', text)
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """分句处理"""
        if not text:
            return []
        
        # 优化分句正则，更好地处理网络文本
        sentences = re.split(r'([。！？；\.\!\?;]+(?:[😀-🙏\s]*)|(?:[😀-🙏]+))', text)
        # 合并分句结果
        merged = []
        temp = ''
        for s in sentences:
            if s:
                if re.match(r'[。！？；\.\!\?;]+[😀-🙏\s]*|[😀-🙏]+', s):
                    temp += s
                    if temp.strip():  # 只添加非空句子
                        merged.append(temp)
                    temp = ''
                else:
                    temp += s
        if temp and temp.strip():  # 添加最后一个非空句子
            merged.append(temp)
        
        return [s.strip() for s in merged if s.strip()]
    
    def count_keywords(self, sentence: str, keywords: set) -> int:
        """统计句子中关键词数量，考虑否定词"""
        words = jieba.lcut(sentence)
        count = 0
        has_negation = is_negation_present(sentence)
        
        for word in words:
            if word in keywords:
                if has_negation:
                    count -= 1  # 否定词存在时，关键词计数为负
                else:
                    count += 1
        return max(0, count)  # 确保不会返回负值
    
    def has_numbers_and_units(self, sentence: str) -> bool:
        """检测是否包含数字和单位"""
        # 检测数字+单位的模式
        patterns = [
            r'\d+\s*元',
            r'\d+\s*块',
            r'\d+\s*分钟',
            r'\d+\s*小时',
            r'\d+\s*米',
            r'\d+\s*公里',
            r'\d+\s*km',
            r'\d+\s*min',
            r'\d+\s*h',
            r'\d+:\d+',  # 时间格式
            r'\d+点',
            r'\d+号'
        ]
        
        for pattern in patterns:
            if re.search(pattern, sentence):
                return True
        return False
    
    def has_first_second_person(self, sentence: str) -> bool:
        """检测是否包含第一、二人称"""
        pronouns = ['我', '我们', '咱们', '你', '你们', '大家', '小伙伴', '朋友们', '宝贝们']
        for pronoun in pronouns:
            if pronoun in sentence:
                return True
        return False
    
    def has_emotional_punctuation(self, sentence: str) -> bool:
        """检测是否包含情感标点"""
        patterns = [
            r'！{2,}',  # 多个感叹号
            r'哈{2,}',  # 哈哈哈
            r'呜{2,}',  # 呜呜呜
            r'555+',    # 555
            r'[😀-🙏]'  # emoji表情
        ]
        
        for pattern in patterns:
            if re.search(pattern, sentence):
                return True
        return False
    
    def rule_override(self, sentence: str) -> Optional[int]:
        """规则覆盖判断，返回0(功能)/1(情感)或None"""
        func_count = self.count_keywords(sentence, self.functional_kw)
        emo_count = self.count_keywords(sentence, self.emotional_kw)
        
        # 功能型规则判断
        if func_count >= 2:  # 功能词≥2个
            return 0
        
        if func_count >= 1 and self.has_numbers_and_units(sentence):
            return 0
        
        # 情感型规则判断
        if emo_count >= 1:
            return 1
            
        if self.has_first_second_person(sentence) and self.has_emotional_punctuation(sentence):
            return 1
            
        # 如果功能词多于情感词且有数字单位，倾向功能型
        if func_count > emo_count and func_count >= 1 and self.has_numbers_and_units(sentence):
            return 0
            
        return None
    
    def simple_sentiment_predict(self, sentence: str) -> int:
        """简单的情感预测（替代BERT模型）"""
        # 计算功能词和情感词的权重
        func_count = self.count_keywords(sentence, self.functional_kw)
        emo_count = self.count_keywords(sentence, self.emotional_kw)
        
        # 考虑其他特征
        has_numbers = self.has_numbers_and_units(sentence)
        has_person = self.has_first_second_person(sentence)
        has_emo_punct = self.has_emotional_punctuation(sentence)
        
        # 计算得分
        func_score = func_count * 2
        if has_numbers:
            func_score += 1
            
        emo_score = emo_count * 2
        if has_person:
            emo_score += 1
        if has_emo_punct:
            emo_score += 1
            
        # 返回预测结果
        if func_score > emo_score:
            return 0  # 功能型
        else:
            return 1  # 情感型
    
    def classify_sentence(self, sentence: str) -> int:
        """对单个句子进行分类"""
        # 首先尝试规则覆盖
        rule_result = self.rule_override(sentence)
        if rule_result is not None:
            return rule_result
        
        # 否则使用简单预测模型
        return self.simple_sentiment_predict(sentence)
    
    def decide_ratios(self, labels: List[int], n_sent: int) -> Tuple[float, float, float, Optional[str]]:
        """计算三种风格的比例，返回混合类型标记"""
        if n_sent == 0:
            return 0.0, 0.0, 1.0, "imagery"
        
        if n_sent < 2:
            return 0.0, 0.0, 1.0, "imagery"
        
        # 统计各类型数量
        func_count = labels.count(0)
        emo_count = labels.count(1)
        
        # 计算比例
        ratio_functional = func_count / n_sent
        ratio_emotional = emo_count / n_sent
        ratio_imagery = 1.0 - (ratio_functional + ratio_emotional)
        
        # 判断是否为混合型
        ratios = [ratio_functional, ratio_emotional, ratio_imagery]
        max_ratio = max(ratios)
        second_max = sorted(ratios, reverse=True)[1]
        
        if max_ratio - second_max < self.style_threshold:
            # 返回混合类型标记
            if ratio_functional > 0.3 and ratio_emotional > 0.3:
                return ratio_functional, ratio_emotional, ratio_imagery, "mixed_func_emo"
            elif ratio_functional > 0.3 and ratio_imagery > 0.3:
                return ratio_functional, ratio_emotional, ratio_imagery, "mixed_func_img"
            elif ratio_emotional > 0.3 and ratio_imagery > 0.3:
                return ratio_functional, ratio_emotional, ratio_imagery, "mixed_emo_img"
        
        return ratio_functional, ratio_emotional, ratio_imagery, None
    
    def detect_style(self, content: str) -> Tuple[int, float, float, float, int, Optional[str]]:
        """检测单个文本的语言风格"""
        # 预处理
        content = self.preprocess_text(content)
        
        # 检查是否为纯表情符号或空文本
        if is_text_emoji_or_symbol(content):
            return 0, 0.0, 0.0, 1.0, 2, "imagery"  # 意象型
        
        # 检查文本长度
        if len(content.replace(' ', '')) < 5:  # 降低最小长度要求
            return 0, 0.0, 0.0, 1.0, 2, "imagery"  # 意象型
        
        # 分句
        sentences = self.split_sentences(content)
        n_sent = len(sentences)
        
        # 如果句子数量太少但有实际内容，作为单句处理
        if n_sent < 2 and content.strip():
            sentences = [content]
            n_sent = 1
        
        # 对每个句子进行分类
        labels = []
        for sentence in sentences:
            if sentence.strip():
                label = self.classify_sentence(sentence)
                labels.append(label)
        
        if not labels:  # 如果没有有效句子
            return 0, 0.0, 0.0, 1.0, 2, "imagery"
        
        # 计算比例和混合类型
        ratio_functional, ratio_emotional, ratio_imagery, mixed_type = self.decide_ratios(labels, len(labels))
        
        # 确定主导风格
        if mixed_type:
            dominant_style = 3  # 混合型的标识
        else:
            ratios = [ratio_functional, ratio_emotional, ratio_imagery]
            # 如果功能型和情感型都很低，判定为意象型
            if ratio_functional < 0.2 and ratio_emotional < 0.2:
                dominant_style = 2  # 意象型
            else:
                dominant_style = np.argmax(ratios)
        
        return n_sent, ratio_functional, ratio_emotional, ratio_imagery, dominant_style, mixed_type
    
    def process_dataset(self, input_path: str, output_path: str):
        """处理整个数据集
        
        Args:
            input_path: 输入CSV文件路径，需包含'video_id'和'text'列
            output_path: 输出CSV文件路径
        """
        print("正在加载数据...")
        
        # 读取输入数据
        try:
            df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(input_path, encoding='gbk', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding='gb18030', low_memory=False)
        
        # 确保video_id是整数类型
        df['video_id'] = pd.to_numeric(df['video_id'], errors='coerce').fillna(0).astype(int)
        
        print(f"共加载 {len(df)} 条数据")
        
        # 检查必需的列
        required_cols = ['video_id', 'text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        # 按video_id排序并去重
        df = df.drop_duplicates(subset=['video_id']).sort_values('video_id')
        
        # 获取上次处理到的ID
        last_processed_id = get_last_processed_id(output_path)
        if last_processed_id > 0:
            print(f"从ID {last_processed_id} 继续处理")
            df = df[df['video_id'] > last_processed_id]
        
        if df.empty:
            print("没有新数据需要处理")
            return None
        
        # 初始化结果列表和进度条
        results = []
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=len(df), desc="处理进度")
        
        # 按批次处理数据
        for start_idx in range(0, len(df), self.batch_size):
            batch_df = df.iloc[start_idx:start_idx + self.batch_size]
            batch_results = []
            
            for _, row in batch_df.iterrows():
                try:
                    video_id = row['video_id']
                    content = row['text']
                    
                    # 检测语言风格
                    n_sent, ratio_func, ratio_emo, ratio_img, dominant, mixed_type = self.detect_style(content)
                    
                    # 添加风格名称
                    style_names = ['功能型', '情感型', '意象型', '混合型']
                    style_name = style_names[dominant] if dominant < len(style_names) else '未知'
                    
                    batch_results.append({
                        'video_id': video_id,
                        'text': content,
                        'style_name': style_name,
                        'n_sent': n_sent,
                        'ratio_functional': ratio_func,
                        'ratio_emotional': ratio_emo,
                        'ratio_imagery': ratio_img,
                        'dominant_style': dominant,
                        'mixed_type': mixed_type
                    })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n处理视频ID {video_id} 时出错: {str(e)}")
                    continue
            
            # 每批次保存一次结果
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                
                # 调整列的顺序
                columns_order = [
                    'video_id',           # 原始ID
                    'text',              # 原始文案
                    'style_name',        # 风格名称
                    'dominant_style',    # 风格编号
                    'ratio_functional',  # 功能型比例
                    'ratio_emotional',   # 情感型比例
                    'ratio_imagery',     # 意象型比例
                    'mixed_type',        # 混合类型
                    'n_sent'            # 句子数量
                ]
                batch_df = batch_df[columns_order]
                
                # 如果文件存在，则合并结果
                if os.path.exists(output_path):
                    try:
                        existing_df = pd.read_csv(output_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            existing_df = pd.read_csv(output_path, encoding='gbk')
                        except UnicodeDecodeError:
                            existing_df = pd.read_csv(output_path, encoding='gb18030')
                    
                    # 确保video_id是整数类型
                    existing_df['video_id'] = pd.to_numeric(existing_df['video_id'], errors='coerce').fillna(0).astype(int)
                    # 移除可能存在的重复项
                    existing_df = existing_df[~existing_df['video_id'].isin(batch_df['video_id'])]
                    batch_df = pd.concat([existing_df, batch_df])
                
                # 按video_id排序并保存
                batch_df = batch_df.sort_values('video_id')
                batch_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 使用带BOM的UTF-8编码
                
                # 更新总结果列表
                results.extend(batch_results)
        
        pbar.close()
        
        # 打印最终统计信息
        if results:
            result_df = pd.DataFrame(results)
            print("\n本次处理的语言风格分布:")
            style_stats = result_df['style_name'].value_counts()
            for style, count in style_stats.items():
                print(f"{style}: {count} 条 ({count/len(result_df)*100:.1f}%)")
            
            if not result_df['mixed_type'].isna().all():
                print("\n本次处理的混合类型分布:")
                mixed_counts = result_df['mixed_type'].value_counts()
                for mixed_type, count in mixed_counts.items():
                    if pd.notna(mixed_type):
                        print(f"{mixed_type}: {count} 条 ({count/len(result_df)*100:.1f}%)")
            
            # 打印总体统计信息
            if os.path.exists(output_path):
                try:
                    total_df = pd.read_csv(output_path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        total_df = pd.read_csv(output_path, encoding='gbk')
                    except UnicodeDecodeError:
                        total_df = pd.read_csv(output_path, encoding='gb18030')
                
                print("\n累计处理的语言风格分布:")
                total_style_stats = total_df['style_name'].value_counts()
                for style, count in total_style_stats.items():
                    print(f"{style}: {count} 条 ({count/len(total_df)*100:.1f}%)")
        
        return pd.DataFrame(results) if results else None

def main():
    """主函数"""
    # 指定输入输出路径
    input_path = r"xxxxxxxxxxxxxxxxxxx"
    output_path = r"xxxxxxxxxxxxxxxxxxx"
    
    # 创建防休眠对象
    prevent_sleep = PreventSleep()
    
    try:
        # 启用防休眠
        prevent_sleep.enable()
        
        # 创建检测器并处理数据
        detector = LanguageStyleDetector()
        result_df = detector.process_dataset(input_path, output_path)
        
        if result_df is not None:
            print("\n处理成功完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复默认电源设置
        prevent_sleep.disable()

if __name__ == "__main__":
    main() 