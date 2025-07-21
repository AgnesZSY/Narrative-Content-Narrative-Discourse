# 旅游短视频叙事内容 × 叙事话语  
多模态量化工具集

> **子任务**：Narrative Content & Narrative Discourse 变量的大样本自动量化

---

## 📚 理论框架与变量体系

| 范畴 | 变量 | 研究定义 |
|------|------|----------|
| **Narrative Content**<br/>(叙事内容) | **情节逻辑性** Plot Logicality | 事件 / 场景 / 动作的因果链与时间线是否清晰、完整、自然 |
| | **感知真实性** Perceived Authenticity | 表现形式与细节给予用户的真实感、可信度 |
| | **内容价值取向** Value Orientation | 导向功利（实用攻略）、情感（治愈/共鸣）、体验（沉浸/挑战）、或其他（景观展示/文化科普/个人成长…） |
| **Narrative Discourse**<br/>(叙事话语) | **视角** Perspective | 文本人称、镜头拍摄、解说立场、播放形态等多层面 |
| | **节奏** Rhythm | 音乐节拍、剪辑密度、运动密度、音画同步共同决定的整体节奏快慢 |
| | **结构形态** Structural Form | 单元式（松散片段） / 连续式（一气呵成） / 离散式（强断裂感） |
| | **体裁** Genre | Vlog / 攻略 / 合集 / 解说 / 戏剧化短剧等 |
| | **戏剧性** Dramatic Quality | 悬念、反转、高潮、冲突等戏剧元素体现度 |
| | **语言风格** Linguistic Style | 功能型 / 情感型 / 意象型 / 混合 |

---
## 环境与安装

conda create -n travel_narrative python=3.9
conda activate travel_narrative
pip install -r requirements.txt


## 项目结构
data/
├── videos/*.mp4
└── texts/total.csv          # video_id,text

output/                      # 结果表 & 可视化
cache/                       # 中间缓存

code/
├── perspective/             # 视角四维子模块
│   ├── 01a_text_pov.py      # 人称视角（文本）
│   ├── 01b_character_pov.py # 人物视角（出镜/主观镜头）
│   ├── 01c_presentation.py  # 呈现视角（解说/旁白）
│   └── 01d_playback.py      # 播放视角（竖屏/画中画…）
├── 02_rhythm.py
├── 03_structure.py
├── 04_genre.py
├── 05_drama.py
├── 06_lang_style.py
├── 07_plot_logic.py
├── 08_authenticity.py
└── 09_value_orient.py

README.md



## 主要功能模块
### 4.1 视角 Perspective
| 维度                              | 子类型 / 字段                                                                                         | 自动提取方法                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **人称视角**<br/>(Text POV)         | `first_person_ratio` (我/我们) <br/>`second_person_ratio` (你/你们) <br/>`third_person_ratio` (他/她/他们) | 文案 Jieba 分词 → 代词统计；支持祈使句检测(`imperative_count`)        |
| **人物视角**<br/>(Character POV)    | `subjective_cam_ratio` 主观镜头概率 <br/>`onscreen_host_ratio` 出现主讲人比例                                 | OpenCV + 头部运动/视线特征检测；MTCNN / face\_recognition 统计出镜帧数 |
| **呈现视角**<br/>(Presentation POV) | `narration_type`∈{**演绎解说**/ **亲历独白**/ **多方对话**} <br/>`voice_over_ratio`                          | 音轨 VAD + STT 分析说话人数量；文本语气分类                           |
| **播放视角**<br/>(Playback POV)     | `orientation`∈{竖屏,横屏} <br/>`split_screen` bool <br/>`picture_in_picture` bool                    | 解析分辨率/黑边比例；检测多宫格、画中画帧布局                               |


##### 流程

读取文案 → Jieba 分词 → 统计人称 pronoun。

视频入帧 → 检测 FOV & 头部运动 → 主观镜头概率。

音轨 → VAD(Voice Activity Detection) + STT(若可用) → 讲解 / 对话判别。

汇总得分，生成 perspective.csv.

### 4.2 节奏 Rhythm
| 缩写             | 字段    | 说明                                        | 范围       |
| -------------- | ----- | ----------------------------------------- | -------- |
| BPM            | `bpm` | 音频节拍                                      | 0-200 Hz |
| CPS            | `cps` | Cuts per Second，剪辑密度                      | 0-2      |
| MDI            | `mdi` | Motion Density Index，运动密度                 | 0-100    |
| SAS            | `sas` | Sound-Action Sync                         | 0-1      |
| NRI            | `nri` | **Narrative Rhythm Index**<br/>(四指标归一化均值) | 0-1      |
| `rhythm_class` | 0/1/2 | 慢/中/快                                     |          |

##### 算法

ffmpeg ➜ 8 kHz mono 音轨 ➜ librosa.beat_track() → BPM。

opencv 抽帧(fps = 2) ➜ 帧差 > 30 计一次剪辑 → CPS。

帧光流 + 灰度差距 → MDI。

音频 onsets ↔ 场景切换时间差 < 0.1 s 记一次同步 → SAS。

四者 min-max 归一化 → nri → 阈值 0.33/0.66 划三档。

### 4.3 结构形态 Structural Form
| 标签             | code | 判定阈值（平均相似度） |
| -------------- | ---- | ----------- |
| 单元式 Unit       | `0`  | 0.50–0.80   |
| 连续式 Continuous | `1`  | > 0.80      |
| 离散式 Discrete   | `2`  | < 0.50      |

##### 步骤

PySceneDetect → 获取场景片段 min(MIN_SEG=3)。

每段中心帧 → ResNet-50 avg-pool 特征（2048 d）。

相邻片段 cosine_similarity → mean_sim。

对照阈值 → structure_label。

结果持久化 & 缓存 (cache/*_features.npy).

### 4.4 体裁 Genre
| 体裁   | 关键词/特征                | 融合权重 (text/vision/audio/scene) |
| ---- | --------------------- | ------------------------------ |
| 行程攻略 | 行程、路线、攻略… + 地图/路牌     | 0.6/0.3/0.1/—                  |
| 文化解说 | 历史、传说、博物馆… + 遗迹/展品    | 0.6/0.3/0.1/—                  |
| 美食探店 | 美食、餐厅… + Food-Closeup | 0.2/0.6/0.2/—                  |
| 体验叙事 | 体验、记录… + 第一视角         | 0.5/0.3/0.2/—                  |
| 情感抒发 | 感悟、温暖… + 舒缓音乐         | 0.8/0.1/0.1/—                  |
| 快闪剪辑 | 快闪、高能… + 高CPS/BPM     | 0.2/0.6/0.2/scene              |
Pipeline：文本 analyze_text() ▶ 视觉 ResNet → 特征匹配 ▶ 音轨 tempo/声纹 ▶ 场景切换 → 加权决策 → genre_labels + confidence_scores.

### 4.5 戏剧性 Dramatic Quality
| 变量             | 字段                   | 来源   |
| -------------- | -------------------- | ---- |
| `has_suspense` | 悬念关键词 + 出现位置         | 文本   |
| `has_surprise` | 意外关键词 + 情感跳跃         | 文本   |
| `vis_score`    | 视觉变化 & 快切            | 视频   |
| `audio_score`  | RMS/谱质心/对比度综合        | 音频   |
| `drama_label`  | {0:无,1:悬念,2:意外,3:双重} | 综合决策 |
阈值可借助 DramaThresholdOptimizer 自动优化 (Grid/Random Search).

### 4.6 语言风格 Linguistic Style
| style\_code | style\_name    | 判定逻辑                |
| ----------- | -------------- | ------------------- |
| `0`         | 功能型 Functional | 预算/交通/干货关键词≥阈值      |
| `1`         | 情感型 Emotional  | 网络流行感叹词 + Emoji/感叹号 |
| `2`         | 意象型 Imagery    | 句长<5 or 纯表情/风景诗意    |
| `3`         | 混合型 Mixed      | 任两类比例差 < 0.1        |
字段：ratio_functional, ratio_emotional, ratio_imagery, mixed_type.

### 4.7 情节逻辑性 Plot Logicality
| 字段                  | 描述                                   |
| ------------------- | ------------------------------------ |
| `event_count`       | 文本事件抽取数量                             |
| `causal_links`      | 抽取因果对数                               |
| `logic_score` (0-1) | `causal_links / event_count` \* 调整系数 |
| `logic_level`       | {0:弱,1:中,2:强} 按 0.3/0.6 阈值           |
实现：spacy + HanLP 事件抽取 ➜ temporal cue & 连词统计 ➜ 简易因果判别。
若事件<3，则默认为弱逻辑。

### 4.8 感知真实性 Perceived Authenticity
| 信号                | 采样                           | 贡献权重 |
| ----------------- | ---------------------------- | ---- |
| 文本一手体验词 (`亲测/实录`) | Regex                        | 0.4  |
| 画面原声率 (`原声轨/环境音`) | `audio.is_speech` vs `music` | 0.3  |
| 手持抖动/噪点 (非大片质感)   | `Gyro blur metric`           | 0.3  |

### 4.9 内容价值取向 Value Orientation
| 取向              | 关键词簇        | 特征      |
| --------------- | ----------- | ------- |
| 功利 Utilitarian  | 省钱、攻略、必去、排队 | 文本+功能特征 |
| 情感 Hedonic      | 治愈、感动、幸福、浪漫 | 文本情感倾向  |
| 体验 Experiential | 体验、记录、挑战、打卡 | 文本+第一视角 |

## 引用与致谢
PySceneDetect、Librosa、TorchVision、Jieba 等开源项目
