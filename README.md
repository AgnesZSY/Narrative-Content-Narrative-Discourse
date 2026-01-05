# Multimodal Measurement Toolkit for Narrative Content (NC) × Narrative Discourse (ND)
## in Tourism Short Videos (Douyin)

This repository provides a reproducible, multimodal measurement toolkit for operationalizing **Narrative Content (NC)** and **Narrative Discourse (ND)** in large-scale tourism short video datasets.

All indicators are constructed **only from video-intrinsic information**:
- **Visual** (frames / scene segments / key-shot sequences)
- **Text** (subtitles, speech transcripts, on-screen text via OCR)
- **Audio** (signal-based rhythmic features)


---

## 1. Conceptual Framework and Variable System

The toolkit implements two theoretically distinct but complementary narrative dimensions:

### 1.1 Narrative Content (NC): *What is told*
Narrative Content captures the substantive destination/travel information and meaning conveyed in the video.

| Dimension | Variable | What it captures |
|---|---|---|
| Narrative Content | **Plot Logicality** | Coherence and traceability of travel events across **time–space–causality** |
| Narrative Content | **Perceived Authenticity** | Strength of **credible and verifiable reality-anchoring cues** |
| Narrative Content | **Value Orientation** | Relative emphasis on **utilitarian (informational)** vs **hedonic (emotional/aesthetic)** value |

### 1.2 Narrative Discourse (ND): *How it is told*
Narrative Discourse captures the expressive and organizational strategies shaping presentation fluency.

| Dimension | Variable | What it captures |
|---|---|---|
| Narrative Discourse | **Perspective** | Narrative viewpoint across textual, visual, auditory, and playback layers |
| Narrative Discourse | **Structural Form** | Global organization of narrative segments (unit/continuous/discrete) |
| Narrative Discourse | **Rhythm** | Audiovisual pacing and temporal dynamics |
| Narrative Discourse | **Genre** | Dominant narrative contract / communicative function |
| Narrative Discourse | **Dramatic Quality** | Use of suspense, surprise, or tension structure |
| Narrative Discourse | **Linguistic Style** | Dominant mode of language organization |

---

## 2. Data Scope and Preprocessing

### 2.1 Unit of Analysis
Each observation corresponds to **one short video**.

### 2.2 Multimodal Inputs

#### Text Pool (merged)
We build a unified **Text Pool** from:
- speech transcripts (ASR)
- subtitles
- on-screen text (OCR)

Cleaning includes sentence segmentation, denoising, punctuation normalization, and filler removal. OCR text is cleaned by removing links, emojis, repeated symbols, and meaningless characters.

#### Visual Stream
- sample frames at fixed intervals
- detect shot boundaries to obtain key segments
- apply visual quality checks (extreme darkness, severe blur, heavy occlusion)

#### Audio Stream
- extract time-series features from raw audio signals:
  - energy
  - beat fluctuation
  - silence segments
These are used primarily for **Rhythm** indicators.

---

## 3. Missing Modality Handling (Degradation + Flags)

Short videos may lack certain modalities (e.g., no speech, no subtitles, no OCR text, or missing audio). The toolkit applies a **theory-consistent degradation strategy**:

- If **text is missing**:
  - constructs that can be supported by visuals (e.g., some spatial continuity / local cue signals) may still be computed using visual evidence
  - constructs that are text-dependent (e.g., causal explanation; linguistic style; temporal ordering connectors) are **not imputed** and are set to missing
- If **audio is missing**:
  - audio-dependent rhythm components are skipped; rhythm can fall back to visual components where defined
- All degraded measurements are **explicitly flagged** for downstream robustness checks.

---

## 4. Outlier Treatment and Standardization

### 4.1 Outlier Treatment (Continuous Indicators)
All continuous raw indicator scores are processed with **two-sided winsorization** (indicator-wise) **before** any mapping (e.g., to 0–1) and standardization:

1) winsorize raw scores (two-sided)  
2) apply indicator-specific transformations (including optional 0–1 mapping)  
3) z-standardize for modeling (mean = 0, SD = 1)

This provides a transparent mechanism for limiting extreme-value influence while keeping the pipeline auditable.

### 4.2 Standardization and Interaction Construction
For regression-ready outputs:
- all continuous predictors are **z-standardized**
- interaction terms are constructed **after standardization** to maintain interpretability and comparability

---

## 5. Variable Construction Details (1:1 Operational Definitions)

## 5.1 Narrative Content (NC)

### 5.1.1 Plot Logicality (continuous; optionally mapped to 0–1)
**Concept**: internal orderliness and comprehensibility of travel events across **time–space–causality**.

**Observed components (video-intrinsic evidence only):**
- **Temporal logic** (Text Pool primary)
  - density of time entities/time-window terms (e.g., morning, dusk, next day, Day X)
  - density of sequencing connectors (e.g., first, then, next, finally)
  - timeline consistency penalty for unexplained jumps
- **Spatial logic** (Text Pool + visuals)
  - density of place entities and locational cues (place names, landmarks, directional words)
  - density of route/movement cues (from A to B, metro transfer, walking, uphill/downhill)
  - scene continuity bonus from adjacent-segment semantic continuity
- **Causal logic** (Text Pool primary)
  - density of causal/explanatory connectors (because, therefore, so that, in order to)
  - step → outcome chain cues (if you do X, you can get Y)
  - problem → solution structures (avoid X otherwise…, tips/suggestions)

**Aggregation rule**
- compute `TimeScore`, `SpaceScore`, `CausalScore` (standardized)
- `PlotLogicalityRaw = mean(TimeScore, SpaceScore, CausalScore)`
- optional linear mapping to **0–1** for interpretability

**Missing rules**
- if text is missing: only the visual continuity part of spatial logic may be computed; other parts are missing and flagged
- composite uses the mean of available subcomponents and preserves missing flags

---

### 5.1.2 Perceived Authenticity (continuous; optionally mapped to 0–1)
**Concept**: the extent to which the destination situation is **credible and verifiable**, anchored in travel-realistic cues (content-level, not expressive polish).

**Observed components**
- **Verifiable information cues** (Text Pool)
  - density of numeric/metric details (price, duration, distance, time windows)
  - density of decision-relevant terms (transport, tickets, reservations, queues, routes, per-capita cost, opening hours)
  - verifiable instruction-like structures (what to do, where to go, what is needed)
- **Local/situational realism cues** (visual + Text Pool)
  - location-identifying cues (signage, maps, ticket interfaces) aligned with textual place cues
  - realistic situational cues (crowds, weather, transit, service procedures)
  - process depiction cues (arrive → enter → experience → leave)
- **Stylization risk cues (control component)** (visual)
  - extreme filterization / unnatural dominant tones (risk signals)
  - long purely-aesthetic sequences with minimal actionable detail (penalty cue)

**Aggregation rule (main specification)**
- compute and standardize: `VerifiableScore`, `LocalCueScore`, `StylizationRiskScore`
- `AuthenticityRaw = (VerifiableScore + LocalCueScore − StylizationRiskScore) / 3`
- optional linear mapping to **0–1**

**Outputs**
- Perceived Authenticity index
- retain the three component scores for sensitivity analyses

**Missing rules**
- no text: `VerifiableScore` missing; compute from remaining components and flag
- no valid visuals: compute text-based parts and flag

---

### 5.1.3 Value Orientation (continuous + optional grouping)
**Concept**: relative emphasis on **utilitarian** vs **hedonic** content value.

#### (a) Utilitarian Score (0–1)
**Observed cues (Text Pool primary; visuals supplementary)**
- actionable knowledge density (routes, transport, price, steps, cautions)
- entity richness (place/time/price entities)
- structured expression (lists, stepwise instructions, comparisons)
- visual information carriers (maps, price screens, signage) as supplementary evidence

**Output**
- `UtilitarianScore` (optional 0–1 mapping)

**Missing rules**
- text missing: estimate only via visual information carriers and flag

#### (b) Hedonic Score (0–1)
**Observed cues (Text Pool primary; visuals supplementary)**
- emotion/evaluation word density (e.g., healing, romantic, breathtaking)
- imagery/aesthetic cue density (e.g., “like an oil painting,” “cyberpunk vibe”)
- first-person experiential feelings
- visual atmosphere/aesthetic dominance as supplementary evidence

**Output**
- `HedonicScore` (optional 0–1 mapping)

**Missing rules**
- text missing: estimate only via visual atmosphere cues and flag

#### (c) Value Orientation
**Main specification**
- `ValueOrientation = HedonicScore − UtilitarianScore`
  - higher values indicate a more hedonic orientation

**Optional robustness grouping**
- tertiles: utilitarian-dominant / balanced / hedonic-dominant

**Missing rules**
- if either score is missing, Orientation is missing and flagged

---

## 5.2 Narrative Discourse (ND)

### 5.2.1 Perspective (categorical; multi-layer)
Perspective is measured along four layers:

1) **Grammatical person POV** (Text Pool)
- first / second / third / impersonal dominance via pronouns and referential patterns
- rule: dominant relative frequency; if all zero → impersonal

2) **Presentation POV** (Text + audio + visuals)
- self narration / third-party narration / dialogue / no speech
- evidence fusion: speech presence, speaker count, in-scene speaker cues
- output: label + (optional) confidence

3) **Character/camera POV** (visual)
- POV shot / direct-to-camera / objective follow / scene-dominant
- cues: person presence ratio, gaze-to-camera, camera subjectivity (handheld/first-person vs fixed)

4) **Playback orientation** (metadata)
- vertical / horizontal / square based on aspect ratio thresholds
- if metadata missing: infer from resolution and flag

---

### 5.2.2 Structural Form (categorical; optional coherence score)
Classifies the global organization as:
- **Unit-based**
- **Continuous**
- **Discrete**

**Rule**
- segment the video (shot boundary detection or fixed windows as fallback)
- compute adjacent-segment semantic similarity (primarily visual embeddings; text as auxiliary)
- map to three classes by thresholds
- optionally retain a continuous coherence score for sensitivity

**Missing rules**
- segmentation failure: fallback to fixed-window segmentation and flag

---

### 5.2.3 Rhythm (continuous index)
Rhythm captures audiovisual pacing through four standardized components:
- **Editing density** (cuts per unit time)
- **Motion intensity** (visual motion variability)
- **Beat fluctuation** (audio beat/energy dynamics)
- **Audio–visual synchronization** (alignment between audio events and visual transitions)

**Aggregation**
- standardize components
- weighted fusion to form a **Rhythm index**
- retain components for robustness analyses

**Missing rules**
- audio missing: compute rhythm from visual components only; flag audio-missing
- visuals missing: rhythm missing

---

### 5.2.4 Genre (6 classes; label + confidence; text-first)
**Goal**: identify the dominant narrative contract / communicative function.

**Classes (6)**
- fast-paced montage  
- emotional expression  
- food exploration (shop/food review)  
- itinerary guide  
- cultural explanation  
- experiential narrative  

**Evidence principle**
- **text-first functional cues** as primary evidence
- visuals/audio provide consistency adjustment (e.g., storefront/food close-ups; explanatory narration patterns; high beat for montage)

**Outputs**
- Top-1 genre label + confidence score
- Top-2 retained for sensitivity analyses

**Missing rules**
- text missing: degrade to audiovisual evidence and flag low confidence

---

### 5.2.5 Dramatic Quality (4 classes; label + confidence)
Classes:
- no dramatic structure
- suspense-driven sequence
- surprise-driven sequence
- combined suspense + surprise

**Evidence**
- text cues primary (suspense: delayed reveal; surprise: reversal markers)
- audiovisual shifts (energy spikes, abrupt transitions, contrast blocks) as supporting evidence

**Outputs**
- label + confidence score

---

### 5.2.6 Linguistic Style (3 classes; optional continuous scores)
Classes:
- functional-descriptive
- emotional-narrative
- imagery-symbolic

**Evidence (Text Pool primary)**
- functional: high density of instructions, entities, checklists, “remember/should” cues
- emotional: emotion-heavy first-person narration, rhetorical emphasis
- imagery-symbolic: minimal, symbolic/imagistic language, strong “blank space” style

**Outputs**
- label (and optionally three continuous style scores)

**Missing rules**
- text missing: style missing (no visual imputation recommended)

---

## 6. Composite Indices Used for Modeling (NC and ND)

### 6.1 Narrative Content Composite Index (NC)
NC is computed from the Narrative Content indicators (Plot Logicality, Perceived Authenticity, Value Orientation) following the main-text specification:

- primarily continuous scores
- a composite index is formed using a dimension-reduction approach aligned with the paper’s modeling (e.g., PCA first component) and then standardized

> Practical note: if Value Orientation is encoded as a binary direction in a specific model variant, keep only **one** dummy to avoid perfect multicollinearity.

### 6.2 Narrative Discourse Composite Index (ND)
ND is computed from discourse-side indicators:
- primarily categorical strategy variables (Perspective, Structural Form, Genre, Dramatic Quality, Linguistic Style)
- plus continuous Rhythm features

A dimension-reduction approach aligned with the paper’s modeling (e.g., MCA for categorical sets + fusion with continuous rhythm) yields ND, which is then standardized.

---

## 7. Repository Structure (Suggested)

```text
code/
├── perspective/                    # Four-layer perspective submodules
│   ├── 01a_text_pov.py             # Grammatical person POV (text-based)
│   ├── 01b_character_pov.py        # Character/camera POV (on-screen presence & subjective shots)
│   ├── 01c_presentation.py         # Presentation POV (narration / voiceover / dialogue)
│   └── 01d_playback.py             # Playback POV (vertical / horizontal / picture-in-picture, etc.)
├── 02_rhythm.py                    # Rhythm (audiovisual pacing)
├── 03_structure.py                 # Structural form
├── 04_genre.py                     # Genre
├── 05_drama.py                     # Dramatic quality
├── 06_lang_style.py                # Linguistic style
├── 07_plot_logic.py                # Plot logicality
├── 08_authenticity.py              # Perceived authenticity
└── 09_value_orient.py              # Value orientation

```
---
## 8. Intended Use

This toolkit is designed to support:

regression analysis (main effects, interactions, nonlinear terms)

robustness checks under missing-modality conditions

machine learning feature screening (e.g., Lasso, XGBoost)

configurational analysis (e.g., fsQCA) using transparent, auditable indicators

It is not intended as a black-box classifier; it is a theory-aligned measurement system with explicit rules, flags, and validation.



