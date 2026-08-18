I've deduplicated by title, merging angle tags. Here is the final bibliography.

# AraSeg 2026 Data-Scarcity Bibliography — 37 unique papers

Deduplicated from 119 collected entries by title (angle tags merged). Where duplicate entries carried differing applicability tags, the most closed-track-favorable defensible tag is listed with variants noted.

## Synthetic generation

**[1] DAGA: Data Augmentation with a Generation Approach for Low-resource Tagging Tasks** — Bosheng Ding, Linlin Liu, Lidong Bing, Canasai Kruengkrai, Thien Hai Nguyen, Shafiq Joty, Luo Si, Chunyan Miao (2020), *EMNLP 2020*. https://aclanthology.org/2020.emnlp-main.488/
_Single most on-target method: linearizes tag+token sequences and trains an LM to GENERATE new labeled tagging data from only the gold set — directly synthesizes boundary-labeled sentences from the 174 docs with zero external labels; gains largest when gold data is smallest. Runs supervised or semi-supervised over unlabeled Arabic._ · closed-legal · code: https://github.com/ntunlp/daga
_(merged from 7 duplicate entries across angles: low-resource token labeling, sequence-labeling augmentation, self-training/semi-sup, few-shot NER transfer, weak supervision, controllable generation)_

**[2] GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation** — Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, Woomyoung Park (2021), *Findings of EMNLP 2021*. https://aclanthology.org/2021.findings-emnlp.192/
_Mixes real seed samples in the prompt so an LLM produces in-distribution examples with soft labels — the seed-conditioned + soft-label recipe (adaptable from classification to boundary spans by regenerating text then re-deriving boundaries) for faithful Hadith/legal pseudo-docs._ · closed-legal / gray · code: none
_(merged from 2 entries: synthetic-generation, back-translation/paraphrase angles)_

**[3] AugGPT: Leveraging ChatGPT for Text Data Augmentation** — Haixing Dai et al. (2023), *IEEE Transactions on Big Data (2025); arXiv:2302.13007*. https://arxiv.org/abs/2302.13007
_Rephrases each seed sentence into multiple surface-different variants for few-shot settings — a cheap way to multiply the 174 docs' distinct token contexts (the 88-89% unseen-context error source) with boundaries recoverable after alignment._ · closed-legal · code: https://github.com/yhydhx/AugGPT

**[4] Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning (FewGen)** — Yu Meng, Martin Michalski, Jiaxin Huang, Yu Zhang, Tarek Abdelzaher, Jiawei Han (2023), *ICML 2023*. https://arxiv.org/abs/2211.03044
_Prefix-tunes a generative LM ON the few-shot examples themselves then synthesizes novel augmentations — closest match to our exact constraint (learn a generator from only the 174 labeled docs, no outside labels); 3-5pt gains over prior augmentation on low-data GLUE._ · closed-legal · code: https://github.com/yumeng5/FewGen

**[5] ZeroGen: Efficient Zero-shot Learning via Dataset Generation** — Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, Lingpeng Kong (2022), *EMNLP 2022*. https://aclanthology.org/2022.emnlp-main.801/
_Synthesizes a full labeled dataset from a PLM with no gold data, then trains a tiny task model on it — a fallback/complement when even the 174 seeds are too few to condition on, and the data-then-distill framing informs how much synthetic seg data actually helps a small segmenter._ · closed-legal · code: https://github.com/HKUNLP/ZeroGen

**[6] Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning (SunGen)** — Jiahui Gao, Renjie Pi, Yong Lin, Hang Xu, Jiacheng Ye, Zhiyong Wu, Weizhong Zhang, Xiaodan Liang, Zhenguo Li, Lingpeng Kong (2023), *ICLR 2023*. https://openreview.net/forum?id=h5OpjGd_lo6
_Addresses the failure mode of ZeroGen/DAGA-style data (generated examples are noisy) via a bi-level sample-reweighting scheme learning per-example quality weights with no human annotation — essential for filtering mislabeled synthetic boundaries before they poison our small segmenter._ · closed-legal · code: https://github.com/SumilerGAO/SunGen

**[7] Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias (AttrPrompt)** — Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang (2023), *NeurIPS 2023 (Datasets & Benchmarks)*. https://arxiv.org/abs/2306.15895
_Naive class-conditional generation is biased/low-diversity; attribute-controlled prompts (length, style, sub-genre) restore diversity at ~5% cost — the lever for forcing synthetic Hadith/legal/exam-stem contexts to cover the long-tail token contexts our dev errors concentrate in._ · closed-legal · code: https://github.com/yueyu1030/AttrPrompt

**[8] Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification** — (preprint authors as listed on arXiv) (2026), *arXiv:2601.16278*. https://arxiv.org/abs/2601.16278
_Empirical evidence that LLMs help low-resource multilingual tasks more as synthetic-DATA GENERATORS than as direct classifiers — supports the closed-track Opus generative-augmentation strategy over zero-shot LLM segmentation, and flags the diversity-stagnation ceiling as synthetic Arabic scales._ · closed-legal · code: none

## Augmentation for sequence labeling (label-preserving)

**[9] MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER** — Ran Zhou, Xin Li, Ruidan He, Lidong Bing, Erik Cambria, Luo Si, Chunyan Miao (2022), *ACL 2022 (Long)*. https://aclanthology.org/2022.acl-long.160/
_Canonical label-preserving augmentation: injects labels into context and masks/regenerates label-bearing tokens so token-label alignment is guaranteed. Adapt an Arabic MLM (AraBERT/CAMeLBERT) with boundary markers in context to mint new in-distribution contexts around boundary vs non-boundary positions; has a cross-lingual/code-mix variant._ · closed-legal · code: https://github.com/RandyZhouRan/MELM
_(merged from 5 entries across sequence-labeling, punctuation/discourse, back-translation, weak-supervision, Hadith-legal angles)_

**[10] ACLM: A Selective-Denoising based Generative Data Augmentation Approach for Low-Resource Complex NER** — Sreyan Ghosh, Utkarsh Tyagi, Manan Suri, Sonal Kumar, S Ramaneswaran, Dinesh Manocha (2023), *ACL 2023 (Long)*. https://aclanthology.org/2023.acl-long.8/ (also listed at https://aclanthology.org/2023.acl-long.512/)
_SOTA generative augmentation that fixes MELM's context-label mismatch via attention-map-guided selective masking on BART, keeping label-bearing tokens fixed while regenerating context — the mismatch it solves is exactly our failure mode (models fail on unseen contexts), so generating coherent NEW contexts around fixed boundary labels is the target intervention._ · closed-legal · code: https://github.com/Sreyan88/ACLM (also https://github.com/betterzhou/ACLM)
_(note: two entries listed different author lists — Ghosh et al. vs Das et al. — and two ACL-anthology IDs; treated as one title)_

**[11] An Analysis of Simple Data Augmentation for Named Entity Recognition** — Xiang Dai, Heike Adel (2020), *COLING 2020*. https://aclanthology.org/2020.coling-main.343/
_Reference baseline for label-preserving sequence-labeling augmentation (label-wise token replacement, synonym replacement, mention replacement, shuffle-within-segments), largest gains precisely on small sets — the cheap no-external-data primitives to try first, plus a fair comparison protocol and a warning about token-label misalignment directly relevant to boundary labels._ · closed-legal · code: https://github.com/boschresearch/data-augmentation-coling2020
_(merged from 3 entries)_

**[12] SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup** — Rongzhi Zhang, Yue Yu, Chao Zhang (2020), *EMNLP 2020 (Long)*. https://aclanthology.org/2020.emnlp-main.691/
_Mixup for token-level labels: interpolates paired sequences and their per-token label distributions in embedding space with a plausibility discriminator, plus active-learning selection; densifies the sparse boundary-context manifold for a binary boundary-after-token task; +2.3–3.8 F1, strongest in low-data regimes._ · closed-legal · code: https://github.com/rz-zhang/SeqMix
_(merged from 2 entries: sequence-labeling, curriculum/active angles)_

**[13] Local Additivity Based Data Augmentation for Semi-supervised NER (LADA)** — Jiaao Chen, Zhenghui Wang, Ran Tian, Zichao Yang, Diyi Yang (2020), *EMNLP 2020 (Long)*. https://aclanthology.org/2020.emnlp-main.95/
_Intra-/Inter-LADA interpolate nearby token sequences for virtually infinite labeled data plus a consistency loss for UNLABELED text — doubly relevant: mixup densifies contexts, and the semi-supervised consistency loss exploits unlabeled in-genre Arabic (Hadith/legal) without external seg labels._ · closed-legal · code: https://github.com/GT-SALT/LADA

**[14] Data Augmentation for Low-Resource Named Entity Recognition Using Backtranslation** — Usama Yaseen, Stefan Langer (2021), *arXiv:2108.11703 (ICON 2021 workshop)*. https://arxiv.org/abs/2108.11703
_Adapts backtranslation to token labeling with label-projection heuristics; largest gains at 50-500 training sentences — our exact scale. Cautionary reference: projection/span drift under MT alignment is the main risk to control for Arabic boundaries; informs whether backtranslation beats the generative methods._ · gray · code: none
_(merged from 2 entries)_

**[15] Sentence Boundary Augmentation for Neural Machine Translation Robustness** — Daniel Li, Te I, Naveen Arivazhagan, Colin Cherry, Dirk Padfield (2020), *arXiv:2010.11132 (later ICASSP 2021)*. https://arxiv.org/abs/2010.11132
_Synthesizes boundary-varied examples by re-splitting/joining existing text to teach segmentation robustness — a concrete label-preserving augmentation generated purely from the 174 docs to cover unseen boundary contexts._ · closed-legal · code: none

## Self-training & semi-supervised sequence labeling

**[16] Uncertainty-aware Self-training for Low-resource Neural Sequence Labeling (SeqUST)** — Jianing Wang, Chengyu Wang, Jun Huang, Ming Gao, Aoying Zhou (2023), *AAAI 2023*. https://ojs.aaai.org/index.php/AAAI/article/view/26603 (arXiv:2302.08659)
_The most directly transferable self-training method for our task shape: MC-dropout token-level uncertainty selects reliable pseudo-labels on unlabeled text, plus noise-robust masked-labeling loss and Gaussian consistency regularization — pseudo-label unlabeled Arabic/Hadith/legal and keep only low-uncertainty boundary decisions, attacking the 88-89% unseen-context error head-on._ · closed-legal · code: https://github.com/wjn1996/SeqUST (also cited as SLNER; some entries list none)
_(merged from 8 duplicate entries — the single most-repeated paper across angles; note one entry gave a different AAAI ID view/26471)_

**[17] Meta Self-training for Few-shot Neural Sequence Labeling (MetaST)** — Yaqing Wang, Subhabrata Mukherjee, Haoda Chu, Yuancheng Tu, Ming Wu, Jing Gao, Ahmed Hassan Awadallah (2021), *KDD 2021 (earlier arXiv:2010.03680, "Adaptive Self-training…")*. https://dl.acm.org/doi/10.1145/3447548.3467235
_Canonical few-shot self-training for taggers with ~10 labels/class: meta-learned per-token re-weighting damps error propagation from noisy pseudo-labels + adaptive uncertainty-based validation selection — our 174-doc track is precisely this few-label regime, counters teacher noise on unseen contexts._ · closed-legal / gray · code: https://github.com/microsoft/MetaST
_(merged from 2 entries)_

**[18] Self-Training Pre-Trained Language Models for Zero- and Few-Shot Multi-Dialectal Arabic Sequence Labeling** — Muhammad Khalifa, Muhammad Abdul-Mageed, Khaled Shaalan (2021), *EACL 2021*. https://aclanthology.org/2021.eacl-main.65/
_Directly Arabic + sequence-labeling + self-training: self-training on a fine-tuned Arabic PLM adds up to ~10 F1 in low/zero-shot NER/POS — the archetype (pseudo-label unlabeled Arabic, confidence-filter, retrain) for squeezing signal from unlabeled in-genre corpora under our constraint._ · closed-legal · code: https://github.com/mohammadKhalifa/zero-shot-arabic-dialects

**[19] Revisiting Self-Training for Neural Sequence Generation** — Junxian He, Jiatao Gu, Jiajun Shen, Marc'Aurelio Ranzato (2020), *ICLR 2020*. https://arxiv.org/abs/1909.13788
_The mechanistic "why self-training works" paper: dropout/hidden-state noise on the student is the critical ingredient (noisy-student intuition) turning pseudo-labeled data into a regularizer, not error amplification — design guidance for any self-training loop on unlabeled Arabic._ · background · code: https://github.com/jxhe/self-training-text-generation

**[20] Unsupervised Data Augmentation for Consistency Training (UDA)** — Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le (2020), *NeurIPS 2020*. https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html
_Foundational semi-supervised result: with ~20 labels, consistency training against strong augmentations beats fully-supervised baselines — motivates a consistency-regularization arm enforcing boundary-prediction invariance on unlabeled Arabic under text-preserving perturbations._ · gray · code: https://github.com/google-research/uda

**[21] STAD: Self-Training with Ambiguous Data for Low-Resource Relation Extraction** — Junjie Yu, Xing Wang, Jiangjiang Zhao, Chunjie Yang, Wenliang Chen (2022), *COLING 2022*. https://aclanthology.org/2022.coling-1.178/
_Unlike confidence-threshold self-training that discards uncertain pseudo-labels, STAD extracts signal from AMBIGUOUS predictions via candidate-/negative-label set training — recovers training signal for exactly our hard unseen-token boundary cases where a teacher is uncertain._ · gray · code: none

**[22] Noisy Self-Training with Data Augmentations for Offensive and Hate Speech Detection Tasks** — João A. Leite, Carolina Scarton, Diego F. Silva (2023), *RANLP 2023*. https://arxiv.org/abs/2307.16609
_Important NEGATIVE-result control: on real low-resource text, adding augmentation noise to self-training HURT vs plain self-training, while plain self-training still gave up to +1.5 F1 — argues for testing vanilla self-training first, a cheap pre-registered baseline._ · background · code: none

**[23] BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision** — Chen Liang, Yue Yu, Haoming Jiang, Siawpeng Er, Ruijia Wang, Tuo Zhao, Chao Zhang (2020), *KDD 2020*. https://dl.acm.org/doi/10.1145/3394486.3403149
_Two-stage teacher-student self-training: first fit noisy distant labels, then drop them and self-train on confident predictions with early stopping — template for bootstrapping boundary supervision on unlabeled Arabic from a small 174-doc seed._ · closed-legal / gray · code: https://github.com/cliang1453/BOND
_(merged from 2 entries)_

**[24] Distantly-Supervised Named Entity Recognition with Noise-Robust Learning and Language Model Augmented Self-Training (RoSTER)** — Yu Meng, Yunyi Zhang, Jiaxin Huang, Xuan Wang, Yu Zhang, Heng Ji, Jiawei Han (2021), *EMNLP 2021*. https://aclanthology.org/2021.emnlp-main.810/
_Pairs a noise-robust loss with LM-augmented self-training to learn token labels from noisy weak supervision — turns rule/heuristic boundary labels on unlabeled Arabic into extra training signal for the closed track._ · closed-legal · code: https://github.com/yumeng5/RoSTER

**[25] STraTA: Self-Training with Task Augmentation for Better Few-shot Learning** — Tu Vu, Minh-Thang Luong, Quoc V. Le, Grady Simon, Mohit Iyyer (2021), *EMNLP 2021*. https://aclanthology.org/2021.emnlp-main.462/
_Combines task augmentation (synthesize auxiliary-task data from the target's own unlabeled text) with self-training; 8 examples/class matches 67K-example fine-tuning — synthesize an auxiliary boundary-style task from unlabeled Arabic, then self-train, no external seg data._ · closed-legal · code: https://github.com/google-research/google-research/tree/master/STraTA

## Cross-lingual / multilingual sentence boundary detection

**[26] Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation (SaT)** — Markus Frohmann, Igor Sterner, Ivan Vulić, Benjamin Minixhofer, Markus Schedl (2024), *EMNLP 2024 (Main), pp. 11908–11941*. https://aclanthology.org/2024.emnlp-main.665/ (arXiv:2406.16678)
_Strongest external pretrained segmenter: punctuation-robust self-supervised pretraining (no labeled seg data) + parameter-efficient LoRA adaptation hitting SOTA on legal docs/lyrics from as few as 16 gold examples — the ceiling and the backbone of our SaT-fusion arm (task #19). Full-model use is OPEN-track; the self-supervised backbone + tiny-data adaptation is the gray/closed-defensible method to mine._ · open-only / gray · code: https://github.com/segment-any-text/wtpsplit
_(merged from 9 duplicate entries; applicability varied open-only↔gray↔closed-legal across angles — most restrictive load-bearing reading is open-only for the full model)_

**[27] Where's the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation (WtP)** — Benjamin Minixhofer, Jonas Pfeiffer, Ivan Vulić (2023), *ACL 2023 (Long)*. https://aclanthology.org/2023.acl-long.398/ (arXiv:2305.18893)
_Learns boundaries self-supervised from raw newline-delimited text across 85 languages incl. Arabic, punctuation-agnostic, adapts from 64-256 labeled examples — the canonical recipe for adding seg signal with zero external labels; motivated our v16-B2 newline-recovery pre-finetuning. A from-scratch reproduction is closed-track-defensible._ · closed-legal / gray / open-only · code: https://github.com/segment-any-text/wtpsplit (older: https://github.com/bminixhofer/wtpsplit)
_(merged from 8 duplicate entries; applicability tag varied by angle)_

**[28] A Unified Approach to Sentence Segmentation of Punctuated Text in Many Languages (Ersatz)** — Rachel Wicks, Matt Post (2021), *ACL-IJCNLP 2021 (Long), pp. 3995–4007*. https://aclanthology.org/2021.acl-long.309/
_Canonical modern multilingual SBD: regex candidate generation + Transformer binary boundary-after-context classifier over 87 languages — the exact task shape as AraSeg's binary boundary-after-token; trained on noisily-annotated data, validating a weak/self-labeled path beyond 174 gold docs; standard baseline/ablation reference._ · background / closed-legal · code: https://github.com/rewicks/ersatz
_(merged from 2 entries)_

**[29] Self-Augmentation Improves Zero-Shot Cross-Lingual Transfer (SALT)** — Fei Wang, Kuan-Hao Huang, Kai-Wei Chang, Muhao Chen (2023), *IJCNLP-AACL 2023 (Short), pp. 1–9*. https://aclanthology.org/2023.ijcnlp-short.1/
_Code-switching + embedding-mixup self-augmentation lifts cross-lingual transfer of a multilingual PLM with strictly NO external data — a closed-track-compliant augmentation to distill boundary signal into our Arabic classifier from its multilingual backbone._ · closed-legal · code: none

**[30] Multilingual unsupervised sequence segmentation transfers to extremely low-resource languages (XLSLM)** — C. M. Downey, Shannon Drizin, Levon Haroutunian, Shivin Thukral (2022), *ACL 2022 (Long), pp. 5331–5346*. https://aclanthology.org/2022.acl-long.366/
_Pretraining a segmental model on typologically related languages transfers segmentation to a target with tiny/zero labels (20.6 F1 zero-shot, biggest gains at small target sizes) — motivates multilingual/cross-lingual pretraining before the 174-doc fine-tune._ · background · code: https://github.com/cmdowney88/XLSLM

**[31] Zero-Shot Cross-Lingual Transfer with Meta Learning (X-MAML)** — Farhad Nooralahzadeh, Giannis Bekoulis, Johannes Bjerva, Isabelle Augenstein (2020), *EMNLP 2020*. https://aclanthology.org/2020.emnlp-main.368/
_MAML-style meta-initialization across languages so a model adapts to a new language/task from very few examples — supports meta-learning a boundary-detection initialization transferable into low-resource Arabic segmentation._ · gray · code: https://github.com/copenlu/X-MAML

**[32] Improving Self-training for Cross-lingual Named Entity Recognition with Contrastive and Prototype Learning (ContProto)** — Ran Zhou, Xin Li, Lidong Bing, Erik Cambria, Chunyan Miao (2023), *ACL 2023*. https://aclanthology.org/2023.acl-long.222/
_SOTA for the self-training half: contrastive representation + prototype-based pseudo-labeling to denoise pseudo-labels on unlabeled target text — reusable to pseudo-label unlabeled Arabic for segmentation; only the self-training/denoising component is closed-track-safe (its cross-lingual source labels are external)._ · gray · code: none

**[33] A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers** — Aditi Chaudhary, Jiateng Xie, Zaid Sheikh, Graham Neubig, Jaime G. Carbonell (2019), *EMNLP-IJCNLP 2019*. https://aclanthology.org/D19-1520/
_Cross-lingual transfer + entity-targeted active annotation reaches competitive accuracy with ~1/10 the data — canonical evidence that transfer + targeted labeling of uncertain spans bootstraps a tagger from a tiny corpus like our 174 docs._ · gray · code: https://github.com/Aditi138/EntityTargetedActiveLearning

## Topic & neural text segmentation

**[34] Text Segmentation as a Supervised Learning Task** — Omri Koshorek, Adir Cohen, Noam Mor, Michael Rotman, Jonathan Berant (2018), *NAACL-HLT 2018 (Short)*. https://aclanthology.org/N18-2075/
_Founding supervised-segmentation paper and the Wiki-727K recipe: labels harvested FREE from document structure (headings) — a template for manufacturing synthetic boundary supervision from unlabeled Arabic corpora to escape the 174-doc ceiling._ · closed-legal · code: https://github.com/koomri/text-segmentation

**[35] Text Segmentation by Cross Segment Attention** — Michal Lukasik, Boris Dadachev, Kishore Papineni, Goncalo Simoes (2020), *EMNLP 2020 (Main)*. https://aclanthology.org/2020.emnlp-main.380/
_The cross-segment BERT formulation — classify a boundary from k tokens left / k right — is exactly our binary boundary-after-token setup; a local-context transformer beats hierarchical models and can be shrunk to few parameters (less overfitting on 174 docs)._ · closed-legal · code: none

**[36] Two-Level Transformer and Auxiliary Coherence Modeling for Improved Text Segmentation (CATS)** — Goran Glavaš, Swapna Somasundaran (2020), *AAAI 2020*. https://ojs.aaai.org/index.php/AAAI/article/view/6284
_Adds a free auxiliary self-supervised objective (distinguish coherent vs corrupted sentence orders) as multi-task signal, paired with cross-lingual embeddings for ZERO-SHOT transfer — two levers (pretext task + cross-lingual transfer) for adding signal without more Arabic seg labels._ · gray · code: https://github.com/EducationalTestingService/CATS

**[37] Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence (Transformer²)** — Kelvin Lo, Yuan Jin, Weicong Tan, Ming Liu, Lan Du, Wray Buntine (2021), *Findings of EMNLP 2021*. https://aclanthology.org/2021.findings-emnlp.283/
_Studies transferring knowledge from external single-/pair-wise NLP tasks into the sentence encoder and finds language-specific pretraining beats domain-specific — an argument for building on Arabic-pretrained encoders (AraBERT/CamelBERT) plus adjacent-task pretraining._ · gray · code: https://github.com/kelvinlo-uni/Transformer-squared

**[38] Unsupervised Text Segmentation Using Semantic Relatedness Graphs (GraphSeg)** — Goran Glavaš, Federico Nanni, Simone Paolo Ponzetto (2016), *\*SEM 2016*. https://aclanthology.org/S16-2016/
_Label-free baseline: builds a semantic-relatedness graph over sentences and cliques them into segments using only word embeddings — usable as a weak-supervision/pseudo-label generator over unlabeled Arabic, or a zero-label fallback where the 174 docs give no coverage._ · gray · code: https://github.com/Dobatymo/graphseg-python

## Low-resource token labeling (few-shot / prompt-based)

**[39] Template-Based Named Entity Recognition Using BART** — Leyang Cui, Yu Wu, Jian Liu, Sen Yang, Yue Zhang (2021), *Findings of ACL-IJCNLP 2021*. https://aclanthology.org/2021.findings-acl.161/
_Recasts token labeling as LM template ranking, extracting far more signal per labeled token in few-shot regimes — a prompt-based way to squeeze the 174 docs harder for the boundary/no-boundary decision that dominates our error._ · closed-legal · code: https://github.com/Nealcly/templateNER

## Arabic NLP (encoders, punctuation, morphology)

**[40] AraBERT: Transformer-based Model for Arabic Language Understanding** — Wissam Antoun, Fady Baly, Hazem Hajj (2020), *LREC 2020 (OSACT4 Workshop)*. https://arxiv.org/abs/2003.00104 (also https://aclanthology.org/2020.osact-1.2/)
_The MSA-pretrained encoder backbone (AraBERTv02) inside our ensemble (task #8); its 24GB pretraining corpus is the free monolingual signal we lean on when labeled seg data is capped, and the natural MLM backbone for MELM-style boundary-context augmentation._ · closed-legal · code: https://github.com/aub-mind/arabert
_(merged from 2 entries)_

**[41] ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic** — Muhammad Abdul-Mageed, AbdelRahim Elmadany, El Moatez Billah Nagoudi (2021), *ACL 2021 (Main)*. https://arxiv.org/abs/2101.01785
_MARBERT (dialect+MSA, 1B tweets) and ARBERT give complementary encoders whose diversity our recurrent-memory ensemble (task #8) and multi-scale decorrelation voters (task #16) exploit to cover unseen token contexts — representation-level signal at no data cost._ · closed-legal · code: https://github.com/UBC-NLP/marbert

**[42] Employing a Multilingual Transformer Model for Segmenting Unpunctuated Arabic Text (PDTS)** — Abdullah M. Alshanqiti, Sami Albouq, Ahmad B. Alkhodre, Abdallah Namoun, Emad Nabil (2022), *Applied Sciences 12(20):10559 (MDPI)*. https://www.mdpi.com/2076-3417/12/20/10559
_Arabic-specific prior art: mBERT punctuation detector + generic linguistic rules to split unpunctuated Arabic into clauses — relevant genre/rule context for our NoPnx tracks and a source of hand rules to pair with synthetic data on the legal/Hadith style._ · background · code: none

**[43] Automated Sentence Boundary Detection in Modern Standard Arabic Transcripts using Deep Neural Networks** — Carlos-Emiliano Gonzalez-Gallardo, Elvys Linhares Pontes, Fatiha Sadat, Juan-Manuel Torres-Moreno (2018), *Procedia Computer Science Vol. 142 (ACLing 2018)*. https://www.sciencedirect.com/science/article/pii/S1877050918321896
_One of few Arabic-specific punctuation-agnostic SBD papers: character embeddings + CNN/RNN-attention to recover boundaries on unpunctuated MSA transcripts, matching our NoPnx track; character-level features add morphological signal when token-context labels are scarce._ · closed-legal · code: none

**[44] AraPunc: Arabic Punctuation Restoration Using Transformers** — Abdelrahman Sakr, Marwan Torki (2023), *AICCSA 2023*. https://ieeexplore.ieee.org/document/10479326/
_Punctuation restoration is the closest adjacent self-supervised task: a token-wise Arabic classifier from the Tashkeela corpus (freely stripped/reconstructed, no manual labels), benchmarking AraBERT/MARBERT/XLM-R — a restoration head pretrained on stripped Tashkeela transfers boundary-cue features into our NoPnx segmenter._ · gray · code: none

**[45] Towards the Development of Balanced Synthetic Data for Correcting Grammatical Errors in Arabic** — Ahlam Alrehili, Areej Alhothali (2025), *arXiv:2502.05312*. https://arxiv.org/abs/2502.05312
_A concrete Arabic synthetic-data pipeline (DeBERTa-v3 error-tagging model + ARAT5 back-translation generator) producing 30M+ labeled Arabic pairs from clean text — the template for our semicolon/legal-list generative augmentation (task #18): generate boundary-labeled synthetic Arabic without external seg data._ · closed-legal · code: none

**[46] On the effectiveness of limited-data large language model fine-tuning for Arabic** — Mohamed Alkaoud (2025), *PLOS One*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332419
_Arabic tasks reach/beat SOTA after fine-tuning on ~500 examples with logarithmic scaling — Arabic-specific evidence that our 174-doc budget can be sufficient and lets us forecast gains from adding synthetic data._ · background · code: none

**[47] Data augmentation for Arabic text classification: a review of current methods, challenges and prospective directions** — Samia F. Abdhood, Nazlia Omar, Sabrina Tiun (2025), *PeerJ Computer Science 11:e2685*. https://peerj.com/articles/cs-2685/
_Arabic-specific catalogue of augmentation techniques (back-translation, EDA-style edits, AraBERT/AraGPT2 MLM insertion, GAN/generative, up-to-+13 F1) with pitfalls under Arabic morphology/diacritics — the practical filter for adapting English-centric DAGA/MELM/Dai-Adel methods and flags morphology-driven label-corruption near boundaries._ · background · code: none
_(merged from 3 entries; one entry mis-attributed authors as "Alhanouf Alsaeed et al." — same title/venue/URL)_

## Arabic sentence segmentation — the benchmark

**[48] Arabic Sentence Segmentation Across Genres and Punctuation Conditions (AraSEG)** — Mohammed Elkholy, Khalid N. Elmadani, Nizar Habash, Bashar Alhafni (2026), *arXiv:2606.08025*. https://arxiv.org/abs/2606.08025
_The AraSeg benchmark/task paper itself: defines the eight-genre corpus (incl. Quran/religious text), punctuation conditions, and the binary boundary-after-token setup we compete on; finds lightweight encoders and dependency-parser models beat LLMs in the hardest NoPnx settings, that performance SATURATES with training data, and cross-genre generalization stays hard — diagnoses our 174-doc / unseen-context error wall and rules out naive LLM scaling. Baseline and design of record._ · closed-legal / background · code: https://github.com/mbzuai-nlp/araseg-shared-task-2026 (also cited as /araseg; some entries none)
_(merged from 7 duplicate entries)_

## Punctuation restoration & discourse/EDU segmentation (auxiliary tasks)

**[49] Real-world sentence boundary detection using multitask learning: A case study on French** — KyungTae Lim, Jungyeul Park (2024), *Natural Language Engineering 30(1):150-170*. https://www.cambridge.org/core/journals/natural-language-engineering/article/realworld-sentence-boundary-detection-using-multitask-learning-a-case-study-on-french/BBD2C47C7ADC6B29B556F78DB09D7F50
_Most on-target for the multitask angle: jointly trains POS + NER auxiliary heads with the SBD head; joint signal lifts unpunctuated-boundary F1 to 89.41% where Punkt-style tools collapse — exactly the NoPnx regime driving our errors; POS/NER labels are producible on the closed corpus with off-the-shelf Arabic taggers, no external seg data._ · closed-legal · code: none

**[50] Punctuation Restoration Improves Structure Understanding Without Supervision** — Junghyun Min, Minho Lee, Woochul Lee, Yeonsoo Lee (2025), *RepL4NLP @ ACL 2025; arXiv:2402.08382*. https://aclanthology.org/2025.repl4nlp-1.10/
_Validates the core hypothesis: punctuation restoration as an unsupervised pre-training/auxiliary objective improves in- and out-of-distribution structure tasks (NER, chunking, POS); targets are free to derive from raw Arabic by masking punctuation — a pure label-free signal source for pretraining our encoder before the 174-doc fine-tune._ · closed-legal · code: none

**[51] Boosting Punctuation Restoration with Data Generation and Reinforcement Learning** — Viet Dac Lai, Abel Salinas, Hao Tan, Trung Bui, Quan Tran, Seunghyun Yoon, Hanieh Deilamsalehy, Franck Dernoncourt, Thien Huu Nguyen (2023), *Interspeech 2023; arXiv:2307.12949*. https://arxiv.org/abs/2307.12949
_A recipe for generating synthetic supervision for a boundary-marking task from abundant unlabeled in-topic text via a large generative LM, then filtering with RL — a template for producing synthetic boundary-labeled Arabic (legal/Hadith) data without external seg corpus._ · closed-legal · code: none

**[52] The DISRPT 2023 Shared Task on Elementary Discourse Unit Segmentation, Connective Detection, and Relation Classification** — Chloé Braud, Yang Janet Liu, Eleni Metheniti, Philippe Muller, Laura Rivière, Attapol Rutherford, Amir Zeldes (2023), *DISRPT 2023 @ ACL*. https://aclanthology.org/2023.disrpt-1.1/
_Defines EDU-segmentation (finer-than-sentence boundaries over 26 datasets/13 languages incl. an Arabic RST treebank) and reports winning multilingual-transformer transfer systems — a near-adjacent auxiliary/pretraining task whose boundaries co-occur with sentence boundaries; the Arabic RST corpus is a candidate extra boundary-supervision source._ · gray · code: https://github.com/disrpt/sharedtask2023

**[53] Discriminative Self-training for Punctuation Prediction** — Qian Chen, Wen Wang, Mengzhe Chen, Qinglin Zhang (2021), *Interspeech 2021*. https://arxiv.org/abs/2104.10339
_Self-training with a weighted loss + discriminative label smoothing exploiting unlabeled transcripts for a boundary/punctuation task near-isomorphic to boundary-after-token — shows how to weight noisy pseudo-labels so self-training on unlabeled Arabic helps rather than amplifying errors._ · gray · code: none

**[54] Punctuation Restoration using Transformer Models for High- and Low-Resource Languages** — Tanvirul Alam, Akib Khan, Firoj Alam (2020), *W-NUT @ EMNLP 2020*. https://aclanthology.org/2020.wnut-1.18/
_Adjacent-task transformer token-classification with an augmentation strategy tuned for low-resource languages — a reusable augmentation-factor recipe and an adjacent-task pretraining objective for boundary prediction when labeled seg data is capped._ · gray · code: https://github.com/xashru/punctuation-restoration

## Weak & distant supervision

**[55] Named Entity Recognition without Labelled Data: A Weak Supervision Approach** — Pierre Lison, Aliaksandr Hubin, Jeremy Barnes, Samia Touileb (2020), *ACL 2020 (Long)*. https://aclanthology.org/2020.acl-main.679/
_Canonical recipe for our bottleneck: write many noisy labelling functions over UNLABELED target text and merge with an HMM that learns each source's accuracy, then train a sequence model on the aggregate. Portable to boundary-after-token: LFs = punctuation, connectives (wa-/fa-/thumma), verb-initial cues, length priors — zero external seg data._ · closed-legal · code: https://github.com/NorskRegnesentral/weak-supervision-for-NER

**[56] skweak: Weak Supervision Made Easy for NLP** — Pierre Lison, Jeremy Barnes, Aliaksandr Hubin (2021), *ACL 2021: System Demonstrations*. https://aclanthology.org/2021.acl-demo.40/
_The engineering path for Lison-2020: a maintained toolkit implementing labelling functions + unsupervised HMM aggregation for sequence labelling (the Snorkel-for-sequences gap) — stand up a rule-voter over unlabeled Arabic quickly instead of hand-rolling the aggregation model._ · closed-legal · code: https://github.com/NorskRegnesentral/skweak

**[57] A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios** — Michael A. Hedderich, Lukas Lange, Heike Adel, Jannik Strötgen, Dietrich Klakow (2021), *NAACL-HLT 2021*. https://aclanthology.org/2021.naacl-main.201/
_The map of the whole solution space: categorizes data augmentation, distant/weak supervision, and cross-lingual transfer as the three levers for few-labels-plentiful-unlabeled — use to pick/sequence methods and cite the pre-registered rationale for closed vs open track._ · background · code: none

## Distillation & controllable generation from model data

**[58] Symbolic Knowledge Distillation: from General Language Models to Commonsense Models** — Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi (2022), *NAACL 2022 (arXiv:2110.07178)*. https://aclanthology.org/2022.naacl-main.341/
_The canonical machine-to-corpus-to-machine recipe: a big LM authors a text corpus, a trained critic filters it for quality, a small student learns from the filtered synthetic corpus and beats human data — the exact template for generating boundary-labeled Arabic examples with a strong LM + quality critic._ · gray · code: https://github.com/peterwest/symbolic-knowledge-distillation

**[59] Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations** — Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, Ming Yin (2023), *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.647/
_Rigorously measures when LLM-generated synthetic data helps vs hurts: subjectivity/ambiguity at task and instance level predicts failure — warns that ambiguous boundary contexts (our 88-89% error region) are precisely where synthetic labels are least trustworthy._ · gray · code: none

**[60] DiLM: Distilling Dataset into Language Model for Text-level Dataset Distillation** — Aru Maekawa, Satoshi Kosugi, Kotaro Funakoshi, Manabu Okumura (2024), *Findings of NAACL 2024*. https://aclanthology.org/2024.findings-naacl.199/
_Dataset distillation for discrete text: trains a generator LM to emit informative synthetic text samples (not embeddings), so the distilled set transfers across architectures — a principled way to synthesize a compact high-signal Arabic seg corpus from the 174 docs that trains encoders other than the distiller._ · gray · code: https://github.com/arumaekawa/DiLM

**[61] Dataset Distillation: A Comprehensive Review** — Ruonan Yu, Songhua Liu, Xinchao Wang (2023), *IEEE TPAMI 2023 (arXiv:2301.07014)*. https://arxiv.org/abs/2301.07014
_The reference survey: taxonomy, gradient/trajectory/distribution-matching methods, and the known hard case of discrete text — use to scope which distillation family (if any) is realistic for a 174-doc token-labeled Arabic corpus and avoid methods known not to transfer to NLP._ · background · code: none

## Curriculum & active learning

**[62] Data-efficient Active Learning for Structured Prediction with Partial Annotation and Self-Training** — Zhisong Zhang, Emma Strubell, Eduard Hovy (2023), *Findings of EMNLP 2023*. https://arxiv.org/abs/2305.12634
_Marries partial (sub-structure) annotation with self-training and an adaptive error estimator for structured prediction — the framework to squeeze maximum signal from the fixed 174 docs by hand-labeling only the most uncertain boundaries and auto-filling the rest._ · closed-legal · code: none

## Hadith / Islamic / legal-genre corpora

**[63] Text Segmentation Using N-grams to Annotate Hadith Corpus** — Shatha Altammami, Eric Atwell, Ammar Alsalka (2019), *WANLP/WACL @ ACL 2019 (Fourth/3rd Arabic NLP Workshop)*. https://aclanthology.org/W19-5605/
_Genre-matched precedent: segments Classical Arabic Hadith into Isnad/Matn using cheap n-gram cue phrases and builds an annotated corpus — a template for rule/weak-supervision boundary cues in the classical/legal register, and a candidate adjacent-domain corpus for transfer or synthetic pretraining._ · background · code: none
_(merged from 2 entries)_

**[64] Noor-Ghateh: A Benchmark Dataset for Evaluating Arabic Word Segmenters in Hadith Domain** — Huda AlShuhayeb, Behrouz Minaei-Bidgoli, Mohammad E. Shenassa, Sayyed-Ali Hossayni (2023), *arXiv:2307.09630*. https://arxiv.org/abs/2307.09630
_A ~223k-word expert-annotated Arabic word-segmentation benchmark in the classical Hadith/legal (Shariat al-Islam) genre, benchmarking Farasa/CAMeL/ALP — characterizes classical/religious-legal tokenization, a candidate in-genre unlabeled/structured text source and a sanity check for morphological pre-tokenization feeding the segmenter._ · background · code: none

**[65] Fabricated Hadith Detection: A Novel Matn-Based Approach With Transformer Language Models** — Mohammad Al-Sarem, Faisal Saeed, Zeyad Ghaleb Al-Mekhlafi, et al. (2022), *IEEE Access, vol. 10*. https://ieeexplore.ieee.org/document/9931123/
_Genre-matched Hadith transformer classification on a curated matn corpus — evidence that fine-tuning Arabic PLMs on small Hadith/religious-legal text is feasible, and a pointer to Hadith text as an unlabeled/adjacent classical-Arabic domain to pretrain or augment segmentation for that genre._ · background · code: none

**[66] MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset** — Tobias Brugger, Matthias Stürmer, Joel Niklaus (2023), *ICAIL 2023*. https://dl.acm.org/doi/10.1145/3594536.3595132
_Legal-genre sentence structure breaks off-the-shelf SBD; benchmarks CRF/BiLSTM-CRF/transformer + zero-shot cross-lingual transfer — genre-matched prior art and a transfer source for our legal-doc segmentation failures (Arabic not among its 6 languages), useful baseline design._ · background · code: https://github.com/tobiasbrugger/MultiLegalSBD

**[67] Open-Source Boundary-Annotated Qur'an Corpus for Arabic and Phrase Breaks Prediction in Classical and MSA Text** — Majdi Sawalha, Claire Brierley, Eric Atwell (2014), *LREC 2014 (corpus; phrase-break method LREC 2012)*. https://aclanthology.org/L14-1114/
_A boundary-annotated Classical/MSA Arabic corpus (77k words, 8,230 sentences) with prosodic-syntactic break tags — a rare adjacent-task Arabic boundary resource to seed adjacent-task pretraining or weak boundary supervision without touching external sentence-seg labels._ · background · code: none

---

## TOP 8 TO READ FIRST for the data-scarcity attack

1. **[1] DAGA** — the single most on-target method: an LM that generates fresh boundary-labeled sentences from only the 174 docs, no external labels; biggest gains at smallest gold.
2. **[16] SeqUST** — the most directly transferable self-training recipe for our per-token boundary task; MC-dropout uncertainty gates pseudo-labels on unlabeled Arabic, attacking unseen-context errors.
3. **[9] MELM** — label-preserving masked-LM augmentation (adaptable on an Arabic MLM) that guarantees boundary-label alignment while minting novel unseen contexts.
4. **[48] AraSEG benchmark** — the task/error-wall paper; defines the corpus, confirms the 174-doc saturation and NoPnx difficulty every intervention must beat. Read to align method choices with the official eval.
5. **[55] Lison 2020 weak supervision** (+ **[56] skweak** toolkit) — turns hand-written boundary rules (connectives, verb-initial cues, punctuation, length) over unlabeled Arabic into an aggregated training signal with zero external seg data.
6. **[27] WtP** — self-supervised newline-recovery segmentation reproducible from unlabeled Arabic; the closed-track-defensible pretraining that already motivated our v16-B2 arm.
7. **[11] Dai & Adel 2020** — the cheap label-preserving augmentation primitives (largest gains on small sets) to benchmark first before heavier generative methods, with the token-label-misalignment caveat spelled out.
8. **[6] SunGen** (pair with **[59]** limitations study) — the noise-filtering/quality-reweighting layer that stops noisy synthetic boundaries from poisoning the segmenter — the guardrail that makes [1]/[9] safe to deploy.