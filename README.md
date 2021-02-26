# A Survey on Spoken Language Understanding: Recent Advances and New Frontiers

![](https://img.shields.io/badge/Status-building-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) 

This repo contains a list of papers, codes, datasets, leaderboards in SLU field. If you found any error, please don't hesitate to open an issue.

## Contributor

Contributed by [Libo Qin](https://github.com/yizhen20133868), [Tianbao Xie](https://github.com/Timothyxxx), [Yudi Zhang](https://github.com/zyd-project),  [Lehan Wang](https://github.com/luxuriant0116)


Thanks for supports from my adviser [Wanxiang Che](http://ir.hit.edu.cn/~car/english.htm)!

## Introduction

Spoken language understanding (SLU) is a critical component in task-oriented dialogue systems. It usually consists of intent and slot filling task to extract semantic constiuents from the natrual language utterances.

For the purpose of alleviating our pressure in article/dataset collation, we worked on sorting out the relevant data sets, papers, codes and lists of SLU in this project.

At present, the project has been completely open source, including:
1. **SLU domain dataset sorting table:** you can quickly index the data set you want to use in it to help you quickly understand the general scale, basic structure, content, characteristics, source and acquisition method of this dataset.
2. **Articles and infos in different directions in the field of SLU:** we classify and arrange the papers according to the current mainstream directions. Each line of the list contains not only the title of the paper, but also the year of publication, the source of publication, the paper link and code link for quick indexing, as well as the data set used.
3. **Leaderboard list on the mainstream datasets of SLU:** we sorted out the leaderboard on the mainstream datasets, and distinguished them according to pre-trained or not. In addition to the paper name and related scores, each line also has links to year, paper and code.

The taxonomy of our survey can be summarized into this picture below.

![SLUs-taxonomy](./SLUs-taxonomy.png)

## Quick path

- [Resources](#resources)
  * [survey paper links](#survey-paper-links)
  * [recent open-sourced code](#recent-open-sourced-code)
  * [Single Model](#single-model)
  * [Joint Model](#joint-model)
  * [Complex SLU Model](#complex-slu-model)
- [Dataset](#dataset)
- [Direction](#direction)
  * [Single Slot Filling](#single-slot-filling)
  * [Single Intent Detection](#single-intent-detection)
  * [Joint Model](#joint-model-1)
  * [Contextual SLU](#contextual-slu)
  * [Chinese SLU](#chinese-slu)
  * [Cross-domain SLU](#cross-domain-slu)
  * [Cross-lingual SLU](#cross-lingual-slu)
  * [Low-resource SLU](#low-resource-slu)
    + [Few-shot SLU](#few-shot-slu)
    + [Zero-shot SLU](#zero-shot-slu)
    + [Unsupervised SLU](#unsupervised-slu)
- [LeaderBoard](#leaderboard)
  * [ATIS](#atis)
    + [Non-pretrained model](#non-pretrained-model)
    + [+ Pretrained model](#--pretrained-model)
  * [SNIPS](#snips)
    + [Non-pretrained model](#non-pretrained-model-1)
    + [+ Pretrained model](#--pretrained-model-1)

## Resources

### survey paper links

1. **Spoken language understanding: Systems for extracting semantic information from speech** `book` [[pdf]](https://ieeexplore.ieee.org/book/8042134)
2. **Recent Neural Methods on Slot Filling and Intent Classification**  `COLING 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) 
3. **A survey of joint intent detection and slot-filling models in natural**  `arxiv 2021` [[pdf]](https://arxiv.org/pdf/2101.08091.pdf) 

### recent open-sourced code

### Single Model

1. **Few-shot Slot Tagging with  Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection  Network** (SNIPS) `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging) 
2. **Sequence-to-Sequence Data  Augmentation for Dialogue Language Understanding** (ATIS/Stanford Dialogue Dataset) `COLING 2018` [[pdf]](https://arxiv.org/pdf/1807.01554.pdf) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) 

### Joint Model

1. **A Co-Interactive Transformer for Joint Slot Filling and Intent Detection**(ATIS/SNIPS) `ICASSP 2021` [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) [[code]](https://github.com/kangbrilliant/DCA-Net)
2. **Joint Slot Filling and Intent  Detection via Capsule Neural Networks** (ATIS/SNIPS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) [[code]](https://github.com/czhang99/Capsule-NLU) 
3. **BERT for Joint Intent  Classification and Slot Filling** (ATIS/SNIPS/Stanford Dialogue Dataset) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) [[code]](https://github.com/monologg/JointBERT) 
4. **A Novel Bi-directional  Interrelated Model for Joint Intent Detection and Slot Filling** (ATIS/Stanford Dialogue Dataset,SNIPS) `ACL 2019` [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) [[code]](https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU) 
5. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 
6. **Slot-Gated Modeling for Joint  Slot Filling and Intent Prediction** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) [[code]](https://github.com/MiuLab/SlotGated-SLU) 
7. **Joint Online Spoken Language  Understanding and Language Modeling with Recurrent Neural Networks** (ATIS) `SIGDIAL 2016` [[pdf]](https://www.aclweb.org/anthology/W16-3603.pdf) [[code]](https://github.com/HadoopIt/joint-slu-lm)

### Complex SLU Model

1. **How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues** (DSTC4) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-1194.pdf) [[code]](https://github.com/MiuLab/Time-Decay-SLU) 
2. **Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning** (DSTC4) `IJCNLP 2017` [[pdf]](https://www.aclweb.org/anthology/I17-2028.pdf) [[code]](https://github.com/MiuLab/Spk-Dialogue) 
3. **Dynamic time-aware attention to speaker roles and contexts for spoken language understanding** (DSTC4) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1710.00165.pdf) [[code]](https://github.com/MiuLab/Time-SLU) 
4. **Injecting Word Information with Multi-Level Word Adapter for Chinese Spoken Language Understanding** (CAIS/ECDT-NLU) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.03903.pdf) [[code]](https://github.com/AaronTengDeChuan/MLWA-Chinese-SLU) 
5. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 
6. **Coach: A Coarse-to-Fine  Approach for Cross-domain Slot Filling** (SNIPS) `ACL 2020` [[pdf]](https://arxiv.org/pdf/2004.11727.pdf) [[code]](https://github.com/zliucr/coach)
7. **CoSDA-ML: Multi-Lingual  Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP** (SC2/4/MLDoc/Multi WOZ/Facebook Multilingual SLU Dataset) `IJCAI 2020` [[pdf]](https://arxiv.org/pdf/2006.06402.pdf) [[code]](https://github.com/kodenii/CoSDA-ML) 
8. **Cross-lingual Spoken Language  Understanding with Regularized Representation Alignment** (Multilingual spoken language understanding (SLU) dataset) `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.14510.pdf) [[code]]([https://github.com/zliucr/crosslingual-slu](https://github.com/zliucr/crosslingual-slu.)) 
9. **Attention-Informed  Mixed-Language Training for Zero-shot Cross-lingual Task-oriented Dialogue  Systems** (Facebook Multilingual SLU Dataset/(DST)MultiWOZ) `AAAI 2020` [[pdf]](https://arxiv.org/pdf/1911.09273.pdf) [[code]](https://github.com/zliucr/mixedlanguage-training) 
10. **MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark** (MTOP/Multilingual ATIS) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2008.09335.pdf) [[code]]() 
11. **Neural Architectures for  Multilingual Semantic Parsing** (GEO/ATIS) `ACL 2017` [[pdf]](https://www.aclweb.org/anthology/P17-2007.pdf) [[code]](http://statnlp.org/research/sp/) 
12. **Few-shot Learning for Multi-label Intent Detection** `AAAI 2021` [[pdf]](https://arxiv.org/abs/2010.05256) [[code]](https://github.com/AtmaHou/FewShotMultiLabel) 
13. **Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network** `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging)



## Dataset

| Name                               | Intro                                                        | Links                                                        | Multi/Single  Turn(M/S) | Detail                                                       | Size & Stats                                                 | Label                                                        |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ATIS                               | 1. The ATIS (Airline Travel Information Systems) dataset (Tur  et al., 2010) is widely used in SLU research 2. For natural language  understanding | Download:         1.https://github.com/yizhen20133868/StackPropagation-SLU/tree/master/data/atis         2.https://github.com/yvchen/JointSLU/tree/master/data      Paper:      https://www.aclweb.org/anthology/H90-1021.pdf | S                       | Airline Travel Information     However, this data set has been shown to have a serious skew problem on intent | Train: 4478 Test: 893 120 slot and 21 intent                 | Intent Slots                                                 |
| SNIPS                              | 1. Collected by Snips for model evaluation. 2. For natural   language understanding 3. Homepage:   https://medium.com/snips-ai/benchmarking-natural-language-understanding-systems-google-facebook-microsoft-and-snips-2b8ddcf9fb19 | Download:       https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines      Paper:      https://arxiv.org/pdf/1805.10190.pdf | S                       | 7 task: Weather,play music, search, add to list, book, moive | Train:13,084 Test:700 7 intent 72 slot labels                | Intent Slots                                                 |
| Facebook Multilingual SLU  Dataset | 1 Contains English, Spanish, and Thai across the weather,  reminder, and alarm domains      2 For cross-lingual SLU | Download:      https://fb.me/multilingual_task_oriented_data      Paper:      https://www.aclweb.org/anthology/N19-1380.pdf | S                       | Utterances are manually translated and annotated             | Train: English 30,521; Spanish 3,617; Thai 2,156     Dev: English 4,181; Spanish 1,983; Thai 1,235     Test: English 8,621; Spanish 3,043; Thai 1,692     11 slot and 12 intent | Intent Slots                                                 |
| MIT Restraunt Corpus               | MIT corpus contains train set and test set in BIO format for  NLU | Download:      https://groups.csail.mit.edu/sls/downloads/restaurant/ | S                       | It is a single-domain dataset, which is associated with  restaurant reservations. MR contains ‘open-vocabulary’ slots, such as  restaurant names | Train:7760      Test:1521                                    | Slots                                                        |
| MIT Movie Corpus                   | The MIT Movie Corpus is a semantically tagged training and  test corpus in BIO format.      The eng corpus are simple queries, and the trivia10k13 corpus are more  complex queries. | Download:      https://groups.csail.mit.edu/sls/downloads/movie/ | S                       | The MIT movie corpus consists of two single-domain datasets:  the movie eng (ME) and movie trivia (MT) datasets. While both datasets  contain queries about film information, the trivia queries are more complex  and specific | eng Corpus:     Train:9775     Test:2443     Trivia Corpus:     Train:7816     Test:1953 | Slots                                                        |
| Multilingual ATIS                  | ATIS was manually translated into Hindi and Turkish          | Download:      It has been put into LDC, and you can download it if you are own a   membership or pay for it      Paper:      http://shyamupa.com/papers/UFTHH18.pdf | S                       | 3 languages                                                  | On the top of ATIS dataset, 893 and 715 utterances from the  ATIS test split were translated     and annotated for Hindi and Turkish evaluation respectively     also translated and annotated 600(each language     separately) utterances from the ATIS train split to use as  supervision     In total 37,084 training examples  and 7,859 test examples | Intent Slots                                                 |
| Multilingual ATIS++                | Extends Multilingual ATIS corpus to     nine languages across four language families | Download:      contact multiatis@amazon.com.      Paper:      https://arxiv.org/abs/2004.14353 | S                       | 10 languages                                                 | check the paper to find the full table of description     (to many info ,have no enough space here) | Intent Slots                                                 |
| Almawave-SLU                       | 1. A dataset for Italian SLU     2. Was generated through a semi-automatic procedure from SNIPS | Download:      contact [first name initial\].[last name]@almawave.it for the dataset      (any author in this paper)      Paper:      https://arxiv.org/pdf/1907.07526.pdf | S                       | 6 domains: Music, Restaurants, TV, Movies,     Books, Weather | Train: 7,142     Validation: 700     Test: 700     7 intents and 39 slots | Intent Slots                                                 |
| Chatbot Corpus                     | 1. Chatbot Corpus is based on questions gathered by a Telegram  chatbot which answers questions about public transport connections,  consisting of 206 questions     2. For intent classification test | Download:      https://github.com/sebischair/NLU-Evaluation-Corpora      Paper:      https://www.aclweb.org/anthology/W17-5522.pdf | S                       | 2 Intents: Departure Time, Find Connection     5 entity types: StationStart, StationDest, Criterion, Vehicle, Line | Train: 100     Test: 106                                     | Intent Entity                                                |
| StackExchange Corpus               | 1. StackExchange Corpus is based on data from two  StackExchange platforms: ask ubuntu and Web Applications     2. Gathers 290 questions and answers in total, 100 from Web Applications  and 190 from ask ubuntu     3. For intent classification test | Download:         https://github.com/sebischair/NLU-Evaluation-Corpora       Paper:      https://www.aclweb.org/anthology/W17-5522.pdf | S                       | Ask ubuntu Intents: “Make Update”, “Setup Printer”, “Shutdown  Computer”, and “Software Recommendation”     Web Applications Intents: “Change  Password”, “Delete Account”, “Download Video”, “Export Data”, “Filter Spam”,  “Find Alternative”, and “Sync Accounts” | Total: 290     Ask ubuntu: 190     Web Application: 100      | Intent Entity                                                |
| MixSNIPS/MixATIS                   | multi-intent dataset based on SNIPS and ATIS                 | Download:      https://github.com/LooperXX/AGIF/tree/master/data      Paper:      https://www.aclweb.org/anthology/2020.findings-emnlp.163.pdf | S                       | using conjunctions, connecting sentences with different  intents forming a ratio of 0.3,0.5 and 0.2 for sentences has which 1,2 and 3  intents, respectively | Train:12,759 utterances     Dev:4,812 utterances     Test:7,848 utterances | Intent(Multi),Slots                                          |
| TOP semantic parsing               | 1,Hierarchical annotation scheme for semantic parsing     2,Allows the representation of compositional queries     3,Can be efficiently and accurately parsed by standard constituency parsing  models | Download:      http://fb.me/semanticparsingdialog      Paper:      https://www.aclweb.org/anthology/D18-1300.pdf | S                       | focused on navigation, events, and navigation to events     evaluation script can be run from evaluate.py within the dataset | 44783 annotations     Train:31279     Dev:4462     Test:9042 | Inten ,Slots in Tree  format                                 |
| MTOP: Multilingual TOP             | 1.An almost-parallel multilingual task-oriented semantic  parsing dataset covering 6 languages and 11 domains.     2.the first multilingual dataset that contain compositional representations  that allow complex nested queries.     3.the dataset creation: i) generating synthetic utterances and annotating  in English, ii) translation, label transfer, post-processing, post editing  and filtering for other languages | Download:      https://fb.me/mtop_dataset      Paper:      https://arxiv.org/pdf/2008.09335.pdf | S                       | 6 languages (both high  and low resource): English, Spanish, French, German, Hindi and Thai.       a mix of both simple as well as  compositional nested queries across 11 domains, 117 intents and 78 slots. | 100k examples in total for 6 languages.     Roughly divided into 70:10:20 percent splits for train,eval and test. | Two kinds of  representations:     1.flat representatiom: Intent and slots     2.compositional decoupled representations:nested intents inside slots     More details 3.2 section in the paper |
| CAIS                               | Collected from real world speaker systems  with manual annotations of slot tags and intent labels | [https://github.com/Adaxry/CM-Net](https://github.com/Adaxry/CM-Net/tree/master/CAIS) | S                       | 1.The utterances were collected from the Chinese Artificial Intelligence Speakers 2.Adopt the BIOES tagging scheme for slots instead of the BIO2 used in the ATIS 3.intent labels are partial to the PlayMusic option | Train: 7,995 utterances Dev: 994 utterances Test: 1024 utterances | slots tags and intent labels                                 |
| Simulated Dialogues dataset        | machines2machines (M2M)                                      | Download: https://github.com/google-research-datasets/simulated-dialogue Paper: http://www.colips.org/workshop/dstc4/papers/60.pdf | M                       | Slots: Sim-R (Restaurant)        price_range, location, restaurant_name, category, num_people, date, time Sim-M (Movie)        theatre_name, movie, date, time, num_people Sim-GEN (Movie):theatre_name, movie, date, time, num_people | Train: Sim-R:1116 Sim-M:384 Sim-GEN:100k Dev: Sim-R:349 Sim-M:120 Sim-GEN:10k Test: Sim-R:775 Sim-M:264 Sim-GEN:10k | Dialogue state User's act,slot,intent System's act,slot      |
| Schema-Guided Dialogue Dataset(SGD) | dialogue simulation(auto based on identified scenarios), word-replacement and human intergration as paraphrasing| Download:  https://github.com/google-researchdatasets/dstc8-schema-guided-dialogue Paper: https://arxiv.org/pdf/1909.05855.pdf | M                       | domains:16,dialogues:16142,turns:329964,acg turns per dialogue:20.44,total unique tokens:30352,slots:214,slot values:14319| NA |   Scheme Representation: service_name;description;slot's name,description,is_categorial,possible_values;intent's name,description,is_transactional,required_slots,optional_slots,result_slots. Dialogue Representation: dialogue_id,services,turns,speaker,utterance,frame,service,slot's name,start,exclusive_end;action's act,slot,values,canonical_values;service_call's method,parameters;service_results,state's active_intent,requested_slots,slot_values |


## Direction

### Single Slot Filling

1. **Few-shot Slot Tagging with  Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection  Network** (SNIPS) `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging) 
2. **A Hierarchical Decoding Model  For Spoken Language Understanding From Unaligned Data** (DSTC2) `ICASSP 2019` [[pdf]](https://arxiv.org/pdf/1904.04498.pdf) 
3. **Utterance Generation With  Variational Auto-Encoder for Slot Filling in Spoken Language Understanding** (ATIS/SNIPS/MIT Corpus) `IEEE Signal Processing Letters 2019` [[pdf]]([https://ieeexplore.ieee.org/document/8625384](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8625384)) 
4. **Data Augmentation with Atomic  Templates for Spoken Language Understanding** (ATIS) `EMNLP 2019` [[pdf]](https://arxiv.org/pdf/1908.10770.pdf) 
5. **A New Concept of Deep  Reinforcement Learning based Augmented General Sequence Tagging System** (ATIS/CNLL-2003) `COLING 2018` [[pdf]](https://www.aclweb.org/anthology/C18-1143.pdf) 
6. **Improving Slot Filling in  Spoken Language Understanding with Joint Pointer and Attention** (DSTC2) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/P18-2068.pdf) 
7. **Sequence-to-Sequence Data  Augmentation for Dialogue Language Understanding** (ATIS/Stanford Dialogue Dataset) `COLING 2018` [[pdf]](https://arxiv.org/pdf/1807.01554.pdf) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) 
8. **ENCODER-DECODER WITH  FOCUS-MECHANISM FOR SEQUENCE LABELLING BASED SPOKEN LANGUAGE UNDERSTANDING** (ATIS) `ICASSP 2017` [[pdf]]([https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=79532433](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7953243)) 
9. **Neural Models for Sequence  Chunking** (ATIS/LARGE) `AAAI 2017` [[pdf]](https://arxiv.org/pdf/1701.04027.pdf) 
10. **Bi-directional recurrent  neural network with ranking loss for spoken language understanding** (ATIS) `IEEE 2016` [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) 
11. **Labeled Data Generation with  Encoder-decoder LSTM for Semantic Slot Filling** () `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/7ffe/83d7dd3a474e15ccc2aef412009f100a5802.pdf) 
12. **SYNTAX OR SEMANTICS?  KNOWLEDGE-GUIDED JOINT SEMANTIC FRAME PARSING** (ATIS/Cortana) `IEEE Workshop on Spoken Language Technology 2016` [[pdf]](https://www.csie.ntu.edu.tw/~yvchen/doc/SLT16_SyntaxSemantics.pdf) 
13. **BI-DIRECTIONAL RECURRENT  NEURAL NETWORK WITH RANKING LOSS FOR SPOKEN LANGUAGE UNDERSTANDING** (ATIS) `ICASSP 2016` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7472841) 
14. **Leveraging Sentence-level  Information with Encoder LSTM for Semantic Slot Filling** (ATIS) `EMNLP 2016` [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) 
15. **Labeled Data Generation with  Encoder-decoder LSTM for Semantic Slot Filling** (ATIS) `INTERSPEECH 2016` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0727.PDF) 
16. **Using Recurrent Neural  Networks for Slot Filling in Spoken Language Understanding** (ATIS) `IEEE/ACM TASLP 2015` [[pdf]](https://ieeexplore.ieee.org/document/6998838) 
17. **Using Recurrent Neural  Networks for Slot Filling in Spoken Language Understanding** (ATIS) `IEEE/ACM Transactions on Audio, Speech, and Language  Processing 2015` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6998838) 
18. **Recurrent  Neural Network Structured Output Prediction for Spoken Language Understanding** (ATIS) `- 2015` [[pdf]](http://speech.sv.cmu.edu/publications/liu-nipsslu-2015.pdf) 
19. **Spoken Language Understanding  Using Long Short-Term Memory Neural Networks** (ATIS) `IEEE 2014` [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) 
20. **Recurrent conditional random  field for language understanding** (ATIS) `IEEE 2014` [[pdf]](https://ieeexplore.ieee.org/document/6854368) 
21. **Recurrent Neural Networks for  Language Understanding** (ATIS) `INTERSPEECH 2013` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kaisheny-338_file_paper.pdf) 
22. **Investigation of  Recurrent-Neural-Network Architectures and Learning Methods for Spoken  Language Understanding** (ATIS) `ISCA 2013` [[pdf]](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_3771.pdf) 
23. **Large-scale personal assistant  technology deployment: the siri experience** () `INTERSPEECH 2013` [[pdf]](https://isca-speech.org/archive/archive_papers/interspeech_2013/i13_2029.pdf) 

### Single Intent Detection

1. **Zero-shot User Intent  Detection via Capsule Neural Networks** (SNIPS/CVA) `EMNLP 2018` [[pdf]](https://arxiv.org/pdf/1809.00385.pdf) 
3. **Intention Detection Based on Siamese Neural Network With Triplet Loss** (SNIPS/ATIS/Facebook multilingual datasets/ Daily Dialogue/MRDA) `IEEE Acess 2020` [[pdf]](https://ieeexplore.ieee.org/document/9082602) 
5. **Multi-Layer Ensembling Techniques for Multilingual Intent Classification** (ATIS) `arXiv 2018` [[pdf]](https://arxiv.org/pdf/1806.07914.pdf) 
6. **Deep Unknown Intent Detection with Margin Loss** (SNIPS/ATIS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1906.00434.pdf) 
7. **Subword Semantic Hashing for Intent Classification on Small Datasets** (The Chatbot Corpus/The AskUbuntu Corpus) `IJCNN 2019` [[pdf]](https://arxiv.org/pdf/1810.07150.pdf) 
8. **Dialogue intent classification with character-CNN-BGRU networks** (the Chinese Wikipedia dataset) `Multimedia Tools and Applications 2018` [[pdf]](https://link.springer.com/article/10.1007/s11042-019-7678-1)  
9. **Joint Learning of Domain Classification and Out-of-Domain Detection with Dynamic Class Weighting for Satisficing False Acceptance Rates** (Alexa) `InterSpeech 2018` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1581.pdf)  
8. **Recurrent neural network and  LSTM models for lexical utterance classification** (ATIS/CB) `INTERSPEECH 2015` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) 
9. **Adversarial Training for Multi-task and Multi-lingual Joint Modeling of Utterance Intent Classification** (collected by the author) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/D18-1064.pdf) 
10. **Exploiting Shared Information for Multi-Intent Natural Language Sentence Classification** (collected by the author) `ISCA 2013` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/08/double_intent.pdf)  


### Joint Model

1.  **A Co-Interactive Transformer for Joint Slot Filling and Intent Detection**(ATIS/SNIPS) `ICASSP 2021` [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) [[code]](https://github.com/kangbrilliant/DCA-Net)
2. **Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding**(ATIS/SNIPS) `AAAI 2020` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6499/6355) 
3. **Joint Slot Filling and Intent  Detection via Capsule Neural Networks** (ATIS/SNIPS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) [[code]](https://github.com/czhang99/Capsule-NLU) 
4. **A Stack-Propagation Framework  with Token-Level Intent Detection for Spoken Language Understanding** (ATIS/SNIPS) `EMNLP 2019` [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) [[code]](https://github.com/LeePleased/StackPropagation-SLU) 
5. **A Joint Learning Framework  With BERT for Spoken Language Understanding** (ATIS/SNIPS/Facebook's Multilingual dataset) `IEEE 2019` [[pdf]](https://ieeexplore.ieee.org/document/8907842) 
6. **BERT for Joint Intent  Classification and Slot Filling** (ATIS/SNIPS/Stanford Dialogue Dataset) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) [[code]](https://github.com/monologg/JointBERT) 
7. **A Novel Bi-directional  Interrelated Model for Joint Intent Detection and Slot Filling** (ATIS/Stanford Dialogue Dataset,SNIPS) `ACL 2019` [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) [[code]](https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU) 
8. **Joint Multiple Intent  Detection and Slot Labeling for Goal-Oriented Dialog** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2019` [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf) 
9. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 
10. **Leveraging Non-Conversational  Tasks for Low Resource Slot Filling: Does it help?** (ATIS/MIT Restaurant, and Movie/OntoNotes 5.0/OPUS   News Commentary) `SIGDIAL 2019` [[pdf]](https://www.aclweb.org/anthology/W19-5911.pdf) 
11. **A Bi-model based RNN Semantic  Frame Parsing Model for Intent Detection and Slot Filling** (ATIS) `NAACL 2018` [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) 
12. **Slot-Gated Modeling for Joint  Slot Filling and Intent Prediction** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) [[code]](https://github.com/MiuLab/SlotGated-SLU) 
13. **A Self-Attentive Model with  Gate Mechanism for Spoken Language Understanding** (ATIS) `EMNLP 2018` [[pdf]](https://www.aclweb.org/anthology/D18-1417.pdf) 
14. **Multi-task learning for Joint  Language Understanding and Dialogue State Tracking** (M2M/DSTC2) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5045.pdf) 
15. **A Joint Model of Intent  Determination and Slot Filling for Spoken Language Understanding** (ATIS/CQUD) `IJCAI 2016` [[pdf]](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) 
16. **Joint Online Spoken Language  Understanding and Language Modeling with Recurrent Neural Networks** (ATIS) `SIGDIAL 2016` [[pdf]](https://www.aclweb.org/anthology/W16-3603.pdf) [[code]](https://github.com/HadoopIt/joint-slu-lm)
17. **Multi-Domain Joint Semantic  Frame Parsing using Bi-directional RNN-LSTM** (ATIS) `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/d644/ae996755c803e067899bdd5ea52498d7091d.pdf) 
18. **Attention-Based Recurrent  Neural Network Models for Joint Intent Detection and Slot Filling** (ATIS) `INTERSPEECH 2016` [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) 
19. **Multi-domain joint semantic  frame parsing using bi-directional RNN-LSTM** `INTERSPEECH 2016` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/IS16_MultiJoint.pdf) 
20. **JOINT SEMANTIC UTTERANCE  CLASSIFICATION AND SLOT FILLING WITH RECURSIVE NEURAL NETWORKS** (ATIS/Stanford Dialogue Dataset,Microsoft Cortana  conversational understanding task(-)) `IEEE SLT 2014` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7078634) 
21. **CONVOLUTIONAL NEURAL NETWORK  BASED TRIANGULAR CRF FOR JOINT INTENT DETECTION AND SLOT FILLING** (ATIS) `IEEE Workshop on Automatic Speech Recognition and  Understanding 2013` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6707709) 

### Contextual SLU

2. **Knowing Where to Leverage: Context-Aware Graph Convolutional Network with An Adaptive Fusion Layer for Contextual Spoken Language Understanding** (Simulated Dialogues dataset) `IEEE 2021` [[pdf]](https://ieeexplore.ieee.org/document/9330801) 
2. **Dynamically Context-sensitive Time-decay Attention for Dialogue Modeling** (DSTC4) `IEEE 2019` [[pdf]](https://arxiv.org/pdf/1809.01557.pdf) 
3. **Multi-turn Intent Determination for Goal-oriented Dialogue systems** (Frames/Key-Value Retrieval) `IJCNN 2019` [[pdf]](https://ieeexplore.ieee.org/document/8852246) 
4. **Transfer Learning for Context-Aware Spoken Language Understanding** (single-turn: ATIS/SNIPS multi-turn: Simulated Dialogues dataset) `IEEE 2019` [[pdf]](https://ieeexplore.ieee.org/document/9003902) 
5. **How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues** (DSTC4) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-1194.pdf) [[code]](https://github.com/MiuLab/Time-Decay-SLU) 
6. **An Efficient Approach to Encoding Context for Spoken Language Understanding** (Simulated Dialogues dataset) `InterSpeech 2018` [[pdf]](https://arxiv.org/pdf/1807.00267.pdf) 
7. **Speaker-sensitive dual memory networks for multi-turn slot tagging** (Microsoft Cortana) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1711.10705.pdf) 
8. **Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning** (DSTC4) `IJCNLP 2017` [[pdf]](https://www.aclweb.org/anthology/I17-2028.pdf) [[code]](https://github.com/MiuLab/Spk-Dialogue) 
9. **Sequential dialogue context modeling for spoken language understanding** (collected by the author) `SIGDIAL 2017` [[pdf]](https://arxiv.org/pdf/1705.03455.pdf) 
10. **End-to-end joint learning of natural language understanding and dialogue manager** (DSTC4) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1612.00913.pdf) [[code]](https://github.com/XuesongYang/end2end_dialog.git) 
11. **Dynamic time-aware attention to speaker roles and contexts for spoken language understanding** (DSTC4) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1710.00165.pdf) [[code]](https://github.com/MiuLab/Time-SLU) 
12. **An Intelligent Assistant for High-Level Task Understanding** (collected by the author) `IUI 2016` [[pdf]](http://www.cs.cmu.edu/~mings/papers/IUI16IntelligentAssistant.pdf) 
13. **End-to-End Memory Networks with Knowledge Carryover for Multi-Turn Spoken Language Understanding** (Collected from Microsoft Cortana) `INTEERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/df07/45ce821007cb3122f00509cc18f2885fa8bd.pdf) 
14. **Leveraging behavioral patterns of mobile applications for personalized spoken language understanding** (collected by the author) `ICMI 2015` [[pdf]](https://www.csie.ntu.edu.tw/~yvchen/doc/ICMI15_MultiModel.pdf) 
15. **Contextual spoken language understanding using recurrent neural networks** (single-turn: ATIS multi-turn: Microsoft Cortana) ` 2015` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/0005271.pdf) 
16. **Contextual domain classification in spoken language understanding systems using recurrent neural network** (collected by the author) `IEEE 2014` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/05/rnn_dom.pdf) 
17. **Easy contextual intent prediction and slot detection** (collected by the author) `IEEE 2013` [[pdf]](https://ieeexplore.ieee.org/document/6639291) 

### Chinese SLU

1. **Injecting Word Information with Multi-Level Word Adapter for Chinese Spoken Language Understanding** (CAIS/ECDT-NLU) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.03903.pdf) [[code]](https://github.com/AaronTengDeChuan/MLWA-Chinese-SLU) 
2. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 

### Cross-domain SLU

1. **Coach: A Coarse-to-Fine  Approach for Cross-domain Slot Filling** (SNIPS) `ACL 2020` [[pdf]](https://arxiv.org/pdf/2004.11727.pdf) [[code]](https://github.com/zliucr/coach)
2. **Towards  Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue  Dataset** (SGD) `AAAI 2020` [[pdf]](https://arxiv.org/pdf/1909.05855.pdf) 
3. **Unsupervised Transfer Learning  for Spoken Language Understanding in Intelligent Agents** (ATIS/SINPS) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) 
4. **Zero-Shot Adaptive Transfer  for Conversational Language Understanding** (collected by author) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1808.10059.pdf) 
5. **Robust Zero-Shot Cross-Domain  Slot Filling with Example Values** (SNIPS/XSchema) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1906.06870.pdf) 
6. **Concept Transfer Learning for  Adaptive Language Understanding** (ATIS/DSTC2&3) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5047.pdf) 
7. **Fast and Scalable Expansion of  Natural Language Understanding Functionality for Intelligent Agents** (generated by the author) `NAACL 2018` [[pdf]](https://arxiv.org/pdf/1805.01542.pdf) 
8. **Bag of Experts Architectures  for Model Reuse in Conversational Language Understanding** (generated by the author) `NAACL-HLT 2018` [[pdf]](https://www.aclweb.org/anthology/N18-3019.pdf) 
9. **Domain Attention with an  Ensemble of Experts** () `ACL 2017` [[pdf]](https://www.aclweb.org/anthology/P17-1060.pdf) 
10. **Towards Zero-Shot Frame  Semantic Parsing for Domain Scaling** `INTERSPEECH 2017` [[pdf]](https://arxiv.org/pdf/1707.02363.pdf) 
11. **Zero-Shot Learning across  Heterogeneous Overlapping Domains** `INTERSPEECH 2017` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0516.PDFF) 
12. **Domainless Adaptation by  Constrained Decoding on a Schema Lattice** (Cortana) `COLING 2016` [[pdf]](https://www.aclweb.org/anthology/C16-1193.pdf) 
13. **Domain Adaptation of Recurrent  Neural Networks for Natural Language Understanding** (United Airlines/Airbnb/Grey-hound bus service/OpenTable (Data  obtained from App)) `INTERSPEECH 2016` [[pdf]](https://arxiv.org/pdf/1604.00117.pdf) 
14. **Natural Language Model  Re-usability for Scaling to Different Domains** (ATIS/MultiATIS) `EMNLP 2016` [[pdf]](https://www.aclweb.org/anthology/D16-1222.pdf) 
15. **Frustratingly Easy Neural  Domain Adaptation** (Cortana) `COLING 2016` [[pdf]](https://www.aclweb.org/anthology/C16-1038.pdf) 
16. **Multi-Domain Joint Semantic  Frame Parsing using Bi-directional RNN-LSTM** (ATIS) `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/d644/ae996755c803e067899bdd5ea52498d7091d.pdf) 
17. **A Model of Zero-Shot Learning  of Spoken Language Understanding** (generated by the auth) `EMNLP 2015` [[pdf]](https://www.aclweb.org/anthology/D15-1027.pdf) 
18. **Online adaptative zero-shot learning spoken language understanding using word-embedding** (DSTC2) `IEEE 2015` [[pdf]](https://ieeexplore.ieee.org/document/7178987) 
19. **Multi-Task Learning for Spoken  Language Understanding with Shared Slots** `INTERSPEECH 2011` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/08/Xiao-IS11.pdf) 

### Cross-lingual SLU

1. **CoSDA-ML: Multi-Lingual  Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP** (SC2/4/MLDoc/Multi WOZ/Facebook Multilingual SLU Dataset) `IJCAI 2020` [[pdf]](https://arxiv.org/pdf/2006.06402.pdf) [[code]](https://github.com/kodenii/CoSDA-ML) 
2. **Cross-lingual Spoken Language  Understanding with Regularized Representation Alignment** (Multilingual spoken language understanding (SLU) dataset) `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.14510.pdf) [[code]]([https://github.com/zliucr/crosslingual-slu](https://github.com/zliucr/crosslingual-slu.)) 
3. **End-to-End Slot Alignment and  Recognition for Cross-Lingual NLU** (ATIS/MultiATIS) `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2004.14353.pdf) 
4. **Multi-Level Cross-Lingual  Transfer Learning With Language Shared and Specific Knowledge for Spoken  Language Understanding** `IEEE Access 2020` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8990095) 
5. **Attention-Informed  Mixed-Language Training for Zero-shot Cross-lingual Task-oriented Dialogue  Systems** (Facebook Multilingual SLU Dataset/(DST)MultiWOZ) `AAAI 2020` [[pdf]](https://arxiv.org/pdf/1911.09273.pdf) [[code]](https://github.com/zliucr/mixedlanguage-training) 
6. **MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark** (MTOP /Multilingual ATIS) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2008.09335.pdf) [[code]]() 
7. **Cross-lingual Transfer  Learning with Data Selection for Large-Scale Spoken Language Understanding** (ATIS) `EMNLP-IJCNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1153.pdf) 
8. **Zero-shot Cross-lingual  Dialogue Systems with Transferable Latent Variables** (Facebook Multilingual SLU Dataset) `EMNLP-IJCNLP 2019` [[pdf]](https://arxiv.org/pdf/1911.04081.pdf) 
9. **Cross-Lingual Transfer  Learning for Multilingual Task Oriented Dialog** (Facebook Multilingual SLU Dataset) `NAACL 2019` [[pdf]](https://arxiv.org/pdf/1810.13327.pdf) 
10. **Almawave-SLU: A new dataset  for SLU in Italian** (Valentina.Bellomaria@almawave.it) `CEUR Workshop 2019` [[pdf]]() 
11. **Multi-lingual Intent Detection  and Slot Filling in a Joint BERT-based Model** (ATIS/SNIPS) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) 
12. **(Almost) Zero-Shot  Cross-Lingual Spoken Language Understanding** (ATIS manually translated into Hindi and Turkish) `IEEE/ICASSP 2018` [[pdf]](http://shyamupa.com/papers/UFTHH18.pdf) 
14. **Neural Architectures for  Multilingual Semantic Parsing** (GEO/ATIS) `ACL 2017` [[pdf]](https://www.aclweb.org/anthology/P17-2007.pdf) [[code]](http://statnlp.org/research/sp/) 
15. **Multi-style adaptive training  for robust cross-lingual spoken language understanding** (English-Chinese ATIS) `IEEE 2013` [[pdf]](https://ieeexplore.ieee.org/abstract/document/6639292) 
16. **ASGARD: A PORTABLE  ARCHITECTURE FOR MULTILINGUAL DIALOGUE SYSTEMS** (collected from crowd-sourcing platform) `ICASSP 2013` [[pdf]](https://groups.csail.mit.edu/sls/publications/2013/Liu_ICASSP-2013.pdf) 
17. **Combining multiple translation  systems for Spoken Language Understanding portability** (MEDIA) `IEEE 2012` [[pdf]](https://ieeexplore.ieee.org/document/6424221) 

### Low-resource SLU

#### Few-shot SLU

1. **Few-shot Learning for Multi-label Intent Detection** `AAAI 2021` [[pdf]](https://arxiv.org/abs/2010.05256) [[code]](https://github.com/AtmaHou/FewShotMultiLabel) 
2. **Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network** `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/ AtmaHou/FewShotTagging)
3. **Data Augmentation for Spoken  Language Understanding via Pretrained Models** (ATIS/SNIPS) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2004.13952.pdf) 
4. **Data augmentation by data  noising for open vocabulary slots in spoken language understanding** (ATIS/Snips/MIT-Restaurant.) `NAACL-HLT 2019` [[pdf]](https://www.aclweb.org/anthology/N19-3014.pdf) 
5. **Data Augmentation for Spoken  Language Understanding via Joint Variational Generation** (ATIS) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) 
6. **Marrying Up Regular  Expressions with Neural Networks: A Case Study for Spoken Language  Understanding** (ATIS) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/P18-1194.pdf) 
7. **Concept Transfer Learning for  Adaptive Language Understanding** (ATIS/DSTC2&3) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5047.pdf) 

#### Zero-shot SLU
1. **Coach: A Coarse-to-Fine  Approach for Cross-domain Slot Filling** (SNIPS) `ACL 2020` [[pdf]](https://arxiv.org/pdf/2004.11727.pdf) [[code]](https://github.com/zliucr/coach)
2. **Zero-Shot Adaptive Transfer  for Conversational Language Understanding** `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1808.10059.pdf) 
3. **Toward zero-shot Entity  Recognition in Task-oriented Conversational Agents** `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5036.pdf) 
4. **Zero-shot User Intent  Detection via Capsule Neural Networks** (SNIPS/CVA) `EMNLP 2018` [[pdf]](https://arxiv.org/pdf/1809.00385.pdf) 
5. **Towards Zero-Shot Frame  Semantic Parsing for Domain Scaling** `INTERSPEECH 2017` [[pdf]](https://arxiv.org/pdf/1707.02363.pdf) 
6. **Zero-Shot Learning across  Heterogeneous Overlapping Domains** `INTERSPEECH 2017` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0516.PDFF) 
7. **A Model of Zero-Shot Learning  of Spoken Language Understanding** (generated by the auth) `EMNLP 2015` [[pdf]](https://www.aclweb.org/anthology/D15-1027.pdf) 
8. **Zero-shot semantic parser for  spoken language understanding** (DSTC2&3) `INTERSPEECH 2015` [[pdf]](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1403.pdf) 

#### Unsupervised SLU

1. **Dialogue State Induction Using Neural Latent Variable Models**  `IJCAI 2020`  [[pdf]](https://www.ijcai.org/proceedings/2020/0532.pdf)

## LeaderBoard
### ATIS

#### Non-pretrained model

| Model                                                        | Intent Acc | Slot F1 | Paper / Source                                               | Code link                                               | Conference                                  |
| ------------------------------------------------------------ | ---------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------- |
| Co-Interactive(Qin et al., 2021)                         | 97.7       | 95.9    | A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) | https://github.com/kangbrilliant/DCA-Net      | ICASSP  |
| Graph LSTM(Zhang et al., 2021)                         | 97.20      | 95.91    | Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding  [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6499/6355) | -      | AAAI  |
| Stack  Propagation(Qin et al., 2019)                         | 96.9       | 95.9    | A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding  [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | https://github.com/LeePleased/StackPropagation-SLU      | EMNLP  |
| SF-ID+CRF(SF first)(E et al., 2019)         | 97.76      | 95.75   | A Novel   Bi-directional Interrelated Model for Joint Intent Detection and Slot   Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) |                                                       | ACL        |
| SF-ID+CRF(ID first)(E et al., 2019)         | 97.09      | 95.8    | A Novel   Bi-directional Interrelated Model for Joint Intent Detection and Slot   Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) | https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU | ACL        |
| Capsule-NLU(Zhang  et al. 2019)                              | 95         | 95.2    | Joint Slot   Filling and Intent Detection via Capsule Neural Networks [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) | https://github.com/czhang99/Capsule-NLU                 | ACL                                         |
| Utterance  Generation With Variational Auto-Encoder(Guo et al., 2019) | -          | 95.04   | Utterance  Generation With Variational Auto-Encoder for Slot Filling in Spoken Language  Understanding [[pdf]](https://ieeexplore.ieee.org/document/8625384) | -                                                       | IEEE Signal Processing Letters              |
| JULVA(full)(Yoo  et al., 2019)                               | 97.24      | 95.51   | Data Augmentation   for Spoken Language Understanding via Joint Variational Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) | -                                                       | AAAI                                        |
| CM-Net(Liu  et al., 2019)                               | 99.1      | 96.20   | CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding[[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf)| https://github.com/Adaxry/CM-Net    | EMNLP                                        |
| Data  noising method(Kim et al., 2019)                       | 98.43      | 96.20    | Data  augmentation by data noising for open vocabulary slots in spoken language  understanding [[pdf]](https://www.aclweb.org/anthology/N19-3014.pdf) | -                                                       | NAACL-HLT                                   |
| ACD(Zhu  et al., 2018)                                       | -          | 96.08   | Concept   Transfer Learning for Adaptive Language Understanding [[pdf]](https://www.aclweb.org/anthology/W18-5047.pdf) | -                                                       | SIGDIAL                                     |
| A Self-Attentive Model with Gate Mechanism(Li et al., 2018)  | 98.77      | 96.52   | A   Self-Attentive Model with Gate Mechanism for Spoken Language   Understanding [[pdf]](https://www.aclweb.org/anthology/D18-1417.pdf) | -                                                       | EMNLP                                       |
| Slot-Gated(Goo  et al., 2018)                                | 94.1       | 95.2    | Slot-Gated   Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) | https://github.com/MiuLab/SlotGated-SLU                 | NAACL                                       |
| DRL based Augmented Tagging System(Wang et al., 2018)        | -          | 97.86   | A  New Concept of Deep Reinforcement Learning based Augmented General Sequence  Tagging System [[pdf]](https://www.aclweb.org/anthology/C18-1143.pdf) | -                                                       | COLING      |
| Bi-model(Wang  et al., 2018)                                 | 98.76      | 96.65   | A Bi-model based   RNN Semantic Frame Parsing Model for Intent Detection and Slot Filling [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) | -                                                       | NAACL                                       |
| Bi-model+decoder(Wang  et al., 2018)        | 98.99      | 96.89   | A Bi-model based   RNN Semantic Frame Parsing Model for Intent Detection and Slot Filling [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) | -                                                     | NAACL      |
| Seq2Seq DA for LU(Hou et al., 2018)                          | -          | 94.82   | Sequence-to-Sequence  Data Augmentation for Dialogue Language Understanding [[pdf]](https://arxiv.org/pdf/1807.01554.pdf) | https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU | COLING                                      |
| BLSTM-LSTM(Zhu  et al., 2017)                                | -          | 95.79   | ENCODER-DECODER  WITH FOCUS-MECHANISM FOR SEQUENCE LABELLING BASED SPOKEN LANGUAGE  UNDERSTANDING  [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7953243) | -                                                       | ICASSP                                      |
| neural  sequence chunking model(Zhai et al., 2017)           | -          | 95.86   | Neural  Models for Sequence Chunking [[pdf]](https://arxiv.org/pdf/1701.04027.pdf) | -                                                       | AAAI                                        |
| Joint  Model of ID and SF(Zhang et al., 2016)                | 98.32      | 96.89   | A   Joint Model of Intent Determination and Slot Filling for Spoken Language   Understanding [[pdf]](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) | -                                                       | IJCAI                                       |
| Attention Encoder-Decoder NN (with aligned inputs)           | 98.43      | 95.87   | Attention-Based   Recurrent Neural Network Models for Joint Intent Detectionand Slot   Filling      [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) | -                                                       | InterSpeech                                 |
| Attention  BiRNN(Liu et al., 2016)                           | 98.21      | 95.98   | Attention-Based   Recurrent Neural Network Models for Joint Intent Detectionand Slot   Filling      [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) | -                                                       | InterSpeech                                 |
| Joint  SLU-LM model(Liu ei al., 2016)                        | 98.43      | 94.64   | Joint Online   Spoken Language Understanding and Language Modeling with Recurrent Neural   Networks [[pdf]](https://arxiv.org/pdf/1609.01462.pdf) | http://speech.sv.cmu.edu/software.html                  | SIGDIAL                                     |
| RNN-LSTM(Hakkani-Tur  et al., 2016)                          | 94.3       | 92.6    | Multi-Domain Joint Semantic Frame Parsing using   Bi-directional RNN-LSTM [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/IS16_MultiJoint.pdf) | -                                                       | InterSpeech                                 |
| R-biRNN(Vu  et al., 2016)                                    | -          | 95.47   | Bi-directional   recurrent neural network with ranking loss for spoken language   understanding      [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) | -                                                       | IEEE                                        |
| Encoder-labeler  LSTM(Kurata et al., 2016)                   | -          | 95.4    | Leveraging Sentence-level Information with  Encoder LSTM for Semantic Slot Filling [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) | -                                                       | EMNLP                                       |
| Encoder-labeler  Deep LSTM(Kurata et al., 2016)              | -          | 95.66   | Leveraging Sentence-level Information with  Encoder LSTM for Semantic Slot Filling [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) |                                                         | EMNLP                                       |
| 5xR-biRNN(Vu  et al., 2016)                 | -          | 95.56   | Bi-directional  recurrent neural network with ranking loss for spoken language  understanding [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) | -                                                     | IEEE       |
| Data  Generation for SF(Kurata et al., 2016)                 | -          | 95.32   | Labeled  Data Generation with Encoder-decoder LSTM for Semantic Slot Filling [[pdf]](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0727.PDF) | -                                                       | InterSpeech                                 |
| RNN-EM(Peng  et al., 2015)                                   | -          | 95.25   | Recurrent Neural   Networks with External Memory for Language Understanding [[pdf]](https://arxiv.org/pdf/1506.00195.pdf) | -                                                       | InterSpeech                                 |
| RNN  trained with sampled label(Liu et al., 2015)            | -          | 94.89   | Recurrent Neural Network Structured Output Prediction for   Spoken Language Understanding      [[pdf]](http://speech.sv.cmu.edu/publications/liu-nipsslu-2015.pdf) | -                                                       | -                                           |
| RNN(Ravuri  et al., 2015)                                    | 97.55      | -       | Recurrent neural network and LSTM models for  lexical utterance classification [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) | -                                                       | InterSpeech                                 |
| LSTM(Ravuri  et al., 2015)                                   | 98.06      | -       | Recurrent neural network and LSTM models for  lexical utterance classification [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) | -                                                       | InterSpeech                                 |
| Hybrid  RNN(Mesnil et al., 2015)                             | -          | 95.06   | Using  Recurrent Neural Networks for Slot Filling in Spoken Language  Understanding [[pdf]](https://ieeexplore.ieee.org/document/6998838) | -                                                       | IEEE/ACM-TASLP                              |
| RecNN(Guo  et al., 2014)                                     | 95.4       | 93.22   | Joint semantic utterance classification and slot filling with   recursive neural networks [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/12/RecNNSLU.pdf) | -                                                       | IEEE-SLT                                    |
| LSTM(Yao  et al., 2014)                                      | -          | 94.85   | Spoken Language Understading Using Long  Short-Term Memory Neural Networks [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) | -                                                       | IEEE                                        |
| Deep  LSTM(Yao et al., 2014)                                 | -          | 95.08   | Spoken Language Understading Using Long  Short-Term Memory Neural Networks [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) | -                                                       | IEEE                                        |
| R-CRF(Yao  et al., 2014)                                     | -          | 96.65   | Recurrent  conditional random field for language understanding [[pdf]](https://ieeexplore.ieee.org/document/6854368) | -                                                       | IEEE                                        |
| RecNN+Viterbi(Guo  et al., 2014)            | 95.4       | 93.96   | Joint semantic utterance classification and slot filling with   recursive neural networks [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/12/RecNNSLU.pdf) | -                                                     | IEEE-SLT   |
| CNN  CRF(Xu et al., 2013)                                    | 94.09      | 5.42   | Convolutional neural network based triangular crf for joint   intent detection and slot filling [[pdf]]((http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.7548&rep=rep1&type=pdf)) | -                                                       | IEEE                                        |
| RNN(Yao  et al., 2013)                                       | -          | 94.11   | Recurrent  Neural Networks for Language Understanding [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kaisheny-338_file_paper.pdf) | -                                                       | InterSpeech                                 |
| Bi-dir.  Jordan-RNN(2013)                                    | -          | 93.98   | Investigation  of Recurrent-Neural-Network Architectures and Learning Methods for Spoken  Language Understanding [[pdf]](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_3771.pdf) | -                                                       | ISCA                                        |



#### + Pretrained model

| Model                                       | Intent Acc | Slot F1 | Paper/Source                                                 | Code link                                             | Conference |
| ------------------------------------------- | ---------- | ------- | ------------------------------------------------------------ | ----------------------------------------------------- | ---------- |
| Co-Interactive(Qin et al., 2021)                         | 98.0       | 96.1    | A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) | https://github.com/kangbrilliant/DCA-Net      | ICASSP  |
| Stack  Propagation+BERT(Qin et al., 2019)   | 97.5       | 96.1    | A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | https://github.com/LeePleased/StackPropagation-SLU    | EMNLP      |
| Bert-Joint(Castellucci  et al., 2019)       | 97.8       | 95.7    | Multi-lingual  Intent Detection and Slot Filling in a Joint BERT-based Model [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) | -                                                     | arXiv      |
| BERT-SLU(Zhang  et al., 2019)               | 99.76      | 98.75   | A Joint   Learning Framework With BERT for Spoken Language Understanding [[pdf]](https://ieeexplore.ieee.org/document/8907842) | -                                                     | IEEE       |
| Joint  BERT(Chen et al., 2019)              | 97.5       | 96.1    | BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) | https://github.com/monologg/JointBERT                 | arXiv      |
| Joint  BERT+CRF(Chen et al., 2019)          | 97.9       | 96      | BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) | https://github.com/monologg/JointBERT                 | arXiv      |
| ELMo-Light  (ELMoL) (Siddhant et al., 2019) | 97.3       | 95.42   | Unsupervised   Transfer Learning for Spoken Language Understanding in Intelligent Agents [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) | -                                                     | AAAI       |


### SNIPS

#### Non-pretrained model

| Model                                                        | Intent Acc | Slot F1 | Paper / Source                                               | Code link                                                    | Conference                     |
| ------------------------------------------------------------ | ---------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------ |
| Co-Interactive(Qin et al., 2021)                         | 98.8       | 95.9    | A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) | https://github.com/kangbrilliant/DCA-Net      | ICASSP  |
| Graph LSTM(Zhang et al., 2021)                         | 98.29      | 95.30    | Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding  [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6499/6355) | -      | AAAI  |
| SF-ID  Network(E et al, 2019)                                | 97.43      | 91.43   | A  Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot  Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) | https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU        | ACL                            |
| CAPSULE-NLU(Zhang  et al, 2019)                              | 97.3       | 91.8    | Joint  Slot Filling and Intent Detection via Capsule Neural Networks [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) | https://github.com/czhang99/Capsule-NLU                      | ACL                            |
| StackPropagation(Qin  et al, 2019)                           | 98         | 94.2    | A  Stack-Propagation Framework with Token-Level Intent Detection for Spoken  Language Understanding     [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | [https://github.com/LeePleased/StackPropagation-SLU. ](https://github.com/LeePleased/StackPropagation-SLU.) | EMNLP                          |
| CM-Net(Liu  et al., 2019)                               | 99.29      | 97.15   | CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding[[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf)| https://github.com/Adaxry/CM-Net    | EMNLP                                        |
| Joint  Multiple(Gangadharaiah et al, 2019)                   | 97.23      | 88.03   | Joint  Multiple Intent Detection and Slot Labeling for Goal-Oriented Dialog [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf) | -                                                            | NAACL                          |
| Utterance  Generation With Variational Auto-Encoder(Guo et al., 2019) | -          | 93.18   | Utterance  Generation With Variational Auto-Encoder for Slot Filling in Spoken Language  Understanding        [[pdf]](https://ieeexplore.ieee.org/document/8625384) | -                                                            | IEEE Signal Processing Letters |
| Slot  Gated Intent Atten.(Goo et al, 2018)                   | 96.8       | 88.3    | Slot-Gated   Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) | https://github.com/MiuLab/SlotGated-SLU                      | NAACL                          |
| Slot  Gated Fulled Atten.(Goo et al, 2018)                   | 97         | 88.8    | Slot-Gated  Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) | https://github.com/MiuLab/SlotGated-SLU                      | NAACL                          |
| Joint  Variational Generation + Slot Gated Intent Atten(Yoo et al., 2018) | 96.7       | 88.3    | Data  Augmentation for Spoken Language Understanding via Joint Variational  Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) | -                                                            | AAAI                           |
| Joint  Variational Generation + Slot Gated Full Atten(Yoo et al., 2018) | 97.3       | 89.3    | Data Augmentation  for Spoken Language Understanding via Joint Variational Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) | -                                                            | AAAI                           |



#### + Pretrained model

| Model                                           | Intent Acc | Slot F1 | Paper/Source                                                 | Code link                                                    | Conference |
| ----------------------------------------------- | ---------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| Co-Interactive(Qin et al., 2021)                         | 98.8       | 97.1    | A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) | https://github.com/kangbrilliant/DCA-Net      | ICASSP  |
| StackPropagation  + Bert(Qin et al, 2019)       | 99         | 97      | A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | [https://github.com/LeePleased/StackPropagation-SLU. ](https://github.com/LeePleased/StackPropagation-SLU.) | EMNLP      |
| Bert-Joint(Castellucci  et al, 2019)            | 99         | 96.2    | Multi-lingual  Intent Detection and Slot Filling in a Joint BERT-based Mode [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) | -                                                            | arXiv      |
| Bert-SLU(Zhang  et al, 2019)                    | 98.96      | 98.78   | A Joint Learning  Framework With BERT for Spoken Language Understanding [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8907842) | -                                                            | IEEE       |
| Joint  BERT(Chen et al, 2019)                   | 98.6       | 97      | BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) | https://github.com/monologg/JointBERT                        | arXiv      |
| Joint  BERT + CRF(Chen et al, 2019)             | 98.4       | 96.7    | BERT  for Joint Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) | https://github.com/monologg/JointBERT                        | arXiv      |
| ELMo-Light(Siddhant  et al, 2019)               | 98.38      | 93.29   | Unsupervised   Transfer Learning for Spoken Language Understanding in Intelligent Agents         [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) | -                                                            | AAAI       |
| ELMo(Peters  et al, 2018;Siddhant et al, 2019 ) | 99.29      | 93.9    | Deep   contextualized word representations      [[pdf]](https://arxiv.org/pdf/1802.05365.pdf)Unsupervised Transfer Learning for Spoken Language Understanding in   Intelligent Agents [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) | -                                                            | NAACL/AAAI |

