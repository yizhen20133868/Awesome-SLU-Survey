SLU Survey

![](https://img.shields.io/badge/Status-building-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) 

This repo contains a list of papers, codes, datasets, leaderboards in SLU field. If you found any error, please don't hesitate to open an issue.

## Introduction

Spoken language understanding (SLU) is a critical component in task-oriented dialogue systems. It usually consists of intent and slot filling task to extract semantic constiuents from the natrual language utterances.

For the purpose of alleviating our pressure in article/dataset collation, we worked on sorting out the relevant data sets, papers, codes and lists of SLU in this project.

At present, the project has been completely open source, including:
1. **Articles and infos in different directions in the field of SLU:** we classify and arrange the papers according to the current mainstream directions, which you can also give a quick check from the `Content` section below. Each line of the list contains not only the title of the paper, but also the year of publication, the source of publication, the paper link and code link for quick indexing, as well as the data set used.

2. **SLU domain dataset sorting table:** you can quickly index the data set you want to use in it to help you quickly understand the general scale, basic structure, content, characteristics, source and acquisition method of this dataset.

3. **Leaderboard list on the mainstream datasets of SLU:** we sorted out the leaderboard on the mainstream datasets, and distinguished them according to pre-trained or not. In addition to the paper name and related scores, each line also has links to year, paper and code.

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
2. **Recurrent neural network and  LSTM models for lexical utterance classification** (ATIS/CB) `INTERSPEECH 2015` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) 
3. **Intention Detection Based on Siamese Neural Network With Triplet Loss** (SNIPS/ATIS/Facebook multilingual datasets/ Daily Dialogue/MRDA) `IEEE Acess 2020` [[pdf]](https://ieeexplore.ieee.org/document/9082602) 
4. **Adversarial Training for Multi-task and Multi-lingual Joint Modeling of Utterance Intent Classification** (collected by the author) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/D18-1064.pdf) 
5. **Multi-Layer Ensembling Techniques for Multilingual Intent Classification** (ATIS) `arXiv 2018` [[pdf]](https://arxiv.org/pdf/1806.07914.pdf) 
6. **Deep Unknown Intent Detection with Margin Loss** (SNIPS/ATIS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1906.00434.pdf) 
7. **Subword Semantic Hashing for Intent Classification on Small Datasets** (The Chatbot Corpus/The AskUbuntu Corpus) `IJCNN 2019` [[pdf]](https://arxiv.org/pdf/1810.07150.pdf) 
8. **Dialogue intent classification with character-CNN-BGRU networks** (the Chinese Wikipedia dataset) `Multimedia Tools and Applications 2018` [[pdf]](https://link.springer.com/article/10.1007/s11042-019-7678-1)  
9. **Joint Learning of Domain Classification and Out-of-Domain Detection with Dynamic Class Weighting for Satisficing False Acceptance Rates** (Alexa) `InterSpeech 2018` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1581.pdf)  
10. **Exploiting Shared Information for Multi-Intent Natural Language Sentence Classification** (collected by the author) `ISCA 2013` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/08/double_intent.pdf)  


### Joint Model

1. **Joint Slot Filling and Intent  Detection via Capsule Neural Networks** (SNIPS/ATIS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) [[code]](https://github.com/czhang99/Capsule-NLU) 
2. **A Stack-Propagation Framework  with Token-Level Intent Detection for Spoken Language Understanding** (SNIPS/ATIS) `EMNLP 2019` [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) [[code]](https://github.com/LeePleased/StackPropagation-SLU) 
3. **A Joint Learning Framework  With BERT for Spoken Language Understanding** (ATIS/SNIPS/Facebook's Multilingual dataset) `IEEE 2019` [[pdf]](https://ieeexplore.ieee.org/document/8907842) 
4. **BERT for Joint Intent  Classification and Slot Filling** (ATIS/Stanford Dialogue Dataset,SNIPS) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) [[code]](https://github.com/monologg/JointBERT) 
5. **A Novel Bi-directional  Interrelated Model for Joint Intent Detection and Slot Filling** (ATIS/Stanford Dialogue Dataset,SNIPS) `ACL 2019` [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) [[code]](https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU) 
6. **Joint Multiple Intent  Detection and Slot Labeling for Goal-Oriented Dialog** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2019` [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf) 
7. **Leveraging Non-Conversational  Tasks for Low Resource Slot Filling: Does it help?** (ATIS/MIT Restaurant, and Movie/OntoNotes 5.0/OPUS   News Commentary) `SIGDIAL 2019` [[pdf]](https://www.aclweb.org/anthology/W19-5911.pdf) 
8. **A Bi-model based RNN Semantic  Frame Parsing Model for Intent Detection and Slot Filling** (ATIS) `NAACL 2018` [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) 
9. **Slot-Gated Modeling for Joint  Slot Filling and Intent Prediction** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) [[code]](https://github.com/MiuLab/SlotGated-SLU) 
10. **A Self-Attentive Model with  Gate Mechanism for Spoken Language Understanding** (ATIS) `EMNLP 2018` [[pdf]](https://www.aclweb.org/anthology/D18-1417.pdf) 
11. **Multi-task learning for Joint  Language Understanding and Dialogue State Tracking** (M2M/DSTC2) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5045.pdf) 
12. **A Joint Model of Intent  Determination and Slot Filling for Spoken Language Understanding** (ATIS/CQUD) `IJCAI 2016` [[pdf]](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) 
13. **Joint Online Spoken Language  Understanding and Language Modeling with Recurrent Neural Networks** (ATIS) `SIGDIAL 2016` [[pdf]](https://www.aclweb.org/anthology/W16-3603.pdf) [[code]](https://github.com/HadoopIt/joint-slu-lm)
14. **Multi-Domain Joint Semantic  Frame Parsing using Bi-directional RNN-LSTM** (ATIS) `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/d644/ae996755c803e067899bdd5ea52498d7091d.pdf) 
15. **Attention-Based Recurrent  Neural Network Models for Joint Intent Detection and Slot Filling** (ATIS) `INTERSPEECH 2016` [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) 
16. **Multi-domain joint semantic  frame parsing using bi-directional RNN-LSTM** `INTERSPEECH 2016` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/IS16_MultiJoint.pdf) 
17. **JOINT SEMANTIC UTTERANCE  CLASSIFICATION AND SLOT FILLING WITH RECURSIVE NEURAL NETWORKS** (ATIS/Stanford Dialogue Dataset,Microsoft Cortana  conversational understanding task(-)) `IEEE SLT 2014` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7078634) 
18. **CONVOLUTIONAL NEURAL NETWORK  BASED TRIANGULAR CRF FOR JOINT INTENT DETECTION AND SLOT FILLING** (ATIS) `IEEE Workshop on Automatic Speech Recognition and  Understanding 2013` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6707709) 

## LeaderBoard
### ATIS

#### Non-pretrained model

| Model                                                        | Intent Acc | Slot F1 | Paper / Source                                               | Code link                                               | Conference                                  |
| ------------------------------------------------------------ | ---------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------- |
| Stack  Propagation(Qin et al., 2019)                         | 96.9       | 95.9    | A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding  [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | https://github.com/LeePleased/StackPropagation-SLU      | EMNLP  |
| SF-ID+CRF(SF first)(E et al., 2019)         | 97.76      | 95.75   | A Novel   Bi-directional Interrelated Model for Joint Intent Detection and Slot   Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) |                                                       | ACL        |
| SF-ID+CRF(ID first)(E et al., 2019)         | 97.09      | 95.8    | A Novel   Bi-directional Interrelated Model for Joint Intent Detection and Slot   Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) | https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU | ACL        |
| Capsule-NLU(Zhang  et al. 2019)                              | 95         | 95.2    | Joint Slot   Filling and Intent Detection via Capsule Neural Networks [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) | https://github.com/czhang99/Capsule-NLU                 | ACL                                         |
| Utterance  Generation With Variational Auto-Encoder(Guo et al., 2019) | -          | 95.04   | Utterance  Generation With Variational Auto-Encoder for Slot Filling in Spoken Language  Understanding [[pdf]](https://ieeexplore.ieee.org/document/8625384) | -                                                       | IEEE Signal Processing Letters              |
| JULVA(full)(Yoo  et al., 2019)                               | 97.24      | 95.51   | Data Augmentation   for Spoken Language Understanding via Joint Variational Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) | -                                                       | AAAI                                        |
| Data  noising method(Kim et al., 2019)                       | 98.43      | 96.2    | Data  augmentation by data noising for open vocabulary slots in spoken language  understanding [[pdf]](https://www.aclweb.org/anthology/N19-3014.pdf) | -                                                       | NAACL-HLT                                   |
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
| K-SAN(Chen  et al., 2016)                                    | -          | 95.38   | SYNTAX  OR SEMANTICS? KNOWLEDGE-GUIDED JOINT SEMANTIC FRAME PARSING [[pdf]](https://www.csie.ntu.edu.tw/~yvchen/doc/SLT16_SyntaxSemantics.pdf) | -                                                       | IEEE Workshop on Spoken Language Technology |
| Encoder-labeler  LSTM(Kurata et al., 2016)                   | -          | 95.4    | Leveraging Sentence-level Information with  Encoder LSTM for Semantic Slot Filling [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) | -                                                       | EMNLP                                       |
| Encoder-labeler  Deep LSTM(Kurata et al., 2016)              | -          | 95.66   | Leveraging Sentence-level Information with  Encoder LSTM for Semantic Slot Filling [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) |                                                         | EMNLP                                       |
| 5xR-biRNN(Vu  et al., 2016)                 | -          | 95.56   | Bi-directional  recurrent neural network with ranking loss for spoken language  understanding [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) | -                                                     | IEEE       |
| Data  Generation for SF(Kurata et al., 2016)                 | -          | 95.32   | Labeled  Data Generation with Encoder-decoder LSTM for Semantic Slot Filling [[pdf]](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0727.PDF) | -                                                       | InterSpeech                                 |
| RNN-EM(Peng  et al., 2015)                                   | -          | 94.96   | Recurrent Neural   Networks with External Memory for Language Understanding [[pdf]](https://arxiv.org/pdf/1506.00195.pdf) | -                                                       | InterSpeech                                 |
| RNN  trained with sampled label(Liu et al., 2015)            | -          | 94.89   | Recurrent Neural Network Structured Output Prediction for   Spoken Language Understanding      [[pdf]](http://speech.sv.cmu.edu/publications/liu-nipsslu-2015.pdf) | -                                                       | -                                           |
| RNN(Ravuri  et al., 2015)                                    | 97.55      | -       | Recurrent neural network and LSTM models for  lexical utterance classification [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) | -                                                       | InterSpeech                                 |
| LSTM(Ravuri  et al., 2015)                                   | 98.06      | -       | Recurrent neural network and LSTM models for  lexical utterance classification [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) | -                                                       | InterSpeech                                 |
| Hybrid  RNN(Mesnil et al., 2015)                             | -          | 95.06   | Using  Recurrent Neural Networks for Slot Filling in Spoken Language  Understanding [[pdf]](https://ieeexplore.ieee.org/document/6998838) | -                                                       | IEEE/ACM-TASLP                              |
| RecNN(Guo  et al., 2014)                                     | 95.4       | 93.22   | Joint semantic utterance classification and slot filling with   recursive neural networks [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/12/RecNNSLU.pdf) | -                                                       | IEEE-SLT                                    |
| LSTM(Yao  et al., 2014)                                      | -          | 94.85   | Spoken Language Understading Using Long  Short-Term Memory Neural Networks [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) | -                                                       | IEEE                                        |
| Deep  LSTM(Yao et al., 2014)                                 | -          | 95.08   | Spoken Language Understading Using Long  Short-Term Memory Neural Networks [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) | -                                                       | IEEE                                        |
| R-CRF(Yao  et al., 2014)                                     | -          | 96.65   | Recurrent  conditional random field for language understanding [[pdf]](https://ieeexplore.ieee.org/document/6854368) | -                                                       | IEEE                                        |
| RecNN+Viterbi(Guo  et al., 2014)            | 95.4       | 93.96   | Joint semantic utterance classification and slot filling with   recursive neural networks [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/12/RecNNSLU.pdf) | -                                                     | IEEE-SLT   |
| CNN  CRF(Xu et al., 2013)                                    | 94.09      | 94.35   | Convolutional neural network based triangular crf for joint   intent detection and slot filling [[pdf]]((http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.7548&rep=rep1&type=pdf)) | -                                                       | IEEE                                        |
| RNN(Yao  et al., 2013)                                       | -          | 94.11   | Recurrent  Neural Networks for Language Understanding [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kaisheny-338_file_paper.pdf) | -                                                       | InterSpeech                                 |
| Bi-dir.  Jordan-RNN(2013)                                    | -          | 93.98   | Investigation  of Recurrent-Neural-Network Architectures and Learning Methods for Spoken  Language Understanding [[pdf]](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_3771.pdf) | -                                                       | ISCA                                        |



#### + Pretrained model

| Model                                       | Intent Acc | Slot F1 | Paper/Source                                                 | Code link                                             | Conference |
| ------------------------------------------- | ---------- | ------- | ------------------------------------------------------------ | ----------------------------------------------------- | ---------- |
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
| SF-ID  Network(E et al, 2019)                                | 97.43      | 91.43   | A  Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot  Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) | https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU        | ACL                            |
| CAPSULE-NLU(Zhang  et al, 2019)                              | 97.3       | 91.8    | Joint  Slot Filling and Intent Detection via Capsule Neural Networks [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) | https://github.com/czhang99/Capsule-NLU                      | ACL                            |
| StackPropagation(Qin  et al, 2019)                           | 98         | 94.2    | A  Stack-Propagation Framework with Token-Level Intent Detection for Spoken  Language Understanding     [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | [https://github.com/LeePleased/StackPropagation-SLU. ](https://github.com/LeePleased/StackPropagation-SLU.) | EMNLP                          |
| Joint  Multiple(Gangadharaiah et al, 2019)                   | 97.23      | 88.03   | Joint  Multiple Intent Detection and Slot Labeling for Goal-Oriented Dialog [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf) | -                                                            | NAACL                          |
| Utterance  Generation With Variational Auto-Encoder(Guo et al., 2019) | -          | 93.18   | Utterance  Generation With Variational Auto-Encoder for Slot Filling in Spoken Language  Understanding        [[pdf]](https://ieeexplore.ieee.org/document/8625384) | -                                                            | IEEE Signal Processing Letters |
| Slot  Gated Intent Atten.(Goo et al, 2018)                   | 96.8       | 88.3    | Slot-Gated   Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) | https://github.com/MiuLab/SlotGated-SLU                      | NAACL                          |
| Slot  Gated Fulled Atten.(Goo et al, 2018)                   | 97         | 88.8    | Slot-Gated  Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) | https://github.com/MiuLab/SlotGated-SLU                      | NAACL                          |
| Joint  Variational Generation + Slot Gated Intent Atten(Yoo et al., 2018) | 96.7       | 88.3    | Data  Augmentation for Spoken Language Understanding via Joint Variational  Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) | -                                                            | AAAI                           |
| Joint  Variational Generation + Slot Gated Full Atten(Yoo et al., 2018) | 97.3       | 89.3    | Data Augmentation  for Spoken Language Understanding via Joint Variational Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) | -                                                            | AAAI                           |



#### + Pretrained model

| Model                                           | Intent Acc | Slot F1 | Paper/Source                                                 | Code link                                                    | Conference |
| ----------------------------------------------- | ---------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| StackPropagation  + Bert(Qin et al, 2019)       | 99         | 97      | A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) | [https://github.com/LeePleased/StackPropagation-SLU. ](https://github.com/LeePleased/StackPropagation-SLU.) | EMNLP      |
| Bert-Joint(Castellucci  et al, 2019)            | 99         | 96.2    | Multi-lingual  Intent Detection and Slot Filling in a Joint BERT-based Mode [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) | -                                                            | arXiv      |
| Bert-SLU(Zhang  et al, 2019)                    | 98.96      | 98.78   | A Joint Learning  Framework With BERT for Spoken Language Understanding [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8907842) | -                                                            | IEEE       |
| Joint  BERT(Chen et al, 2019)                   | 98.6       | 97      | BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) | https://github.com/monologg/JointBERT                        | arXiv      |
| Joint  BERT + CRF(Chen et al, 2019)             | 98.4       | 96.7    | BERT  for Joint Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) | https://github.com/monologg/JointBERT                        | arXiv      |
| ELMo-Light(Siddhant  et al, 2019)               | 98.38      | 93.29   | Unsupervised   Transfer Learning for Spoken Language Understanding in Intelligent Agents         [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) | -                                                            | AAAI       |
| ELMo(Peters  et al, 2018;Siddhant et al, 2019 ) | 99.29      | 93.9    | Deep   contextualized word representations      [[pdf]](https://arxiv.org/pdf/1802.05365.pdf)Unsupervised Transfer Learning for Spoken Language Understanding in   Intelligent Agents [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) | -                                                            | NAACL/AAAI |

