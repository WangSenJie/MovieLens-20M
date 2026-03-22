# 简历描述与面试讲稿

## 简历项目描述

### 版本 1：适合一行项目经历

> 基于 MovieLens 20M 搭建推荐系统项目，完成 Content、UserCF、ItemCF、SVD、ALS、BPR、LightFM 和两阶段召回重排实验，构建 genres、year、tag genome、人物标签等多源特征，采用 Leave-One-Out 与 HR@K、NDCG@K、MRR、Coverage 做离线评估，并通过 FastAPI、前端页面和 Docker 实现服务化展示。

### 版本 2：适合简历 3 条要点

- 基于 MovieLens 20M 构建完整推荐系统项目，覆盖内容召回、协同过滤、矩阵分解与两阶段 `Recall + Rerank` 架构，并统一实现训练、评估、推荐和服务化接口。
- 设计 `genres / year / decade / user tags / tag genome / director / actors` 等多源内容特征，使用 Leave-One-Out 协议和 `HR@10/20`、`NDCG@10/20`、`MAP@10`、`MRR`、`Coverage@10` 进行离线实验对比。
- 使用 FastAPI 提供推荐接口与相似电影接口，构建前端展示页和 Docker 部署方案；当前实验中 `LightFM` 取得最佳准确性表现，`SVD` 与 `ALS` 在准确率和覆盖率之间表现稳定。

## 1 分钟项目介绍

> 我做了一个基于 MovieLens 20M 的推荐系统项目，目标不是只复现一个算法，而是做一个完整可展示的推荐系统。项目里实现了内容模型、UserCF、ItemCF、SVD、ALS、BPR、LightFM，以及一个两阶段的召回加重排架构。特征上除了 genres 和 tags，还补了 year、decade、tag genome，以及导演和演员相关的人物标签。评估上采用 Leave-One-Out，用 HR@K、NDCG@K、MRR 和 Coverage 做统一对比。实验结果里 LightFM 的准确率最好，SVD 和 ALS 也明显优于热门榜，说明协同过滤在这个数据集上更有效。工程上我把训练、评估、推荐 API、前端展示页和 Docker 部署都串起来了，所以这个项目既能讲算法，也能讲工程落地。

## 面试时建议重点讲的 4 个点

### 1. 为什么要做两阶段

- 单阶段模型更容易实现，但候选集太大时效率和灵活性都受限
- 两阶段先召回再重排，更接近工业界推荐系统结构
- 这个项目里默认用 `content + itemcf + popularity` 做召回，再用 `svd` 做重排

### 2. 为什么 `content` 指标不高

- 内容特征只覆盖电影显式属性和标签语义，缺少更强的协同信号
- 但内容模型覆盖率通常更高，更适合补长尾和冷启动
- 所以它更适合做召回源，而不是单独做最终排序

### 3. 为什么离线指标看起来不算很高

- 当前协议是 `Leave-One-Out + 全量未见物品候选集`
- 这种评估比“随机采样少量负样本”更严格
- 所以 `HR@10` 在 `0.08 - 0.10` 左右并不奇怪，重点是相对提升是否稳定

### 4. 线上如果要做 A/B Test 看什么

- 北极星指标看 `CTR / Watch Rate / Favorite Rate`
- 护栏指标看 `P95 latency / Bounce Rate / Complaint Rate`
- 核心对比是单阶段模型和两阶段模型在用户体验与长尾曝光上的差异
