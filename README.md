# MovieLens 推荐系统项目

项目基于 `MovieLens 20M` 数据集实现了完整的推荐系统工作流：

- 单阶段模型：`Popularity / Content / UserCF / ItemCF / SVD`
- 可选增强模型：`ALS / BPR / LightFM`
- 两阶段架构：`Recall + Rerank`
- 特征工程：`genres / year / director / actors / user tags / tag genome`
- 统一离线评估指标与实验汇总
- `FastAPI + 前端展示页`
- `Notebook` 可视化实验报告
- `Docker` 部署

## 当前实验结果

下面展示的是本地实验中得到的一组正式离线结果，评估协议为 `rating >= 4.0 + Leave-One-Out + 全量未见物品候选集`：

| Model | HR@10 | HR@20 | NDCG@10 | MRR | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightfm | 0.0995 | 0.1633 | 0.0503 | 0.0483 | 0.5493 |
| svd | 0.0967 | 0.1616 | 0.0490 | 0.0469 | 0.2864 |
| als | 0.0902 | 0.1551 | 0.0440 | 0.0431 | 0.5605 |
| two_stage | 0.0885 | 0.1433 | 0.0444 | 0.0409 | 0.2745 |
| usercf | 0.0878 | 0.1461 | 0.0448 | 0.0437 | 0.1409 |
| itemcf | 0.0844 | 0.1327 | 0.0440 | 0.0422 | 0.1854 |
| bpr | 0.0583 | 0.1046 | 0.0294 | 0.0311 | 0.6545 |
| popularity | 0.0520 | 0.0908 | 0.0259 | 0.0256 | 0.0345 |
| content | 0.0319 | 0.0563 | 0.0157 | 0.0164 | 0.4898 |

可以直接总结成三点：

- `LightFM` 当前在准确性指标上最好，是项目里的最强单阶段模型
- `SVD / ALS / UserCF / ItemCF` 整体显著优于 `Popularity`，说明协同过滤信号有效
- `Content / BPR / ALS / LightFM` 覆盖率更高，说明多样性更强，但不同模型在准确性和覆盖率之间存在 trade-off

## 项目结构

```text
movielens_recsys/
├── data/                            # 本地自行下载的数据，默认不提交
├── artifacts/                       # 本地训练生成的模型与指标，默认不提交
├── docs/
│   └── ab_test.md                   # A/B Test 思路说明
├── notebooks/
│   └── experiment_report.ipynb      # 实验报告 Notebook
├── static/                          # 前端展示页
├── movielens_recsys/
│   ├── artifacts.py                 # artifact 读写
│   ├── data.py                      # 数据读取、过滤、特征构建
│   ├── evaluation.py                # 评估逻辑
│   ├── evaluate.py                  # 离线评估入口
│   ├── metrics.py                   # HR / Recall / NDCG / MAP / MRR / Coverage
│   ├── models.py                    # 各类推荐模型 + two_stage
│   ├── recommend.py                 # 命令行推荐入口
│   ├── serve.py                     # FastAPI 服务 + 前端入口
│   └── train.py                     # 训练与实验入口
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-optional.txt
└── README.md
```

## 项目亮点

### 1. 两阶段推荐架构

- Recall：默认使用 `content + itemcf + popularity`
- Rerank：默认使用 `svd`，也可切换成 `lightfm`

### 2. 更完整的特征工程

内容特征不再只用 `genres + tags`，还加入：

- `year`：从标题中解析年份
- `decade`：年代桶
- `tag genome`：从 `genome_scores.csv` 提取高相关标签
- `director / actors`：优先读取 `data/movie_people.csv`；若没有则用 `tags + genome tags` 做启发式推断

### 3. 工程化交付

- API：FastAPI
- 展示页：静态前端直接消费 API
- 部署：Docker / docker-compose
- 文档：A/B Test 思路说明

## 环境准备

建议使用 Python 3.9+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果启用 `ALS / BPR / LightFM`：

```bash
pip install -r requirements-optional.txt
```

如果可选依赖没有安装，训练脚本不会报致命错误，而是把这些模型标记为 `skipped`。

## 数据准备

把 MovieLens 相关 CSV 文件放到 `data/` 目录下，至少需要：

- `ratings.csv`
- `movies.csv`
- `tags.csv`

可选增强数据：

- `genome_scores.csv`
- `genome_tags.csv`
- `movie_people.csv`

目录格式见：

- [data/README.md](data/README.md)

## 快速开始

### 1. 训练并对比多个模型

```bash
python3 -m movielens_recsys.train \
  --preset quick \
  --data-dir data \
  --artifacts-dir artifacts \
  --models popularity,content,usercf,itemcf,svd,two_stage,als,bpr,lightfm \
  --include-user-ids 1
```

### 2. 训练两阶段推荐

```bash
python3 -m movielens_recsys.train \
  --preset quick \
  --data-dir data \
  --artifacts-dir artifacts \
  --models two_stage \
  --two-stage-recall-models content,itemcf,popularity \
  --two-stage-reranker svd \
  --recall-k-per-model 120 \
  --candidate-limit 300 \
  --include-user-ids 1
```

### 3. 重新评估已有模型

```bash
python3 -m movielens_recsys.evaluate \
  --artifacts-dir artifacts \
  --models all
```

### 4. 给指定用户生成推荐结果

```bash
python3 -m movielens_recsys.recommend \
  --artifacts-dir artifacts \
  --model two_stage \
  --user-id 1 \
  --top-k 10
```

### 5. 启动推荐 API 和前端展示页

```bash
python3 -m movielens_recsys.serve \
  --artifacts-dir artifacts \
  --host 127.0.0.1 \
  --port 8000
```

启动后可访问：

- [http://127.0.0.1:8000](http://127.0.0.1:8000)
- `GET /health`
- `GET /models`
- `GET /users/{user_id}/recommendations?model=two_stage&top_k=10`
- `GET /items/{movie_id}/similar?model=content&top_k=10`
- `GET /metrics/latest`
- `GET /ab-test/plan`

## 特征工程说明

项目当前使用以下特征：

- `genres`
- `year`
- `decade`
- `user tags TF-IDF`
- `genome tags TF-IDF`
- `director / actors`

默认逻辑：

- `year` 从 `movies.csv` 标题中解析
- `genome tags` 从 `genome_scores.csv + genome_tags.csv` 中抽取高相关标签
- `director / actors`：
  - 如果存在 `data/movie_people.csv`，优先读取
  - 如果不存在，则根据 `tags` 和 `genome tags` 进行启发式推断

如果后面有更高质量的电影元数据，可以新增：

```text
data/movie_people.csv
columns: movieId,director,actors
```

## 离线评估指标

项目默认输出以下指标：

- `HitRate@10`
- `HitRate@20`
- `Recall@10`
- `Recall@20`
- `NDCG@10`
- `NDCG@20`
- `MAP@10`
- `MRR`
- `Item Coverage@10`

评估协议：

- 正反馈定义：`rating >= 4.0`
- 切分方式：`Leave-One-Out`
- 测试样本：每个用户最后一次正反馈
- 候选集：过滤后所有未看过电影

## 模型说明

核心稳定模型：

- `popularity`
- `content`
- `usercf`
- `itemcf`
- `svd`
- `two_stage`

可选增强模型：

- `als`
- `bpr`
- `lightfm`

## A/B Test 思路

详细文档见：
- [docs/ab_test.md](docs/ab_test.md)


## Docker 部署

构建镜像：

```bash
docker build -t movielens-recsys .
```

启动服务：

```bash
docker compose up --build
```

如果你希望容器内也加载 `LightFM`，在 Apple Silicon 上默认通过 `linux/amd64` 兼容层运行，所以首次构建会更慢。

## 训练产物

训练完成后会在 `artifacts/` 下生成：

- `manifest.json`
- `metadata.json`
- `metrics_summary.csv`
- `metrics_summary.json`
- `movies_filtered.csv`
- `shared.pkl`
- `models/<model_name>/model.pkl`
- `models/<model_name>/metadata.json`

## Notebook 实验报告

Notebook 位置：

- `notebooks/experiment_report.ipynb`

它包含：

- 数据集概览
- 内容特征说明
- 多模型指标对比图
- 推荐案例
- 项目总结

## 常用参数

- `--preset quick|full`
- `--models popularity,content,usercf,itemcf,svd,two_stage,als,bpr,lightfm`
- `--include-user-ids 1,2,3`
- `--two-stage-recall-models content,itemcf,popularity`
- `--two-stage-reranker svd`
- `--recall-k-per-model 100`
- `--candidate-limit 300`
- `--max-tag-features 3000`
- `--max-genome-features 1000`
- `--genome-top-n 8`
- `--genome-min-relevance 0.6`
