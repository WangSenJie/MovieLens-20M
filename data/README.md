# 数据目录说明

这个目录默认不提交原始数据，只保留目录结构说明。

## 必需文件

请把以下文件放到当前目录下：

- `ratings.csv`
- `movies.csv`
- `tags.csv`

## 可选增强文件

如果你希望启用更丰富的内容特征，可以额外放入：

- `genome_scores.csv`
- `genome_tags.csv`
- `movie_people.csv`

其中 `movie_people.csv` 的推荐格式为：

```text
movieId,director,actors
1,John Lasseter,Tom Hanks|Tim Allen
2,Joe Johnston,Robin Williams|Kirsten Dunst
```

## 说明

- 缺少 `genome_*` 文件时，项目仍可运行，只是内容特征会更弱
- 缺少 `movie_people.csv` 时，项目会退化为基于 `tags + genome tags` 的启发式人物特征
- 训练命令默认读取 `data/` 目录，因此文件名请保持一致
