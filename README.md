# text-classification

## 1 简介

`文本分类`是NLP的基本任务之一，通过学习此任务熟悉NLP的工作流程，同时了解NLP算法所能实现的效果。

## 2 数据

### 2.1 THUCNews 数据集

> [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

### 2.2 我们的数据集（THUCNews-5_2000）

我们选取 THUCNews 数据集中类别的词义相差比较大的 5 组：
`体育`、`房产`、`股票`、`时政`、`游戏`。

从每1组中随机取 2000 条加入我们的数据集，最后得到 1 万条新闻数据。
这些数据保存在`/data/THUCNews-5_2000/`中。

每个单独的文件是1条数据，文件名使用`-`切分后，前半部分是类别名称，后半部分是原 THUCNews 数据集中的 id 。

## 3 项目结构

```
.
├── script
├── data
│   └── THUCNews-5_2000
├── model
│   ├── pytorch
│   ├── sklearn
│   └── tensorflow
├── config
└── util
```

## 4 安装

```bash
# 克隆代码
git clone https://github.com/kinggreenhall/text-classification.git

cd text-classification

# 安装依赖
pip install -r requirements.txt

# 准备数据
bash prepare_data.sh

# 开始训练
bash train_with_tensorflow.sh
```