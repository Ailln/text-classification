# text-classification

## 1 简介

`文本分类`是 NLP 的基本任务之一，通过此任务的学习可以快速熟悉 NLP 的工作流程，同时了解 NLP 算法所能实现的效果。

## 2 数据

### 2.1 THUCNews 数据集

> [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)
是根据新浪新闻 RSS 订阅频道 2005~2011 年间的历史数据筛选过滤生成，包含 74 万篇新闻文档（2.19 GB），均为 UTF-8 纯文本格式。
我们在原始新浪新闻分类体系的基础上，重新整合划分出 14 个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

### 2.2 新数据集（THUCNews-5_2000）

从 THUCNews 数据集中选取类别名词词义相差比较大的 5 组数据，分别为：
`体育`、`房产`、`股票`、`时政`、`游戏`。
接下来，从每 1 组数据中随机取 2000 条放入的新的数据集中，最终得到 1 万条新闻数据。
将新数据集按照 8:2:2 划分成训练集（train set）、验证集（validate set）、测试集（test set），保存在`/data/THUCNews-5_2000/`中。

### 2.3 单个数据介绍

每个单独的文件是 1 条数据，文件名使用`-`切分后，前半部分是类别名称，后半部分是原 THUCNews 数据集中的 id 。

## 3 项目结构

```
.
├── config # 配置
├── data # 数据
│   └── THUCNews-5_2000
│       ├── train
│       ├── validate
│       └── test
├── model # 模型
│   └── tensorflow
├── output # 输出
├── run # 入口
├── script # 脚本
├── server # 部署
└── util # 工具
```

## 4 安装

```bash
# 克隆代码
git clone https://github.com/kinggreenhall/text-classification.git

# 进入项目
cd text-classification

# 安装依赖
pip install -r requirements.txt

# 准备数据
bash ./script/prepare_data.sh

# 开始训练
python -m run.tensorflow_cnn
# 或者（需要替换 $config_path）
python -m run.tensorflow_cnn train $config_path

# 测试模型（需要替换 $train_time）
python -m run.tensorflow_cnn test $train_time
```

## 5 可视化

```bash
# 使用 Tensorboard 进行可视化（需要替换 $train_time）
tensorboard --logdir ./output/$train_time/log
```

## 6 部署

```bash
# 启动 flask 服务
python -m server.flask_app
# 发送 test 数据
pythom -m server.send_data
```

## 7 TODO

- [x] tensorflow cnn
- [x] flask server
- [ ] tensorflow rnn

## 8 联系方式

kinggreenhall@gmail.com

## 9 LICENSE

[MIT License](./LICENSE)
