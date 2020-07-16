# text-classification

## 1 简介

`文本分类`　是 NLP 的基本任务之一，通过此任务的学习可以快速熟悉 NLP 的工作流程，同时了解 NLP 算法所能实现的效果。

本项目包含了 `tensorflow` `pytorch` `sklearn` 三个框架的各种不同实现，包括 `cnn` `rnn` 等。
除此之外，还提供了基本部署方法，可以使用 RESTful API 访问！

## 2 数据

### 2.1 THUCNews 数据集

我们选用了了开源的中文文本分类数据集`THUCNews`，介绍如下：

> [THUCNews](http://thuctc.thunlp.org/) 是根据新浪新闻 RSS 订阅频道 2005~2011 年间的历史数据筛选过滤生成，包含 74 万篇新闻文档（2.19 GB），均为 UTF-8 纯文本格式。
> 我们在原始新浪新闻分类体系的基础上，重新整合划分出 14 个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

| 序号 | 类别 | 数量 |
| :-: | :-: | :-: |
| 1 | 财经 | 37098 |
| 2 | 彩票 | 7588 |
| 3 | 房产 | 20050 |
| 4 | 股票 | 154398 |
| 5 | 家居 | 32586 |
| 6 | 教育 | 41936 |
| 7 | 科技 | 162929 |
| 8 | 社会 | 50849 |
| 9 | 时尚 | 13368 |
| 10 | 时政 | 63086 |
| 11 | 体育 | 131604 |
| 12 | 星座 | 3578 |
| 13 | 游戏 | 24373 |
| 14 | 娱乐 | 92632 |

### 2.2 新数据集 1（THUCNews-5_2000，低配置）

由于原数据集较大，对硬件要求较高，因此我们先从中提取部分数据再使用。操作过程如下：

1. 我们从 THUCNews 数据集中选取类别名词词义相差比较大的 5 组数据，分别为： `体育, 房产, 股票, 时政, 游戏`。
2. 从每 1 组数据中随机取 2000 条放入的新的数据集中，最终得到 1 万条新闻数据。
3. 将新数据集按照 `8:2:2` 划分成训练集（train set）、验证集（validate set）、测试集（test set），保存在 `/data/THUCNews-5_2000/` 中。

数据集数量为：

- train set: 1600 * 5
- validate set: 200 * 5
- test set: 200 * 5

获取数据方式见 `/script/prepare_data.sh`。

### 2.2 新数据集 2（THUCNews-14_10000，中配置）

TODO

### 2.3 单个数据介绍

每个单独的文件是 1 条数据，文件名使用`-`切分后，前半部分是类别名称，后半部分是原 THUCNews 数据集中的 id 。
举例：
```bash
体育-76.txt
```
## 3 结构

```
.
├── config # 配置
├── data # 数据
│   └── THUCNews-5_2000
│       ├── train
│       ├── validate
│       └── test
├── model # 模型
│   ├── pytorch
│   ├── tensorflow
│   └── sklearn
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
bash script/tf1-env.sh

# 准备数据
bash script/prepare_data.sh

# 开始训练
python -m run.tensorflow -m cnn

# 测试模型（需要替换 $train_date）
python -m run.tensorflow -r test -d $train_date
```

## 5 可视化

```bash
# 使用 Tensorboard 进行可视化（需要替换 $train_date）
tensorboard --logdir ./output/$train_date/log
```

> 当前仅 tensorflow 可用。

## 6 部署

```bash
# 启动 flask 服务
python -m server.flask_app
# 发送 test 数据
pythom -m server.send_data
```

> 当前仅 tensorflow 可用。

## 7 TODO

- [x] tensorflow cnn
- [ ] tensorflow rnn
- [ ] tensorflow_v2 cnn
- [ ] tensorflow_v2 rnn
- [x] tensorflow flask server
- [x] pytorch cnn
- [x] pytorch rnn
- [ ] pytorch flask server
- [x] sklearn svm
- [x] sklearn bayes

## 8 参考

- [THUCTC: 一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)
- [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
- [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf)

## 9 许可证

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)
[![](https://award.dovolopor.com?lt=Ailln's&rt=idea&lbc=lightgray&rbc=red&ltc=red)](https://github.com/Ailln/award)
