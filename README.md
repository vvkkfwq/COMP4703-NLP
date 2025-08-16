# COMP4703 Natural Language Processing

这个仓库包含COMP4703自然语言处理课程的实验作业和项目。

## Week 3: 平滑方法与作者识别

### 概述
探索语言模型的平滑方法及其在作者识别中的应用。

### 实现的方法
- **最大似然估计 (MLE)**
- **拉普拉斯平滑 (Add-1)**
- **Lidstone平滑 (Add-α)**
- **回退模型 (Backoff Models)**

### 作者识别实验
使用交叉熵进行作者识别：
- 训练数据：Jane Austen的《理智与情感》
- 测试数据：《爱玛》vs. G.K. Chesterton的《球与十字架》

### 文件结构
- `week3/week3.ipynb`: 主要的Jupyter notebook实现
- `week3/nltk_model/`: NLTK2 n-gram模型移植版本

### 实验结果
成功使用trigram语言模型与Lidstone平滑识别出Jane Austen为《爱玛》的作者。

### 结果分析
- **Emma** (同作者): 每词交叉熵 = 14.61 (较低)
- **Chesterton** (不同作者): 每词交叉熵 = 17.30 (较高)

系统成功通过统计语言模型识别了作者身份！

## 环境设置
```bash
pip install nltk
# 下载所需的NLTK数据
import nltk
nltk.download('gutenberg')
```

## 使用方法
在Jupyter Notebook或JupyterLab中打开 `week3/week3.ipynb`。

## 技术要点
- **N-gram语言建模**
- **概率平滑技术**
- **交叉熵计算**
- **统计文本分析**
