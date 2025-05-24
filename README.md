# Semantic Pivots Enable Cross-Lingual Transfer in Large Language Models 	
## Word-Level Cross-Lingual Translation Task
### 数据集构建
- 获取字典中的词汇和词性
- 构建平行的单词翻译数据集
- 为每条数据的每个语言对 构建干扰项
- 过滤带有不合理干扰项的数据
### 计算跨语言能力分数
## Semantic Pivot behavior
- 利用 logit lens 观察模型中间层输出
- 以共现频率划分模型的两种推理方式，并计算AUC值衡量其划分的准确性
- 发现潜在的语义中枢
  - 求取与原词语和目标词语均有较高共现频率的
  - 过滤没有语义关联的token
  - 观察语义中枢集合在中间层的出现概率总和

## semantic pivot-aware pre-training dataset
- 根据预训练文档，构建共现频率图
- 选取与较多token有较高共现频率的token 组成语义中枢集合
- 选取含有高语义中枢集合的文档作为预训练语料
