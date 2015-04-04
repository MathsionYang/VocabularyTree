# VocabularyTree
VocabularyTree. My graduation project

我的毕业设计：乱序图像集的快速匹配和支撑结构生成


通过词汇树得到大量的视觉单词，基于这些视觉单词将训练集当中的图像转换成为tf-idf向量，通过tf-idf向量的检索来找到最相似的图像

数据结构：

class vocabularyTreeNode: 词汇树当中的节点

class vocabularyTree:  词汇树

class imageRetriver: 对于词汇树进行封装，用于图像检索
