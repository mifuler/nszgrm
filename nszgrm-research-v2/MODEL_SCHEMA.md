# `models_fp32.json` 模型接口说明（format_version=3）

本文件说明训练器输出给独立 HTML 前端使用的模型参数结构。训练包本身不生成 HTML、JavaScript 或浏览器资源。

## 顶层结构

```json
{
  "format_version": 3,
  "trainer_version": "3.0.0",
  "generated_utc": "ISO-8601 UTC 时间",
  "seed": 20261117,
  "precision": "original-FP32; NO parameter quantization",
  "models": {
    "writing": { "task": "writing", "classes": [], "models": {}, "fusion_weights": {}, "score_semantics": "..." },
    "language": { "task": "language", "classes": [], "models": {}, "fusion_weights": {}, "score_semantics": "..." }
  },
  "provenance": {}
}
```

`models.language` 只有在八种语言全部达到最低数据量、训练和数据划分检查通过时才会存在。`required` 模式下缺少任意一类会使训练失败；`best_effort` 模式可以只产出书写模型，并在报告中说明原因。

## 固定类别

书写模型：

- `zh_native`
- `ja_kanji_proxy`
- `foreign_phonetic_proxy`

语言模型：

- `zh`
- `en`
- `fr`
- `de`
- `es`
- `ru`
- `ja`
- `ko`

语言模型的 `zh` 来自 Wikidata P103 的 Chinese（Q7850）、Mandarin（Q9192）和 Standard Chinese（Q727694）人物母语记录合并。P103 是人物母语属性，不是姓名词源标注。

## 每个任务的字段

- `task`：`writing` 或 `language`。
- `classes`：概率数组所对应的类别顺序；前端必须按此数组索引，不能自行重排后再读取模型参数。
- `models`：四个独立模型，键固定为 `linear`、`complement_nb`、`cnn`、`bigru`。
- `fusion_weights`：四个模型的融合权重，非负且总和为 1。
- `score_semantics`：该任务分数的语义和限制。

每个模型均有 `temperature`。单模型先产生原始概率 `p`，再按以下方式校准：

\[
 p'_i = \operatorname{softmax}\left(\frac{\log(\max(p_i,10^{-30}))}{T}\right)
\]

融合结果为：

\[
 p_{fusion} = \sum_{m\in\{linear,complement\_nb,cnn,bigru\}} w_m p'_m
\]

融合概率按 `classes` 顺序排列。最终类别为最大融合概率对应的类别。

## 子模型参数

### `linear`

- `tfidf.type`：`tfidf_char_1_3`
- `tfidf.vocabulary`：字符 1–3 gram 到特征索引的映射。
- `tfidf.idf`：按特征索引排列的 IDF 数组。
- `coef`：类别 × 特征的 FP32 权重。
- `intercept`：每类偏置。
- `mode`：多分类通常为 `sigmoid_ovr`。

输入姓名按字符切分，统计 1–3 gram，乘以 IDF 后进行 L2 归一化。多分类模型对每类线性 logit 取 sigmoid，再归一化为概率。

### `complement_nb`

复用同一 `tfidf` 特征；`feature_log_prob` 是类别 × 特征矩阵。前端将每类矩阵行与 TF-IDF 特征向量点积，再对类别分数做 softmax。

### `cnn`

- `vocab`：汉字到 token ID 的映射，`<PAD>` 为 0、`<UNK>` 为 1。
- `max_len`：最大字符长度。
- `hyperparameters.emb`：Embedding 维度。
- `hyperparameters.channels`：每个卷积窗口的通道数。
- `state`：PyTorch `state_dict`。每个参数包含 `shape` 和展平的 `values`；按 C-order reshape 回 `shape`。

TextCNN 使用 2、3、4 字符窗口、ReLU、masked max pooling 和线性分类头。

### `bigru`

字段结构与 `cnn` 类似。`hyperparameters.emb` 为 Embedding 维度，`hyperparameters.hidden` 为单方向 GRU 隐藏维度。`state` 保存双向 GRU 和分类头的全部 FP32 参数，按 C-order reshape。

## 音译指数

书写模型中 `foreign_phonetic_proxy` 的融合分数乘以 100，可作为前端显示的音译指数。它是公开语料弱监督分类分数，尚未经过代表性真实姓名人群的概率校准，不能解释为真实词源概率或真实世界准确率。

## 文件体积

`models_fp32.json` 的原始字节数上限为 200,000,000。训练器不在导出前量化参数。前端可另行开发量化版本，但量化文件不属于本训练包的训练输出。
