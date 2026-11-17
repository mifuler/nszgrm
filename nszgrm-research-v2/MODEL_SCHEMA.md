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

`models_fp32.json` 的原始字节数上限为 200,000,000。正式训练工作流还会生成独立的 `models_int8.json`，其量化格式见下文。


## `models_int8.json` 量化格式（format_version=4）

量化模型保留 FP32 模型的任务结构、类别顺序、温度系数和融合权重，因此前端可以沿用相同的四模型推理流程。文件名为 `models_int8.json`，但失败模型可能使用 INT16；具体精度以每个权重对象的 `quant_format` 为准。

二维及更高维权重采用对称逐输出行量化。INT8 对象格式：

```json
{
  "quant_format": "sym_int8_per_row_v1",
  "shape": [3, 60000],
  "scales": [0.001, 0.002, 0.003],
  "data_b64": "..."
}
```

INT16 使用 `sym_int16_per_row_v1`，其余字段相同。`data_b64` 按 C-order 保存有符号小端整数。还原时，先按 `shape` 展平为行，再对第 i 行的整数乘以 `scales[i]`。

TF-IDF 的 vocabulary、idf、偏置、温度和融合权重保留原结构与 FP32 数值；ComplementNB 的 `feature_log_prob`、线性模型的 `coef`、CNN/BiGRU 的二维及以上 state 权重会量化。CNN/BiGRU 的一维 bias 保留 FP32。

量化文件硬上限为 15,000,000 字节（严格小于）。训练器对每个模型和融合结果分别计算相对 Accuracy 与 Macro-F1 损耗，门槛均为 10%。INT8 不通过时按模型回退 INT16 并重新评测。评测数据来自训练过程预留的弱标签测试划分，不是人工金标准。
