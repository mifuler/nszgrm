# 中文姓名音译检测 · GitHub Actions 训练工程

使用 **Python、scikit-learn、GitHub Actions Linux CPU** 进行训练。Actions 一键执行：下载公开姓名语料 → 清洗与冲突去除 → 监督训练 → 分类器校准 → 独立测试 → INT8/INT16 自动权重量化 → 打包浏览器离线推理文件 → 保存全部训练结果。

**状态：可运行的训练工程，不附带已训练的真实模型。** 只有 Actions 真正成功执行、产生指标后才可以评价模型质量。

## 自动部署到新仓库（可选）

已安装 `git`、`gh` 且执行过 `gh auth login` 时，可在解压后的工程目录执行：

```bash
# Linux / Termux，创建一个新的公开仓库并尝试立即触发训练
bash deploy.sh iqonli/ChineseName-Training
```

Windows PowerShell：

```powershell
.\deploy.ps1 -Repository iqonli/ChineseName-Training
```

仓库名只是示例，可自行替换。脚本会创建**公开**仓库、提交代码并请求执行首次工作流；若仓库已存在则直接停止，不覆盖原有数据。首次提交后 GitHub 尚未识别工作流时，进入仓库 Actions 手动运行即可。脚本不会接收或保存 GitHub 密码，使用已有的 `gh` 登录凭据。

## 如何一键启动

1. 新建一个 GitHub 仓库，将本 ZIP 内文件**连同 `.github/workflows/train.yml`** 一起提交到仓库根目录。建议先用独立仓库验证。仓库可以公开，下载的原始姓名数据不会提交进去。
2. 打开仓库 **Actions → Train Chinese-name models (CPU / Linux) → Run workflow**。
3. 保持默认参数点击绿色的 **Run workflow**。无需提供密钥、GPU 或外部存储服务。
4. 作业结束，打开对应运行记录底部 **Artifacts**，下载 `chinese-name-models-<run_id>`。可在 Summary 查看分类指标。默认还会创建一个不设置为 latest 的 **GitHub Release**，长期保存模型及报告（可在 Actions 表单取消勾选 `publish_release`）。
5. 解压 Artifacts 后，打开 `browser/index.html` 即可用刚训练的模型离线查询姓名。 `browser/models.js` 是自动生成的模型权重。

运行环境是 `ubuntu-24.04` + Python 3.11。如果开启 Release 发布，仓库的 Actions 需要允许工作流获得 `contents: write` 权限；如果发布被组织权限阻止，训练产物通常仍保存在 Artifacts 中。主要依赖 `scikit-learn`、`numpy`、`scipy`、`joblib`，不需要 PyTorch、TensorFlow 或 GPU。工作流 Artifact 保留 30 天；如果仓库策略更短，则以实际限制为准。Release 资产在未人为删除的情况下可以长期保留。**建议另外下载 ZIP 做备份。**

## 默认训练内容与标签

| 任务 | 监督样本来源 | 类别 / 实际含义 | 结果文件 |
| --- | --- | --- | --- |
| `writing_type` | wainshine/Chinese-Names-Corpus | `zh_native` 中文姓名库、`foreign_phonetic_proxy` 翻译名库、`ja_kanji_proxy` 日本姓名汉字库 | `writing_type.joblib/json/*_metrics.json` |
| `language_proxy` | Wikidata P103 人物母语属性 + 中文名称 | `en`、`ru`、`ja`、`fr`、`es`、`de`、`ko` 等有实际样本的语言 | `language_proxy.joblib/json/*_metrics.json`（成功时） |

`foreign_phonetic_proxy` **不能等同于“英语音译”**：源词典包含地名，也不保证所有姓名源自英语。`ja_kanji_proxy` 指日本姓名汉字表记模式，不能被视为中文音译。P103 表示人物母语，**不是姓名词源的金标准**。不同源语言可能产生同一个中文译名，模型必须承认不确定性。

默认对每个书写模式类别最多采样 120,000 条去重姓名进行训练（提高 GitHub Hosted Linux CPU 的可行性）；原始三组语料仍完整下载并扫描。可通过 Actions 界面调整上限（设为 0 表示不截取，但可能大幅增加内存和时间）。按姓名哈希确定训练/验证/测试分组（约 80%/10%/10%）；同一个规范化汉字串不会跨拆分集；交叉类别同形串会剔除。数据仍可能存在同一人的不同名称变体造成的语义泄漏。

特征为带首尾边界符的 **汉字字符 1—3 gram**，二值化计数；训练采用 SGD Logistic 分类器和验证集 Logistic 校准器。优先使用 INT8 线性分类器权重；若验证集量化误差超标则自动改用 INT16。校准器系数保留浮点数。量化精度只根据验证集选择，再在独立测试集计算指标和与非量化模型的差异（准确率变化、Macro-F1 变化、预测类别变化率、概率平均/最大绝对偏差）。**量化不是无损的**：训练产物始终同时保存非量化版 `.joblib`，实际损失由报告中的测量值决定。没有独立外部代表性数据前，0—100 的**音译指数是模型分数，不是真实世界音译概率**。

### Wikidata 训练的失败处理

默认 `wikidata=best_effort`：依次查询七种语言的中文人名，通过 P103 构建多语言弱监督标签。如果公共查询服务失败或数据量不足，保留训练成功的书写模式模型，记录具体失败原因，不生成虚假的语言模型。浏览器也将显示“无语言模型”。

可将参数改为 `required`，要求所有 Wikidata 查询成功且至少两类语言拥有足够样本，否则 Actions 报错。改为 `off`，只训练三类写法分类器。

数据服务依赖网络，Wikidata 对批量 SPARQL 查询有流量与执行时间限制。复杂查询可能失败，因此语言模型尚不能保证每次都能训练；`source_manifest.json` 会如实记录各语言数量和原因。

## 训练产物结构

```text
outputs/
  REPORT.md                      # 本次真实训练结果概览、限制及混淆矩阵
  source_manifest.json           # 来源、文件 SHA256、清洗与失败记录
  writing_type.joblib            # Python: 向量器、线性分类器、校准器
  writing_type.json              # INT8/INT16 浏览器端模型
  writing_type_metrics.json      # 独立测试准确率、Macro-F1、各类 Brier、混淆矩阵、量化误差
  writing_type_test_errors.csv   # 错误样本
  language_proxy.joblib          # 仅成功训练语言分类器后出现
  language_proxy.json
  language_proxy_metrics.json
  language_proxy_test_errors.csv
  browser/
    index.html                   # 完全离线页面，无 CDN
    models.js                    # 包含真实训练出的 量化模型权重
    README.txt
```

浏览器版采用 `<script src="models.js">` 加载本地模型，支持从文件管理器直接打开 `index.html`，无需 HTTP 服务器。推理只接受 **2—8 个连续汉字**，不默默删除姓名中的间隔号、字母或其他符号。完整外国姓名请按实际成分拆分；批量文本/CSV 支持导出结果。

由于标签含混及人工词典的抽样偏差，`*_metrics.json` 反映的是**抽取语料内**的分布。不能用高分直接推断现实世界中一个人的文化背景或国籍。

## 本地运行 / 调试

Linux、macOS 或 Windows 上使用 Python 3.11：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/pipeline.py --output outputs --cache .cache/data --max-per-class 120000 --wikidata best_effort
```

无需联网的开发调试：

```bash
python src/pipeline.py --demo --output demo_outputs --wikidata off
```

**`--demo` 只验证训练和导出流程，使用固定的手选测试姓名，不可作为有效模型或算法性能展示。** Actions 的正式工作流始终运行公开语料真实训练，绝不悄悄切换到 demo。

## 下载来源及许可证

- 主语料：[Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus)，仓库许可证 Apache-2.0，但其中原始汇编语料的上游来源不同，需单独核实再分发权利。下载后，使用的三个原始 `.txt` 文件路径、大小、SHA256 和 commit 写入 `source_manifest.json`。
- 多语言弱标签：[Wikidata P103](https://www.wikidata.org/wiki/Property:P103)、[Wikidata Query Service](https://query.wikidata.org/)，Wikidata 结构化数据 CC0；任务中只使用姓名、实体编号和母语分类，不把国籍当语言标签。
- GitHub Actions：[工作流语法](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)、[Artifact 保留策略](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts)。

自动下载的原始公开数据放在 `.cache/`，默认不会上传 GitHub Artifacts 或纳入源代码 ZIP。可通过 `source_manifest.json` 复核来源与校验值。

## 当前边界 / 后续任务

本包完成三类**弱监督**分类器的训练闭环，并尽可能训练 P103 多语言模型。尚未引入可靠人工标注的跨语言姓名金标准、特征音韵匹配以及神经网络 ONNX 备选模型，也未做校外数据的实际准确率验证。下一阶段建议人工复核分层抽样测试集（至少分别检验中、英、俄、日姓名及歧义译名），再决定是否训练字符 CNN、BiGRU 和 ONNX 模型。不要把这一阶段的测试集准确率称为“跨语言姓名来源识别准确率”。
