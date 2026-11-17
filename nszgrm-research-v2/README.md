# NSZGRM · 中文姓名多模型监督训练工程（v3.1）

本工程使用 Python 在 GitHub Actions 的 Ubuntu Linux CPU 环境训练中文姓名模型。它只负责**数据获取、监督训练、模型融合、评估和结果保存**；不生成 HTML，不包含浏览器前端。HTML 由独立项目开发，可按需读取 `models_fp32.json` 或 `models_int8.json`。

训练分为两个相互独立的任务：

- **书写模型：3 类**——中国本土姓名、日语汉字姓名、外国姓名音译写法。
- **语言模型：8 类**——中文、英语、法语、德语、西班牙语、俄语、日语、韩语。

每项任务分别训练四种架构：字符 TF-IDF + SGD Logistic、Complement Naive Bayes、TextCNN、BiGRU。四个模型独立输出类别分数，在验证集上分别校准，再学习融合权重。测试集不参与温度校准或融合权重选择。

> **标签边界：** 公开语料的类别属于弱监督标签。Wikidata `P103` 表示人物母语，不等于姓名词源。因此八分类模型识别的是“人物母语相关的中文姓名线索”，不能直接宣称它已可靠判定任意姓名的真实词源。只有增加来源可查、人工审核、按人物及别名分组的金标准测试集，才能进一步验证真实泛化能力。

## 模型定义

### 书写模型

| 标签 | 含义 |
|---|---|
| `zh_native` | 中国本土姓名语料 |
| `ja_kanji_proxy` | 日本姓名的汉字写法语料 |
| `foreign_phonetic_proxy` | 外文姓名中文译写语料 |

这三类描述训练语料中的书写类别。尤其 `foreign_phonetic_proxy` 不是“所有外国姓名”，48 万翻译名语料可能包含地名或其他非人名记录，清洗后仍需人工抽查。

### 语言模型

| 标签 | 含义 | Wikidata P103 项目 |
|---|---|---|
| `zh` | 中文 | [Chinese `Q7850`](https://www.wikidata.org/wiki/Q7850)、[Mandarin `Q9192`](https://www.wikidata.org/wiki/Q9192)、[Standard Chinese `Q727694`](https://www.wikidata.org/wiki/Q727694) |
| `en` | 英语 | `Q1860` |
| `fr` | 法语 | `Q150` |
| `de` | 德语 | `Q188` |
| `es` | 西班牙语 | `Q1321` |
| `ru` | 俄语 | `Q7737` |
| `ja` | 日语 | `Q5287` |
| `ko` | 韩语 | `Q9176` |

中文类把 Wikidata 中的 Chinese、Mandarin 和 Standard Chinese 母语记录合并到 `zh`，并按实体和姓名去重。这样可以覆盖 `P103` 中采用不同中文语言项目标注的人物。

**完整八分类是默认训练要求。** `required` 模式只有在八类都达到最低样本量、且数据划分可用时才会训练并发布语言模型；缺少任意类别就让训练失败，不会把七分类结果冒充八分类。`best_effort` 可用于网络故障排查：如果覆盖不全，仍可保存书写模型，但语言模型会标记为 unavailable。`off` 则关闭语言任务。

### 四种架构与融合

| 模型 | 方法 | 作用 |
|---|---|---|
| `linear` | 字符 1–3 gram TF-IDF + SGD Logistic | 字符组合、位置和频率特征 |
| `complement_nb` | Complement Naive Bayes | 提供不同于线性模型的概率估计 |
| `cnn` | TextCNN，2/3/4 字符窗口 | 学习局部姓名片段 |
| `bigru` | 双向 GRU | 学习字符顺序及上下文 |

每个模型分别输出 Accuracy、Balanced Accuracy、Macro-F1、Log Loss、ECE、分类报告和混淆矩阵。温度校准只使用验证集 A；融合权重只使用不重叠的验证集 B。四个模型均参与融合，权重下限为 3%。

默认神经网络参数：Embedding 128、CNN Channels 96、BiGRU Hidden 128、词表上限 6,000；字符 TF-IDF 最多 60,000 个特征。TextCNN 和 BiGRU 每类最多抽样 45,000 条训练样本，最多训练 10 轮并启用早停。

## 数据与划分

默认自动下载：

- [Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus)：中文姓名、外文中文译名、日本姓名及古代中文姓名补充语料。
- [Wikidata Query Service](https://query.wikidata.org/)：按人物 `P103` 母语属性获取中文标签。

训练过程保存上游仓库 revision、原始文件 SHA256、文件大小、各阶段清洗数量、每类样本量和划分统计。原始语料不随训练结果重新分发。

同一任务中相同汉字姓名不会跨训练、验证和测试集；Wikidata 数据按人物实体及相同姓名连通分组。仍可能存在同一人物的其他译名、不同译名对应同一原名等跨样本关系，必须继续用人工审核数据检查。

## 模型体积与精度

每次正式训练同时交付两份模型：`models_fp32.json` 保留完整 FP32 参数，硬上限为 200,000,000 字节；`models_int8.json` 使用逐输出行对称 INT8 权重量化，必要时对个别模型回退到 INT16，偏置、IDF、温度和融合权重保留 FP32。量化文件原始大小必须严格小于 **15,000,000 字节**。

量化不是只看文件大小。训练后会从本轮训练的独立测试划分中分层抽样，分别对四个子模型和融合模型计算 Accuracy、Macro-F1；任一模型的相对损耗超过 10%，就把失败模型切换为 INT16 后重新评测。若融合模型仍失败，则扩大回退范围；最终仍不能通过，Actions 不会发布 Release。评测样本属于语料弱标签，不能代替人工金标准。

Wikidata 下载优先使用 aria2c，失败时由 urllib 继续重试。HTTP 截断、超时、无效 JSON 不会写入缓存；有效响应通过原子替换保存，最多六轮指数退避。

本训练包不生成 HTML。独立前端可以分别加载 FP32 和量化模型，具体张量格式见 [`MODEL_SCHEMA.md`](MODEL_SCHEMA.md)。

## 一键推送到 `mifuler/nszgrm`

在 Android Termux 中使用已经登录 GitHub 的账号。先确认当前账号和目标仓库权限：

```bash
pkg update -y
pkg install -y git gh rsync unzip
termux-setup-storage

gh api user --jq .login
gh repo view mifuler/nszgrm --json viewerPermission --jq '.viewerPermission'
```

当前账号必须对 `mifuler/nszgrm` 具有 `WRITE`、`MAINTAIN` 或 `ADMIN` 权限，并允许修改 `.github/workflows`。

把 ZIP 下载到 Android `Download`，执行：

```bash
mkdir -p ~/nszgrm-v3
unzip -o ~/storage/downloads/NSZGRM-ResearchTrainer-v3-zh8-no-html.zip -d ~/nszgrm-v3

cd ~/nszgrm-v3/NSZGRM-ResearchTrainer
bash push-existing.sh mifuler/nszgrm
```

推送脚本会克隆目标仓库，将训练工程放在独立目录 `nszgrm-research-v2/`，并更新 `.github/workflows/research-train.yml`。它不会删除仓库根目录 `LICENSE`、旧版模型文件或其他工作流；仅同步新版训练工程目录。

## 推送与启动 Actions

首次使用 Termux 执行 `push-existing.sh` 时，会把一次性触发标记一并推送到 `main`；工作流因此自动启动，使用 150000 / 45000 / 10 epochs / required / publish release 的默认配置。以后重新训练时，使用 `mifuler` 账号打开：

<https://github.com/mifuler/nszgrm/actions/workflows/research-train.yml>

选择 `Run workflow`：

| 参数 | 首次建议 |
|---|---:|
| Branch | `main` |
| `max_per_class` | `150000` |
| `neural_per_class` | `45000` |
| `epochs` | `10` |
| `wikidata` | **`required`** |
| `publish_release` | `true` |

`required` 会要求八个语言类别全部达到最低样本量。若 Wikidata 服务临时故障，先检查日志；不要把 `best_effort` 产生的部分语言模型当作八分类成品。

工作流使用标准 `ubuntu-24.04` CPU，最多运行 350 分钟。会先执行单元测试和合成数据烟雾训练，再安装 aria2、下载真实公开语料、训练、量化、评估损耗，并验证 200 MB FP32 与 15 MB 量化上限。

## 输出内容

训练成功后，Actions Artifact 和 GitHub Release ZIP 仅包含训练结果，例如：

```text
outputs/
├── models_fp32.json
├── models_int8.json
├── LOSS_REPORT.json
├── QUANT_AUDIT.json
├── REPORT.json
├── REPORT.md
├── train_manifest.json
├── review_queue.csv
├── writing_errors.csv
├── writing_disagreement_review_queue.csv
├── language_errors.csv
└── language_disagreement_review_queue.csv
```

实际文件会根据人工审核数据和语言模型是否成功生成而略有不同。不会包含 `.html` 文件。

- `models_fp32.json`：两项任务的原始 FP32 参数、类别、温度系数和融合权重。
- `models_int8.json`：逐行 INT8/INT16 量化参数；模型布局与 FP32 一致。
- `LOSS_REPORT.json`：各子模型和融合模型在分层弱标签测试样本上的量化损耗。
- `QUANT_AUDIT.json`：量化权重误差、INT16 回退名单和文件体积。
- `REPORT.json`：逐模型与融合指标、混淆矩阵、校准指标、数据计数和模型体积。
- `REPORT.md`：可直接阅读的训练摘要。
- `train_manifest.json`：下载来源、版本、SHA256 和数据清洗记录。
- `*_errors.csv`：弱标签测试集中的模型错误。
- `*_disagreement_review_queue.csv`：模型意见分歧较大或融合置信度较低的待人工审核姓名。
- `review_queue.csv`：书写语料抽样审核队列。

Artifact 默认保留 30 天；开启 `publish_release` 后，工作流会发布非 latest 的 `research-v3-<run_id>` Release。

## 本地测试与运行

推荐 Python 3.11。Linux / Termux 可使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu

python -m compileall -q src tests
python -m unittest discover -s tests -v

python src/train_all.py --smoke --output smoke-artifacts \
  --epochs 1 --neural-per-class 30 --max-features 600 \
  --vocab-limit 150 --embedding-dim 16 --cnn-channels 16 \
  --gru-hidden 16 --batch-size 128

python tests/verify_outputs.py smoke-artifacts --allow-demo --require-language --skip-quantized
```

`--smoke` 只验证代码结构和训练流程，合成数据指标不能作为研究结果。

正式训练建议在 GitHub Actions 执行。需要本地运行时：

```bash
python src/train_all.py --output outputs --cache .cache/nszgrm \
  --max-per-class 150000 --neural-per-class 45000 \
  --max-features 60000 --vocab-limit 6000 \
  --embedding-dim 128 --cnn-channels 96 --gru-hidden 128 \
  --epochs 10 --batch-size 256 --wikidata required \
  --wikidata-limit 5000 --min-language 80 --seed 20261117

python src/quantize_existing.py --source outputs/models_fp32.json --out outputs --int16-models language/linear
node tests/quant_loss_gate.js outputs/models_fp32.json outputs/models_int8.json \
  .cache/quant_eval_samples.json outputs/LOSS_REPORT.json
python src/record_quant_report.py outputs
python tests/verify_outputs.py outputs --require-language
```

本地命令展示首轮量化和验收；如果某个模型损耗超过 10%，需要根据 LOSS_REPORT.json 将该模型加入 INT16 回退名单并重新量化。GitHub Actions 已自动执行选择性回退和复测。

## 当前研究限制

1. `P103` 是人物母语标签，不是姓名起源语言。中文标签可能是人物的中文译名；模型学习到的是弱监督关联。
2. 书写语料存在来源差异、非人名混入和现代/古代姓名分布差异。
3. 八分类只覆盖指定的七种外语及中文，不代表所有语言；阿拉伯语、印度语言等不在本轮模型范围内。
4. 如果没有足量人工核验的独立测试数据，不能宣称模型已达到真实世界可靠性。
5. 该版本不生成 HTML。量化模型由训练工作流生成；浏览器界面仍由独立前端工程开发。

### v3.1.1 Wikidata 下载修复

先前德国德语 Q188 的 SPARQL JSON 响应含未正确转义的控制字符，普通 `json.loads` 会在固定位置反复报 `Invalid control character`。新版仅在该特定解析错误下允许 JSON 字符串中存在原始控制字符，其余 JSON 结构仍严格验证；后续汉字姓名过滤会自动排除含控制字符的名字。对仍无法正确解析的响应，第二轮后自动改用 SPARQL CSV 格式，保留 aria2 和 urllib 两种下载方式。

每次 Actions 运行后保存已经完整验证的语言缓存，后续运行尝试恢复。缓存损坏会被丢弃，不会以不完整语料训练模型。八语言强制齐全的规则、FP32 上限 200 MB、量化模型小于 15 MB、Accuracy 和 Macro-F1 相对损耗不超过 10% 的验收标准均不改变。
