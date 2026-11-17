# NSZGRM · 中文姓名多模型音译研究训练工程（v2）

一套 **Python 监督训练 + GitHub Actions Linux CPU + 完全离线 JavaScript 推理** 的研究工程。训练两个不同任务：中文姓名的书写类型与来源语言线索；每个任务最多训练四种独立架构，逐模型评测，再学习融合权重。所有部署模型均为 **原始 FP32 精度**，所有模型参数包原始文件大小 **严格不超过 200,000,000 字节**。

> **性能约束：** 这是一套可执行、可复现的训练和评估程序，**不是已经证明真实世界可靠的姓名词源识别器**。主训练语料是弱监督标签，Wikidata P103 只是人物母语线索。评估采用外部人工审核样本时才可对真实泛化作更有依据的判断。不要把不同语料库之间的识别准确率冒充任意姓名的真实准确率。

## 模型结构与体积

| 组成 | 架构 | 输入 | 作用 |
|---|---|---|---|
| linear | 字符 1–3 gram TF-IDF + SGD Logistic | 字符 n-gram | 高频/低频字组合、位置模式的统计判别 |
| complement_nb | Complement Naive Bayes | 共享 TF-IDF 字符特征 | 与线性模型有不同的似然归纳偏置 |
| cnn | 原生 PyTorch TextCNN，2/3/4 窗口 | 汉字嵌入序列 | 学习局部名字片段 |
| bigru | 原生 PyTorch 双向 GRU | 汉字嵌入序列 | 学习前后位置的顺序关系 |
| fusion | 验证集独立拟合的非负集成权重 | 四模型概率 | 平均/校准并融合四模型分类得分 |

每个任务都训练和评估四个独立模型；各模型会先在验证集 A 上做温度校准，再在不重叠的验证集 B 上优化加权融合的 log loss。各模型的权重下限是 3%，因此每一个模型都参与最终评分。训练后的 **弱监督测试集不会用于训练或选择融合权重**。

默认神经网络为 embedding=128、CNN channels=96、BiGRU hidden=128、词表 6000；字符模型最多保留 60000 个 n-gram。默认神经网络每类抽样至多 45000 条，最多训练 10 轮，并启用早停。可在工作流输入中调节训练规模和轮数。

输出 `models_fp32.json` 包括所有独立模型的 FP32 参数、TF-IDF 数据、类别顺序、温度系数和融合权重。程序计算文件的**未压缩原始字节数**，不得超过 **200 MB（十进制）**；**不会量化后假装满足上限**。`browser/index.html` 是为双击本地打开而嵌入相同参数的副本，不是额外训练模型；它可能使发布 ZIP 的展开体积增加。

## 一键推送到现有 mifuler/nszgrm

在 Android Termux 使用已经登录、且对目标仓库有写权限的 `iqonli` GitHub CLI：

```sh
pkg install -y gh git rsync unzip
termux-setup-storage
gh api user --jq .login
gh repo view mifuler/nszgrm --json viewerPermission --jq .viewerPermission
```

把本 ZIP 放在手机 `Download`，执行：

```sh
mkdir -p ~/nszgrm-v2
unzip -o ~/storage/downloads/NSZGRM-ResearchTrainer-v2-200MB.zip -d ~/nszgrm-v2
cd ~/nszgrm-v2/NSZGRM-ResearchTrainer
bash push-existing.sh mifuler/nszgrm
```

脚本克隆 `main`，将代码写入独立目录 `nszgrm-research-v2/`，只新增或更新工作流 `.github/workflows/research-train.yml`，**不删除旧版训练器、旧工作流或仓库根目录 LICENSE**。修改提交并推送后显示训练工作流地址。如遇到 403，检查 `iqonli` 在目标仓库的写权限、GitHub CLI 登录凭据，以及 Token 是否允许修改 `.github/workflows`。

如果已经手工推送了工程，也可以直接在 GitHub 网页进入：

<https://github.com/mifuler/nszgrm/actions/workflows/research-train.yml>

选择 **Run workflow → main → Run workflow**。工作流只使用标准 `ubuntu-24.04` CPU；**不假装存在免费的 GitHub GPU**。内含标准测试、合成语料端到端烟雾测试、自动下载、真正训练、模型体积验证、离线 JS 验证、Artifact 和 Release。

工作流默认在训练完成且所有检查成功后，创建非 latest 的 Release：`research-v2-<run_id>`，包含 `outputs/` 的 ZIP。`outputs/REPORT.md` 是人工阅读报告；`REPORT.json` 包括每个独立模型的 Accuracy、Balanced Accuracy、Macro-F1、交叉熵、逐类精度/召回率/F1、混淆矩阵、期望校准误差、融合权重和大小。`*_disagreement_review_queue.csv` 抽出模型分歧明显的姓名，供人工进一步审核。

## 数据获取与标签含义

默认从 [wainshine/Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus) 下载三套主数据和古代中文姓名补充语料。源代码先以浅克隆 Git 仓库下载；失败才尝试 GitHub codeload tarball。数据只在 Actions 环境训练，不随训练包重新分发。执行后保存 Git commit / 文件 SHA256 / 清洗计数到 `train_manifest.json`。汇编语料上游许可可能不统一，需要按应用场景再审查。

- `zh_native`：中文姓名来源语料中的记录（**不是个人国籍或词源的真值**）。
- `foreign_phonetic_proxy`：外文翻译名词典中的记录，其中可能混有地名；不是全部英语名。
- `ja_kanji_proxy`：日本姓名词典中的汉字写法；**不是音译名**。
- `en / ru / ja / fr / es / de / ko`：Wikidata `P103` 人物母语标签；**不等于姓名起源语言**，仅能作为初步来源线索。

按姓名汉字字符串划分弱监督训练/验证/测试，删除跨类别完全同形词，Wikidata 按人物 ID 和相同汉字写法的联通分组进行划分。**仍可能有同一人的其他中文译名变体泄漏**。必须用外部有证据的、按人物或原文姓名分组的人工审核样本进行额外验证。

若 Wikidata 暂时无法访问，`best_effort` 模式会继续完成主书写分类模型，并在报告中明确标注多语言模型不可用，不会伪造七语言模型或指标。`required` 模式严格要求每个查询成功；`off` 可以关闭来源语言分支。

## 独立人工金标准集

`data/gold_reviewed.csv` 提供空白模板，列为：

```csv
name,writing_type,origin_lang,group_id,reference_url,reviewer,split
```

必须有中文姓名、书写类型、标识同一人物或同源译名的 `group_id`、可查证的 `reference_url`、审核者标识及 `train/val/test` 划分。同一人物的各名称变体不能跨越划分。仅包含标题行时**不会生成虚假的人工测试指标**。

加入带有 `split=test` 的人工审核行后，程序会排除这批精确同名的弱监督记录，并单独计算 `human_review_metrics`；如果人工数据不足或来源无法核实，结果仅供探索。不同组的同源别名仍须由人工统一 `group_id` 并人工排查。

## 本地跑通 / 开发测试

推荐 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python src/train_all.py --smoke --output smoke-artifacts \
  --epochs 1 --neural-per-class 30 --max-features 600 \
  --vocab-limit 150 --embedding-dim 16 --cnn-channels 16 \
  --gru-hidden 16 --batch-size 128
node tests/verify_browser.js smoke-artifacts/models_fp32.json
```

`--smoke` 只是内部合成数据检查，严禁引用其准确率作为科研指标。没有 Node 环境时，训练仍可执行，离线网页的额外数学检查需要安装 Node.js。

正式训练：

```bash
python src/train_all.py --output outputs --cache .cache/nszgrm \
  --max-per-class 150000 --neural-per-class 45000 \
  --max-features 60000 --vocab-limit 6000 \
  --embedding-dim 128 --cnn-channels 96 --gru-hidden 128 \
  --epochs 10 --batch-size 256 --wikidata best_effort \
  --wikidata-limit 5000 --min-language 80 --seed 20261117
```

建议在 GitHub Actions 中执行正式训练。CPU 训练时间取决于语料下载和实例性能，可能比旧版线性模型慢得多，GitHub 托管 Job 有执行时长限制；运行时间或内存不满足时，应降低训练样本数和 epoch 或改用符合预算的 GPU 环境。当前程序未使用预训练 MacBERT，也没有训练 Transformer；不能把这些尚未实施的模型写成已完成。

## 完全本地部署

解压 Release，直接打开 `outputs/browser/index.html`；无需联网，无需 CDN、Python 服务或后端。支持单条姓名、四模型独立得分、融合指数、各类别概率、语言线索和 TXT/CSV 批量输入导出。支持 2–8 个连续汉字，复杂外文全名需拆分。输出的音译指数是来源语料类的融合概率型分数，**尚未经代表性真实人群概率校准，不代表真实世界概率**。

## 交付界限

此工程已经通过合成语料端到端训练及浏览器推理一致性检查。**正式大型公开语料训练需要在你的 Actions 上运行，实际模型的 200 MB 体积、真实性能、七语言覆盖情况均由那次运行产生的报告确认。** 禁止用合成样本的高分替代正式测试结论。
