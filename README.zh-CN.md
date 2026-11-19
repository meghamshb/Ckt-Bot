**禁止分发：HKU ECE 知识产权**

**禁止擅自复制、泄露、重新上传或出售。使用本项目拥有权利的材料须获得书面授权，并受下述权利和例外规定约束。**

**版权 2026。本项目材料的合法权利人保留其享有的全部权利。**

**本仓库包含保密材料，仅供获授权的内部研究和审阅使用。未向公众授予使用、复制、修改、披露、分发、发布、再许可或商业化本项目拥有权利的材料的许可。上述行为须事先取得相关权利人的书面同意。**

**未经授权复制、披露或分发本项目拥有权利的材料，可能导致法律行动，包括民事诉讼、禁令、损害赔偿及诉讼费用。**

**更改项目名称、删除本声明，或在网上找到本项目的副本，并不代表获得使用许可。复制或分享本项目拥有权利的材料前，须取得书面授权，但适用许可或法律允许的情况除外。**

**HKU ECE 是项目所有者指定的归属标示。实际权利归属及授权权限仍须依照适用法律、香港大学知识产权政策及相关协议确定。本声明不构成权利转让。**

**第三方组件、许可及声明继续受其原有条款约束。本声明不主张拥有这些组件的权利，也不限制其许可已授予的权利。法定例外及不可放弃的权利不受影响。**

**民事权利救济：[香港《版权条例》（第 528 章），第 107 至 108 条](https://www.elegislation.gov.hk/hk/cap528)。**

**国际版权框架：[《伯尔尼公约》第 2、5 条](https://www.wipo.int/wipolex/en/text/283698) · [《与贸易有关的知识产权协定》（TRIPS）第 9 至 10、41 至 45 条](https://www.wto.org/english/docs_e/legal_e/trips_e.htm) · [《世界知识产权组织版权条约》第 4、8、14 条](https://www.wipo.int/wipolex/en/text/295166)。权利救济须依照适用的本地法律进行。**

**[完整许可条款](LICENSE) · [版权指引](https://www.ipd.gov.hk/en/copyright/what-is-copyright/index.html) · [香港大学知识产权政策](https://www.rss.hku.hk/contracts/ipr)**

---

[English](README.md) · [廣東話（繁體）](README.yue-HK.md) · **普通话（简体）**

<h1 align="center">K-Bot</h1>
<p align="center">上传电路图，输入问题，查看解题过程。</p>
<p align="center"><a href="run.md">安装指南</a> · <a href="#pipeline">处理流程</a> · <a href="https://huggingface.co/Meghamsh111/KBOT-models">模型</a></p>

K-Bot 读取电路图并给出带有解题步骤的答案。网页会展示各个处理阶段，包括识别元件、追踪导线、分析连接、计算及解释结果。

<a name="pipeline"></a>

## 架构

![K-Bot 当前架构](docs/visuals/architecture.png)

## 工作流程

以下图片使用已完成的 Sadiku 图 10.52 运行结果。这是一个已知示例。

### 识别元件

**YOLO26n** 识别元件，并用边界框标出其位置。

![原始电路图、实际 YOLO 检测结果及用于读取元件的局部图片](docs/visuals/components.png)

### 追踪导线并连接元件

**SAM2.1** 提供线条掩膜。**连接分析程序**追踪元件引脚之间的连接。

![处理后的导线掩膜、连接点及最终电气节点映射](docs/visuals/topology.png)

### 读取、计算并说明

**视觉语言模型**读取数值和标签。问题理解模型确定需要计算的量。**SymPy** 计算答案，**ngspice** 核对计算结果，解释模型给出解题步骤。

![实际 JSON 结构定义及电路输出节选](docs/visuals/json-output.png)

![初始连接、生成的 SPICE 网表及经过核对的答案](docs/visuals/analysis.png)

这些图片使用保存的运行数据，通过 [HTML 和 CSS](docs/visuals/index.html) 渲染生成。源代码及渲染脚本位于同一文件夹。

## 运行 K-Bot

按照 **[run.md](run.md)** 安装软件、下载模型、登录并启动网页。每位团队成员都需要自己的托管模型登录账号及访问权限。

模型存放在 [Hugging Face](https://huggingface.co/Meghamsh111/KBOT-models)。未包含在 GitHub 仓库中的文件见 [run.md](run.md#what-is-included)。Docker 仍处于实验阶段，尚未验证。

## 当前状态

在原有 Mac 的一个新环境中，通过浏览器完成一个已知交流电路的处理，用时 **55.8 秒**。答案与独立计算一致。详见[复现记录](provenance/HANDOFF_REPRODUCTION_20261119.json)。

模糊的导线、交叉点及错误读取的标签仍可能导致电路识别错误。一个示例成功并不能证明对新电路的准确率。目前尚未在第二台实体电脑上测试。
