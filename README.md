# 9. deepresearch 设计

# 引言

> 笔者以为, Deepresearch 是人工智能领域模仿人思维的一个尝试方向, 即模拟了人在面对一个问题时, 因为无法给出一个理想的答案而寻求其他信息的帮助, 通过大模型与搜索引擎的配合获取到更合适的答案。这样的搜索方案适合制作调研报告，带领大家获取到更有价值到答案。这个概念的提出也在悄悄改变互联网的格局，也为 rag 的优化打开了新局面。
---
| 目录 |
| ---- |
| [一、DeepResearch 初窥](#一deepresearch-初窥) |
| [二、deepresearch 发展脉络](#二deepresearch-发展脉络) |
| [三、N8n 实现 Deepresearch](#三n8n-实现-deepresearch) |
| [四、深入设计](#四深入设计) |
| [五、参考文献](#五参考文献) |
---

## 一、DeepResearch 初窥
```
这个技术的出名好像离我们并不遥远（本文定稿于25年10月）...
```
![](static/performance.png)

*图 1：通义DeepResearch模型表现（图源:通义实验室Tongyi lab）*

九月末通义实验室开发的DeepResearch在多项测试中表现着实优异：

> [!TIP]
> “为了让 AI 真正具备“做研究”的能力，我们针对通义 DeepResearch 的数据、Agent范式、训练、基础设施（Infra）、Test Time Scaling 进行了系统性创新。所有技术方案均已开源，欢迎开发者共建。”[<sup>[1]</sup>](#ref-1)
>
>"阿里通义实验室悄悄（其实动静不小）发布了一个叫 Tongyi DeepResearch 的 Agent 项目。它没有开发布会，没请明星站台，甚至没发通稿——但它在 GitHub 上架当天，就登顶了“每日趋势榜”。这速度，比人类发现“咖啡因无效”后换第三杯咖啡还快。\
>"圈内讨论随之而来。有人兴奋地复制粘贴，有人皱眉点开论文附录，还有人默默关掉网页——因为文档里出现了“后训练”、“工具调用”、“强化学习”这些词。它们像宇宙射线一样，精准穿透了部分读者的知识防护层。"[<sup>[2]</sup>](#ref-2)



## 二、deepresearch 发展脉络

| 目录 |
| ---- |
| [2.1 对于 Deepresearch 框架等索引有哪些？](#21-对于-deepresearch-框架等索引有哪些) |
| [2.2 Deepresearch 的三个发展时期](#22-deepresearch-的三个发展时期) |
| [2.3 Deepresearch 的层次化技术框架](#23-deepresearch-的层次化技术框架) |
| [2.4 Deepresearch 的实现架构](#24-deepresearch-的实现架构) |
| [2.5 Deepresearch 存在的几个问题](#25-deepresearch-存在的几个问题) |
| [2.6 Deepresearch 值得做的几个研究方向](#26-deepresearch-值得做的几个研究方向) |
| [2.7 deepresearch 的发展方向](#27-deepresearch-的发展方向) |

下面借用 @老刘说NLP 的文章~

核心内容来自综述：《**A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications**》[<sup>[3]</sup>](#ref-3)，分析了自 2023 年以来涌现的 80 多个商业和非商业实现，包括 OpenAI、Gemini、Perplexity以及众多开源替代方案，如 dzhng、HKUDS等开源实现，**是个很不错的读物**。

对于这份技术总结中，**选择 5 个核心问题来看**。

### **2.1 对于 Deepresearch 框架等索引有哪些？**

相关的索引列表在：[https://github.com/scienceaix/deepresearch](https://github.com/scienceaix/deepresearch)，对 Projects 做了很好的整理，如下：

![](static/OoHcbWIPWo7Dn3x3TZMcZuW4nwh.png)
*网页部分截图*

### **2.2 Deepresearch 的三个发展时期**

![](static/RDCAbZrxeoI01GxxuC0cVnvJn7b.png)

**原初期探索期**（2023 年-2025 年 2 月），以 n8n、QwenLM/QwenAgent 等工作流自动化框架为代表。这一时期是DeepResearch概念的孕育和尝试阶段。一些早期工作将自动化研究工作流的想法付诸实践，或基于开源大模型推出Agent框架。这些探索
  **验证了“让 AI 去做研究”的可行性**
  ，但当时尚未形成统一的框架与成熟产品。

**竞争角逐期**（2025 年 2 月-3 月），DeepSeek-R1 开源以及 2025 年 2 月 OpenAI 发布了 DeepResearch 为标志。短短一个月内，DeepResearch 从概念走向实用，
  **多方竞相发布产品**
  ，标志着这一领域进入白热化竞争。

**扩展整合**期（2025 年 3 月至今），以如 Jina-AI/node-DeepResearch、Manus、AutoGLM-Research、MGX、DevinAnthropic 于 2025 年 4 月推出的 Claude/Research 为代表。整体上，这一阶段特征是
  **功能整合和生态扩张：**
  不同模式和优势开始在新系统中兼容并存，DeepResearch 系统被应用到更广泛的场景中。

### **2.3 Deepresearch 的层次化技术框架**

尽管不同 DeepResearch 实现各有侧重，但从技术上看普遍包含层次分明的四个核心模块 ￼：基座模型与推理引擎、工具使用与环境交互、任务规划与执行控制、以及知识综合与输出生成。每一层各司其职，共同支撑起深度研究的完整流程：
![](static/NofobSleaoNIlXxaGx5cuU3bnMf.png)

### **2.4 Deepresearch 的实现架构**

包括四种基础架构模式：单体式、基于流水线的、多智能体以及混合实现。

整体大的逻辑在：
![](static/Abshb4Di0oMC2wxdMlHcWrTvnAf.png)

细分的逻辑可以拆解下：

---
- **单体模式** 将所有深度研究能力整合在一个以核心推理引擎为中心的统一的架构框架内，采用集中控制机制，直接集成专用模块。这种架构采用集中式控制，组件间紧密耦合，共享统一的内存和上下文。优点是推理过程一致、整体实现简单，适合小规模快速部署。但缺点是扩展性和并行能力受限，难以灵活替换模块。
![](static/YZ8WbivleoiXo4xuNNhc4DosnAg.png)

- **流水线模式** 通过一系列通过明确定义的接口连接的专业处理阶段实现深度研究能力，将研究工作流分解为离散的处理组件，并在各阶段之间进行明确的数据转换。例如可以依次设置 “意图理解 → 查询扩展 → 信息抓取 → 观点抽取 → 证据核查 → 报告整合”等模块，每一步由独立组件完成[<sup>[4]</sup>](#ref-4)。流水线架构的优点是模块解耦、接口标准化，各阶段易于替换或改进，便于监控和调试，特别适合需要定制工作流的企业应用。但是由于严格的顺序流程，遇到复杂推理任务时可能不如其他架构灵活。

- **多智能体模式** 通过由明确通信协议协调的专门自主智能体生态系统，在具有不同角色和责任的协作智能体之间分配研究功能。

- **混合模式** 结合多种架构模式，以统一实现中平衡各自的优势。
![](static/Dhpdbf0i5oYmkAx4VZscbh5MnrX.png)
---

### **2.5 Deepresearch 存在的几个问题**

当前 DeepResearch 面临诸多挑战，主要可归纳为以下四大类：

- 信息完整性与幻觉（Information Accuracy and Hallucination）
- 隐私与数据安全（Privacy and Data Security）
- 来源归属与知识产权（Source Attribution and IP Issues）
- 可访问性与数字鸿沟（Accessibility and Digital Divide）

![](static/Je7NbaaLMoAvsFxdnDScObIRnZg.png)

### **2.6 Deepresearch 值得做的几个研究方向**

未来 DeepResearch 的发展可围绕以下四大方向展开（见图 1），旨在提升系统的推理深度、信息多样性、领域适应性和人机协作能力：

- 高级推理架构（Advanced Reasoning Architectures）
- 多模态深度研究（Multi-Modal Deep Research）
- 领域特化优化（Domain-Specific Optimization）
- 人机协作标准（Human-AI Collaboration Standards）

![](static/Zyk5bXqXBoKDK2xryZlcnOFUnie.png)
*图 ：DeepResearch 未来四大研究方向示意（图源:@老刘说NLP）*

### **2.7 deepresearch 的发展方向**

从上面的发展脉络我们可以看到，deepresearch 在不断地进化。从「单体实现」到「规划任务多智能体混合实现」。那么这里给大家提个小问题，「为什么大家要挖空心思做 deepresearch 呢？」

我们首先知道 deepresearch 的出现是在两种基础技术之上，搜索引擎和大模型。

**搜索引擎的缺点是「整理的内容过于碎片化」，对问题不会理解导致获取的内容和问题不匹配。**

**大模型的缺点是由于训练导致的「存量语料数据永远与真实资料有时间差」，而且因为大语言模型是概率模型导致在回答问题时会有幻觉。**

deepresearch 的出现一定程度上解决了两个技术的痛点，并为我们带来了一个更新的技术。

那么下一步我们需要一个更强更好的 deepresearch 产品~未来可以卷的方向从目前的产品也能初见端倪。

- **可能是卷表现形式，让结果更适合大家接受。**比如秘塔处理完搜索后会给出报告，大家都很喜欢这样的结果呈现形式。

![](static/UPGPbfwIQoCRQOxG1fXcK5ICnHc.png)

- **也有可能是卷策略的，让「deepresearch」更「deep」一些**，比如这篇文章《[用子模优化法为 DeepResearch 生成多样性查询](https://mp.weixin.qq.com/s/ZB4Rc9GErYsx13-UqDjU3A)》。

![](static/HJbObcBdro5M2lxq7XHcoYKlnxd.png)

- **最后一种是卷垂直领域，垂直领域优势也许会被放大。**领域的特定工具需要定制化开发，也许是个不错的方向~

![](static/VFApb9Eabob5eox9ND1cRtgXn7b.png)

## 三、N8n 实现 Deepresearch

| 目录 |
| ---- |
| [3.1 deepresearch 结构拆解](#31-deepresearch-结构拆解) |
| [3.2 DeepResearch 工作流相关铺垫 <-- 如何配置环境](#32-deepresearch-工作流相关铺垫) |
| [3.3 理顺 DeepResearch 流程 <-- 工作流节点详解](#33-理顺-deepresearch-流程) |
| [3.4 深度研究效果 <-- 深度研究,效果如何](#34-深度研究效果) |

### **3.1 deepresearch 结构拆解**

这里我们使用 N8n 搭建一个简易版的 deepresearch，以便大家学习理解，快速掌握~当然后面还有个复杂版的也给大家简单介绍一下。

首先明确一下输入与输出，我们做个简易版的 deepresearch 需要定义一个深度查询的轮数，接着我们需要让他明确深度查询的主题。这样我们需要输入{轮数、查询主题}。输出内容就是我们的 deepresearch 调研结果。返回内容即可。

过程设计。我们输入查询主题后，大模型处理我们需要查询的内容，交给搜索引擎查询，然后判断是否完成的任务。如果完成返回调研结果即可，如果没有完成继续重复执行处理与查询的任务。

大概的流程图如下：
![](static/1.png)

> 有同学会问为啥要设计一个轮数判断，这个为啥设计在这里。如果第一次我们执行完就查到了结果。第二次大家并不希望流程跑完，所以尽可能把这个判断往前放。这样不会浪费查询资源，尽可能短的流程满足需求。

### **3.2 DeepResearch 工作流相关铺垫**

- **引入工作流**

  大家从github的仓库中下载data/示例工作流.json, 然后打开 n8n 的工作区, 点击右上角的省略号”...“, 点击「import from file」, 导入刚刚下载 json 文件即可见到上图工作流界面. 

- **工作流预设**

  - 研究深度设置

    首先很好理解的是: 「研究中期」中的「判断」节点是循环迭代的重要环节, 点开「判断」节点，会看到「判断」节点有两个判断条件:

        {{$json.shouldContinue}} is equal to false
        {{$json.time}} is equal to 3

    第一个判断条件中的「shouldContinue」变量为「研究中期」的「优化回答和逻辑规划」Agent节点给出, 由LLM自行判断答案是否完美呢? 是否要继续研究呢? 回答true or false. 如果回答 false 则迈向「研究后期」.\
    第二个判断条件则**可以由用户决定**, 即为研究深度, 你可以改变目标值来决定研究的最大循环次数, 次数越多, 研究越深入, 同时运行时间也会相应变长, token 使用也相应变多. 目标值默认为3. 
    ![](static/2.png)
  - Model 设置
  
    注意到在「Model」部分，选用了DeepSeek Model节点作为Agent的模型. 你可以点开小鲸鱼, 按照 「Credential to connect with >> Create new credential >> API-key」的顺序, 最后输入个人API即可. 
    
    API获取: [https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
  - Tavily 设置

    Tavily Search 插件用于查询互联网结果，只需要输入查询内容即可返回结果。这个插件使用之前需要到 [https://www.tavily.com/](https://www.tavily.com/)进行注册。每个月赠送 1000 次查询机会。这里我们导入子查询主题「nextSearch」就会执行查询获取结果。

    http设置已经完成啦, 你所需要做的就是给「Header Auth >> Create new credential」补充个人API:

          {
            Name:Authorization,
            Value:<your_Tavily_API_Key>
          }
    ![](static/3.png)

- **模型记忆**

  模型记忆对于深度研究模型是很重要的, 因为在深度研究中我们会搜集不少参考数据, 可以想想自己平常深入思考某一个问题, 常常就是信息的交汇与灵感的迸发, 记忆的唤醒, 理固宜然. 模型的记忆由于其特殊的上下文存在形式, 可以减少一定的token损耗, 省下一点米~(bushi

  在工作流的右下方, 我们会看到Memory部分. 这是n8n自带的记忆内存节点, 不需要API接入, 应该足够我们使用~



### **3.3 理顺 DeepResearch 流程**

OK呀,现在你的环境已经成功配置好了, 你完完全全可以把任何问题抛给对话框, 会得到一个不错的结果.

那么这个「深度研究工作流」是怎样运行的呢?

### 3.3.1 研究初期
![](static/4.png)
- **对话框输入**

  这个节点是触发器, 或者也可以理解为开关, 当你在左下「对话框」中发送prompt时, 开关就已按下, 这个节点接收prompt的内容传递给下一个节点. 

- **初步回答和继续规划**

  这个节点是 Agent 节点，我们将 prompt 内容传递给大模型，大模型会给出初步的回答。

  提示词如下:
  ~~~json
  您是一个深度研究代理，现在是研究初期, 刚遇到用户的问题, 请您先给出初步解答和接下来应该研究的具体方面.
  ## 输出
  - 必须进一步搜索信息，请务必设置nextSearchTopic
  - 请以**json**格式输出如下例:\
  {result: str | None\
  nextSearchTopic: str | None}
  ~~~

  - result: 第一次思考的”结晶“
  - nextSearchTopic: 接下来应该研究的具体方面Topic
  
  
- **提取1**

  这个节点主要是对于「初步回答和继续规划」输出的切割, javascript代码如下:
  ~~~javascript
    // 清洗输入
  let raw = $input.first().json.output;
  if (typeof raw === 'string') {
    raw = raw.match(/\{[\s\S]*\}/);
  }

  // 解析为对象
  const data = JSON.parse(raw);

  // 提取数据
  const nextSearchTopic = data.nextSearchTopic;
  const result = data.result;

  // 返回 n8n 需要的格式
  return [
    {
      json: {
        nextSearchTopic,
        result
      }
    }
  ];
  ~~~

  首先利用「正则表达式」匹配提取有效字段, 然后分别提取出 nextSearchTopic 和 result 字段。

### 3.3.2 研究中期
![](static/5.png)
「研究中期」主要是循环体:

-[-「Tavily」-「优化回答和逻辑规划」-「提取2」 - 判断 -]$_n$-

*上式略似有机高分子的表示式(doge), 不过其中的 n 为上述的研究深度/最大循环次数.*

- **Tavily**

  这个节点是查询节点，我们将 nextSearchTopic 内容传递给 Tavily 进行查询。

- **优化回答和逻辑规划**

  这个节点是 Agent 节点，我们将收集到的内容传递给LLM, LLM进行再加工. 

  提示词如下:
  ~~~json
  您是一个深度研究代理, 

  根据您的主动研究, 您查询到了
  {{ $json.results }}

  您发现了什么？还有哪些问题尚未解答？下一步应该调查哪些具体方面？

  ## 输出
  - 请不要输出与已搜索主题完全相同的主题
  - 如果需要进一步搜索信息，请设置nextSearchTopic
  - 如果已获得足够信息，请将shouldContinue设置为false
  - 请以json格式输出, 如下例:

  {result: str | None
  nextSearchTopic: str | None
  shouldContinue: bool }
  ~~~
  - {{$json.results}} 为「tavily」节点传入的检索内容.
  - result: 这个变量在期待一个更好的研究成果.
  - nextSearchTopic: 如果需要进一步搜索信息，LLM会给出希望的子查询任务.
  - shouldContinue: 这个变量决定是否继续查询, 当为 false 时, 则表示已经获得足够的信息.

- **提取2**

  这个节点主要是对于「优化回答和逻辑规划」输出的切割, javascript代码如下:
  ~~~javascript
  // 清洗输入
  let raw = $input.first().json.output;
  if (typeof raw === 'string') {
    raw = raw.match(/\{[\s\S]*\}/);
  }
  const data = JSON.parse(raw);

  // 提取数据
  const result = data.result;
  const nextSearchTopic = data.nextSearchTopic;
  const shouldContinue = data.shouldContinue;
  let prevTime = 0;
  try {prevTime = $('判断').first().json.time ?? 0} catch (e) {prevTime = 0}
  const time = prevTime + 1;

  // 返回 n8n 需要的格式
  return [
    {
      json: {
        result,
        nextSearchTopic,
        shouldContinue,
        time
      }
    }
  ];
  ~~~
  - time 为当前循环次数计数器, 每进行完成一轮循环, 该变量会计数加一, 直到加到设定的最大循环次数,跳出循环.

- **判断**

  前文有大致讲过该节点的作用, 这里就不再赘述.

### 3.3.3 研究后期

![](static/5.png)

「研究后期」主要是「研究总结」Agent节点对于整个思考过程进行总结归纳, 最后给出回答于左下聊天框.

  提示词如下:
  ~~~json
  对以下的调查结果总结

  - 重要的洞察、结论和剩余的不确定性。
  - 适当引用来源。
  - 该总结应非常全面和详细，预计为长文。
  - 请以json格式输出, 如下例:

  {result: str | None}
  ~~~
LLM 通过读取记忆, 给出总结性回答, 正如我们人一样, 在思考的最后, 试图给出一个让人信服的严谨的回答.

## **3.4 深度研究效果**

笔者主要从检索资源利用率角度上分析深度研究的效果, 基于以下假设:

>收集到的信息越多(这个主要由 研究深度 和 Agent限制长度 控制), 利用率越高, 思考就越全面, 结论就越可靠.

测试如下:

Question
~~~
热木星是什么
~~~
Answer
~~~
重要洞察与结论：

基本特征：热木星是质量与木星相当或更大（通常为0.3-10个木星质量），但轨道周期极短（通常1-10天）的系外气态巨行星。它们与宿主恒星的距离通常小于0.1天文单位，导致表面温度极高（可达700-3000°C）[1]。

大气特性：由于极端接近恒星，热木星的大气层经历强烈的恒星辐射，导致大气膨胀和显著的质量流失。许多热木星显示出温度反转现象，即高层大气温度高于低层[2]。

轨道迁移机制：热木星的存在挑战了经典行星形成理论，因为气态巨行星理论上应在远离恒星的寒冷区域形成。目前主流的解释包括：

盘迁移理论：行星在原行星盘内通过与气体盘的角动量交换向内迁移[3]
高偏心率迁移：通过行星-行星散射或科兹伊机制获得高偏心率轨道，随后由潮汐作用圆化[4]
观测特征：热木星主要通过凌星法和径向速度法探测。它们通常表现出：

较大的行星半径（由于热膨胀）
同步自转（潮汐锁定导致永久昼侧和夜侧）
强烈的昼夜温度差异
剩余不确定性：

形成路径的争议：虽然迁移机制被广泛接受，但具体哪种迁移路径占主导仍存在争议。不同热木星可能通过不同路径形成[5]。

大气逃逸率：热木星大气逃逸的确切速率和长期演化影响仍不确定，这关系到行星的最终命运和大气组成演化[6]。

金属丰度异常：许多热木星宿主恒星显示出较高的金属丰度，这种关联的确切原因尚不完全清楚[7]。

缺失的种群：观测发现中等轨道距离（0.1-1AU）的巨行星数量较少，这一"热木星沙漠"现象的成因仍需进一步研究[8]。

参考文献： [1] Winn & Fabrycky (2015) ARA&A [2] Fortney et al. (2008) ApJ [3] Lin et al. (1996) Nature [4] Rasio & Ford (1996) Science [5] Dawson & Johnson (2018) ARA&A [6] Vidal-Madjar et al. (2003) Nature [7] Fischer & Valenti (2005) ApJ [8] Mazeh et al. (2016) A&A Reviews
~~~

回溯工作流记录, 可以看到工作流一共执行三次, 检索主题有:
  - 热木星的形成机制与迁移理论
  - 热木星的大气组成和物理特性
  - 热木星的大气逃逸机制和观测特征

我们可以从以下三个角度分析:

 - 第一, 参考文献一共摘引了8/15, 体现深度研究模型在信息取舍上并没有一味取用, 能够适当征引.
 - 第二, 三个主题在结论中均有体现, 体现深度研究模型在迭代中发挥了记忆功能, 提高了利用率.
 - 第三, 三个主题环环相扣, 在遍历完第一个主题下检索到的内容后, 模型能敏锐把握与大气组成的关系, 进而联想到逃逸机制, 三个主题的考验显得非常精彩.

 最后, 这样的一次研究, 总共**耗时1m.272s**, **耗2075 tokens**, 确实非常实用()

## 四、深入设计

在探索应用中还有一个更难的实现在循环过程中外层是反思循环，内层是执行循环，智能体在执行循环中针对特定子主题开展研究工作。在执行循环中，通过调用搜索/访问/思考工具的函数来获取答案。

这个过程与人类研究中的 “反思” 阶段非常相似 —— 思考 “我已经知道什么”、“我还需要知道什么” 以及 “我接下来应该查询什么”。整个系统的创新之处就在于这种迭代方法：收集信息；分析现有信息与原始问题之间的 “差距”；生成新的查询以填补这些差距；重复这个过程，直到差距被填补。

这部分作为探索内容留给大家尝试，可能比较难如果做不下来也没事。主要帮大家开拓眼界~

## 五、参考文献
<span id="ref-1"></span>[1] 通义实验室, 不止SOTA！通义 DeepResearch模型、框架、方案全开源, https://mp.weixin.qq.com/s/23b-aWTArhATJRupaTYC8A, 20250917 \
<span id="ref-2"></span>[2] 罗智凌, Tongyi DeepResearch的技术报告探秘, https://mp.weixin.qq.com/s/9MfLzDimxLphCdfuK7KWrA?scene=1&click_id=8, 20250929 \
<span id="ref-3"></span>[3]Xu R, Peng J. A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications[J]. arXiv preprint arXiv:2506.12594, 2025. \
<span id="ref-4"></span>[4] PaperAgent, 一篇95页最新80种Deep Research系统全面综述, https://developer.volcengine.com/articles/7519779273176318015, 20250625
