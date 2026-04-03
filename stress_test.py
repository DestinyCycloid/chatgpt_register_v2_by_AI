#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI兼容API高强度压力测试脚本
支持多线程并发请求、流式响应、高token消耗测试

功能说明:
---------
本脚本用于对OpenAI兼容的大模型API进行高强度压力测试，专门设计用于快速消耗token。
支持多线程并发、流式响应、长文本输入等特性。

配置文件:
---------
需要在项目根目录创建 .env 文件，包含以下配置：
    OPENAI_API_BASE=http://127.0.0.1:8317/v1     # API地址
    OPENAI_API_KEY=test                          # API密钥
    MODEL_NAME=gpt-5.2-codex                     # 模型名称

安装依赖:
---------
pip install openai python-dotenv

使用方法:
---------
1. 基础测试（10线程，默认消息）:
   python stress_test.py
   python stress_test.py -t 10

2. 高强度测试（50线程 + 超长文本，每次请求约12000+ tokens）:
   python stress_test.py -t 50 --long-text
   python stress_test.py -t 50 --long-text --max-tokens 8000

3. 超高强度测试（100线程 + 长文本 + 流式响应）:
   python stress_test.py -t 100 --long-text --stream

4. 持续测试模式（不停发送请求，直到手动Ctrl+C停止）:
   python stress_test.py -t 20 --continuous
   python stress_test.py -t 50 --long-text --continuous --stream

5. 静默模式（适合高并发，只显示最终统计）:
   python stress_test.py -t 200 --long-text --quiet
   python stress_test.py -t 100 --continuous --quiet

6. 自定义消息测试:
   python stress_test.py -t 20 -m "请写一篇5000字的文章"
   python stress_test.py -t 30 -m "详细解释量子计算" --max-tokens 5000

7. 组合使用（推荐用于快速消耗token）:
   python stress_test.py -t 100 --long-text --stream --continuous
   python stress_test.py -t 50 --long-text --max-tokens 8000 --continuous

参数说明:
---------
-t, --threads      : 并发线程数，默认10
                     建议值：轻度测试10-20，中度测试50-100，重度测试100-500
                     
-m, --message      : 发送的测试消息（自定义问题）
                     如果不指定，默认使用中等长度的消息
                     
--long-text        : 使用超长文本消息（约8000+ tokens输入）
                     配合--max-tokens 4000，每次请求可消耗12000+ tokens
                     推荐用于快速消耗token
                     
--stream           : 使用流式响应（SSE）
                     优点：提高并发效率，减少等待时间
                     适合持续测试模式
                     
--continuous       : 持续测试模式（不停发送请求）
                     线程池会持续工作，直到按Ctrl+C停止
                     适合长时间压力测试和token消耗
                     
--max-tokens       : 最大生成token数，默认4000
                     可设置更高值（如8000）来增加每次请求的token消耗
                     注意：某些模型有上限限制
                     
--quiet            : 静默模式，只显示最终统计信息
                     适合高并发测试，减少输出开销
                     推荐在100+线程时使用

输出说明:
---------
正常模式会实时显示：
- 每个请求的状态（✓成功 / ✗失败）
- 每个请求的耗时和token消耗
- 实时统计：总请求数、成功率、token消耗速率

最终统计包括：
- 总耗时、总请求数、成功率
- Token统计：总消耗、输入tokens、输出tokens
- 性能指标：token消耗速率（tokens/秒）、请求速率（请求/秒）
- 平均响应时间、平均每请求token数

使用场景:
---------
1. 快速消耗token（推荐配置）:
   python stress_test.py -t 100 --long-text --stream --continuous
   
2. 测试API并发能力:
   python stress_test.py -t 50 --stream
   
3. 测试API稳定性（长时间运行）:
   python stress_test.py -t 20 --continuous
   
4. 批量测试（一次性发送N个请求）:
   python stress_test.py -t 100 --long-text

注意事项:
---------
- 高线程数会对服务器造成较大压力，请谨慎使用
- 持续测试模式需要手动Ctrl+C停止
- 建议先小规模测试（10-20线程），确认API稳定后再增加并发
- 使用--long-text + 高--max-tokens可以最大化token消耗
- 流式响应（--stream）可以提高并发效率，但token统计是估算值
- 静默模式（--quiet）适合高并发测试，减少终端输出开销

性能参考:
---------
在典型配置下（假设API响应时间2-5秒）：
- 10线程：约 2-5 请求/秒，24,000-60,000 tokens/分钟
- 50线程 + --long-text：约 10-25 请求/秒，120,000-300,000 tokens/分钟
- 100线程 + --long-text：约 20-50 请求/秒，240,000-600,000 tokens/分钟
- 持续模式可以维持稳定的高token消耗速率

实际性能取决于：API响应速度、网络延迟、服务器负载、线程数等因素
"""

import argparse
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 统计数据
stats = {
    'success': 0,
    'failed': 0,
    'total_time': 0.0,
    'total_tokens': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_requests': 0
}
stats_lock = Lock()
stop_event = Event()

# 超长文本模板（用于高token消耗测试，约8000+ tokens）
LONG_TEXT_TEMPLATE = """
请详细分析以下主题，并提供深入的见解和例子。这是一个非常重要的研究项目，需要你提供最详细、最全面的分析报告。

主题：人工智能在现代社会中的全面应用与深远影响

第一部分：医疗健康领域的AI革命

人工智能在医疗健康领域的应用正在引发一场前所未有的革命。从疾病诊断到药物研发，从个性化治疗到医疗资源优化，AI技术正在深刻改变着医疗行业的方方面面。

在疾病诊断方面，深度学习算法已经在多个领域展现出超越人类专家的诊断能力。例如，在医学影像分析领域，卷积神经网络可以从X光片、CT扫描、MRI图像中识别出微小的病变，其准确率在某些特定任务上已经超过了经验丰富的放射科医生。斯坦福大学的研究团队开发的皮肤癌检测算法，通过分析皮肤病变的照片，能够以与皮肤科医生相当的准确率判断是否为恶性肿瘤。在眼科领域，Google的DeepMind开发的AI系统可以通过分析视网膜扫描图像，诊断出50多种眼部疾病，包括糖尿病视网膜病变、青光眼、年龄相关性黄斑变性等，其诊断准确率达到94%以上。

在药物研发领域，AI技术正在显著缩短新药开发周期并降低研发成本。传统的新药研发通常需要10-15年时间，耗资数十亿美元，而AI可以通过虚拟筛选、分子设计、临床试验优化等方式大幅提高效率。DeepMind的AlphaFold系统在蛋白质结构预测方面取得了突破性进展，能够准确预测蛋白质的三维结构，这对于理解疾病机制和设计靶向药物具有重要意义。在新冠疫情期间，AI技术被广泛应用于疫苗研发、药物筛选、疫情预测等方面，展现了其在应对公共卫生危机中的巨大潜力。

个性化医疗是AI在医疗领域的另一个重要应用方向。通过分析患者的基因组数据、病史、生活方式等信息，AI系统可以为每个患者制定个性化的治疗方案。IBM Watson for Oncology系统可以分析患者的医疗记录和最新的医学文献，为癌症患者推荐最适合的治疗方案。在精准医疗领域，AI可以根据患者的基因型预测其对特定药物的反应，从而避免无效治疗和药物副作用。

远程医疗和AI的结合正在改善医疗资源分配不均的问题。在医疗资源匮乏的偏远地区，AI辅助诊断系统可以帮助基层医生提高诊断水平，减少误诊和漏诊。智能问诊系统可以通过自然语言处理技术，与患者进行交互，收集症状信息，初步判断病情，并给出就医建议。这不仅可以提高医疗服务的可及性，还能减轻医生的工作负担。

然而，医疗AI的发展也面临着诸多挑战。数据隐私是一个重要问题，医疗数据涉及患者的敏感信息，如何在保护隐私的前提下进行数据共享和模型训练是一个亟待解决的问题。医疗责任认定也是一个法律难题，当AI系统做出错误诊断或治疗建议时，责任应该由谁来承担？算法的可解释性也是一个关键问题，医生和患者需要理解AI系统是如何得出诊断结论的，而不是简单地接受一个"黑箱"的判断。

第二部分：教育领域的智能化转型

人工智能正在深刻改变教育的方式和内容。个性化学习是AI在教育领域最重要的应用之一。传统的"一刀切"教学模式无法满足每个学生的个性化需求，而AI可以根据学生的学习进度、理解能力、兴趣爱好等因素，为每个学生定制专属的学习路径和内容。

自适应学习平台利用机器学习算法，实时分析学生的学习行为和表现，动态调整教学内容的难度和呈现方式。例如，当系统检测到学生在某个知识点上遇到困难时，会自动提供更多的练习题和讲解材料；当学生掌握得很好时，会加快学习进度，避免重复学习已经掌握的内容。这种个性化的学习方式可以显著提高学习效率和学习效果。

智能教学系统可以扮演虚拟教师的角色，为学生提供24小时的学习支持。这些系统可以回答学生的问题，解释复杂的概念，提供学习建议，甚至可以进行苏格拉底式的对话，引导学生深入思考。Carnegie Learning开发的数学教学系统，通过认知模型和机器学习技术，可以像人类教师一样理解学生的思维过程，并提供针对性的指导。

AI在自动批改作业和考试方面也展现出巨大潜力。对于选择题、填空题等客观题，AI可以实现即时批改和反馈。对于作文等主观题，自然语言处理技术可以分析文章的结构、语法、逻辑、创意等多个维度，给出评分和改进建议。虽然AI在主观题批改方面还无法完全替代人类教师，但可以作为辅助工具，帮助教师减轻工作负担。

在线教育平台利用AI技术提供更好的学习体验。推荐算法可以根据学生的学习历史和兴趣，推荐合适的课程和学习资源。学习分析技术可以追踪学生的学习进度，识别学习困难，预测学习结果，帮助教师和学生及时调整学习策略。

然而，AI教育也带来了一些担忧。教育公平性是一个重要问题，先进的AI教育工具往往价格昂贵，可能会加剧教育资源的不平等。师生关系的变化也值得关注，过度依赖AI可能会削弱师生之间的情感联系和人文关怀。创造力的培养也是一个挑战，AI擅长处理结构化的知识和技能，但在培养学生的创造力、批判性思维、情感智能等方面，人类教师仍然不可替代。

第三部分：金融科技中的AI应用

金融行业是AI技术应用最广泛、最深入的领域之一。从风险管理到投资决策，从客户服务到监管合规，AI正在重塑金融服务的各个环节。

在风险评估和信用评分方面，机器学习算法可以分析海量的数据，包括传统的财务数据和非传统的行为数据，更准确地评估借款人的信用风险。蚂蚁金服的芝麻信用利用大数据和AI技术，综合考虑用户的身份特质、履约能力、信用历史、人脉关系、行为偏好等多个维度，计算出信用分数。这种新型的信用评估方式可以为传统金融机构难以覆盖的人群提供金融服务，促进普惠金融的发展。

欺诈检测是AI在金融领域的另一个重要应用。传统的基于规则的欺诈检测系统往往存在高误报率和低检出率的问题，而机器学习算法可以从历史交易数据中学习欺诈模式，实时识别异常交易。深度学习模型可以检测出复杂的欺诈行为，如账户盗用、洗钱、内幕交易等。PayPal使用机器学习技术，每天分析数十亿笔交易，欺诈率降低到了0.32%以下。

算法交易和量化投资是AI在投资领域的典型应用。高频交易系统利用AI算法，在毫秒级别内分析市场数据，执行交易决策。量化对冲基金使用机器学习模型，从海量的市场数据中挖掘投资机会，构建投资组合。Renaissance Technologies等顶级量化基金，通过AI技术实现了远超市场平均水平的投资回报。

智能投顾（Robo-advisor）为普通投资者提供了低成本、专业化的理财服务。这些系统通过问卷调查了解投资者的风险偏好、投资目标、财务状况等信息，然后利用现代投资组合理论和机器学习算法，为投资者构建和管理投资组合。Betterment、Wealthfront等智能投顾平台，管理着数百亿美元的资产。

在客户服务方面，AI聊天机器人可以处理大量的客户咨询，提供24小时服务，显著降低人工成本。自然语言处理技术使得聊天机器人能够理解客户的问题，提供准确的答案，甚至可以处理复杂的业务流程，如开户、转账、理赔等。

监管科技（RegTech）利用AI技术帮助金融机构满足日益严格的监管要求。反洗钱系统使用机器学习算法，从大量交易数据中识别可疑的洗钱行为。合规监控系统可以自动审查交易记录、通信记录等，确保符合监管规定。摩根大通开发的COiN（Contract Intelligence）系统，利用机器学习技术，可以在几秒钟内完成原本需要律师花费36万小时才能完成的合同审查工作。

然而，金融AI也面临着挑战。算法黑箱问题使得监管机构和投资者难以理解AI系统的决策逻辑，这可能导致信任危机。系统性风险也是一个担忧，如果大量金融机构使用相似的AI算法，可能会导致市场行为的同质化，增加系统性风险。算法偏见可能导致歧视性的金融服务，例如，如果训练数据中存在种族或性别偏见，AI系统可能会对某些群体做出不公平的信用评估。

第四部分：交通运输领域的智能化革命

自动驾驶技术是AI在交通领域最引人注目的应用。自动驾驶汽车利用计算机视觉、传感器融合、深度学习等技术，可以感知周围环境，做出驾驶决策，控制车辆行驶。根据SAE（美国汽车工程师学会）的分类，自动驾驶分为L0到L5六个级别，目前大多数量产车型处于L2级别（部分自动化），而L4级别（高度自动化）的自动驾驶出租车已经在部分城市开始试运营。

Waymo是自动驾驶领域的领先者，其自动驾驶出租车已经在美国多个城市提供商业服务，累计行驶里程超过2000万英里。特斯拉的Autopilot和FSD（Full Self-Driving）系统，虽然目前仍需要驾驶员监督，但已经可以在高速公路和城市道路上实现自动驾驶。中国的百度Apollo、小马智行等公司也在自动驾驶领域取得了显著进展。

自动驾驶技术面临着诸多挑战。技术挑战包括复杂场景的感知和理解、长尾问题的处理、恶劣天气条件下的可靠性等。法律问题包括责任认定、保险制度、道路法规等。伦理困境包括"电车难题"等道德选择问题，当事故不可避免时，自动驾驶系统应该如何做出选择？

智能交通系统利用AI技术优化交通流量，减少拥堵。智能信号灯系统可以根据实时交通流量动态调整信号灯时长，提高道路通行效率。交通预测系统可以预测未来的交通状况，帮助驾驶员选择最优路线。新加坡、巴塞罗那等城市已经部署了智能交通系统，显著改善了交通状况。

在物流配送领域，AI技术被广泛应用于路径规划、仓储管理、需求预测等方面。亚马逊的仓库使用了大量的机器人，通过AI算法协调机器人的工作，提高了仓储效率。顺丰、京东等物流公司使用AI技术优化配送路线，减少配送时间和成本。无人配送车和无人机配送也在逐步推广。

共享出行平台如滴滴、Uber等，利用AI技术进行供需匹配和动态定价。机器学习算法可以预测不同时间、不同地点的出行需求，提前调配车辆。动态定价算法根据供需关系实时调整价格，平衡供需，提高平台效率。

第五部分：制造业的智能化转型

工业4.0时代，AI成为智能制造的核心驱动力。预测性维护利用机器学习算法，分析设备运行数据，预测设备故障，提前进行维护，避免意外停机。通用电气的Predix平台，为工业设备提供预测性维护服务，帮助客户减少停机时间，降低维护成本。

质量检测是AI在制造业的重要应用。计算机视觉技术可以检测产品的外观缺陷，如划痕、裂纹、色差等，其检测速度和准确率远超人工检测。深度学习模型可以识别复杂的缺陷模式，适应不同的产品和缺陷类型。

供应链优化利用AI技术进行需求预测、库存管理、物流规划等。机器学习算法可以分析历史销售数据、市场趋势、季节因素等，准确预测未来需求，帮助企业优化库存水平，减少库存成本和缺货风险。

协作机器人（Cobot）可以与人类工人协同工作，提高生产效率和灵活性。与传统的工业机器人不同，协作机器人配备了传感器和AI算法，可以感知周围环境，避免与人类发生碰撞，安全地与人类共享工作空间。

数字孪生技术结合AI，可以创建物理系统的虚拟副本，进行仿真和优化。西门子的数字化工厂利用数字孪生技术，在虚拟环境中测试和优化生产流程，然后应用到实际生产中，显著提高了生产效率和产品质量。

智能制造对就业结构产生了深远影响。一方面，自动化和AI技术替代了大量的重复性、低技能工作；另一方面，也创造了新的工作岗位，如数据分析师、AI工程师、机器人维护工程师等。工人需要不断学习新技能，适应智能制造时代的要求。

请基于以上内容，继续深入分析AI伦理、全球AI战略、技术发展趋势、人机协作模式等方面，并提供详细的案例研究和数据支持。每个部分都要详细展开，提供具体的例子、数据、分析和见解。总字数要求不少于10000字，确保内容充实、论据充分、分析透彻。
"""


def send_request(client, model, message, thread_id, stream=False, max_tokens=4000):
    """
    发送单个请求到API
    
    Args:
        client: OpenAI客户端实例
        model: 模型名称
        message: 要发送的消息
        thread_id: 线程ID
        stream: 是否使用流式响应
        max_tokens: 最大生成token数
    
    Returns:
        tuple: (是否成功, 响应时间, 响应内容或错误信息, token统计)
    """
    start_time = time.time()
    token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    try:
        if stream:
            # 流式响应
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                stream=True,
                max_tokens=max_tokens
            )
            
            content = ""
            for chunk in response:
                if stop_event.is_set():
                    break
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            elapsed_time = time.time() - start_time
            # 流式响应通常不返回token统计，需要估算
            token_info['prompt_tokens'] = len(message) // 4  # 粗略估算
            token_info['completion_tokens'] = len(content) // 4
            token_info['total_tokens'] = token_info['prompt_tokens'] + token_info['completion_tokens']
            
        else:
            # 非流式响应
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens
            )
            elapsed_time = time.time() - start_time
            
            content = response.choices[0].message.content
            
            # 获取token统计
            if hasattr(response, 'usage') and response.usage:
                token_info['prompt_tokens'] = response.usage.prompt_tokens
                token_info['completion_tokens'] = response.usage.completion_tokens
                token_info['total_tokens'] = response.usage.total_tokens
            else:
                # 如果没有usage信息，估算token数
                token_info['prompt_tokens'] = len(message) // 4
                token_info['completion_tokens'] = len(content) // 4
                token_info['total_tokens'] = token_info['prompt_tokens'] + token_info['completion_tokens']
        
        return True, elapsed_time, content, token_info
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, elapsed_time, str(e), token_info


def update_stats(success, elapsed_time, token_info):
    """更新统计数据"""
    with stats_lock:
        stats['total_requests'] += 1
        if success:
            stats['success'] += 1
            stats['total_tokens'] += token_info['total_tokens']
            stats['prompt_tokens'] += token_info['prompt_tokens']
            stats['completion_tokens'] += token_info['completion_tokens']
        else:
            stats['failed'] += 1
        stats['total_time'] += elapsed_time

def print_stats(start_time, quiet=False):
    """打印实时统计信息"""
    if quiet:
        return
    
    with stats_lock:
        elapsed = time.time() - start_time
        if elapsed > 0:
            tokens_per_sec = stats['total_tokens'] / elapsed
            requests_per_sec = stats['total_requests'] / elapsed
            
            print(f"\r实时统计 | 请求: {stats['total_requests']} | 成功: {stats['success']} | "
                  f"失败: {stats['failed']} | Tokens: {stats['total_tokens']:,} | "
                  f"速率: {tokens_per_sec:.1f} tokens/s | {requests_per_sec:.2f} req/s", 
                  end='', flush=True)


def run_stress_test(threads, message, stream=False, continuous=False, max_tokens=4000, quiet=False):
    """
    运行压力测试
    
    Args:
        threads: 线程数量
        message: 要发送的消息
        stream: 是否使用流式响应
        continuous: 是否持续测试
        max_tokens: 最大生成token数
        quiet: 是否静默模式
    """
    # 从环境变量读取配置
    api_base = os.getenv('OPENAI_API_BASE')
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('MODEL_NAME')
    
    if not all([api_base, api_key, model]):
        print("错误: 请在.env文件中配置OPENAI_API_BASE, OPENAI_API_KEY和MODEL_NAME")
        return
    
    if not quiet:
        print(f"=== OpenAI兼容API高强度压力测试 ===")
        print(f"API地址: {api_base}")
        print(f"模型: {model}")
        print(f"线程数: {threads}")
        print(f"流式响应: {'是' if stream else '否'}")
        print(f"持续测试: {'是' if continuous else '否'}")
        print(f"最大tokens: {max_tokens}")
        print(f"消息长度: {len(message)} 字符 (约 {len(message)//4} tokens)")
        print(f"{'='*80}\n")
    
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    # 开始测试
    start_time = time.time()
    
    try:
        if continuous:
            # 持续测试模式
            if not quiet:
                print(f"持续测试模式启动，按 Ctrl+C 停止...\n")
            
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = set()
                request_id = 0
                
                while not stop_event.is_set():
                    # 保持线程池满载
                    while len(futures) < threads and not stop_event.is_set():
                        request_id += 1
                        future = executor.submit(send_request, client, model, message, request_id, stream, max_tokens)
                        futures.add(future)
                    
                    # 处理完成的任务
                    done_futures = {f for f in futures if f.done()}
                    for future in done_futures:
                        try:
                            success, elapsed_time, result, token_info = future.result()
                            update_stats(success, elapsed_time, token_info)
                            
                            if not quiet:
                                status = "✓" if success else "✗"
                                print(f"\r[{status}] 请求 #{stats['total_requests']} | "
                                      f"耗时: {elapsed_time:.2f}s | Tokens: {token_info['total_tokens']} | "
                                      f"总计: {stats['total_tokens']:,} tokens", end='')
                        except Exception as e:
                            if not quiet:
                                print(f"\n处理结果时出错: {e}")
                        
                        futures.remove(future)
                    
                    time.sleep(0.01)  # 短暂休眠避免CPU占用过高
        
        else:
            # 单次批量测试
            with ThreadPoolExecutor(max_workers=threads) as executor:
                future_to_thread = {
                    executor.submit(send_request, client, model, message, i, stream, max_tokens): i + 1
                    for i in range(threads)
                }
                
                if not quiet:
                    print(f"所有 {threads} 个线程已同时启动，等待响应...\n")
                    print(f"{'='*80}")
                
                # 实时处理完成的任务
                for future in as_completed(future_to_thread):
                    thread_num = future_to_thread[future]
                    success, elapsed_time, result, token_info = future.result()
                    update_stats(success, elapsed_time, token_info)
                    
                    if not quiet:
                        status = "✓ 成功" if success else "✗ 失败"
                        print(f"\n[线程 {thread_num}] {status} | 耗时: {elapsed_time:.2f}秒 | "
                              f"Tokens: {token_info['total_tokens']} "
                              f"(输入: {token_info['prompt_tokens']}, 输出: {token_info['completion_tokens']})")
                        
                        if success and len(result) < 500:
                            print(f"响应: {result[:200]}...")
                        elif not success:
                            print(f"错误: {result}")
                        
                        print(f"{'-'*80}")
                        print_stats(start_time, quiet)
    
    except KeyboardInterrupt:
        if not quiet:
            print("\n\n收到停止信号，正在结束测试...")
        stop_event.set()
    
    total_elapsed = time.time() - start_time
    
    # 输出最终统计结果
    print(f"\n\n{'='*80}")
    print(f"=== 测试完成 ===")
    print(f"总耗时: {total_elapsed:.2f}秒")
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功请求: {stats['success']}")
    print(f"失败请求: {stats['failed']}")
    if stats['total_requests'] > 0:
        print(f"成功率: {stats['success']/stats['total_requests']*100:.1f}%")
    print(f"\n--- Token统计 ---")
    print(f"总Token消耗: {stats['total_tokens']:,}")
    print(f"输入Tokens: {stats['prompt_tokens']:,}")
    print(f"输出Tokens: {stats['completion_tokens']:,}")
    if total_elapsed > 0:
        print(f"\n--- 性能指标 ---")
        print(f"Token消耗速率: {stats['total_tokens']/total_elapsed:.1f} tokens/秒")
        print(f"请求速率: {stats['total_requests']/total_elapsed:.2f} 请求/秒")
    if stats['success'] > 0:
        print(f"平均响应时间: {stats['total_time']/stats['success']:.2f}秒")
        print(f"平均每请求Token: {stats['total_tokens']/stats['success']:.0f}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='OpenAI兼容API高强度压力测试工具 - 快速消耗Token',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础测试
  python stress_test.py -t 10
  
  # 高强度测试（50线程 + 长文本）
  python stress_test.py -t 50 --long-text
  
  # 流式响应测试
  python stress_test.py -t 20 --stream
  
  # 持续测试模式（不停发送请求）
  python stress_test.py -t 10 --continuous
  
  # 静默模式（只显示最终统计）
  python stress_test.py -t 100 --long-text --quiet
        """
    )
    
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=10,
        help='并发线程数 (默认: 10)'
    )
    
    parser.add_argument(
        '-m', '--message',
        type=str,
        default=None,
        help='发送的测试消息（默认: 短消息）'
    )
    
    parser.add_argument(
        '--long-text',
        action='store_true',
        help='使用超长文本消息（约8000+ tokens输入，适合快速消耗token）'
    )
    
    parser.add_argument(
        '--stream',
        action='store_true',
        help='使用流式响应（提高并发效率）'
    )
    
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='持续测试模式（不停发送请求，按Ctrl+C停止）'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4000,
        help='最大生成token数 (默认: 4000)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，只显示最终统计信息'
    )
    
    args = parser.parse_args()
    
    # 确定使用的消息
    if args.message:
        message = args.message
    elif args.long_text:
        message = LONG_TEXT_TEMPLATE
    else:
        message = "请详细介绍一下你自己，包括你的能力、特点、应用场景等，尽可能详细地展开说明。"
    
    # 运行压力测试
    run_stress_test(
        threads=args.threads,
        message=message,
        stream=args.stream,
        continuous=args.continuous,
        max_tokens=args.max_tokens,
        quiet=args.quiet
    )


if __name__ == '__main__':
    main()
