import json
from typing import List, Dict, Any

# 导入 TinkerPop 核心库
from gremlin_python.structure.graph import Graph
from gremlin_python.process.traversal import T
# 导入 anonymous_traversal 模块，用于获取 __ 对象作为 T_ANON，避免 IDE 报错
from gremlin_python.process.anonymous_traversal import traversal as anonymous_traversal_source


# --- 1. TinkerGraph 客户端：负责硬编码加载和查询执行 ---

class TinkerGraphClient:
    """
    使用 TinkerGraph (内存图数据库)，通过硬编码 Gremlin 遍历来加载知识。
    """

    def __init__(self):
        print("🔧 初始化：TinkerGraph 内存知识图谱客户端就绪。")
        self.graph = Graph()
        self.g = self.graph.traversal()
        self._load_knowledge()

    def _load_knowledge(self):
        """
        硬编码加载知识图谱结构。
        使用最稳定的模式：先添加所有节点和属性（用 toList() 提交），再查询节点并添加边。
        """
        print("   -> 🔨 正在加载硬编码的工业知识图谱...")

        # 定义一个字典，用于存储我们创建的节点引用 (通过 name 属性)
        node_map = {}

        # --- 1. 添加所有节点和属性 (使用 toList() 提交，避免 iterate() 错误) ---

        # 节点数据列表：[标签, 唯一的名称属性, 其他属性字典]
        node_data = [
            # Device
            ('Device', 'CNC-M500', {'location': 'Line-A', 'model': 'M500-Pro'}),
            ('Device', 'Robot-KUKA-KR40', {'location': 'Line-B', 'model': 'KR40-V2'}),
            # Component
            ('Component', 'Spindle-Unit-001', {'lifespan': '5000h', 'supplier': 'Siemens'}),
            ('Component', 'Bearing-Group-A', {'material': 'Steel'}),
            ('Component', 'Motor-Unit-A', {'power': '15kW'}),
            # FaultMode
            ('FaultMode', 'Overheating', {'severity': 'High', 'description': 'Temperature exceeds safety threshold.'}),
            ('FaultMode', 'AbnormalVibration',
             {'severity': 'Medium', 'description': 'Vibration values exceed ISO standard.'}),
            ('FaultMode', 'PowerLoss', {'severity': 'High', 'description': 'Motor fails to achieve rated power.'}),
            # RepairAction
            ('RepairAction', 'CheckCoolantLevel', {'estimated_time': '30min'}),
            ('RepairAction', 'ReplaceBearing', {'estimated_time': '2h', 'required_tool': 'Puller-T80'}),
            ('RepairAction', 'InspectWiring', {'estimated_time': '1h'}),
        ]

        for label, name, properties in node_data:
            # 创建节点，并设置名称属性
            t = self.g.addV(label).property('name', name)
            # 添加其他属性
            for key, value in properties.items():
                t = t.property(key, value)

            # **核心修改：使用 toList() 提交更改**
            t.toList()

            # 将节点名称添加到映射列表，以便下一步查询
            node_map[name] = None

            # --- 2. 重新查询节点对象用于添加边 ---

        # 批量查询所有节点并存储在 node_map 中
        for name in node_map.keys():
            # 通过唯一的 'name' 属性查询节点对象，并使用 .next() 获取单个结果
            node_map[name] = self.g.V().has('name', name).next()

        # --- 3. 边创建 (使用查询到的对象) ---

        # 定义边数据：[源节点名称, 关系类型, 目标节点名称]
        edge_data = [
            ('CNC-M500', 'HAS_COMPONENT', 'Spindle-Unit-001'),
            ('CNC-M500', 'HAS_COMPONENT', 'Bearing-Group-A'),
            ('Robot-KUKA-KR40', 'HAS_COMPONENT', 'Motor-Unit-A'),

            ('Spindle-Unit-001', 'CAN_CAUSE', 'Overheating'),
            ('Bearing-Group-A', 'CAN_CAUSE', 'AbnormalVibration'),
            ('Motor-Unit-A', 'CAN_CAUSE', 'PowerLoss'),

            ('Overheating', 'CAN_BE_SOLVED_BY', 'CheckCoolantLevel'),
            ('AbnormalVibration', 'CAN_BE_SOLVED_BY', 'ReplaceBearing'),
            ('PowerLoss', 'CAN_BE_SOLVED_BY', 'InspectWiring'),
        ]

        for source_name, label, target_name in edge_data:
            source_node = node_map[source_name]
            target_node = node_map[target_name]
            # **核心修改：使用 toList() 提交更改**
            self.g.V(source_node).addE(label).to(target_node).toList()

        print("   ✅ 知识图谱硬编码加载完成。")

    def run_query(self, gremlin_query_string: str) -> List[Dict[str, Any]]:
        """
        接收 LLM 生成的 Gremlin 语句字符串，并直接在 TinkerGraph 上执行。
        """
        print(f"   -> ⚙️ 正在执行 Gremlin 遍历...")

        try:
            # 1. 核心步骤：获取 __ 匿名遍历对象
            T_ANON = anonymous_traversal_source().__

            # 2. 替换字符串：将 LLM 输出的 '__.' 替换为我们定义的 'T_ANON.'
            executed_query_string = gremlin_query_string.replace('__.', 'T_ANON.')

            # 3. 创建 eval 的本地命名空间。
            eval_scope = {
                'g': self.g,
                'T': T,
                'T_ANON': T_ANON  # 注入 T_ANON 供 eval 使用
            }

            # 使用 eval 执行 Gremlin 遍历
            results = eval(executed_query_string, globals(), eval_scope)

            # 清理 valueMap() 返回的列表包装的值
            cleaned_results = []
            for res in results:
                cleaned_res = {}
                for key, value in res.items():
                    # 结果清理：将列表包装的值（如 ['5000h']）解包为 '5000h'
                    cleaned_res[key] = value[0] if isinstance(value, list) else value
                cleaned_results.append(cleaned_res)

            return cleaned_results

        except Exception as e:
            print(f"   ❌ Gremlin 执行失败: {e}")
            return []


# --- 2. 模拟 LLM：负责语义解析 (保持不变) ---

def llm_natural_language_to_gremlin(question: str) -> str:
    """LLM 直接输出可执行的 Gremlin Python 语句字符串。"""
    print("   -> 🧠 LLM 正在进行语义解析并生成 Gremlin 遍历...")

    if "CNC-M500" in question and ("Spindle" in question or "主轴" in question) and (
            "寿命" in question or "lifespan" in question):
        return """g.V().has('Device', 'name', 'CNC-M500').out('HAS_COMPONENT').has('name', 'Spindle-Unit-001').valueMap('name', 'lifespan', 'supplier').toList()"""

    elif ("过热" in question or "温度太高" in question) and ("维修操作" in question or "怎么修" in question):
        return """g.V().has('FaultMode', 'name', 'Overheating').in_('CAN_BE_SOLVED_BY').valueMap('name', 'estimated_time').toList()"""

    elif ("轴承异响" in question or "振动异常" in question):
        return """g.V().has('FaultMode', 'name', 'AbnormalVibration').in_('CAN_BE_SOLVED_BY').valueMap('name', 'estimated_time', 'required_tool').toList()"""

    elif "Puller-T80" in question or "特定工具" in question:
        return """g.V().has('RepairAction', 'required_tool', 'Puller-T80').in_('CAN_BE_SOLVED_BY').out('CAN_CAUSE').values('name').toList()"""

    return ""


# --- 3. LLM 润色和主流程 (保持不变) ---

def llm_refine_answer(question: str, kg_results: List[Dict[str, Any]]) -> str:
    # ... (保持不变) ...
    print("   -> 💡 LLM 正在润色最终答案...")

    if not kg_results:
        return f"抱歉，知识图谱中没有找到针对 '{question}' 的精确结构化信息。"

    if "寿命" in question or "lifespan" in question:
        res = kg_results[0]
        comp_name = res.get('name', '组件')
        lifespan = res.get('lifespan', '未知')
        supplier = res.get('supplier', '未知')
        return f"✅ **精确事实：** 根据知识图谱，设备 {comp_name} (供应商: {supplier}) 的推荐寿命为 **{lifespan}**。"

    elif "维修操作" in question or "怎么修" in question or "异响" in question:
        actions = []
        for r in kg_results:
            name = r.get('name', '未知操作')
            time = r.get('estimated_time', '未知时间')
            tool = r.get('required_tool', '无特殊工具')
            actions.append(f"操作名称: **{name}** (耗时: {time}, 所需工具: {tool})")

        return f"🛠️ **诊断建议：** 针对您提出的故障，知识图谱推荐以下维修操作：\n* " + "\n* ".join(actions)

    elif "Puller-T80" in question or "特定工具" in question:
        fault_names = [name for result in kg_results for name in result.values()]
        return f"🔧 **工具查询结果：** 需要使用 **Puller-T80** 的维修动作是解决以下故障：{', '.join(fault_names)}。"

    return f"原始结构化数据：{json.dumps(kg_results, ensure_ascii=False, indent=2)}"


def run_hybrid_query(question: str, kg_client: TinkerGraphClient) -> str:
    print(f"\n====================================================")
    print(f"🔍 用户提问: {question}")
    print(f"====================================================")

    gremlin_query_string = llm_natural_language_to_gremlin(question)

    if gremlin_query_string:
        print(f"✅ 语义解析成功。LLM 生成 Gremlin 语句:\n{gremlin_query_string.strip()}")

        kg_results = kg_client.run_query(gremlin_query_string)

        print(f"📊 知识图谱返回结构化数据: {kg_results}")

        final_answer = llm_refine_answer(question, kg_results)

    else:
        print("⚠️ 无法将问题转化为精确查询。")
        final_answer = "无法进行精确的事实查询。作为通用模型，我建议您立即断电并查阅设备安全手册。"

    return final_answer


# --- 4. 运行示例 ---

if __name__ == "__main__":

    try:
        tinker_graph_client = TinkerGraphClient()
    except Exception as e:
        print("\nFATAL ERROR: 无法初始化 TinkerGraph 客户端。")
        print("最终诊断：您的环境中的 'gremlinpython' 库，在嵌入式模式下，无法正确处理图修改操作（addV, addE）。")
        print("建议：尝试将 'gremlinpython' 降级或升级到稳定版本，例如 3.4.x 或 3.6.x。")
        print(f"原始错误: {e}")
        exit()

    # 运行示例 1: 事实性查询
    question_1 = "CNC-M500 那台机器，主轴单元的推荐寿命是多久？"
    result_1 = run_hybrid_query(question_1, tinker_graph_client)
    print(f"\n--- 最终回答 ---:\n{result_1}")

    # 运行示例 2: 推理查询
    question_2 = "如果传感器反馈主轴温度太高，我应该执行什么维修操作？"
    result_2 = run_hybrid_query(question_2, tinker_graph_client)
    print(f"\n--- 最终回答 ---:\n{result_2}")

    # 运行示例 3: 逆向查询
    question_3 = "哪个故障的维修需要 Puller-T80 这个工具？"
    result_3 = run_hybrid_query(question_3, tinker_graph_client)
    print(f"\n--- 最终回答 ---:\n{result_3}")