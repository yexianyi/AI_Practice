import pandas as pd
import plotly.graph_objects as go

# 读取数据（假设数据已按期号升序排列）
data = pd.read_csv("daletou.csv")

# 定义颜色和标签
colors = ['red', 'blue', 'green', 'orange', 'purple']
labels = ['R1', 'R2', 'R3', 'R4', 'R5']

# 创建画布
fig = go.Figure()

# 对每个位置（R1-R5）分别添加轨迹（交换X/Y轴）
for i in range(5):
    ball_column = f'R{i+1}'
    fig.add_trace(
        go.Scatter(
            x=data[ball_column],  # X轴：球号（1-35）
            y=data['No'],         # Y轴：期号（1, 2, ..., N）
            mode='lines+markers',
            name=labels[i],
            line=dict(color=colors[i]),
            marker=dict(size=6)
        )
    )

# 更新布局
fig.update_layout(
    title='大乐透历史中奖号码趋势（X轴：球号 1-35，Y轴：期号）',
    xaxis_title='红色球号码',
    yaxis_title='期号',
    xaxis=dict(range=[1, 35]),  # 固定X轴范围（球号）
    hovermode='closest',        # 悬停时显示最近的数据点
    template='plotly_white'
)

# 显示图表
fig.show()