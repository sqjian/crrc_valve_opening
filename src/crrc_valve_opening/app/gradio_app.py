from pathlib import Path

import gradio as gr
import numpy as np
import polars as pl


def load_data(file_path: str, preview_rows: int = 10) -> tuple[pl.DataFrame | None, pl.DataFrame | None, str]:
    """
    加载 CSV 数据文件

    Args:
        file_path: CSV 文件路径
        preview_rows: 预览的行数

    Returns:
        (完整DataFrame, 预览DataFrame, 状态信息字符串)
    """
    try:
        df = pl.read_csv(file_path)
        preview_df = df.head(preview_rows)
        status_info = f"✅ 数据加载成功! 共 {len(df)} 行, {len(df.columns)} 列"
        return df, preview_df, status_info
    except Exception as e:
        return None, None, f"❌ 数据加载失败: {str(e)}"


def predict(df: pl.DataFrame) -> pl.DataFrame:
    """
    使用 mock 模块模拟预测逻辑

    Args:
        df: 输入的 DataFrame

    Returns:
        包含预测结果 y1-y5 的 DataFrame
    """
    # 为每一行生成模拟预测值
    num_rows = len(df)

    # 基于 x1 列生成带有一定规律的预测值
    if "x1" in df.columns:
        base_values = df["x1"].to_numpy()
    else:
        base_values = np.random.random(num_rows)

    # 生成 y1-y5 预测列,添加不同的变化模式
    y1 = base_values * 100 + np.random.normal(0, 5, num_rows)  # 线性趋势
    y2 = base_values * 80 + np.sin(np.arange(num_rows) * 0.5) * 10  # 正弦波动
    y3 = base_values * 120 + np.random.normal(0, 8, num_rows)  # 更大波动
    y4 = base_values * 90 + np.cos(np.arange(num_rows) * 0.3) * 15  # 余弦波动
    y5 = base_values * 110 + (np.arange(num_rows) * 0.1)  # 递增趋势

    # 将预测结果添加到 DataFrame
    result_df = df.clone()
    result_df = result_df.with_columns(
        [pl.Series("y1", y1), pl.Series("y2", y2), pl.Series("y3", y3), pl.Series("y4", y4), pl.Series("y5", y5)]
    )

    return result_df


def create_original_plots(df: pl.DataFrame):
    """
    创建原始数据的曲线图 (x1, x2, x3 vs 时间)

    Args:
        df: 原始 DataFrame

    Returns:
        三个 Plotly 图表对象的元组 (plot_x1, plot_x2, plot_x3)
    """
    import plotly.graph_objects as go

    # 检查是否包含时间列
    if "ts" not in df.columns:
        return None, None, None

    # 创建 x1 曲线图
    plot_x1 = None
    if "x1" in df.columns:
        plot_x1 = go.Figure()
        plot_x1.add_trace(
            go.Scatter(
                x=df["ts"].to_list(),
                y=df["x1"].to_list(),
                mode="lines+markers",
                name="x1",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )
        plot_x1.update_layout(
            height=300,
            title_text="x1 随时间变化",
            xaxis_title="时间 (ts)",
            yaxis_title="x1",
            showlegend=False,
        )
        plot_x1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        plot_x1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # 创建 x2 曲线图
    plot_x2 = None
    if "x2" in df.columns:
        plot_x2 = go.Figure()
        plot_x2.add_trace(
            go.Scatter(
                x=df["ts"].to_list(),
                y=df["x2"].to_list(),
                mode="lines+markers",
                name="x2",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=4),
            )
        )
        plot_x2.update_layout(
            height=300,
            title_text="x2 随时间变化",
            xaxis_title="时间 (ts)",
            yaxis_title="x2",
            showlegend=False,
        )
        plot_x2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        plot_x2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # 创建 x3 曲线图
    plot_x3 = None
    if "x3" in df.columns:
        plot_x3 = go.Figure()
        plot_x3.add_trace(
            go.Scatter(
                x=df["ts"].to_list(),
                y=df["x3"].to_list(),
                mode="lines+markers",
                name="x3",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=4),
            )
        )
        plot_x3.update_layout(
            height=300,
            title_text="x3 随时间变化",
            xaxis_title="时间 (ts)",
            yaxis_title="x3",
            showlegend=False,
        )
        plot_x3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        plot_x3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return plot_x1, plot_x2, plot_x3


def create_plot(df: pl.DataFrame):
    """
    创建预测结果的动态曲线图

    Args:
        df: 包含预测结果的 DataFrame

    Returns:
        Plotly 图表对象
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 检查是否包含时间列和预测列
    if "ts" not in df.columns or "y1" not in df.columns:
        return None

    # 创建子图
    fig = make_subplots(rows=1, cols=1, subplot_titles=["预测结果随时间变化趋势"])

    # 添加 y1-y5 的曲线
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    y_cols = ["y1", "y2", "y3", "y4", "y5"]

    for i, (y_col, color) in enumerate(zip(y_cols, colors)):
        if y_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["ts"].to_list(),
                    y=df[y_col].to_list(),
                    mode="lines+markers",
                    name=y_col,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )

    # 更新布局
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode="x unified",
        title_text="预测值随时间变化",
        xaxis_title="时间 (ts)",
        yaxis_title="预测值",
    )

    # 添加动画效果
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def process_data(file_input, preview_rows: int):
    """
    处理数据的主函数:加载、预测、可视化

    Args:
        file_input: 上传的文件对象或None(使用默认数据)
        preview_rows: 预览的行数

    Returns:
        (状态信息, 原始数据预览表格, x1曲线图, x2曲线图, x3曲线图, 预测结果表格, 预测曲线图)
    """
    # 确定数据文件路径
    if file_input is not None:
        # 使用用户上传的文件
        file_path = file_input.name
    else:
        # 使用默认数据文件
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        file_path = str(data_dir / "demo.csv")

    # 1. 加载原始数据
    df_original, preview_df, status_info = load_data(file_path, preview_rows)

    if df_original is None or preview_df is None:
        return status_info, None, None, None, None, None, None

    # 2. 创建原始数据的曲线图
    plot_x1, plot_x2, plot_x3 = create_original_plots(df_original)

    # 3. 执行预测
    df_predicted = predict(df_original)

    # 4. 创建预测结果可视化图表
    plot = create_plot(df_predicted)

    # 5. 转换为 pandas DataFrame 用于 Gradio 显示
    preview_table = preview_df.to_pandas()
    predicted_table = df_predicted.to_pandas()

    return status_info, preview_table, plot_x1, plot_x2, plot_x3, predicted_table, plot


def create_interface():
    """
    创建 Gradio 界面
    """
    with gr.Blocks(title="阀门开度预测系统") as demo:
        gr.Markdown("# 📊 阀门开度预测系统")
        gr.Markdown("阀门开度数据预测与可视化平台")

        with gr.Row():
            with gr.Column(scale=1):
                # 数据上传区
                gr.Markdown("## 1️⃣ 数据上传")
                file_upload = gr.File(
                    label="上传CSV文件 (可选)", file_types=[".csv"], type="filepath", file_count="single"
                )
                gr.Markdown("💡 **提示**: 如果不上传文件,将使用默认数据 (data/step1.csv)")

                # 数据预览设置
                gr.Markdown("## 2️⃣ 数据预览设置")
                preview_rows_slider = gr.Slider(
                    minimum=5, maximum=50, value=10, step=1, label="预览行数", info="选择要预览的数据行数"
                )

                # 处理按钮区
                gr.Markdown("## 3️⃣ 执行预测")
                process_btn = gr.Button("🚀 开始预测", variant="primary", size="lg")

                # 状态信息
                status_info = gr.Textbox(label="状态信息", lines=2, interactive=False)

        # 原始数据预览区
        gr.Markdown("## 4️⃣ 原始数据预览")
        with gr.Row():
            data_preview = gr.Dataframe(label="原始数据预览 (前N行)", interactive=False, wrap=True)

        # 原始数据曲线图
        gr.Markdown("### 原始数据曲线图")
        with gr.Row():
            plot_x1 = gr.Plot(label="x1 随时间变化")
            plot_x2 = gr.Plot(label="x2 随时间变化")
            plot_x3 = gr.Plot(label="x3 随时间变化")

        # 预测结果展示区
        gr.Markdown("## 5️⃣ 预测结果展示")

        with gr.Row():
            # 预测结果表格
            result_table = gr.Dataframe(label="预测结果表格 (包含 y1-y5 预测列)", interactive=False, wrap=True)

        with gr.Row():
            # 预测曲线图
            result_plot = gr.Plot(label="预测趋势图")

        # 绑定事件
        process_btn.click(
            fn=process_data,
            inputs=[file_upload, preview_rows_slider],
            outputs=[status_info, data_preview, plot_x1, plot_x2, plot_x3, result_table, result_plot],
        )

    return demo


def main():
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
