from pathlib import Path

import gradio as gr
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
        df = pl.read_csv(file_path, infer_schema_length=10000)
        preview_df = df.head(preview_rows)
        status_info = f"✅ 数据加载成功! 共 {len(df)} 行, {len(df.columns)} 列"
        return df, preview_df, status_info
    except Exception as e:
        return None, None, f"❌ 数据加载失败: {str(e)}"


def load_predicted_data(file_path: str) -> tuple[pl.DataFrame | None, str]:
    """
    加载预测数据文件 (step2.csv)

    Args:
        file_path: CSV 文件路径

    Returns:
        (完整DataFrame, 状态信息字符串)
    """
    try:
        df = pl.read_csv(file_path, infer_schema_length=10000)
        status_info = f"✅ 预测数据加载成功! 共 {len(df)} 行, {len(df.columns)} 列"
        return df, status_info
    except Exception as e:
        return None, f"❌ 预测数据加载失败: {str(e)}"


def create_original_plots(df: pl.DataFrame):
    """
    创建原始数据的曲线图 (海拔、坡度、速度 vs 时间)

    Args:
        df: 原始 DataFrame

    Returns:
        三个 Plotly 图表对象的元组 (plot_altitude, plot_slope, plot_speed)
    """
    import plotly.graph_objects as go

    # 检查是否包含时间列
    if "时间量(s)" not in df.columns:
        return None, None, None

    # 创建海拔曲线图
    plot_altitude = None
    if "海拔(m)" in df.columns:
        plot_altitude = go.Figure()
        plot_altitude.add_trace(
            go.Scatter(
                x=df["时间量(s)"].to_list(),
                y=df["海拔(m)"].to_list(),
                mode="lines+markers",
                name="海拔(m)",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )
        plot_altitude.update_layout(
            height=300,
            title_text="海拔(m) 随时间变化",
            xaxis_title="时间量(s)",
            yaxis_title="海拔(m)",
            showlegend=False,
        )
        plot_altitude.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        plot_altitude.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # 创建坡度曲线图
    plot_slope = None
    if "坡度(‰)" in df.columns:
        plot_slope = go.Figure()
        plot_slope.add_trace(
            go.Scatter(
                x=df["时间量(s)"].to_list(),
                y=df["坡度(‰)"].to_list(),
                mode="lines+markers",
                name="坡度(‰)",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=4),
            )
        )
        plot_slope.update_layout(
            height=300,
            title_text="坡度(‰) 随时间变化",
            xaxis_title="时间量(s)",
            yaxis_title="坡度(‰)",
            showlegend=False,
        )
        plot_slope.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        plot_slope.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # 创建速度曲线图
    plot_speed = None
    if "速度v（km/h）" in df.columns:
        plot_speed = go.Figure()
        plot_speed.add_trace(
            go.Scatter(
                x=df["时间量(s)"].to_list(),
                y=df["速度v（km/h）"].to_list(),
                mode="lines+markers",
                name="速度v（km/h）",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=4),
            )
        )
        plot_speed.update_layout(
            height=300,
            title_text="速度v（km/h） 随时间变化",
            xaxis_title="时间量(s)",
            yaxis_title="速度v（km/h）",
            showlegend=False,
        )
        plot_speed.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        plot_speed.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return plot_altitude, plot_slope, plot_speed


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
    if "时间量(s)" not in df.columns or "预测最优开度" not in df.columns:
        return None

    # 创建子图 - 显示实际阀门开度和预测最优开度
    fig = make_subplots(rows=1, cols=1, subplot_titles=["阀门开度预测结果"])

    # 添加实际阀门开度曲线
    if "阀门开度(%)" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["时间量(s)"].to_list(),
                y=df["阀门开度(%)"].to_list(),
                mode="lines",
                name="实际阀门开度(%)",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1,
            col=1,
        )

    # 添加预测最优开度曲线
    fig.add_trace(
        go.Scatter(
            x=df["时间量(s)"].to_list(),
            y=df["预测最优开度"].to_list(),
            mode="lines+markers",
            name="预测最优开度",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # 更新布局
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode="x unified",
        title_text="阀门开度预测趋势",
        xaxis_title="时间量(s)",
        yaxis_title="开度(%)",
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
        (状态信息, 原始数据预览表格, 海拔曲线图, 坡度曲线图, 速度曲线图, 预测结果表格, 预测曲线图)
    """
    # 确定数据文件路径
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    if file_input is not None:
        # 使用用户上传的文件作为原始数据
        original_file_path = file_input.name
    else:
        # 使用默认原始数据文件 (step1.csv)
        original_file_path = str(data_dir / "step1.csv")

    # 预测数据文件路径 (step2.csv)
    predicted_file_path = str(data_dir / "step2.csv")

    # 1. 加载原始数据
    df_original, preview_df, status_info = load_data(original_file_path, preview_rows)

    if df_original is None or preview_df is None:
        return status_info, None, None, None, None, None, None

    # 2. 创建原始数据的曲线图
    plot_altitude, plot_slope, plot_speed = create_original_plots(df_original)

    # 3. 加载预测数据
    df_predicted, pred_status = load_predicted_data(predicted_file_path)
    
    if df_predicted is None:
        return status_info + "\n" + pred_status, preview_df.to_pandas(), plot_altitude, plot_slope, plot_speed, None, None

    # 4. 创建预测结果可视化图表
    plot = create_plot(df_predicted)

    # 5. 转换为 pandas DataFrame 用于 Gradio 显示
    preview_table = preview_df.to_pandas()
    predicted_table = df_predicted.to_pandas()

    return status_info + "\n" + pred_status, preview_table, plot_altitude, plot_slope, plot_speed, predicted_table, plot


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
            plot_x1 = gr.Plot(label="海拔(m) 随时间变化")
            plot_x2 = gr.Plot(label="坡度(‰) 随时间变化")
            plot_x3 = gr.Plot(label="速度v（km/h） 随时间变化")

        # 预测结果展示区
        gr.Markdown("## 5️⃣ 预测结果展示")

        with gr.Row():
            # 预测结果表格
            result_table = gr.Dataframe(label="预测结果表格 (包含预测最优开度)", interactive=False, wrap=True)

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
