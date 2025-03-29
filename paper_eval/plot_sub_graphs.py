import argparse
import pandas as pd
from parser import Parser
from sub_grouped_bar import SubGroupedBar
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Parse simulation file and plot two grouped bar charts side by side for y1 and y2, with a single global legend on top center and individual subplot titles displayed below each subplot."
    )
    parser.add_argument("input_file", help="Path to input file with simulation data")
    parser.add_argument("--x_expr", required=True, help="Expression for x-axis (e.g., 'm + k')")
    parser.add_argument("--y1_expr", required=True, help="Expression for left subplot y-axis (e.g., 'availability')")
    parser.add_argument("--y2_expr", required=True, help="Expression for right subplot y-axis (e.g., 'throughput')")
    parser.add_argument("--z_expr", default=None, help="Optional expression for grouping (e.g., 'qlc')")
    parser.add_argument("--xlabel", default=None, help="X-axis label")
    parser.add_argument("--y1_label", default=None, help="Y-axis label for left subplot")
    parser.add_argument("--y2_label", default=None, help="Y-axis label for right subplot")
    parser.add_argument("--y1_title", default="", help="Title for left subplot (displayed below the subgraph)")
    parser.add_argument("--y2_title", default="", help="Title for right subplot (displayed below the subgraph)")
    parser.add_argument("--legend", default=None, help="Legend title (if grouping is used)")
    parser.add_argument("--filter_expr", default=None, help="Optional filter expression (e.g., 'm > 5')")
    parser.add_argument("--output_file", default=None, help="Optional output file path (PDF format)")
    parser.add_argument("--y_thousands", action="store_true", help="Format y-axis tick labels with thousand separators")
    parser.add_argument("--y1_min", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--y1_max", type=float, default=None, help="Maximum y-axis value")
    parser.add_argument("--y1_interval", type=float, default=None, help="Y-axis tick interval")
    parser.add_argument("--y2_min", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--y2_max", type=float, default=None, help="Maximum y-axis value")
    parser.add_argument("--y2_interval", type=float, default=None, help="Y-axis tick interval")

    args = parser.parse_args()

    p = Parser()
    df = p.parse_file_to_dataframe(args.input_file)
    print("Parsed DataFrame:")
    print(df.head())

    if args.filter_expr:
        try:
            df = df.query(args.filter_expr)
            print("DataFrame after filtering:")
            print(df.head())
        except Exception as e:
            print(f"Error applying filter expression '{args.filter_expr}': {e}")

    try:
        df["x"] = df.eval(args.x_expr)
    except Exception as e:
        print(f"Error evaluating x_expr '{args.x_expr}': {e}")
        df["x"] = df[args.x_expr]

    try:
        df["y1"] = df.eval(args.y1_expr)
    except Exception as e:
        print(f"Error evaluating y1_expr '{args.y1_expr}': {e}")
        df["y1"] = df[args.y1_expr]

    try:
        df["y2"] = df.eval(args.y2_expr)
    except Exception as e:
        print(f"Error evaluating y2_expr '{args.y2_expr}': {e}")
        df["y2"] = df[args.y2_expr]

    if args.z_expr:
        try:
            df["z"] = df.eval(args.z_expr)
        except Exception as e:
            print(f"Error evaluating z_expr '{args.z_expr}': {e}")
            df["z"] = df[args.z_expr]
    else:
        df["z"] = None

    xlabel = args.xlabel if args.xlabel else args.x_expr
    y1_label = args.y1_label if args.y1_label else args.y1_expr
    y2_label = args.y2_label if args.y2_label else args.y2_expr

    plotter = SubGroupedBar(df)
    z_col = "z" if args.z_expr else None

    # 좌우 서브플롯 생성 (1행 2열) 및 서브플롯 사이에 여백 추가
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    plt.subplots_adjust(wspace=0.2)

    # 왼쪽 서브플롯: y1 데이터 (범례 정보를 반환받음)
    legend_info = plotter.plot_grouped_bar(
        x_col="x",
        y_col="y1",
        z_col=z_col,
        title=args.y1_title,   # 서브플롯 아래에 제목 표시
        xlabel=xlabel,             # x축 라벨은 오른쪽에만 표시
        ylabel=y1_label,
        legend_title=args.legend,
        y_thousands=args.y_thousands,
        ax=axs[0],
        show_legend=True,
        y_min=args.y1_min,         # 추가
        y_max=args.y1_max,         # 추가
        y_interval=args.y1_interval  # 추가
    )

    # 오른쪽 서브플롯: y2 데이터 (범례는 표시하지 않음)
    plotter.plot_grouped_bar(
        x_col="x",
        y_col="y2",
        z_col=z_col,
        title=args.y2_title,   # 서브플롯 아래에 제목 표시
        xlabel=xlabel,
        ylabel=y2_label,
        legend_title=args.legend,
        y_thousands=args.y_thousands,
        ax=axs[1],
        show_legend=False,
        y_min=args.y2_min,         # 추가
        y_max=args.y2_max,         # 추가
        y_interval=args.y2_interval  # 추가
    )

    # z 그룹 범례가 있을 경우, 전역 범례를 상단 중앙에 테두리 없이 추가
    if legend_info is not None:
        handles, _ = legend_info
        labels = [f"{args.legend}={i+1}" for i in range(len(handles))]
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                   ncol=len(labels), frameon=False, fontsize=13)

    if args.output_file:
        plt.savefig(args.output_file, format="pdf")
        plt.close()
        print(f"Graph saved to {args.output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
