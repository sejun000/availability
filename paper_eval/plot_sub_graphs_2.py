import argparse
import pandas as pd
from parser import Parser
from sub_grouped_bar import SubGroupedBar
from stacked_bar import StackedBar
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Parse simulation file and plot two grouped bar charts side by side for y1 and y2, with a single global legend on top center and individual subplot titles displayed below each subplot."
    )
    parser.add_argument("input_file", help="Path to input file with simulation data")
    parser.add_argument("--x1_expr", required=True, help="Expression for x-axis (e.g., 'm + k')")
    parser.add_argument("--x2_expr", required=True, help="X2 Expression for x-axis (e.g., 'm + k')")
    parser.add_argument("--y1_expr", required=True, help="Expression for left subplot y-axis (e.g., 'availability')")
    parser.add_argument("--y2_expr", required=True, help="Comma-separated expressions for stacked bar (e.g., 'a,b,c')")
    parser.add_argument("--z_expr", default=None, help="Optional expression for grouping (e.g., 'qlc')")
    parser.add_argument("--x1_label", default=None, help="X1-axis label")
    parser.add_argument("--x2_label", default=None, help="X2-axis label")
    parser.add_argument("--y1_label", default=None, help="Y-axis label for left subplot")
    parser.add_argument("--y2_label", required=True, help="Comma-separated labels for stacked bar (e.g., 'A,B,C')")
    parser.add_argument("--y1_title", default="", help="Title for left subplot (displayed below the subgraph)")
    parser.add_argument("--y2_title", default="", help="Title for right subplot (displayed below the subgraph)")
    parser.add_argument("--y2_total_label", default=None, help="Label for total stacked bar (if needed)")
    parser.add_argument("--legend", default=None, help="Legend title (if grouping is used)")
    parser.add_argument("--filter1_expr", default=None, help="Optional filter expression (e.g., 'm > 5')")
    parser.add_argument("--filter2_expr", default=None, help="Optional filter expression (e.g., 'm > 5')")
    parser.add_argument("--output_file", default=None, help="Optional output file path (PDF format)")
    parser.add_argument("--y_thousands", action="store_true", help="Format y-axis tick labels with thousand separators")
    parser.add_argument("--y1_min", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--y1_max", type=float, default=None, help="Maximum y-axis value")
    parser.add_argument("--y1_interval", type=float, default=None, help="Y-axis tick interval")
    parser.add_argument("--y2_min", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--y2_max", type=float, default=None, help="Maximum y-axis value")
    parser.add_argument("--y2_interval", type=float, default=None, help="Y-axis tick interval")
    parser.add_argument("--legend_location1", default="upper right", help="Location of the legend (default: 'upper right')")
    parser.add_argument("--legend_location2", default="upper right", help="Location of the legend (default: 'upper right')")


    args = parser.parse_args()

    p = Parser()
    df = p.parse_file_to_dataframe(args.input_file)
    print("Parsed DataFrame:")
    print(df.head())

    if args.filter1_expr:
        try:
            df = df.query(args.filter1_expr)
            print("DataFrame after filtering:")
            print(df.head())
        except Exception as e:
            print(f"Error applying filter expression '{args.filter_expr}': {e}")

    try:
        df["x1"] = df.eval(args.x1_expr)
    except Exception as e:
        print(f"Error evaluating x_expr '{args.x1_expr}': {e}")
        df["x1"] = df[args.x1_expr]

    try:
        df["x2"] = df.eval(args.x2_expr)
    except Exception as e:
        print(f"Error evaluating x_expr '{args.x2_expr}': {e}")
        df["x2"] = df[args.x2_expr]
    try:
        df["y1"] = df.eval(args.y1_expr)
    except Exception as e:
        print(f"Error evaluating y1_expr '{args.y1_expr}': {e}")
        df["y1"] = df[args.y1_expr]

    # 기존 df["y2"] 처리 대신, StackedBar의 경우 여러 열로 분리합니다.
    y2_expr_list = [expr.strip() for expr in args.y2_expr.split(',')]
    y2_label_list = [label.strip() for label in args.y2_label.split(',')]
    # 각 열에 대해 DataFrame에 새 컬럼 생성
    for i, expr in enumerate(y2_expr_list):
        col_name = f"y2_{i}"
        try:
            df[col_name] = df.eval(expr)
        except Exception as e:
            print(f"Error evaluating y2_expr '{expr}': {e}")
            df[col_name] = df[expr]

    if args.z_expr:
        try:
            df["z"] = df.eval(args.z_expr)
        except Exception as e:
            print(f"Error evaluating z_expr '{args.z_expr}': {e}")
            df["z"] = df[args.z_expr]
    else:
        df["z"] = None
    
    # ... (기존 df 처리 후)
    x1_label = args.x1_label if args.x1_label else args.x1_expr
    y1_label = args.y1_label if args.y1_label else args.y1_expr
    y2_total_label = args.y2_total_label if args.y2_total_label else None
    # GroupedBar의 경우, 기존 y1_expr 사용
    plotter_group = SubGroupedBar(df)
    # StackedBar: y2_expr_list의 각 항목이 열로 생성됨 -> 열 이름은 "y2_0", "y2_1", ...
    y2_cols = [f"y2_{i}" for i in range(len(y2_expr_list))]

    # 좌우 서브플롯 생성 (1행 2열)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(left=0.08, right=0.94)
    # 왼쪽: GroupedBar (기존대로, legend 등 필요하면 처리)
    legend_info = plotter_group.plot_grouped_bar(
        fig=fig,
        x_col="x1",
        y_col="y1",
        z_col="z" if args.z_expr else None,
        title=args.y1_title,   # 서브플롯 아래 제목
        xlabel=x1_label,             # x축 라벨은 오른쪽에만 표시
        ylabel=y1_label,
        legend_title=args.legend,
        y_thousands=args.y_thousands,
        ax=axs[0],
        show_legend=True,
        local_legend=True,  # 지역 범례 사용
        legend_location=args.legend_location1,
        y_min=args.y1_min,         # 추가
        y_max=args.y1_max,         # 추가
        y_interval=args.y1_interval  # 추가
    )
    if args.filter2_expr:
        try:
            df = df.query(args.filter2_expr)
            print("DataFrame after filtering:")
            print(df.head())
        except Exception as e:
            print(f"Error applying filter expression '{args.filter_expr}': {e}")
    plotter_stack = StackedBar(df)
    x2_label = args.x2_label if args.x2_label else args.x2_expr

    # 오른쪽: StackedBar (legend는 각 서브플롯에 개별적으로 표시)
    plotter_stack.plot_stacked_bar(
        fig=fig,
        x_col="x2",
        y_cols=y2_cols,
        y_labels=y2_label_list,
        title=args.y2_title,   # 서브플롯 아래 제목
        xlabel=x2_label,
        ylabel=y2_total_label,  # 필요에 따라 y2 label을 설정
        y_thousands=args.y_thousands,
        legend_location=args.legend_location2,
        ax=axs[1],
        y_min=args.y2_min,         # 추가
        y_max=args.y2_max,         # 추가
        y_interval=args.y2_interval  # 추가
    )
    plt.subplots_adjust(top=0.94, bottom=0.34)
    # 만약 GroupedBar의 legend를 글로벌로 처리할 필요가 있으면 추가 (여기서는 별도 legend를 두지 않음)
    if args.output_file:
        plt.savefig(args.output_file, format="pdf", dpi=600, bbox_inches='tight')
        plt.close()
        df.to_csv(args.output_file+".txt", sep='\t', index=False)
        print(f"Graph saved to {args.output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
