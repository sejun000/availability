import argparse
import pandas as pd
from csv_parser import Parser
from grouped_bar import GroupedBar

def main():
    parser = argparse.ArgumentParser(
        description="Parse simulation file and plot grouped bar chart with optional grouping and y-axis thousand separator."
    )
    parser.add_argument("input_file", help="Path to input file with simulation data")
    parser.add_argument("--x_expr", required=True, help="Expression for x-axis (e.g., 'm + k')")
    parser.add_argument("--y_expr", required=True, help="Expression for y-axis (e.g., 'availability')")
    # z_expr는 옵션: 입력되지 않으면 그룹핑 없이 단순 바 차트를 그립니다.
    parser.add_argument("--z_expr", default=None, help="Optional expression for grouping (e.g., 'qlc')")
    parser.add_argument("--x_label", default=None, help="X-axis label")
    parser.add_argument("--y_label", default=None, help="Y-axis label")
    parser.add_argument("--legend", default=None, help="Legend title (if grouping is used)")
    parser.add_argument("--filter_expr", default=None, help="Optional filter expression (e.g., 'm > 5')")
    parser.add_argument("--output_file", default=None, help="Optional output file path (PDF format)")
    # 새 옵션: y축 눈금 라벨에 천 단위 콤마 포맷 적용 여부
    parser.add_argument("--y_thousands", action="store_true", help="Format y-axis tick labels with thousand separators")
    parser.add_argument("--y_min", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--y_max", type=float, default=None, help="Maximum y-axis value")
    parser.add_argument("--y_interval", type=float, default=None, help="Y-axis tick interval")
    parser.add_argument("--legend_location", default="upper right", help="Location of the legend (default: 'upper right')")
    parser.add_argument("--legend_type", type=str, default="equal", help="legend type")
    parser.add_argument("--legend_labels", type=str, default=None, help="Comma-separated legend labels for z_expr (if grouping is used)")
    parser.add_argument("--legend_ncol", type=int, default=2, help="Number of columns in the legend (default: 2)")
    
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
        df["x"] = df.eval(args.x_expr, engine="python")
    except Exception as e:
        try:
            df["x"] = df.apply(lambda r: eval(args.x_expr, {}, r.to_dict()), axis=1)
        except Exception as e:
            print(f"Error evaluating x_expr '{args.x_expr}': {e}")
            df["x"] = df[args.x_expr]

    try:
        df["y"] = df.eval(args.y_expr, engine="python")
    except Exception as e:
        try:
            df["y"] = df.apply(lambda r: eval(args.y_expr, {}, r.to_dict()), axis=1)
        except Exception as e:
            print(f"Error evaluating y_expr '{args.y_expr}': {e}")
            df["y"] = df[args.y_expr]

    if args.z_expr:
        try:
            df["z"] = df.eval(args.z_expr, engine="python")
        except Exception as e:
            try:
                df["z"] = df.apply(lambda r: eval(args.z_expr, {}, r.to_dict()), axis=1)
            except Exception as e:
                print(f"Error evaluating z_expr '{args.z_expr}': {e}")
                df["z"] = df[args.z_expr]
    else:
        df["z"] = None

    xlabel = args.x_label if args.x_label else args.x_expr
    ylabel = args.y_label if args.y_label else args.y_expr
    
    plotter = GroupedBar(df)
    # z_expr가 입력되지 않으면 z_col에 None 전달
    z_col = "z" if args.z_expr else None
    plotter.plot_grouped_bar(
        x_col="x",
        y_col="y",
        z_col=z_col,
        title="Grouped Bar Chart",
        xlabel=xlabel,
        ylabel=ylabel,
        legend_title=args.legend,
        output_file=args.output_file,
        y_thousands=args.y_thousands,
        y_min=args.y_min,
        y_max=args.y_max,
        y_interval=args.y_interval,
        legend_location=args.legend_location,
        legend_type=args.legend_type,
        legend_labels=args.legend_labels.split(",") if args.legend_labels else None,
        legend_ncol=args.legend_ncol
    )
    
    if (args.output_file):
        df.to_csv(args.output_file+".txt", sep='\t', index=False)

if __name__ == "__main__":
    main()
