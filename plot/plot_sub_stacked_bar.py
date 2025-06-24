import argparse
import pandas as pd
import matplotlib.pyplot as plt
from parser import Parser
from stacked_bar_2 import StackedBar

def reorder_handles(hl, nc):
    # 원하는 row-major/column-major 변환
    new = sum((hl[i::nc] for i in range(nc)), [])
    return new
def reorder_legend(list, ncol):
    """
    col wise -> row wise
    """
    new_list = []
    for i in range(ncol):
        for j in range((len(list) + ncol - 1)//ncol):
            if i + j*ncol < len(list):
                new_list.append(list[i + j*ncol])
    return new_list

def main():
    parser = argparse.ArgumentParser(
        description="Plot 4 stacked bar subgraphs (2x2 grid) with a single global legend on top center."
    )
    parser.add_argument("input_file", help="Path to input file with simulation data")
    parser.add_argument("--x_expr", required=True, help="Expression for x-axis (e.g., 'cost')")
    parser.add_argument("--y_expr", required=True, 
                        help="Comma-separated expressions for y values (e.g., 'initial_cost, repair_cost')")
    parser.add_argument("--y_label", required=True, 
                        help="Comma-separated labels for y values (e.g., 'Initial, Repair')")
    parser.add_argument("--xlabel", default=None, help="X-axis label")
    parser.add_argument("--y_total_label", default=None, help="Y-axis label")
    parser.add_argument("--legend", default="", help="Prefix for legend labels (e.g., 'Parity')")
    parser.add_argument("--filter_expr1", default=None, help="Filter expression for subplot 1")
    parser.add_argument("--filter_expr2", default=None, help="Filter expression for subplot 2")
    parser.add_argument("--filter_expr3", default=None, help="Filter expression for subplot 3")
    parser.add_argument("--filter_expr4", default=None, help="Filter expression for subplot 4")
    parser.add_argument("--title1", default="", help="Title for subplot 1")
    parser.add_argument("--title2", default="", help="Title for subplot 2")
    parser.add_argument("--title3", default="", help="Title for subplot 3")
    parser.add_argument("--title4", default="", help="Title for subplot 4")
    parser.add_argument("--output_file", default=None, help="Optional output file path (PDF format)")
    parser.add_argument("--y_thousands", action="store_true", 
                        help="Format y-axis tick labels with thousand separators")
    parser.add_argument("--y_min", default=None, help="Comma-separated minimum y-axis values for each subplot (e.g., '0,0,0')")
    parser.add_argument("--y_max", default=None, help="Comma-separated maximum y-axis values for each subplot (e.g., '0.5,0.5,0.5')")
    parser.add_argument("--y_interval", default=None, help="Comma-separated y-axis tick intervals for each subplot (e.g., '0.05,0.05,0.05')")
    parser.add_argument("--legend_location", type=str, default=None, help="Legend location")
    parser.add_argument("--z_col", type=int, default=None, help="Column name for z-axis grouping")
    args = parser.parse_args()

    p = Parser()
    df = p.parse_file_to_dataframe(args.input_file)
    print("Parsed DataFrame:")
    print(df.head())

    y_min_list = [float(min_val.strip()) for min_val in args.y_min.split(",")] if args.y_min else None
    y_max_list = [float(max_val.strip()) for max_val in args.y_max.split(",")] if args.y_max else None
    y_interval_list = [float(interval.strip()) for interval in args.y_interval.split(",")] if args.y_interval else None

    try:
        df["x"] = df.eval(args.x_expr)
    except Exception as e:
        print(f"Error evaluating x_expr '{args.x_expr}': {e}")
        df["x"] = df[args.x_expr]

    # y_expr (comma separated)를 리스트로 분리 후 각 식 평가하여 열 생성
    y_expr_list = [expr.strip() for expr in args.y_expr.split(',')]
    y_label_list = [label.strip() for label in args.y_label.split(',')]
    for i, expr in enumerate(y_expr_list):
        col_name = f"y_{i}"
        try:
            df[col_name] = df.eval(expr)
        except Exception as e:
            print(f"Error evaluating y_expr '{expr}': {e}")
            df[col_name] = df[expr]
    y_cols = [f"y_{i}" for i in range(len(y_expr_list))]
    
    xlabel = args.xlabel if args.xlabel else args.x_expr
    ylabel = args.y_total_label if args.y_total_label else ""
    
    # 2x2 subplot 생성
    fig, axs = plt.subplots(1, 4, figsize=(30, 7), sharex=False)

    
    filters = [args.filter_expr1, args.filter_expr2, args.filter_expr3, args.filter_expr4]
    titles = [args.title1, args.title2, args.title3, args.title4]
    
    legend_info_global = None
    for idx in range(4):
        df_sub = df.copy()
        if filters[idx]:
            try:
                df_sub = df_sub.query(filters[idx])
            except Exception as e:
                print(f"Error applying filter expression for subplot {idx+1}: {e}")
        ax = axs[idx % 4]
        plotter = StackedBar(df_sub)
        # 각 subplot에서는 local legend를 그리지 않음 (show_legend=False)
        legend_info = plotter.plot_stacked_bar(
            fig=fig,
            x_col="x",
            y_cols=y_cols,
            y_labels=[f"{args.legend}={label}" for label in y_label_list] if args.legend else y_label_list,
            title=titles[idx],
            xlabel=xlabel,
            ylabel=ylabel,
            output_file=args.output_file,
            y_thousands=args.y_thousands,
            ax=ax,
            legend_location=args.legend_location,
            y_min=y_min_list[idx] if y_min_list else None,
            y_max=y_max_list[idx] if y_max_list else None,
            y_interval=y_interval_list[idx] if y_interval_list else None,
            show_legend=False
        )
        # global legend info를 첫 번째 subplot에서 받아둡니다.
        if idx == 0:
            legend_info_global = legend_info

    # global legend가 있다면 상단 중앙에 한 번만 표시
    if legend_info_global is not None:
        handles, orig_labels = legend_info_global
        # 만약 args.legend가 주어졌다면, 각 레이블에 접두어를 붙임
        if args.legend:
            labels = [f"{args.legend}={label}" for label in orig_labels]
        else:
            labels = orig_labels
        if args.z_col != None:
            reordered_handle = reorder_handles(handles, args.z_col)
            reordered_labels = reorder_legend(labels, args.z_col)
            fig.legend(reordered_handle, reordered_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                   ncol=args.z_col, frameon=False, fontsize=26)
        else:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                   ncol=len(labels), frameon=False, fontsize=26)
        
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.74, bottom=0.24)
    plt.subplots_adjust(hspace=0.65, wspace=0.3)
    plt.subplots_adjust(left=0.06, right=0.98)
    if args.output_file:
        plt.savefig(args.output_file, format="pdf", dpi=600, bbox_inches='tight')
        plt.close()
        df.to_csv(args.output_file+".txt", sep='\t', index=False)
        print(f"Graph saved to {args.output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
