import argparse
from parser import Parser
from stacked_bar import StackedBar

def main():
    ap = argparse.ArgumentParser(description="Stacked-bar plot helper")
    ap.add_argument("input_file")
    ap.add_argument("--x_expr",  required=True)
    ap.add_argument("--y_exprs", required=True, help="comma-separated list")
    ap.add_argument("--legend", default=None,
                    help="comma-separated legend labels (same count as y_exprs)")
    ap.add_argument("--x_label", default=None)
    ap.add_argument("--y_label", default=None)
    ap.add_argument("--title",  default="")
    ap.add_argument("--output_file", default=None)
    ap.add_argument("--filter_expr", default=None, help="Optional filter expression (e.g., 'm > 5')")
    ap.add_argument("--y_min", type=float, default=None, help="Minimum y-axis value")
    ap.add_argument("--y_max", type=float, default=None, help="Maximum y-axis value")
    ap.add_argument("--y_interval", type=float, default=None, help="Y-axis tick interval")
    ap.add_argument("--legend_location", default="upper right", help="Location of the legend (default: 'upper right')")
    args = ap.parse_args()

    df = Parser().parse_file_to_dataframe(args.input_file)
    
    df["x"] = df.eval(args.x_expr)

    y_expr_list = [e.strip() for e in args.y_exprs.split(",") if e.strip()]
    if args.legend:
        labels = [l.strip() for l in args.legend.split(",")]
        if len(labels) != len(y_expr_list):
            raise ValueError("legend count must match y_exprs count")
    else:
        labels = y_expr_list

    y_cols = []
    
    if args.filter_expr:
        try:
            print("Applying filter expression:", args.filter_expr)
            df = df.query(args.filter_expr)
            print("DataFrame after filtering:")
            print(df.head())
        except Exception as e:
            print(f"Error applying filter expression '{args.filter_expr}': {e}")
            raise
    for expr, lab in zip(y_expr_list, labels):
        try:
            print (f"Evaluating expression: {expr} -> {lab}")
            df[lab] = df.eval(expr, engine="python")
        except Exception as e:
            print("\n[ERROR] Cannot evaluate:", expr)
            print("Reason :", e)
            print("Columns:", list(df.columns))
            raise                      # 재-raise 해서 스택 확인
        y_cols.append(lab)

    ax = StackedBar(df).plot_stacked_bar(
        x_col="x",
        y_cols=y_cols,
        y_labels=labels,
        title=args.title,
        xlabel=args.x_label or args.x_expr,
        ylabel=args.y_label or "",
        output_file=args.output_file,
        y_min=args.y_min,
        y_max=args.y_max,
        y_interval=args.y_interval,
        legend_location=args.legend_location
    )
    # output to file 
    if args.output_file:
        ax.figure.savefig(args.output_file, bbox_inches='tight', dpi=300, format='png')
        print(f"Saved plot to {args.output_file}")
    else:
        print("No output file specified. Plot not saved.")


if __name__ == "__main__":
    main()
