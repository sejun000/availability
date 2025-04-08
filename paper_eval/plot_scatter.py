import argparse
import pandas as pd
import matplotlib.pyplot as plt
from parser import Parser  # 데이터 파싱용 (사용자 정의)
import matplotlib.ticker as ticker
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Plot scatter plots for multiple y expressions arranged in a grid."
    )
    parser.add_argument("input_file", help="Path to input file with simulation data")
    parser.add_argument("--output_file", default=None, help="Output file name (without extension) for saving the DataFrame")
    parser.add_argument("--x_expr", required=True, help="Expression for x-axis (e.g., 'm')")
    parser.add_argument("--x_label", default=None, help="X-axis label")
    parser.add_argument("--legend", default="", help="Legend prefix for groups")
    parser.add_argument("--y_expr", required=True, help="Comma-separated expressions for y values (e.g., 'a, b, c')")
    parser.add_argument("--y_label", required=True, help="Comma-separated labels for y values (e.g., 'A, B, C')")
    parser.add_argument("--y_min", default=None, help="Comma-separated minimum y-axis values for each subplot (e.g., '0,0,0')")
    parser.add_argument("--y_max", default=None, help="Comma-separated maximum y-axis values for each subplot (e.g., '0.5,0.5,0.5')")
    parser.add_argument("--y_interval", default=None, help="Comma-separated y-axis tick intervals for each subplot (e.g., '0.05,0.05,0.05')")
    parser.add_argument("--titles", default="", help="Titles for the entire figure")

    parser.add_argument("--filtered_expr", default=None, help="Global filter expression for data (e.g., 'm > 5')")

    args = parser.parse_args()

    # 데이터 파싱
    p = Parser()
    df = p.parse_file_to_dataframe(args.input_file)
    print("Parsed DataFrame:")
    print(df.head())

    if args.filtered_expr:
        try:
            print("Number of rows after filtering:", df.shape[0])
            df = df.query(args.filtered_expr)
            print("DataFrame after global filtering:")
            print("Number of rows after filtering:", df.shape[0])
            print(df.head())
        except Exception as e:
            print(f"Error applying global filter expression '{args.filtered_expr}': {e}")
    print(df)
    # x_expr 평가
    try:
        df["x"] = df.eval(args.x_expr)
    except Exception as e:
        print(f"Error evaluating x_expr '{args.x_expr}': {e}")
        df["x"] = df[args.x_expr]

    # z_expr 평가 (옵션)
    try:
        df["z"] = np.where((df["k"] == 0) & (df["network_k"] > 0), 'Inter',
                   np.where((df["network_k"] == 0) & (df["k"] > 0), 'Intra', 'Multi'))
        print("DataFrame after applying default z_expr:")
        print(df[["z"]].head())
    except Exception as e:
        print(f"Error evaluating default z_expr: {e}")

    # y_expr와 y_label 처리 (콤마 분리)
    y_expr_list = [expr.strip() for expr in args.y_expr.split(",")]
    y_label_list = [label.strip() for label in args.y_label.split(",")]
    title_list = [title.strip() for title in args.titles.split(",")]
    if len(y_expr_list) != len(y_label_list):
        print("The number of y_expr and y_label must match.")
        return
    if len(y_expr_list) != len(title_list):
        print("The number of y_expr and titles must match.")
        return

    # 각 y_expr 평가하여 새로운 열 생성
    for i, expr in enumerate(y_expr_list):
        col_name = f"y_{i}"
        try:
            df[col_name] = df.eval(expr)
        except Exception as e:
            print(f"Error evaluating y_expr '{expr}': {e}")
            df[col_name] = df[expr]
    y_cols = [f"y_{i}" for i in range(len(y_expr_list))]
    n_y = len(y_cols)

    # 서브플롯 레이아웃 결정 (1~4개)
    if n_y == 1:
        nrows, ncols = 1, 1
    elif n_y == 2:
        nrows, ncols = 1, 2
    elif n_y == 3:
        nrows, ncols = 1, 3
    elif n_y == 4:
        nrows, ncols = 1, 4
    else:
        print("Only support up to 4 y expressions.")
        return

        # n_y는 y_expr에서 분리된 값의 개수 (이미 계산됨)
    if args.y_min is not None:
        y_min_list = [float(x.strip()) for x in args.y_min.split(",")]
        if len(y_min_list) < n_y:
            y_min_list.extend([y_min_list[-1]]*(n_y - len(y_min_list)))
    else:
        y_min_list = [None]*n_y

    if args.y_max is not None:
        y_max_list = [float(x.strip()) for x in args.y_max.split(",")]
        if len(y_max_list) < n_y:
            y_max_list.extend([y_max_list[-1]]*(n_y - len(y_max_list)))
    else:
        y_max_list = [None]*n_y

    if args.y_interval is not None:
        y_interval_list = [float(x.strip()) for x in args.y_interval.split(",")]
        if len(y_interval_list) < n_y:
            y_interval_list.extend([y_interval_list[-1]]*(n_y - len(y_interval_list)))
    else:
        y_interval_list = [None]*n_y


    fig, axs = plt.subplots(nrows, ncols, figsize=(12 * ncols, 8 * nrows), squeeze=False)

    # 만약 z_expr가 제공되면, 고정 색상 팔레트를 사용하여 그룹별 색상 매핑 생성
    groups = sorted(df["z"].dropna().unique())
    colors = plt.cm.tab10.colors  # 최대 10가지 색상 제공
    color_map = {group: colors[i % len(colors)] for i, group in enumerate(groups)}
    # global legend용 handle과 label 생성
    marker_styles = ['o', '^', 'x']

    global_handles = []
    global_labels = []
    for i, group in enumerate(groups):
        handle = plt.Line2D([], [], marker=marker_styles[i % len(marker_styles)], linestyle='',
                            markersize=8, color=color_map[group])
        global_handles.append(handle)
        label = f"{args.legend}={group}" if args.legend else str(group)
        global_labels.append(label)

    # 각 서브플롯에 대해 scatter plot 그리기 (가로 우선 배치)
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if idx >= n_y:
                axs[i, j].axis('off')
                continue
            ax = axs[i, j]
            y_col = y_cols[idx]
            # 그룹별로 scatter 그리기 (기본 조건식으로 생성된 df["z"]를 사용)
            for group in groups:
                subset = df[df["z"] == group]
                ax.scatter(subset["x"], subset[y_col], marker=marker_styles[groups.index(group) % len(marker_styles)], s=80, color=color_map[group], alpha=0.8)

            #ax.set_xlabel(args.x_label if args.x_label else args.x_expr, fontsize=25)
            ax.set_ylabel(y_label_list[idx], fontsize=25)
            ax.tick_params(axis='both', labelsize=23)
            # x축을 정수형으로 표시
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if y_min_list[idx] is not None or y_max_list[idx] is not None:
                ax.set_ylim(bottom=y_min_list[idx], top=y_max_list[idx])
            print (y_interval_list[idx], idx)
            from matplotlib.ticker import MultipleLocator
            if y_interval_list[idx] is not None:
                ax.yaxis.set_major_locator(MultipleLocator(y_interval_list[idx]))
            ax.xaxis.set_major_locator(MultipleLocator(10))
            fig.canvas.draw()

            ymin, ymax = ax.get_ylim()
            ax.text(0.5, -0.64, title_list[idx], transform=ax.transAxes,
                    ha='center', fontsize=26, clip_on=False)
            ax.grid(axis='y', linestyle='--', linewidth=1, color='black')
            # get_ygridlines()로 모든 가로 grid line 순회
            for line in ax.get_ygridlines():
                # x_data, y_data 형태로 반환 (horizontal line이면 y_data가 같은 값 2개)
                x_data, y_data = line.get_data()
                #print (y_data, ymin, ymax)
                # 혹은 line.get_ydata() 만으로도 확인 가능
                
                # 두 점의 y좌표가 모두 ymin(또는 ymax)와 같은 경우가 최솟값/최댓값을 그리는 line
                if (y_data[0] <= ymin + 1e-8):
                    line.set_visible(False)
                elif (y_data[0] >= ymax - 1e-8):
                    line.set_visible(False)
            idx += 1

        # 추가: y_min, y_max, y_interval 적용
    #ax.yaxis.set_major_locator(MultipleLocator(args.y_interval))
    # global legend를 상단 중앙에 표시 (z_expr가 제공된 경우)
            # y축 현재 범위 가져오기
    fig.text(0.5, 0.15, args.x_label if args.x_label else args.x_expr, ha='center', fontsize=26)
    fig.legend(global_handles, global_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                ncol=len(global_labels), frameon=False, fontsize=26)
    
    # 전체 여백 조정 (상단에 여백 확보)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.82, bottom=0.30)
    plt.subplots_adjust(left=0.07, right=0.93)
    plt.subplots_adjust(hspace=0.35, wspace=0.54)
    if (args.output_file):
        df.to_csv(args.output_file+".txt", sep='\t', index=False)
        plt.savefig(args.output_file, format="pdf", dpi=600, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
