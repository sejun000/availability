import argparse
import pandas as pd
import numpy as np
from csv_parser import Parser
from sub_grouped_bar import SubGroupedBar
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    parser = argparse.ArgumentParser(
        description="두 개의 grouped bar 차트를 그립니다. 왼쪽은 고정 표 데이터를 이용( x: Capacity, y: durability nines, z: Parity Disks ), "
                    "오른쪽은 사용자 입력( x, y2, z )으로 플롯합니다."
    )
    # simulation data 관련 인자 (오른쪽 subplot 용)
    parser.add_argument("input_file", help="시뮬레이션 데이터가 있는 파일 경로")
    parser.add_argument("--x_expr", required=True, help="시뮬레이션 데이터용 x축 표현식 (예: 'm + k')")
    parser.add_argument("--y1_expr", required=True, help="시뮬레이션 데이터용 왼쪽 subplot y축 표현식 (이 값은 표 데이터 플롯에는 사용하지 않음)")
    parser.add_argument("--y2_expr", required=True, help="시뮬레이션 데이터용 오른쪽 subplot y축 표현식 (y2는 그대로 사용)")
    parser.add_argument("--z_expr", default=None, help="시뮬레이션 데이터용 그룹화 표현식 (예: 'qlc')")
    parser.add_argument("--xlabel", default=None, help="시뮬레이션 데이터용 x축 라벨")
    parser.add_argument("--y1_label", default=None, help="시뮬레이션 데이터용 왼쪽 subplot y축 라벨 (표 데이터 플롯에는 사용하지 않음)")
    parser.add_argument("--y2_label", default=None, help="시뮬레이션 데이터용 오른쪽 subplot y축 라벨")
    parser.add_argument("--y1_title", default="", help="왼쪽 subplot 제목 (표 데이터용)")
    parser.add_argument("--y2_title", default="", help="오른쪽 subplot 제목 (시뮬레이션 데이터용)")
    parser.add_argument("--legend", default=None, help="시뮬레이션 데이터 범례 제목 (그룹핑이 있을 경우)")
    parser.add_argument("--filter_expr", default=None, help="시뮬레이션 데이터 필터 표현식 (예: 'm > 5')")
    parser.add_argument("--output_file", default=None, help="출력 파일 경로 (PDF 형식)")
    parser.add_argument("--y_thousands", action="store_true", help="y축 눈금에 천 단위 구분자 사용")
    parser.add_argument("--y1_min", type=float, default=None, help="시뮬레이션 데이터용 왼쪽 subplot y축 최소값 (표 데이터 플롯에는 사용하지 않음)")
    parser.add_argument("--y1_max", type=float, default=None, help="시뮬레이션 데이터용 왼쪽 subplot y축 최대값 (표 데이터 플롯에는 사용하지 않음)")
    parser.add_argument("--y1_interval", type=float, default=None, help="시뮬레이션 데이터용 왼쪽 subplot y축 간격 (표 데이터 플롯에는 사용하지 않음)")
    parser.add_argument("--y2_min", type=float, default=None, help="시뮬레이션 데이터용 오른쪽 subplot y축 최소값")
    parser.add_argument("--y2_max", type=float, default=None, help="시뮬레이션 데이터용 오른쪽 subplot y축 최대값")
    parser.add_argument("--y2_interval", type=float, default=None, help="시뮬레이션 데이터용 오른쪽 subplot y축 간격")
    parser.add_argument("--z_col", default=None, help="시뮬레이션 데이터용 z축 그룹화 컬럼명")
    args = parser.parse_args()

    ###############################
    # 1. 시뮬레이션 데이터 로드 (오른쪽 subplot)
    ###############################
    p = Parser()
    df_sim = p.parse_file_to_dataframe(args.input_file)
    if args.filter_expr:
        try:
            df_sim = df_sim.query(args.filter_expr)
        except Exception as e:
            print(f"필터 표현식 '{args.filter_expr}' 적용 에러: {e}")

    try:
        df_sim["x"] = df_sim.eval(args.x_expr)
    except Exception as e:
        print(f"x_expr '{args.x_expr}' 평가 에러: {e}")
        df_sim["x"] = df_sim[args.x_expr]

    try:
        df_sim["y1"] = df_sim.eval(args.y1_expr)
    except Exception as e:
        print(f"y1_expr '{args.y1_expr}' 평가 에러: {e}")
        df_sim["y1"] = df_sim[args.y1_expr]

    try:
        df_sim["y2"] = df_sim.eval(args.y2_expr)
    except Exception as e:
        print(f"y2_expr '{args.y2_expr}' 평가 에러: {e}")
        df_sim["y2"] = df_sim[args.y2_expr]

    if args.z_expr:
        try:
            df_sim["z"] = df_sim.eval(args.z_expr)
        except Exception as e:
            print(f"z_expr '{args.z_expr}' 평가 에러: {e}")
            df_sim["z"] = df_sim[args.z_expr]
    else:
        df_sim["z"] = None

    ###############################
    # 2. 고정 표 데이터 생성 (왼쪽 subplot)
    ###############################
    table_data = {
        "Capacity (TB)": [8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256],
        "Parity Disks": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        "MTTDL (years)": [
            2.843577e+01, 1.205549e+04, 7.849202e+06, 6.971796e+09,
            1.435474e+01, 3.035917e+03, 9.875011e+05, 4.383762e+08,
            7.314228e+00, 7.701573e+02, 1.250425e+05, 2.773158e+07,
            3.793971e+00, 1.982840e+02, 1.604029e+04, 1.775655e+06,
            2.033843e+00, 5.259899e+01, 2.112093e+03, 1.164890e+05,
            1.153779e+00, 1.481937e+01, 2.932408e+02, 8.024645e+03
        ],
        "1-Year Durability (%)": [
            96.544420, 99.991705, 99.999987, 99.99999998565650116689,
            93.270772, 99.967066, 99.999899, 99.99999977188544164619,
            87.221456, 99.870241, 99.9992, 99.999996,
            76.829918, 99.496943, 99.993766, 99.999944,
            61.159800, 98.116781, 99.952665, 99.999142,
            42.032909, 93.474712, 99.659564, 99.987539
        ]
    }
    df_table = pd.DataFrame(table_data)
    # 내림차순 로그 계산을 위해 내림수를 직접 조정(100%인 경우 1e-8 이하로 clip)
    df_table["durability_nines"] = -np.log10((1 - df_table["1-Year Durability (%)"] / 100).clip(lower=1e-13))
    # SubGroupedBar에 맞는 컬럼 할당
    df_table["x"] = df_table["Capacity (TB)"]
    df_table["y1"] = df_table["durability_nines"]
    df_table["z"] = df_table["Parity Disks"]

    ###############################
    # 3. 플롯 생성: 좌측은 표 데이터, 우측은 시뮬레이션 데이터
    ###############################
    # 시뮬레이션 플롯용 x축 라벨 처리
    xlabel_sim = args.xlabel if args.xlabel else args.x_expr
    y2_label = args.y2_label if args.y2_label else args.y2_expr

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    # 왼쪽: 고정 표 데이터 – x: Capacity, y: durability nines, z: Parity Disks
    plotter_table = SubGroupedBar(df_table)
    legend_info_table = plotter_table.plot_grouped_bar(
        fig=fig,
        x_col="x",
        y_col="y1",
        z_col="z",
        title=args.y1_title,      # 예: "Table Data Graph"
        xlabel="Capacity (TB)",
        ylabel="Nines",
        legend_title="Parity Disks",
        y_thousands=args.y_thousands,
        ax=axs[0],
        show_legend=True,
        y_min=args.y1_min,
        y_max=args.y1_max,
        y_interval=args.y1_interval,
        legend_type="s'"
    )

    # 오른쪽: 시뮬레이션 데이터 – x, y2, z는 사용자 입력대로 사용
    z_col_sim = "z" if args.z_expr else None
    plotter_sim = SubGroupedBar(df_sim)
    legend_info_sim = plotter_sim.plot_grouped_bar(
        fig=fig,
        x_col="x",
        y_col="y2",
        z_col=z_col_sim,
        title=args.y2_title,
        xlabel=xlabel_sim,
        ylabel=y2_label,
        legend_title=args.legend,
        y_thousands=args.y_thousands,
        ax=axs[1],
        show_legend=False,
        y_min=args.y2_min,
        y_max=args.y2_max,
        y_interval=args.y2_interval
    )

    # 범례가 있을 경우, 왼쪽 플롯의 범례를 전역 범례로 상단 중앙에 추가합니다.
    if legend_info_table is not None:
        handles, orig_labels = legend_info_table
        labels = [f"K={label}" for label in orig_labels]
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                   ncol=len(labels), frameon=False, fontsize=26)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.28, left=0.08, right=0.96)

    if args.output_file:
        plt.savefig(args.output_file, format="pdf", dpi=600, bbox_inches='tight')
        plt.close()
        print(f"그래프가 {args.output_file}에 저장되었습니다.")
        df_sim.to_csv(args.output_file + ".txt", sep='\t', index=False)
    else:
        plt.show()

if __name__ == "__main__":
    main()
