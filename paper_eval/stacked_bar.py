import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch   # 범례용 패치

class StackedBar:
    """
    StackedBar 클래스는 x축에 따라 여러 y값(누적)을 쌓은 stacked bar chart를 그립니다.
    y_cols와 y_labels는 각각 콤마로 구분된 문자열을 리스트로 변환한 값입니다.
    """
    def __init__(self, df):
        self.df = df

    def plot_stacked_bar(
        self,
        x_col: str,
        y_cols: list,
        y_labels: list,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        output_file: str = None,
        y_thousands: bool = False,
        ax=None,
        legend_location: str = "upper right",
        y_min: float = None,
        y_max: float = None,
        y_interval: float = None,
        fig: plt.Figure = None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # ───── 회색조 & 패턴 정의 ─────
        #gray_sequence  = ['#f0f0f0', '#cccccc', '#999999', '#666666',
        #                  '#4d4d4d', '#333333', '#000000']
        gray_sequence  = ['#aec7e8', '#ffbb78', '#98df8a', '#f7b6d2',
                          '#c5b0d5', '#c7c7c7', '#ffed6f']
        hatch_sequence = ['', '/', '\\\\', 'x', 'o', '.', '*', '|']

        # x축 값 기준 정렬 및 내부 좌표 생성
        df_sorted = self.df.sort_values(by=x_col)
        actual_x = df_sorted[x_col].values
        positions = list(range(len(actual_x)))
        bottoms = [0] * len(actual_x)

        # ───── 스택 막대 그리기 ─────
        for i, col in enumerate(y_cols):
            y_vals = df_sorted[col].values
            ax.bar(
                positions,
                y_vals,
                bottom=bottoms,
                color=gray_sequence[i % len(gray_sequence)],
                hatch=hatch_sequence[i % len(hatch_sequence)],
                edgecolor="black",
                width=0.5,
            )
            bottoms = [b + y for b, y in zip(bottoms, y_vals)]

        # x축 tick 설정
        ax.set_xticks(positions)
        try:
            ax.set_xticklabels([str(int(v)) for v in actual_x], rotation=0)
        except Exception:
            ax.set_xticklabels([str(v) for v in actual_x], rotation=0)

        # 축 및 라벨
        ax.set_xlabel(xlabel if xlabel else x_col, fontsize=26, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else "", fontsize=26)
        ax.tick_params(axis="x", which="both", length=0, pad=10, labelsize=26)
        ax.tick_params(axis="y", labelsize=26)
        if y_thousands:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
            )

        # 범례: 색상·패턴 동일하게 맞춤
        legend_handles = [
            Patch(
                facecolor=gray_sequence[i % len(gray_sequence)],
                edgecolor="black",
                hatch=hatch_sequence[i % len(hatch_sequence)],
            )
            for i in range(len(y_cols))
        ]
        leg = ax.legend(
            legend_handles,
            y_labels,
            loc=legend_location,
            frameon=True,
            edgecolor="black",
            fontsize=26,
        )
        leg.get_frame().set_alpha(1)

        # 제목
        if title:
            ax.text(
                0.5,
                -0.45,
                title,
                transform=ax.transAxes,
                ha="center",
                fontsize=26,
            )
            plt.subplots_adjust(bottom=0.25)

        # 그리드 및 축 보정
        ax.set_frame_on(True)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=1, color="black")

        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if y_interval is not None:
            from matplotlib.ticker import MultipleLocator

            ax.yaxis.set_major_locator(MultipleLocator(y_interval))

        # 최상단/최하단 그리드 숨김
        fig.canvas.draw()
        ymin, ymax = ax.get_ylim()
        for line in ax.get_ygridlines():
            _, y_data = line.get_data()
            if y_data[0] <= ymin + 1e-8 or y_data[0] >= ymax - 1e-8:
                line.set_visible(False)

        if output_file:
            plt.savefig(
                output_file, format="pdf", dpi=600, bbox_inches="tight"
            )
            print(f"Plot saved to {output_file}")

        return ax
