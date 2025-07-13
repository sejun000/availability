import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch, Rectangle   # 범례·색상 패치

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
        show_legend: bool = True,
        fig: plt.Figure = None,
    ):
        if legend_location is None:
            legend_location = "upper right"
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # ───────── 회색조 + 다중 해치(수평 제외) 정의 ─────────
        
        gray_sequence = [
            "#f0f0f0",
            "#d9d9d9",
            "#bdbdbd",
            "#969696",
            "#525252",
            "#737373",
            "#252525",
        ]
        
        #gray_sequence = ['dodgerblue', 'orange', 'green', 'violet', 'yellow', 'purple', 'pink']
        hatch_sequence = [
            "",        # 무늬 없음
            "o",       # / 대각선
            "\\\\",    # \ 대각선
            "|",       # 수직선
            "x",       # X 교차
            "/",       # 작은 원
            ".",       # 점
            "*",       # 별
        ]  # 수평 '-' 는 제외

        # 데이터 정렬 및 좌표 준비
        df_sorted = self.df.sort_values(by=x_col)
        actual_x = df_sorted[x_col].values
        positions = list(range(len(actual_x)))
        bottoms = [0] * len(actual_x)

        # ───────── 스택 막대 그리기 ─────────
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

        # x축 설정
        ax.set_xticks(positions)
        try:
            ax.set_xticklabels([str(int(v)) for v in actual_x], rotation=0)
        except Exception:
            ax.set_xticklabels([str(v) for v in actual_x], rotation=0)

        # 축 라벨·스타일
        ax.set_xlabel(xlabel if xlabel else x_col, fontsize=26, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else "", fontsize=26)
        ax.tick_params(axis="x", which="both", length=0, pad=10, labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        if y_thousands:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
            )

        # 범례 처리
        legend_handles = [
            Patch(
                facecolor=gray_sequence[i % len(gray_sequence)],
                edgecolor="black",
                hatch=hatch_sequence[i % len(hatch_sequence)],
            )
            for i in range(len(y_labels))
        ]
        if show_legend:
            leg = ax.legend(
                legend_handles,
                y_labels,
                loc=legend_location,
                frameon=True,
                edgecolor="black",
                fontsize=26,
            )
            leg.get_frame().set_alpha(1)
            legend_info = (leg.get_handles(), y_labels)
        else:
            legend_info = (
                [
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        fc=gray_sequence[i % len(gray_sequence)],
                        hatch=hatch_sequence[i % len(hatch_sequence)],
                        edgecolor="black",
                    )
                    for i in range(len(y_labels))
                ],
                y_labels,
            )

        # 제목
        if title:
            ax.text(
                0.5,
                -0.37,
                title,
                transform=ax.transAxes,
                ha="center",
                fontsize=26,
                clip_on=False,
            )
            plt.subplots_adjust(bottom=0.3)

        # 그리드·눈금 등
        ax.set_frame_on(True)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=1, color="black")

        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if y_interval is not None:
            from matplotlib.ticker import MultipleLocator

            ax.yaxis.set_major_locator(MultipleLocator(y_interval))

        if y_max is not None:
            effective_ymin = y_min if y_min is not None else 0
            offset = 0.03 * (y_max - effective_ymin)
            for patch in ax.patches:
                bar_top = patch.get_y() + patch.get_height()
                if bar_top > y_max:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        y_max + offset,
                        f"{int(round(bar_top))}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="black",
                        clip_on=False,
                    )

        # 상·하단 그리드 숨김
        fig.canvas.draw()
        ymin, ymax = ax.get_ylim()
        for line in ax.get_ygridlines():
            _, y_data = line.get_data()
            if y_data[0] <= ymin + 1e-8 or y_data[0] >= ymax - 1e-8:
                line.set_visible(False)

        return legend_info
