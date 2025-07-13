import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch                # 범례용 패치

class GroupedBar:
    """
    GroupedBarPlotter 클래스는 단일 y 값을 사용한 그룹형 바 차트를 그립니다.
    - z_col이 제공되면 x와 z 기준으로 그룹핑하여 여러 바를 그리며,
      회색조 + 패턴(hatch)을 활용해 흑백 인쇄에도 구분 가능하도록 함.
    - z_col이 없으면 단순히 x축 값에 대해 y 값을 표시합니다.
    """
    def __init__(self, df):
        self.df = df

    def reorder_legend(self, list, ncol):
        new_list = []
        for i in range(ncol):
            for j in range((len(list) + ncol - 1)//ncol):
                if i + j*ncol < len(list):
                    new_list.append(list[i + j*ncol])
        return new_list

    def reorder_handles(self, hl, nc):
        return sum((hl[i::nc] for i in range(nc)), [])

    def plot_grouped_bar(self, x_col: str, y_col: str, z_col: str = None,
                         title: str = "", xlabel: str = "", ylabel: str = "",
                         legend_title: str = "", output_file: str = None,
                         y_thousands: bool = False, ax=None, show_legend: bool = True,
                         y_min: float = None, y_max: float = None, y_interval: float = None, 
                         legend_location: str = "upper right", legend_type: str = "equal", ext_legend_labels: list = None):
        bar_width = 0.5

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if z_col:
            pivot_df = self.df.pivot(index=x_col, columns=z_col, values=y_col).sort_index()

            # ────────── 색상: 밝은 회색 ↔︎ 진한 회색(검정 포함) + 패턴 정의 ──────────
            #gray_sequence  = ['#f0f0f0', '#cccccc', '#999999', '#666666',
            #                  '#4d4d4d', '#333333', '#000000']               ### ← 수정
            gray_sequence  = ['#aec7e8', '#ffbb78', '#98df8a', '#f7b6d2',
                              '#c5b0d5', '#c7c7c7', '#ffed6f']
            hatch_sequence = ['', '/', '\\\\', 'x', 'o', '.', '*', '|'] ### ← 수정

            n_groups = len(pivot_df.columns)
            n_x      = len(pivot_df.index)

            # 일단 색 지정(무시될 수 있음) 후 개별 패치에서 덮어쓰기
            pivot_df.plot(kind='bar', ax=ax,
                          color=gray_sequence[:n_groups],                  ### ← 수정
                          width=bar_width, edgecolor='black')

            # 각 막대별 facecolor & hatch 재설정
            for idx, patch in enumerate(ax.patches):
                col_idx = idx // n_x
                patch.set_facecolor(gray_sequence[col_idx % len(gray_sequence)])  ### ← 수정
                patch.set_hatch(hatch_sequence[col_idx % len(hatch_sequence)])    ### ← 수정

            # 범례 레이블
            if ext_legend_labels is None:
                if legend_type == "equal":
                    legend_labels = [f"{legend_title}={label}" for label in pivot_df.columns]
                else:
                    legend_labels = [f"{label} {legend_title}" if label != 0 else "No TLC"
                                     for label in pivot_df.columns]
            else:
                legend_labels = [f"{label}" for label in ext_legend_labels]

            if show_legend:
                legend_handles = [
                    Patch(facecolor=gray_sequence[i % len(gray_sequence)],       ### ← 수정
                          edgecolor='black',
                          hatch=hatch_sequence[i % len(hatch_sequence)])
                    for i in range(n_groups)
                ]
                if ext_legend_labels is None:
                    leg = ax.legend(self.reorder_handles(legend_handles, 2),
                                    self.reorder_legend(legend_labels, 2),
                                    loc=legend_location, frameon=True,
                                    edgecolor='black', fontsize=26, ncol=2)
                else:
                    leg = ax.legend(legend_handles, legend_labels,
                                    loc=legend_location, frameon=True,
                                    edgecolor='black', fontsize=26, ncol=2)
                leg.get_frame().set_alpha(1)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
        else:
            # z_col이 없으면 단일 막대 (회색 + 패턴)
            pivot_df = self.df.sort_values(by=x_col)
            pivot_df.plot(x=x_col, y=y_col, kind='bar', ax=ax,
                          color='#999999', edgecolor='black', hatch='//',  ### ← 수정
                          width=bar_width)

        # ────────── 공통 설정 ──────────
        ax.set_xlabel(xlabel if xlabel else x_col, fontsize=26, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else y_col, fontsize=26)
        ax.tick_params(axis='x', which='both', length=0, pad=10, labelsize=26)
        ax.tick_params(axis='y', labelsize=26)

        if y_thousands:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{int(x):,}'))

        ax.set_xticks(range(len(pivot_df.index)))
        ax.set_xticklabels([
            str(int(val)) if abs(val - int(val)) < 1e-9 else str(val)
            for val in pivot_df.index
        ], rotation=0)

        ax.set_frame_on(True)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1, color='black')

        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if y_interval is not None:
            from matplotlib.ticker import MultipleLocator
            ax.yaxis.set_major_locator(MultipleLocator(y_interval))

        if y_max is not None:
            offset = 0.02 * (y_max - (y_min if y_min is not None else 0))
            for patch in ax.patches:
                bar_top = patch.get_y() + patch.get_height()
                if bar_top > y_max:
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            y_max + offset,
                            f'{bar_top:.1f}', ha='center', va='bottom',
                            fontsize=20, color='black')

        plt.subplots_adjust(top=0.96, bottom=0.18, left=0.14, right=0.96)

        for idx, line in enumerate(ax.get_ygridlines()):
            if idx == 0 or idx == len(ax.get_ygridlines()) - 1:
                line.set_visible(False)

        if output_file:
            plt.savefig(output_file, format="pdf", dpi=600, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
