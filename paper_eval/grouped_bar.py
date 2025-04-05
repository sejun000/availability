import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter

class GroupedBar:
    """
    GroupedBarPlotter 클래스는 단일 y 값을 사용한 그룹형 바 차트를 그립니다.
    - z_col이 제공되면 x와 z 기준으로 그룹핑하여 여러 바를 그리며, 
      고정된 색상 순서( dodgerblue, orange, green, violet)를 사용합니다.
    - z_col이 없으면 단순히 x축 값에 대해 y 값을 표시합니다.
    """
    def __init__(self, df):
        self.df = df

    def plot_grouped_bar(self, x_col: str, y_col: str, z_col: str = None,
                         title: str = "", xlabel: str = "", ylabel: str = "",
                         legend_title: str = "", output_file: str = None,
                         y_thousands: bool = False, ax=None, show_legend: bool = True,
                         y_min: float = None, y_max: float = None, y_interval: float = None, 
                         legend_location: str = "upper right"):
        bar_width = 0.5
        # ax가 제공되지 않으면 새 Figure와 축을 생성
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if z_col:
            pivot_df = self.df.pivot(index=x_col, columns=z_col, values=y_col)
            pivot_df = pivot_df.sort_index()
            # 고정된 색상 순서: dodgerblue, orange, green, violet
            color_sequence = ['dodgerblue', 'orange', 'green', 'violet', 'yellow', 'purple', 'pink']
            n = len(pivot_df.columns)
            colors = [color_sequence[i % len(color_sequence)] for i in range(n)]
            pivot_df.plot(kind='bar', ax=ax, color=colors,
                          width=bar_width, edgecolor='black')
            # 각 막대의 테두리를 명시적으로 검은색으로 설정
            for patch in ax.patches:
                patch.set_edgecolor('black')
            legend_labels = [f"{legend_title}={label}" for label in pivot_df.columns]
            #legend_labels = [f"{legend_title}={i+1}" for i in range(n)]
            if show_legend:
                # 범례를 상단 중앙에, 테두리 없이 표시
                leg = ax.legend(legend_labels, loc=legend_location, frameon=True, edgecolor='black', fontsize=13)
                leg.get_frame().set_alpha(1)
            else:
                # 범례 제거
                leg = ax.get_legend()
                if leg:
                    leg.remove()
        else:
            pivot_df = self.df.sort_values(by=x_col)
            pivot_df.plot(x=x_col, y=y_col, kind='bar', ax=ax,
                          color='skyblue', edgecolor='black', width=bar_width)

        #ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel if xlabel else x_col, fontsize=13, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else y_col, fontsize=13)
        ax.tick_params(axis='x', which='both', length=0, pad=10, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

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
            # 각 bar에 대해 실제 bar top 값이 y_max를 초과하면 텍스트로 실제값을 표시
            # (여기서 offset은 y_max 대비 약간의 여백, 필요에 따라 조정)
            offset = 0.02 * (y_max - (y_min if y_min is not None else 0))
            for patch in ax.patches:
                bar_top = patch.get_y() + patch.get_height()
                if bar_top > y_max:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,  # bar 중앙
                        y_max + offset,                         # y_max보다 약간 위쪽에 표시
                        f'{bar_top:.1f}',                        # 실제값 (소수점 없이)
                        ha='center', va='bottom', fontsize=10, color='black'
                    )
        # y축 grid line 객체 순회
        # 가장 첫번째와 마지막 grid line 숨기기
        index=0
        for line in ax.get_ygridlines():
            if index == 0 or index == len(ax.get_ygridlines()) - 1:
                line.set_visible(False)
            index += 1
        if (output_file):
            plt.savefig(output_file, format="pdf", dpi=600, bbox_inches='tight')
            #plt.savefig(output_file, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
