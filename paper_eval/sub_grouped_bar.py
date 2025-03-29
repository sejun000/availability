import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter

class SubGroupedBar:
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
                     y_thousands: bool = False, ax=None, show_legend: bool = True, local_legend: bool = False, legend_location: str = "upper right",
                     y_min: float = None, y_max: float = None, y_interval: float = None):
        bar_width = 0.5
        global_legend = None  # global legend 정보를 담을 변수
        
        # ax가 제공되지 않으면 새 Figure와 축을 생성 (subgraph 배치는 main에서 처리)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if z_col:
            pivot_df = self.df.pivot(index=x_col, columns=z_col, values=y_col)
            pivot_df = pivot_df.sort_index()
            # 고정된 색상 순서: dodgerblue, orange, green, violet
            color_sequence = ['dodgerblue', 'orange', 'green', 'violet']
            n = len(pivot_df.columns)
            colors = [color_sequence[i % len(color_sequence)] for i in range(n)]
            pivot_df.plot(kind='bar', ax=ax, color=colors,
                          width=bar_width, edgecolor='black')
            # 각 막대의 테두리를 명시적으로 검은색으로 설정
            for patch in ax.patches:
                patch.set_edgecolor('black')
            if show_legend:
                # 범례 정보를 캡쳐하고, 자동 생성된 범례는 제거합니다.
                handles, labels = ax.get_legend_handles_labels()
                if local_legend == False:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                else:
                    legend_labels = [f"{legend_title}={i+1}" for i in range(n)]
                    leg = ax.legend(legend_labels, loc=legend_location, frameon=True, edgecolor='black', fontsize=13)
                    leg.get_frame().set_alpha(1)
                global_legend = (handles, labels)
            else:
                # 범례 제거
                leg = ax.get_legend()
                if leg:
                    leg.remove()

        else:
            pivot_df = self.df.sort_values(by=x_col)
            pivot_df.plot(x=x_col, y=y_col, kind='bar', ax=ax,
                          color='skyblue', edgecolor='black', width=bar_width)

        # 기존의 서브플롯 제목은 제거하고, 제목은 서브플롯 아래에 텍스트로 표시합니다.
        if title:
            # y 좌표는 필요에 따라 조정 (여기서는 -0.15로 설정)
            ax.text(0.5, -0.27, title, transform=ax.transAxes,
                    ha='center', fontsize=13)
            plt.subplots_adjust(bottom=0.25)
       # else:
        #    leg = ax.get_legend()
        #    if leg:
        #        leg.remove()

        ax.set_xlabel(xlabel if xlabel else x_col, fontsize=13, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else y_col, fontsize=13)
        ax.tick_params(axis='x', which='both', length=0, pad=10, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_xticks(range(len(pivot_df.index)))  
        ax.set_xticklabels([str(int(val)) for val in pivot_df.index], rotation=0)
        if y_thousands:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        ax.set_frame_on(True)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1, color='black')
        # ... 기존 코드 끝부분 아래에 추가
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
                        f'{bar_top:.0f}',                        # 실제값 (소수점 없이)
                        ha='center', va='bottom', fontsize=10, color='black'
                    )
        return global_legend
