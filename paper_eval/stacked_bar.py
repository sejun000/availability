import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class StackedBar:
    """
    StackedBar 클래스는 x축에 따라 여러 y값(누적)을 쌓은 stacked bar chart를 그립니다.
    y_cols와 y_labels는 각각 콤마로 구분된 문자열을 리스트로 변환한 값입니다.
    """
    def __init__(self, df):
        self.df = df

    def plot_stacked_bar(self, x_col: str, y_cols: list, y_labels: list,
                         title: str = "", xlabel: str = "", ylabel: str = "",
                         output_file: str = None, y_thousands: bool = False, ax=None, legend_location: str = "upper right",
                         y_min: float = None, y_max: float = None, y_interval: float = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
        # x축 값 기준 정렬 및 내부 x 좌표 생성
        df_sorted = self.df.sort_values(by=x_col)
        actual_x = df_sorted[x_col].values  # 실제 x 데이터 (예: [8, 16, 32, 64, 128])
        positions = list(range(len(actual_x)))  # 균일 간격을 위한 내부 좌표: [0,1,2,3,4]
        bottoms = [0] * len(actual_x)
        color_sequence = ['dodgerblue', 'orange', 'green', 'violet', 'red', 'purple']
        for i, col in enumerate(y_cols):
            y_values = df_sorted[col].values
            ax.bar(positions, y_values, bottom=bottoms, color=color_sequence[i % len(color_sequence)],
                   edgecolor='black', width=0.5)
            bottoms = [b + y for b, y in zip(bottoms, y_values)]
        # x축 tick 설정: 내부 좌표를 tick 위치로, 실제 x값을 label로 (정수형으로 변환)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(val)) for val in actual_x], rotation=0)
        
        ax.set_xlabel(xlabel if xlabel else x_col, fontsize=13, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else "", fontsize=13)
        ax.tick_params(axis='x', which='both', length=0, pad=10, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        if y_thousands:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        leg = ax.legend(y_labels, loc=legend_location, frameon=True, edgecolor='black', fontsize=13)
        leg.get_frame().set_alpha(1)
        if title:
            ax.text(0.5, -0.27, title, transform=ax.transAxes,
                    ha='center', fontsize=13)
            plt.subplots_adjust(bottom=0.25)
        ax.set_frame_on(True)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1, color='black')
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if y_interval is not None:
            from matplotlib.ticker import MultipleLocator
            ax.yaxis.set_major_locator(MultipleLocator(y_interval))
        return ax
