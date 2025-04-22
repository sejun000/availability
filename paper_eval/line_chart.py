import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse

parser = argparse.ArgumentParser(
        description="plot line graph"
    )

parser.add_argument("output_file", help="output erasure coding")
# 주어진 데이터
encoding_time_data = [
    # k = 1 데이터
    {'n': 2, 'k': 1, 'Encoding Time': 65},
    {'n': 3, 'k': 1, 'Encoding Time': 91},
    {'n': 4, 'k': 1, 'Encoding Time': 98},
    {'n': 5, 'k': 1, 'Encoding Time': 115},
    {'n': 6, 'k': 1, 'Encoding Time': 138},
    {'n': 7, 'k': 1, 'Encoding Time': 150},
    {'n': 8, 'k': 1, 'Encoding Time': 127},
    {'n': 9, 'k': 1, 'Encoding Time': 141},
    {'n': 10, 'k': 1, 'Encoding Time': 157},
    {'n': 11, 'k': 1, 'Encoding Time': 173},
    {'n': 12, 'k': 1, 'Encoding Time': 199},
    {'n': 13, 'k': 1, 'Encoding Time': 206},
    {'n': 14, 'k': 1, 'Encoding Time': 224},
    {'n': 15, 'k': 1, 'Encoding Time': 240},
    {'n': 16, 'k': 1, 'Encoding Time': 256},
    {'n': 17, 'k': 1, 'Encoding Time': 273},
    {'n': 18, 'k': 1, 'Encoding Time': 290},
    {'n': 19, 'k': 1, 'Encoding Time': 306},
    {'n': 20, 'k': 1, 'Encoding Time': 321},
    {'n': 21, 'k': 1, 'Encoding Time': 363},
    {'n': 22, 'k': 1, 'Encoding Time': 353},
    {'n': 23, 'k': 1, 'Encoding Time': 369},
    {'n': 24, 'k': 1, 'Encoding Time': 387},
    {'n': 25, 'k': 1, 'Encoding Time': 401},
    {'n': 26, 'k': 1, 'Encoding Time': 420},
    {'n': 27, 'k': 1, 'Encoding Time': 449},
    {'n': 28, 'k': 1, 'Encoding Time': 449},
    {'n': 29, 'k': 1, 'Encoding Time': 481},
    {'n': 30, 'k': 1, 'Encoding Time': 497},
    {'n': 31, 'k': 1, 'Encoding Time': 551},
    {'n': 32, 'k': 1, 'Encoding Time': 759},
    {'n': 33, 'k': 1, 'Encoding Time': 808},
    {'n': 34, 'k': 1, 'Encoding Time': 814},
    {'n': 35, 'k': 1, 'Encoding Time': 830},
    {'n': 36, 'k': 1, 'Encoding Time': 914},
    {'n': 37, 'k': 1, 'Encoding Time': 921},
    {'n': 38, 'k': 1, 'Encoding Time': 963},
    {'n': 39, 'k': 1, 'Encoding Time': 914},
    {'n': 40, 'k': 1, 'Encoding Time': 960},
    {'n': 41, 'k': 1, 'Encoding Time': 1004},
    {'n': 42, 'k': 1, 'Encoding Time': 1576},
    {'n': 43, 'k': 1, 'Encoding Time': 1054},
    {'n': 44, 'k': 1, 'Encoding Time': 1032},
    {'n': 45, 'k': 1, 'Encoding Time': 1054},
    {'n': 46, 'k': 1, 'Encoding Time': 1154},
    {'n': 47, 'k': 1, 'Encoding Time': 1103},
    {'n': 48, 'k': 1, 'Encoding Time': 1173},

    # k = 2 데이터
    {'n': 2, 'k': 2, 'Encoding Time': 133},
    {'n': 3, 'k': 2, 'Encoding Time': 157},
    {'n': 4, 'k': 2, 'Encoding Time': 177},
    {'n': 5, 'k': 2, 'Encoding Time': 195},
    {'n': 6, 'k': 2, 'Encoding Time': 273},
    {'n': 7, 'k': 2, 'Encoding Time': 241},
    {'n': 8, 'k': 2, 'Encoding Time': 191},
    {'n': 9, 'k': 2, 'Encoding Time': 219},
    {'n': 10, 'k': 2, 'Encoding Time': 245},
    {'n': 11, 'k': 2, 'Encoding Time': 265},
    {'n': 12, 'k': 2, 'Encoding Time': 278},
    {'n': 13, 'k': 2, 'Encoding Time': 300},
    {'n': 14, 'k': 2, 'Encoding Time': 332},
    {'n': 15, 'k': 2, 'Encoding Time': 361},
    {'n': 16, 'k': 2, 'Encoding Time': 366},
    {'n': 17, 'k': 2, 'Encoding Time': 382},
    {'n': 18, 'k': 2, 'Encoding Time': 404},
    {'n': 19, 'k': 2, 'Encoding Time': 434},
    {'n': 20, 'k': 2, 'Encoding Time': 445},
    {'n': 21, 'k': 2, 'Encoding Time': 474},
    {'n': 22, 'k': 2, 'Encoding Time': 552},
    {'n': 23, 'k': 2, 'Encoding Time': 520},
    {'n': 24, 'k': 2, 'Encoding Time': 576},
    {'n': 25, 'k': 2, 'Encoding Time': 592},
    {'n': 26, 'k': 2, 'Encoding Time': 604},
    {'n': 27, 'k': 2, 'Encoding Time': 640},
    {'n': 28, 'k': 2, 'Encoding Time': 660},
    {'n': 29, 'k': 2, 'Encoding Time': 770},
    {'n': 30, 'k': 2, 'Encoding Time': 991},
    {'n': 31, 'k': 2, 'Encoding Time': 1071},
    {'n': 32, 'k': 2, 'Encoding Time': 1151},
    {'n': 33, 'k': 2, 'Encoding Time': 1128},
    {'n': 34, 'k': 2, 'Encoding Time': 1184},
    {'n': 35, 'k': 2, 'Encoding Time': 1259},
    {'n': 36, 'k': 2, 'Encoding Time': 1299},
    {'n': 37, 'k': 2, 'Encoding Time': 1250},
    {'n': 38, 'k': 2, 'Encoding Time': 1288},
    {'n': 39, 'k': 2, 'Encoding Time': 1316},
    {'n': 40, 'k': 2, 'Encoding Time': 1364},
    {'n': 41, 'k': 2, 'Encoding Time': 1474},
    {'n': 42, 'k': 2, 'Encoding Time': 1692},
    {'n': 43, 'k': 2, 'Encoding Time': 1485},
    {'n': 44, 'k': 2, 'Encoding Time': 1525},
    {'n': 45, 'k': 2, 'Encoding Time': 1521},
    {'n': 46, 'k': 2, 'Encoding Time': 1589},
    {'n': 47, 'k': 2, 'Encoding Time': 1627},
    {'n': 48, 'k': 2, 'Encoding Time': 1679},

    # k = 3 데이터
    {'n': 2, 'k': 3, 'Encoding Time': 152},
    {'n': 3, 'k': 3, 'Encoding Time': 177},
    {'n': 4, 'k': 3, 'Encoding Time': 189},
    {'n': 5, 'k': 3, 'Encoding Time': 203},
    {'n': 6, 'k': 3, 'Encoding Time': 215},
    {'n': 7, 'k': 3, 'Encoding Time': 248},
    {'n': 8, 'k': 3, 'Encoding Time': 279},
    {'n': 9, 'k': 3, 'Encoding Time': 307},
    {'n': 10, 'k': 3, 'Encoding Time': 331},
    {'n': 11, 'k': 3, 'Encoding Time': 367},
    {'n': 12, 'k': 3, 'Encoding Time': 392},
    {'n': 13, 'k': 3, 'Encoding Time': 438},
    {'n': 14, 'k': 3, 'Encoding Time': 459},
    {'n': 15, 'k': 3, 'Encoding Time': 508},
    {'n': 16, 'k': 3, 'Encoding Time': 527},
    {'n': 17, 'k': 3, 'Encoding Time': 606},
    {'n': 18, 'k': 3, 'Encoding Time': 593},
    {'n': 19, 'k': 3, 'Encoding Time': 632},
    {'n': 20, 'k': 3, 'Encoding Time': 705},
    {'n': 21, 'k': 3, 'Encoding Time': 692},
    {'n': 22, 'k': 3, 'Encoding Time': 883},
    {'n': 23, 'k': 3, 'Encoding Time': 774},
    {'n': 24, 'k': 3, 'Encoding Time': 856},
    {'n': 25, 'k': 3, 'Encoding Time': 878},
    {'n': 26, 'k': 3, 'Encoding Time': 889},
    {'n': 27, 'k': 3, 'Encoding Time': 938},
    {'n': 28, 'k': 3, 'Encoding Time': 1242},
    {'n': 29, 'k': 3, 'Encoding Time': 1273},
    {'n': 30, 'k': 3, 'Encoding Time': 1318},
    {'n': 31, 'k': 3, 'Encoding Time': 1384},
    {'n': 32, 'k': 3, 'Encoding Time': 1488},
    {'n': 33, 'k': 3, 'Encoding Time': 1464},
    {'n': 34, 'k': 3, 'Encoding Time': 1545},
    {'n': 35, 'k': 3, 'Encoding Time': 1646},
    {'n': 36, 'k': 3, 'Encoding Time': 1677},
    {'n': 37, 'k': 3, 'Encoding Time': 1671},
    {'n': 38, 'k': 3, 'Encoding Time': 1720},
    {'n': 39, 'k': 3, 'Encoding Time': 1770},
    {'n': 40, 'k': 3, 'Encoding Time': 1809},
    {'n': 41, 'k': 3, 'Encoding Time': 1906},
    {'n': 42, 'k': 3, 'Encoding Time': 2799},
    {'n': 43, 'k': 3, 'Encoding Time': 1995},
    {'n': 44, 'k': 3, 'Encoding Time': 2025},
    {'n': 45, 'k': 3, 'Encoding Time': 2131},
    {'n': 46, 'k': 3, 'Encoding Time': 2109},
    {'n': 47, 'k': 3, 'Encoding Time': 2224},
    {'n': 48, 'k': 3, 'Encoding Time': 2203},

    # k = 4 데이터
    {'n': 2, 'k': 4, 'Encoding Time': 273},
    {'n': 3, 'k': 4, 'Encoding Time': 273},
    {'n': 4, 'k': 4, 'Encoding Time': 359},
    {'n': 5, 'k': 4, 'Encoding Time': 362},
    {'n': 6, 'k': 4, 'Encoding Time': 377},
    {'n': 7, 'k': 4, 'Encoding Time': 437},
    {'n': 8, 'k': 4, 'Encoding Time': 358},
    {'n': 9, 'k': 4, 'Encoding Time': 417},
    {'n': 10, 'k': 4, 'Encoding Time': 437},
    {'n': 11, 'k': 4, 'Encoding Time': 476},
    {'n': 12, 'k': 4, 'Encoding Time': 513},
    {'n': 13, 'k': 4, 'Encoding Time': 566},
    {'n': 14, 'k': 4, 'Encoding Time': 609},
    {'n': 15, 'k': 4, 'Encoding Time': 662},
    {'n': 16, 'k': 4, 'Encoding Time': 698},
    {'n': 17, 'k': 4, 'Encoding Time': 731},
    {'n': 18, 'k': 4, 'Encoding Time': 783},
    {'n': 19, 'k': 4, 'Encoding Time': 857},
    {'n': 20, 'k': 4, 'Encoding Time': 916},
    {'n': 21, 'k': 4, 'Encoding Time': 939},
    {'n': 22, 'k': 4, 'Encoding Time': 968},
    {'n': 23, 'k': 4, 'Encoding Time': 1023},
    {'n': 24, 'k': 4, 'Encoding Time': 1084},
    {'n': 25, 'k': 4, 'Encoding Time': 1098},
    {'n': 26, 'k': 4, 'Encoding Time': 1228},
    {'n': 27, 'k': 4, 'Encoding Time': 1246},
    {'n': 28, 'k': 4, 'Encoding Time': 1610},
    {'n': 29, 'k': 4, 'Encoding Time': 1706},
    {'n': 30, 'k': 4, 'Encoding Time': 1782},
    {'n': 31, 'k': 4, 'Encoding Time': 1853},
    {'n': 32, 'k': 4, 'Encoding Time': 1888},
    {'n': 33, 'k': 4, 'Encoding Time': 1965},
    {'n': 34, 'k': 4, 'Encoding Time': 2056},
    {'n': 35, 'k': 4, 'Encoding Time': 2069},
    {'n': 36, 'k': 4, 'Encoding Time': 2121},
    {'n': 37, 'k': 4, 'Encoding Time': 2203},
    {'n': 38, 'k': 4, 'Encoding Time': 2261},
    {'n': 39, 'k': 4, 'Encoding Time': 2321},
    {'n': 40, 'k': 4, 'Encoding Time': 2393},
    {'n': 41, 'k': 4, 'Encoding Time': 4645},
    {'n': 42, 'k': 4, 'Encoding Time': 2614},
    {'n': 43, 'k': 4, 'Encoding Time': 2657},
    {'n': 44, 'k': 4, 'Encoding Time': 2617},
    {'n': 45, 'k': 4, 'Encoding Time': 2775},
    {'n': 46, 'k': 4, 'Encoding Time': 2752},
    {'n': 47, 'k': 4, 'Encoding Time': 2901},
    {'n': 48, 'k': 4, 'Encoding Time': 2876},

    # k = 5 데이터
    {'n': 2, 'k': 5, 'Encoding Time': 341},
    {'n': 3, 'k': 5, 'Encoding Time': 362},
    {'n': 4, 'k': 5, 'Encoding Time': 386},
    {'n': 5, 'k': 5, 'Encoding Time': 453},
    {'n': 6, 'k': 5, 'Encoding Time': 528},
    {'n': 7, 'k': 5, 'Encoding Time': 589},
    {'n': 8, 'k': 5, 'Encoding Time': 474},
    {'n': 9, 'k': 5, 'Encoding Time': 608},
    {'n': 10, 'k': 5, 'Encoding Time': 589},
    {'n': 11, 'k': 5, 'Encoding Time': 680},
    {'n': 12, 'k': 5, 'Encoding Time': 716},
    {'n': 13, 'k': 5, 'Encoding Time': 807},
    {'n': 14, 'k': 5, 'Encoding Time': 831},
    {'n': 15, 'k': 5, 'Encoding Time': 908},
    {'n': 16, 'k': 5, 'Encoding Time': 939},
    {'n': 17, 'k': 5, 'Encoding Time': 1021},
    {'n': 18, 'k': 5, 'Encoding Time': 1115},
    {'n': 19, 'k': 5, 'Encoding Time': 1146},
    {'n': 20, 'k': 5, 'Encoding Time': 1179},
    {'n': 21, 'k': 5, 'Encoding Time': 1243},
    {'n': 22, 'k': 5, 'Encoding Time': 1362},
    {'n': 23, 'k': 5, 'Encoding Time': 1351},
    {'n': 24, 'k': 5, 'Encoding Time': 1447},
    {'n': 25, 'k': 5, 'Encoding Time': 1475},
    {'n': 26, 'k': 5, 'Encoding Time': 1582},
    {'n': 27, 'k': 5, 'Encoding Time': 1657},
    {'n': 28, 'k': 5, 'Encoding Time': 1743},
    {'n': 29, 'k': 5, 'Encoding Time': 2201},
    {'n': 30, 'k': 5, 'Encoding Time': 2262},
    {'n': 31, 'k': 5, 'Encoding Time': 2298},
    {'n': 32, 'k': 5, 'Encoding Time': 2596},
    {'n': 33, 'k': 5, 'Encoding Time': 2619},
    {'n': 34, 'k': 5, 'Encoding Time': 2771},
    {'n': 35, 'k': 5, 'Encoding Time': 2895},
    {'n': 36, 'k': 5, 'Encoding Time': 2983},
    {'n': 37, 'k': 5, 'Encoding Time': 3074},
    {'n': 38, 'k': 5, 'Encoding Time': 3194},
    {'n': 39, 'k': 5, 'Encoding Time': 3289},
    {'n': 40, 'k': 5, 'Encoding Time': 3446},
    {'n': 41, 'k': 5, 'Encoding Time': 5398},
    {'n': 42, 'k': 5, 'Encoding Time': 3498},
    {'n': 43, 'k': 5, 'Encoding Time': 3683},
    {'n': 44, 'k': 5, 'Encoding Time': 3681},
    {'n': 45, 'k': 5, 'Encoding Time': 3872},
    {'n': 46, 'k': 5, 'Encoding Time': 3855},
    {'n': 47, 'k': 5, 'Encoding Time': 4135},
    {'n': 48, 'k': 5, 'Encoding Time': 4156},

    # k = 6 데이터
    {'n': 2, 'k': 6, 'Encoding Time': 401},
    {'n': 3, 'k': 6, 'Encoding Time': 392},
    {'n': 4, 'k': 6, 'Encoding Time': 520},
    {'n': 5, 'k': 6, 'Encoding Time': 503},
    {'n': 6, 'k': 6, 'Encoding Time': 563},
    {'n': 7, 'k': 6, 'Encoding Time': 639},
    {'n': 8, 'k': 6, 'Encoding Time': 525},
    {'n': 9, 'k': 6, 'Encoding Time': 584},
    {'n': 10, 'k': 6, 'Encoding Time': 713},
    {'n': 11, 'k': 6, 'Encoding Time': 808},
    {'n': 12, 'k': 6, 'Encoding Time': 783},
    {'n': 13, 'k': 6, 'Encoding Time': 950},
    {'n': 14, 'k': 6, 'Encoding Time': 967},
    {'n': 15, 'k': 6, 'Encoding Time': 987},
    {'n': 16, 'k': 6, 'Encoding Time': 1053},
    {'n': 17, 'k': 6, 'Encoding Time': 1119},
    {'n': 18, 'k': 6, 'Encoding Time': 1182},
    {'n': 19, 'k': 6, 'Encoding Time': 1245},
    {'n': 20, 'k': 6, 'Encoding Time': 1303},
    {'n': 21, 'k': 6, 'Encoding Time': 1408},
    {'n': 22, 'k': 6, 'Encoding Time': 1506},
    {'n': 23, 'k': 6, 'Encoding Time': 1570},
    {'n': 24, 'k': 6, 'Encoding Time': 1582},
    {'n': 25, 'k': 6, 'Encoding Time': 1721},
    {'n': 26, 'k': 6, 'Encoding Time': 1805},
    {'n': 27, 'k': 6, 'Encoding Time': 2027},
    {'n': 28, 'k': 6, 'Encoding Time': 2261},
    {'n': 29, 'k': 6, 'Encoding Time': 2770},
    {'n': 30, 'k': 6, 'Encoding Time': 2702},
    {'n': 31, 'k': 6, 'Encoding Time': 2824},
    {'n': 32, 'k': 6, 'Encoding Time': 3092},
    {'n': 33, 'k': 6, 'Encoding Time': 3030},
    {'n': 34, 'k': 6, 'Encoding Time': 3101},
    {'n': 35, 'k': 6, 'Encoding Time': 3238},
    {'n': 36, 'k': 6, 'Encoding Time': 3452},
    {'n': 37, 'k': 6, 'Encoding Time': 3539},
    {'n': 38, 'k': 6, 'Encoding Time': 3667},
    {'n': 39, 'k': 6, 'Encoding Time': 3616},
    {'n': 40, 'k': 6, 'Encoding Time': 3741},
    {'n': 41, 'k': 6, 'Encoding Time': 5460},
    {'n': 42, 'k': 6, 'Encoding Time': 4075},
    {'n': 43, 'k': 6, 'Encoding Time': 4031},
    {'n': 44, 'k': 6, 'Encoding Time': 4336},
    {'n': 45, 'k': 6, 'Encoding Time': 4337},
    {'n': 46, 'k': 6, 'Encoding Time': 4381},
    {'n': 47, 'k': 6, 'Encoding Time': 4498},
    {'n': 48, 'k': 6, 'Encoding Time': 4788},
]

df = pd.DataFrame(encoding_time_data)

# -------------------------------------------
# Throughput(GB/s) = (1,000,000 / EncodingTime(µs)) / 1000 
#                  = 1000 / EncodingTime(µs)
# -------------------------------------------
df['Throughput(GB/s)'] = 1000 / df['Encoding Time'] * 4

# 피벗 테이블: index=n, columns=k, values=Throughput(GB/s)
pivot_df = df.pivot(index='n', columns='k', values='Throughput(GB/s)')

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 6))

markers = ['o', '^', 's', 'D', 'P', '*', 'v', 'x']  # (원하시면 다른 마커 스타일로 변경)

for idx, col in enumerate(sorted(pivot_df.columns)):
    ax.plot(
        pivot_df.index, 
        pivot_df[col], 
        marker=markers[idx % len(markers)], 
        label=f'k={col}', 
        linewidth=2
    )

# --------- 축 라벨 설정 --------- 
ax.set_xlabel('N', fontsize=26, labelpad=10)
ax.set_ylabel('Throughput (GB/s)', fontsize=26)
ax.tick_params(axis='x', which='both', length=0, pad=10, labelsize=26)
ax.tick_params(axis='y', labelsize=26)

# --------- y축 범위 / 간격 설정 ---------
ax.set_ylim(0, 70)
ax.yaxis.set_major_locator(MultipleLocator(10))  # 2 간격

# --------- 그리드(수평선) 스타일 ---------
ax.set_frame_on(True)
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', linewidth=1, color='black')

# 첫 번째/마지막 그리드 라인 숨기기 (옵션)
ygridlines = ax.get_ygridlines()
if len(ygridlines) > 2:
    ygridlines[0].set_visible(False)
    ygridlines[-1].set_visible(False)

# --------- 범례 설정 ---------
leg = ax.legend(
    loc='upper right', 
    frameon=True, 
    edgecolor='black', 
    fontsize=26, 
    ncol=2
)
leg.get_frame().set_alpha(1)
args = parser.parse_args()
# --------- 여백 조정 (왼쪽에 여유 공간) ---------
plt.subplots_adjust(left=0.18, bottom=0.18, right=0.96, top=0.96)
if (args.output_file):
    plt.savefig(args.output_file, format="pdf", dpi=600, bbox_inches='tight')
    #plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {args.output_file}")
else:
    plt.show()
