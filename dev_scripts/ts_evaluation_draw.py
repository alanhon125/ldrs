import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict

REVIEWED_TS_VERSION = 'v4'
TS_VERS = ['v4','v4.1','v4.2','v4.3','v4.4','v5','v5.1','v5.2','v5.3','v5.4','v5.4.1']
OUTPUT_FOLDER = f"/home/data2/ldrs_analytics/data/ts_evaluation_output/antd_{REVIEWED_TS_VERSION}"
PARSED_TS_FOLDERS = ["TS_"+i for i in TS_VERS]
THRESHOLDS = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
ANTD_TS_FOLDER = f"/home/data2/ldrs_analytics/data/reviewed_antd_ts_{REVIEWED_TS_VERSION}"

data = defaultdict(list)
total = 0

for file in [f for f in os.listdir(ANTD_TS_FOLDER) if f!= 'all_merge_results.csv']:
    df_gt = pd.read_csv(os.path.join(ANTD_TS_FOLDER, file))
    df_gt = df_gt[df_gt[["clause_id", "definition", "schedule_id"]].notna().any(axis=1)]
    total += len(df_gt.groupby("section").groups)

print(total)

for thres in THRESHOLDS:

    for parsed_ts_folder in PARSED_TS_FOLDERS:
        folders = os.listdir(OUTPUT_FOLDER + "_" + parsed_ts_folder)
        BoW_less_ratio_list = list()
        distance_less_ratio_list = list()

        df_valid = pd.read_csv(
            os.path.join(OUTPUT_FOLDER + "_" + parsed_ts_folder, "total_valid.csv")
        )
        df_failed = pd.read_csv(
            os.path.join(OUTPUT_FOLDER + "_" + parsed_ts_folder, "total_failed.csv")
        )
        df_valid['ts_version'] = parsed_ts_folder
        df_valid['difference_BoW_less'] = df_valid['difference_BoW_less'].map(lambda x: float(x.rstrip('%'))/100)
        df_valid['difference_distance_less'] = df_valid['difference_distance_less'].map(lambda x: float(x.rstrip('%'))/100)

        BoW_less_ratio_list = df_valid.difference_BoW_less.to_list()
        distance_less_ratio_list = df_valid.difference_distance_less.to_list()

        df_failed = df_failed[~df_failed.section.isnull()]
        failed_cases = len(df_failed)

        total_BoW = len(BoW_less_ratio_list)
        total_distance = len(distance_less_ratio_list)

        BoW_less_ratio_list = [r for r in BoW_less_ratio_list if r >= thres]
        distance_less_ratio_list = [r for r in distance_less_ratio_list if r >= thres]

        Average_BoW_less_ratio = round(
            (sum(BoW_less_ratio_list) + failed_cases) / total * 100, 2
        )
        Average_distance_less_ratio = round(
            (sum(distance_less_ratio_list) + failed_cases) / total * 100, 2
        )

        if thres == 0:
            print(parsed_ts_folder)
            print(failed_cases)
            print("ratio:", failed_cases / total)

        data["TS_version"].append(parsed_ts_folder)
        data["Average_BoW_less_ratio"].append(Average_BoW_less_ratio)
        data["Count_BoW_less_ratio"].append(sum(BoW_less_ratio_list))
        data["Average_distance_less_ratio"].append(Average_distance_less_ratio)
        data["Count_distance_less_ratio"].append(sum(distance_less_ratio_list))
        data["Num_of_sections_fail_to_extract"].append(failed_cases)
        data["Threshold"].append(str(thres))

df = pd.DataFrame(data)
df.to_csv(f"/home/data2/ldrs_analytics/data/ts_evaluation_output/TS_versions_VS_antdTS_{REVIEWED_TS_VERSION}.csv",index=False, encoding='utf-8-sig')

y_labels = ["Average_distance_less_ratio", "Average_BoW_less_ratio"]
num_y_labels = len(y_labels)

fig, axes = plt.subplots(num_y_labels, 1, sharex=True, figsize=(10, 5 * num_y_labels))

if num_y_labels == 1:
    axes = [axes]

palette = list()
for i in range(len(THRESHOLDS)):
    palette.append(sns.color_palette("flare")[i])

for i, y_label in enumerate(y_labels):
    g = sns.lineplot(
        data=df,
        x="TS_version",
        y=y_label,
        hue="Threshold",
        style="Threshold",
        markers=True,
        dashes=False,
        palette=palette,
        ax=axes[i],
    )

    for item, color in zip(df.groupby("Threshold"), palette):
        # item[1] is a grouped data frame
        for x, y in item[1][["TS_version", y_label]].values:
            axes[i].text(x, y + 0.06, f"{y:.2f}", color=color)

    g.set(ylabel=None)
    axes[i].set_title(y_label)

    if i == 0:
        sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1, 1))
    else:
        axes[i].get_legend().remove()
fig.savefig(f"/home/data2/ldrs_analytics/data/ts_evaluation_output/TS_versions_VS_antdTS_{REVIEWED_TS_VERSION}.svg", bbox_inches="tight")
