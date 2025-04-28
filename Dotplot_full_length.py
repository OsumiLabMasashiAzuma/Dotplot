import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import io

st.title("Hourly Movement Processing and Dot Plot Generator")

# --- ファイルアップロード ---
st.header("Step 1: Upload Files")
uploaded_file_a = st.file_uploader("Upload Cage1 File", type=["xlsx"])
uploaded_file_b = st.file_uploader("Upload Cage2 File", type=["xlsx"])

if uploaded_file_a and uploaded_file_b:
    # --- ファイル読み込み ---
    df_a = pd.read_excel(uploaded_file_a)
    df_b = pd.read_excel(uploaded_file_b)

    # --- データワイド化関数 ---
    def wide_format(df):
        return df.pivot(index="Hour", columns="ID", values="Total_Distance")

    # --- Phase付与関数 ---
    def add_phase_column(df):
        df = df.copy()
        df['Hour'] = pd.to_datetime(df['Hour'])
        min_date = df['Hour'].dt.date.min()

        def determine_phase(dt):
            hour = dt.hour
            date = dt.date()
            if 8 <= hour < 20:
                phase = "LightPhase"
                phase_date = date
            else:
                phase = "DarkPhase"
                if hour < 8:
                    phase_date = date - pd.Timedelta(days=1)
                else:
                    phase_date = date
            day_num = (phase_date - min_date).days + 1
            return f"Day{day_num}_{phase}"

        df['Phase'] = df['Hour'].apply(determine_phase)
        return df

    # --- 平均計算関数 ---
    def calculate_hourly_average(df):
        id_columns = df.columns.difference(['Hour', 'Phase'])
        df_grouped_sum = df.groupby("Phase")[id_columns].sum()
        phase_counts = df['Phase'].value_counts().reindex(df_grouped_sum.index)
        df_hourly_average = df_grouped_sum.div(phase_counts, axis=0)
        phase_start_times = df.groupby("Phase")["Hour"].min()
        return df_hourly_average.loc[phase_start_times.sort_values().index]

    # --- プロット関数 ---
    def generate_dot_plot(df_a_avg, df_b_avg, group_a_name, group_b_name):
        df_a_long = df_a_avg.reset_index().melt(id_vars="Phase", var_name="ID", value_name="hourly_movement")
        df_a_long["Group"] = group_a_name

        df_b_long = df_b_avg.reset_index().melt(id_vars="Phase", var_name="ID", value_name="hourly_movement")
        df_b_long["Group"] = group_b_name

        df_combined = pd.concat([df_a_long, df_b_long], ignore_index=True)

        df_combined["Phase"] = pd.Categorical(
            df_combined["Phase"],
            categories=sorted(
                df_combined["Phase"].unique(),
                key=lambda x: (int(x.split("_")[0][3:]), 0 if "Light" in x else 1)
            ),
            ordered=True
        )

        color_a = "#1f77b4"
        color_b = "#ff7f0e"

        def get_significance_asterisks(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''

        plt.figure(figsize=(14, 6))
        offset = 0.11
        phase_order = df_combined["Phase"].cat.categories

        global_max = df_combined["hourly_movement"].max()
        global_min = df_combined["hourly_movement"].min()
        asterisk_offset = (global_max - global_min) * 0.1

        for i, phase in enumerate(phase_order):
            y_a = df_combined[(df_combined["Phase"] == phase) & (df_combined["Group"] == group_a_name)]["hourly_movement"].dropna()
            y_b = df_combined[(df_combined["Phase"] == phase) & (df_combined["Group"] == group_b_name)]["hourly_movement"].dropna()

            jitter_a = np.random.uniform(-0.05, 0.05, len(y_a))
            jitter_b = np.random.uniform(-0.05, 0.05, len(y_b))

            plt.scatter(np.full(len(y_a), i + 1 - offset) + jitter_a, y_a, color=color_a, label=group_a_name if i == 0 else "")
            plt.scatter(np.full(len(y_b), i + 1 + offset) + jitter_b, y_b, color=color_b, label=group_b_name if i == 0 else "")

            if len(y_a) > 1 and len(y_b) > 1:
                stat, pval = ttest_ind(y_a, y_b, equal_var=False)
                star = get_significance_asterisks(pval)
                if star:
                    ymax = max(y_a.max(), y_b.max())
                    higher_group_color = color_a if y_a.mean() > y_b.mean() else color_b
                    plt.text(i + 1, ymax + asterisk_offset, star, ha='center', va='bottom', fontsize=14, color=higher_group_color)

        plt.ylim(global_min - asterisk_offset * 0.5, global_max + asterisk_offset * 1.6)
        plt.xticks(ticks=np.arange(1, len(phase_order) + 1), labels=phase_order, rotation=270)
        plt.ylabel("hourly_movement")
        plt.xlabel("Phase")
        plt.title(f"{group_a_name} vs {group_b_name}")
        plt.legend(title="Group")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        return buf

    # --- 正しいワイド化・Phase付与・平均化 ---
    df_a_wide = wide_format(df_a)
    df_b_wide = wide_format(df_b)

    df_a_wide_reset = df_a_wide.reset_index()
    df_b_wide_reset = df_b_wide.reset_index()

    df_a_phase_added = add_phase_column(df_a_wide_reset)
    df_b_phase_added = add_phase_column(df_b_wide_reset)

    df_a_avg = calculate_hourly_average(df_a_phase_added)
    df_b_avg = calculate_hourly_average(df_b_phase_added)

    # --- プロット生成 ---
    st.header("Step 2: Generated Dot Plot")
    plot_buf = generate_dot_plot(df_a_avg, df_b_avg, "Cage1", "Cage2")
    st.image(plot_buf)

    # --- ダウンロードリンク提供 ---
    st.download_button(
        label="Download Dot Plot Image",
        data=plot_buf,
        file_name="hourly_movement_dotplot.png",
        mime="image/png"
    )

    # --- データもダウンロードできるようにする ---
    st.header("Step 3: Download Processed Data")

    def convert_df(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=True)
        output.seek(0)
        return output

    st.download_button(
        label="Download Cage1 Processed Data",
        data=convert_df(df_a_avg),
        file_name="processed_cage1.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="Download Cage2 Processed Data",
        data=convert_df(df_b_avg),
        file_name="processed_cage2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
