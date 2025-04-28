import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import io

st.title("Hourly Movement Processing and Dot Plot Generator (Split by Day)")

# --- ファイルアップロード ---
st.header("Step 1: Upload Files")
uploaded_file_a = st.file_uploader("Upload Cage1 File", type=["xlsx"])
uploaded_file_b = st.file_uploader("Upload Cage2 File", type=["xlsx"])

if uploaded_file_a and uploaded_file_b:
    df_a = pd.read_excel(uploaded_file_a)
    df_b = pd.read_excel(uploaded_file_b)

    def wide_format(df):
        return df.pivot(index="Hour", columns="ID", values="Total_Distance")

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

    def calculate_hourly_average(df):
        id_columns = df.columns.difference(['Hour', 'Phase'])
        df_grouped_sum = df.groupby("Phase")[id_columns].sum()
        phase_counts = df['Phase'].value_counts().reindex(df_grouped_sum.index)
        df_hourly_average = df_grouped_sum.div(phase_counts, axis=0)
        phase_start_times = df.groupby("Phase")["Hour"].min()
        return df_hourly_average.loc[phase_start_times.sort_values().index]

    def get_significance_asterisks(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    def plot_dotplot(dataframe, title):
        plt.figure(figsize=(14, 6))
        offset = 0.11
        phase_order = dataframe["Phase"].cat.categories

        global_max = dataframe["hourly_movement"].max()
        global_min = dataframe["hourly_movement"].min()
        asterisk_offset = (global_max - global_min) * 0.1

        for i, phase in enumerate(phase_order):
            y_a = dataframe[(dataframe["Phase"] == phase) & (dataframe["Group"] == "Cage1")]["hourly_movement"].dropna()
            y_b = dataframe[(dataframe["Phase"] == phase) & (dataframe["Group"] == "Cage2")]["hourly_movement"].dropna()

            jitter_a = np.random.uniform(-0.05, 0.05, len(y_a))
            jitter_b = np.random.uniform(-0.05, 0.05, len(y_b))

            plt.scatter(np.full(len(y_a), i + 1 - offset) + jitter_a, y_a, color="#1f77b4", label="Cage1" if i == 0 else "")
            plt.scatter(np.full(len(y_b), i + 1 + offset) + jitter_b, y_b, color="#ff7f0e", label="Cage2" if i == 0 else "")

            if len(y_a) > 1 and len(y_b) > 1:
                stat, pval = ttest_ind(y_a, y_b, equal_var=False)
                star = get_significance_asterisks(pval)
                if star:
                    ymax = max(y_a.max(), y_b.max())
                    higher_group_color = "#1f77b4" if y_a.mean() > y_b.mean() else "#ff7f0e"
                    plt.text(i + 1, ymax + asterisk_offset, star, ha='center', va='bottom', fontsize=14, color=higher_group_color)

        plt.ylim(global_min - asterisk_offset * 0.5, global_max + asterisk_offset * 1.6)
        plt.xticks(ticks=np.arange(1, len(phase_order) + 1), labels=phase_order, rotation=270)
        plt.ylabel("hourly_movement")
        plt.xlabel("Phase")
        plt.title(title)
        plt.legend(title="Group")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        return buf

    df_a_wide = wide_format(df_a)
    df_b_wide = wide_format(df_b)

    df_a_wide_reset = df_a_wide.reset_index()
    df_b_wide_reset = df_b_wide.reset_index()

    df_a_phase_added = add_phase_column(df_a_wide_reset)
    df_b_phase_added = add_phase_column(df_b_wide_reset)

    df_a_avg = calculate_hourly_average(df_a_phase_added)
    df_b_avg = calculate_hourly_average(df_b_phase_added)

    df_a_long = df_a_avg.reset_index().melt(id_vars="Phase", var_name="ID", value_name="hourly_movement")
    df_a_long["Group"] = "Cage1"
    df_b_long = df_b_avg.reset_index().melt(id_vars="Phase", var_name="ID", value_name="hourly_movement")
    df_b_long["Group"] = "Cage2"

    df_combined = pd.concat([df_a_long, df_b_long], ignore_index=True)

    df_combined["Day"] = df_combined["Phase"].str.extract(r"Day(\d+)", expand=False).astype(int)
    df_combined["PhaseType"] = df_combined["Phase"].str.extract(r"_(LightPhase|DarkPhase)", expand=False)

    split_day = st.number_input("Enter Day to Split", min_value=1, value=10)

    before_df = df_combined[df_combined["Day"] <= split_day].copy()
    after_df = df_combined[df_combined["Day"] > split_day].copy()

    def sort_phase_key(x):
        day = int(x.split("_")[0][3:])
        phase_type = 0 if "Light" in x else 1
        return (day, phase_type)

    before_categories = sorted(before_df["Phase"].unique(), key=sort_phase_key)
    after_categories = sorted(after_df["Phase"].unique(), key=sort_phase_key)

    before_df["Phase"] = pd.Categorical(before_df["Phase"], categories=before_categories, ordered=True)
    after_df["Phase"] = pd.Categorical(after_df["Phase"], categories=after_categories, ordered=True)

    st.header("Step 2: Generated Dot Plots")
    buf1 = plot_dotplot(before_df, f"Cage1 vs Cage2 (-Day{split_day})")
    buf2 = plot_dotplot(after_df, f"Cage1 vs Cage2 (Day{split_day + 1}-)")

    st.image(buf1)
    st.download_button("Download Before Split Plot", buf1, "before_split_dotplot.png", mime="image/png")

    st.image(buf2)
    st.download_button("Download After Split Plot", buf2, "after_split_dotplot.png", mime="image/png")
