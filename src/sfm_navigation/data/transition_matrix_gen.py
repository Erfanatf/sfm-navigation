import pandas as pd

# -- Paths (adjust if needed) --
run_id = "20260611_203306"
trans_path = f"pipeline_results/{run_id}/transition_analysis/transition_matrix.csv"
rates_path = f"pipeline_results/{run_id}/transition_analysis/mood_poisson_rates.csv"
output_path = "src/sfm_navigation/data/transition_matrix_with_rates_17moods.csv"

# -- Regime code → mood name mapping --
code2mood = {
    "0_1_0_0": "Uninterrupted_speed_walker",
    "1_1_0_0": "Solo_sprinter",
    "0_1_0_1": "Alert_fast_walker_open_space",
    "0_1_1_1": "Engaged_speed_walker",
    "0_1_1_2": "Group_barging_through",
    "0_1_1_0": "Focused_rusher",
    "1_1_1_0": "Crowd_weaving_rusher",
    "1_1_0_1": "Watchful_runner",
    "1_1_1_1": "Alert_crowd_sprinter",
    "1_1_1_2": "Rushing_group_dense",
    "0_0_1_0": "Zoned_out_weaver",
    "1_0_1_2": "Group_in_a_panic",
    "1_1_2_0": "Ruthless_barger",
    "0_1_2_0": "Aggressive_barger",
    "0_1_2_1": "Stressed_pusher",
    "0_1_0_2": "Quiet_pair",
    "1_1_2_1": "Desperate_rusher",
}

# 1) Load Poisson rates (columns: regime, lambda)
rates_df = pd.read_csv(rates_path)
rate_map = dict(zip(rates_df["regime"], rates_df["lambda"]))  # regime → λ

# 2) Load full transition matrix (columns: from, to, prob)
trans = pd.read_csv(trans_path)

# 3) Keep only rows where both from and to are in our 17 regimes
mask = trans["from"].isin(code2mood.keys()) & trans["to"].isin(code2mood.keys())
filtered = trans[mask].copy()

# 4) Map regime codes to mood names
filtered["from"] = filtered["from"].map(code2mood)
filtered["to"] = filtered["to"].map(code2mood)

# 5) Attach the switching rate of the **from** mood
#    (rate is per second for leaving that mood)
filtered["rate"] = filtered["from"].map(
    lambda mood: rate_map.get({v: k for k, v in code2mood.items()}[mood], None)
)

# Reorder and save
final = filtered[["from", "to", "prob", "rate"]]
final.to_csv(output_path, index=False)

print(f"Saved {output_path}")
print("Sample rate values:")
print(final[["from", "rate"]].drop_duplicates().head(10))
