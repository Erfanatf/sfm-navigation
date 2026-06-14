import sys
import argparse
from ..data.atc_loader import load_atc_raw
from ..crowd_analysis.binning import convert_units, bin_data, select_bin_data
from ..crowd_analysis.visualization import create_crowd_animation

def main():
    parser = argparse.ArgumentParser(description="ATC crowd analysis and animation")
    parser.add_argument('--csv', type=str, default='/home/erfanatf/Documents/notebooks/content/drive/MyDrive/ATC_data/atc-20121114.csv',
                        help="Path to the big ATC CSV file (e.g., atc-20121114.csv)")
    parser.add_argument('--bin', type=int, default=4,
                        help="Index of the bin to display (0 = most crowded)")
    parser.add_argument('--subsample', type=int, default=5,
                        help="Frame subsampling factor (1 = no subsampling)")
    args = parser.parse_args()

    df_raw = load_atc_raw(args.csv)
    df = convert_units(df_raw)
    bin_stats_df = bin_data(df)
    df_bin, longest_id = select_bin_data(df, bin_stats_df, selected_bin_index=args.bin)

    fig = create_crowd_animation(df_bin, longest_id, frame_subsample=args.subsample)

    # Save and open in browser
    html_path = "atc_crowd_animation.html"
    fig.write_html(html_path)
    print(f"\nAnimation saved to {html_path}")
    try:
        import webbrowser
        webbrowser.open(html_path, new=1)
        print("Browser window should open now with the interactive animation.")
    except Exception:
        print("Please open the file manually.")

if __name__ == "__main__":
    main()