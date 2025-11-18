import argparse
import os
import json
import math
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



def safe_parse_numeric(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return np.nan
    if s.startswith("<"):
        s2 = s.lstrip("<").strip()
        try:
            return float(s2)
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def parse_pfas_values_field(field):
    out = {}
    cens = {}
    if pd.isna(field):
        return out, cens
    values = None
    if isinstance(field, (list, tuple)):
        values = field
    else:
        try:
            values = json.loads(field)
        except Exception:
            try:
                txt = str(field)
                txt2 = txt.replace("'", "\"")
                values = json.loads(txt2)
            except Exception:
                return out, cens
    if not isinstance(values, (list, tuple)):
        return out, cens
    for item in values:
        try:
            substance = item.get("substance") or item.get("name") or item.get("compound") or item.get("cas_id")
            if not substance:
                continue
            val = item.get("value", None)
            used_censored = False
            if val is None:
                if "less_than" in item:
                    val = item.get("less_than")
                    used_censored = True
                elif "limit" in item:
                    val = item.get("limit")
                    used_censored = True
                elif "operator" in item and item.get("operator") == "<" and "value" in item:
                    val = item.get("value")
                    used_censored = True
            num = safe_parse_numeric(val)
            if math.isnan(num) and isinstance(val, str) and val.strip().startswith("<"):
                num = safe_parse_numeric(val.strip().lstrip("<").strip())
                used_censored = True
            out[substance] = num
            cens[substance] = bool(used_censored and not math.isnan(num))
        except Exception:
            continue
    return out, cens

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def main(args):
    input_csv = args.input
    outdir = args.outdir
    figs_dir = os.path.join(outdir, "figures")
    ensure_dirs(outdir)
    ensure_dirs(figs_dir)

    try:
        df = pd.read_csv(input_csv, dtype=str)
    except Exception as e:
        df = pd.read_csv(input_csv, dtype=str, encoding='latin1')

    for col in ["pfas_values", "date", "year", "lat", "lon", "city", "name", "pfas_sum"]:
        if col not in df.columns:
            df[col] = np.nan

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    parsed_rows = []
    censored_rows = []
    for i, val in df['pfas_values'].items():
        parsed, cens = parse_pfas_values_field(val)
        parsed_rows.append(parsed)
        censored_rows.append(cens)
    pfas_df = pd.DataFrame(parsed_rows).fillna(np.nan)
    cens_df = pd.DataFrame(censored_rows).fillna(False).astype(bool)

    for c in pfas_df.columns:
        pfas_df[c] = pd.to_numeric(pfas_df[c], errors='coerce')

    for c in pfas_df.columns:
        if c in df.columns:
            df[c + "_pfas"] = pfas_df[c]
        else:
            df[c] = pfas_df[c]

    for c in cens_df.columns:
        colname = f"{c}_censored"
        if colname in df.columns:
            df[colname] = cens_df[c]
        else:
            df[colname] = cens_df[c]

    pfas_cols = list(pfas_df.columns)
    if len(pfas_cols) == 0:
        print("Warning: No PFAS compound columns detected in 'pfas_values'. Exiting.")
        return

    computed_sum = pfas_df.sum(axis=1, skipna=True)
    df['pfas_sum'] = pd.to_numeric(df['pfas_sum'], errors='coerce')
    df['pfas_sum'] = df['pfas_sum'].fillna(computed_sum)

    df['pfas_compound_count'] = pfas_df.notna().sum(axis=1)

    stats = OrderedDict()
    cols_for_stats = pfas_cols + ['pfas_sum']
    for col in cols_for_stats:
        series = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors='coerce')
        count_non_na = int(series.notna().sum())
        total = int(len(series))
        miss = total - count_non_na
        pct_miss = 100.0 * miss / total if total > 0 else np.nan
        stats[col] = {
            'n': count_non_na,
            'n_total': total,
            'missing': miss,
            'pct_missing': round(pct_miss, 2),
            'mean': float(series.mean(skipna=True)) if count_non_na > 0 else np.nan,
            'median': float(series.median(skipna=True)) if count_non_na > 0 else np.nan,
            'min': float(series.min(skipna=True)) if count_non_na > 0 else np.nan,
            'max': float(series.max(skipna=True)) if count_non_na > 0 else np.nan,
            'std': float(series.std(skipna=True)) if count_non_na > 1 else np.nan
        }

    top10 = df.nlargest(10, 'pfas_sum')[['name','city','lat','lon','pfas_sum','pfas_compound_count']].copy()

    geo_by_city = df.groupby('city').agg(
        samples=('pfas_sum','count'),
        mean_pfas_sum=('pfas_sum','mean'),
        max_pfas_sum=('pfas_sum','max')
    ).sort_values('mean_pfas_sum', ascending=False).reset_index()

    temp_by_year = df.groupby('year').agg(
        samples=('pfas_sum','count'),
        mean_pfas_sum=('pfas_sum','mean'),
        max_pfas_sum=('pfas_sum','max')
    ).sort_index().reset_index()

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.index.name = 'compound'
    stats_df.to_csv(os.path.join(outdir, 'pfas_summary_stats.csv'))

    top10.to_csv(os.path.join(outdir, 'top10_pfas_samples.csv'), index=False)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10,6))
    valid_loc = df.dropna(subset=['lat','lon','pfas_sum'])
    if len(valid_loc) > 0:
        sc = plt.scatter(valid_loc['lon'], valid_loc['lat'], c=valid_loc['pfas_sum'],
                    cmap='Reds', s=50, alpha=0.8)
        plt.colorbar(sc, label='PFAS_sum (ng/L)')
    plt.title('PFAS Sum by Sampling Location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(figs_dir, 'pfas_scatter_map.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.histplot(df['pfas_sum'].dropna(), bins=30, kde=True)
    plt.title('Histogram of PFAS Sum')
    plt.xlabel('PFAS_sum (ng/L)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figs_dir, 'pfas_sum_histogram.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(max(10, len(pfas_cols)*0.6),6))
    data_for_box = df[pfas_cols].copy()
    sns.boxplot(data=data_for_box, showfliers=True)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Concentration (ng/L)')
    plt.title('PFAS Concentration Distribution by Substance (linear)')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'pfas_boxplot_linear.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(max(10, len(pfas_cols)*0.6),6))
    sns.boxplot(data=np.log1p(data_for_box.fillna(0)))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('log1p(Concentration)')
    plt.title('PFAS Concentration Distribution by Substance (log1p)')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'pfas_boxplot_log1p.png'), dpi=150)
    plt.close()
    if temp_by_year['year'].notna().sum() > 0:
        plt.figure(figsize=(10,5))
        sns.lineplot(data=temp_by_year, x='year', y='mean_pfas_sum', marker='o')
        plt.title('Mean PFAS_sum by Year')
        plt.xlabel('Year')
        plt.ylabel('Mean PFAS_sum (ng/L)')
        plt.savefig(os.path.join(figs_dir, 'pfas_time_series_by_year.png'), dpi=150)
        plt.close()
    try:
        pfas_vals = df[pfas_cols].fillna(0).values
        if pfas_vals.shape[0] >= 4 and pfas_vals.shape[1] >= 1:
            scaler = StandardScaler()
            pfas_scaled = scaler.fit_transform(pfas_vals)
            k = min(6, max(2, int(math.sqrt(max(2, pfas_scaled.shape[0]//2)))))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(pfas_scaled)
            df['cluster'] = clusters
            plt.figure(figsize=(10,6))
            sns.scatterplot(data=df, x='lon', y='lat', hue='cluster', palette='tab10', s=80)
            plt.title(f'KMeans Clusters (k={k}) of Sampling Locations by PFAS profile')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend(title='cluster', loc='best')
            plt.savefig(os.path.join(figs_dir, 'pfas_kmeans_clusters.png'), dpi=150)
            plt.close()
        else:
            df['cluster'] = np.nan
    except Exception as e:
        df['cluster'] = np.nan
    df.to_csv(os.path.join(outdir, 'pfas_parsed_full.csv'), index=False)
    report_lines = []
    def add(s=""):
        report_lines.append(s)

    add("PFAS SURFACE WATER ANALYSIS REPORT")
    add("="*60)
    add(f"Input file: {input_csv}")
    add(f"Total samples (rows): {len(df)}")
    add(f"PFAS compound columns detected: {len(pfas_cols)} -> {', '.join(pfas_cols)}")
    add("")
    add("DATA CLEANING STEPS (summary):")
    add("- Read CSV and coerced lat/lon/year to numeric where possible.")
    add("- Parsed 'pfas_values' JSON field; handled 'value', 'less_than' and '<...' strings.")
    add("- Converted PFAS compound columns to numeric; missing/invalid parsed to NaN.")
    add("- Computed 'pfas_sum' from available PFAS concentrations when original missing.")
    add("")
    add("MISSING DATA SUMMARY:")
    for c, info in stats.items():
        add(f" - {c}: n={info['n']}/{info['n_total']} present, missing={info['missing']} ({info['pct_missing']}%)")
    add("")
    add("SUMMARY STATISTICS (selected):")
    add(stats_df.to_string(float_format="%.3f"))
    add("")
    add("TOP 10 SAMPLES BY PFAS_SUM:")
    if len(top10) > 0:
        add(top10.to_string(index=False, float_format="%.3f"))
    else:
        add(" - No top samples (no data).")
    add("")
    add("GEOGRAPHIC INSIGHTS (top cities by mean PFAS_sum):")
    if len(geo_by_city) > 0:
        add(geo_by_city.head(20).to_string(index=False, float_format="%.3f"))
    else:
        add(" - No geographic aggregation possible.")
    add("")
    add("TEMPORAL INSIGHTS (mean PFAS_sum per year):")
    if len(temp_by_year) > 0:
        add(temp_by_year.to_string(index=False, float_format="%.3f"))
    else:
        add(" - No temporal data (year missing).")
    add("")
    add("FILES SAVED:")
    add(f" - Summary stats CSV: {os.path.join(outdir, 'pfas_summary_stats.csv')}")
    add(f" - Top10 CSV: {os.path.join(outdir, 'top10_pfas_samples.csv')}")
    add(f" - Parsed full data CSV: {os.path.join(outdir, 'pfas_parsed_full.csv')}")
    add(f" - Figures folder: {figs_dir} (files: {', '.join(sorted(os.listdir(figs_dir)) )})")
    add(f" - Text report: {os.path.join(outdir, 'pfas_report.txt')}")
    add("")
    add("SUGGESTED NEXT STEPS / NOTES:")
    add("- Many values may be below detection limits (censored). Document how censored values were treated.")
    add("- Consider replacing censored values with 1/2 LOD or using survival-analysis / Tobit models for statistics.")
    add("- For source attribution, combine with land-use / industrial source layers and hydrology.")
    add("- For ML: consider regression for PFAS_sum using location, year, sector; clustering of PFAS fingerprints; and outlier detection.")
    add("")
    add("END OF REPORT")
    add("="*60)
    report_path = os.path.join(outdir, 'pfas_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print("\nAll outputs saved to:", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PFAS dataset analysis: parse, summarize, plot, and produce textual report.")
    parser.add_argument('--input', '-i', required=True, help='Input CSV filepath (e.g., pfas1.csv)')
    parser.add_argument('--outdir', '-o', default='output', help='Output directory to save results (default: output)')
    args = parser.parse_args()
    main(args)
