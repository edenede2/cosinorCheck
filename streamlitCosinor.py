import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
from CosinorPy import cosinor
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit

# Initialize session state variables
if 'analysed' not in st.session_state:
    st.session_state.analysed = False

if 'results' not in st.session_state:
    st.session_state.results = None

if 'selected_window_label' not in st.session_state:
    st.session_state.selected_window_label = None

def generate_fixed_windows(start_datetime, end_datetime, window_size_hours, window_shift_hours):
    """
    Generate time windows starting every window_shift_hours, with specified window sizes.
    """
    windows = []
    window_id = 0
    delta_window = timedelta(hours=window_size_hours)
    delta_shift = timedelta(hours=window_shift_hours)

    # Adjust current_start to the next multiple of window_shift_hours
    current_start = start_datetime.replace(minute=0, second=0, microsecond=0)
    remainder_hours = current_start.hour % window_shift_hours
    if remainder_hours != 0:
        # Move to the next window_shift_hours increment
        current_start += timedelta(hours=window_shift_hours - remainder_hours)

    while current_start <= end_datetime:
        current_end = current_start + delta_window - timedelta(seconds=1)
        label = f"{current_start.strftime('%Y-%m-%d %H:%M:%S')} to {current_end.strftime('%Y-%m-%d %H:%M:%S')}"
        windows.append({
            'window_id': window_id,
            'start': current_start,
            'end': current_end,
            'label': label
        })
        current_start += delta_shift
        window_id += 1

    return windows

def first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal):
    """
    Preprocess the data by selecting relevant columns, parsing dates,
    and applying filters based on user choices.
    """
    if signal == "BpmMean":
        col = "BpmMean"
    elif signal == "StepsMean":
        col = "StepsInMinute"

    df = (
        pl.DataFrame(dataframe)
        .select([
            "DateAndMinute",
            col,
            "not_in_israel",
            "is_dst_change",
        ])
        .with_columns(
            pl.col("DateAndMinute").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        )
    )

    if remove_not_in_IL:
        df = df.filter(pl.col("not_in_israel") == False)

    if remove_dst_change:
        df = df.filter(pl.col("is_dst_change") == False)

    return df

def downsample_signal(df: pl.DataFrame, window_size: str, signal: str) -> pd.DataFrame:
    """
    Downsample the signal data using the specified window size.
    """
    if signal == "BpmMean":
        col = "BpmMean"
    elif signal == "StepsMean":
        col = "StepsInMinute"

    grouped = df.group_by_dynamic(
        "DateAndMinute",
        every=window_size,
        period=window_size,
        closed='left',
        label='left'
    ).agg([
        pl.col(col).count().alias("count_non_missing"),
        pl.col(col).is_null().sum().alias("count_missing"),
        pl.col(col).mean().alias("mean"),
    ])

    downsampled = grouped.with_columns(
        pl.when(pl.col("count_missing") > pl.col("count_non_missing"))
          .then(None)
          .otherwise(pl.col("mean"))
          .alias("y")
    ).select([
        pl.col("DateAndMinute"),
        pl.col("y")
    ])

    downsampled_df = downsampled.to_pandas()
    return downsampled_df

def cosinor_analysis(data: pd.DataFrame, period: int):
    """
    Perform cosinor analysis on the downsampled data.
    """
    dates = data['test'].unique()
    results = {}

    for date in dates:
        data_for_date = data[data['test'] == date]
        data_for_date = data_for_date.dropna(subset=['y'])

        if 'interpolated_y' in data_for_date.columns:
            data_for_date['y'] = data_for_date['interpolated_y']

        results[date] = cosinor.fit_me(
            data_for_date['x'],
            data_for_date['y'],
            n_components=1,
            period=period,
            plot=False,
            return_model=True,
            params_CI=True
        )

    return results

def generate_polar_ticks(analysis_window_size_hours):
    """
    Generate ticks for the polar plot based on the analysis window size.
    """
    total_hours = analysis_window_size_hours
    num_ticks = 8
    tick_interval = total_hours / num_ticks

    hours = []
    hours_deg = []

    for i in range(num_ticks + 1):
        hour = i * tick_interval
        deg = (i * 360) / num_ticks
        hour_int = int(hour % 24)
        day = int(hour // 24) + 1
        label = f"Day {day} {hour_int:02d}:00"
        hours.append(label)
        hours_deg.append(deg)

    return hours, hours_deg

def plot_cosinor(results, original_data, window_size, window_label, analysis_window_size):
    """
    Plot the cosinor analysis results for a single window.
    """
    data = results[window_label]
    original_data = original_data[original_data['test'] == window_label]

    x_data = [(x * window_size) / 60 for x in original_data['x']]
    y_data = original_data['y']
    y_data_interpolated = original_data['interpolated_y'] if 'interpolated_y' in original_data.columns else None

    x_estimated = [(x * window_size) / 60 for x in data[3]]
    y_estimated = data[4]

    fig = go.Figure()

    if y_data_interpolated is not None:
        fig.add_trace(go.Scatter(x=x_data, y=y_data_interpolated, mode='markers', name='Interpolated Data'))

    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Original Data'))
    fig.add_trace(go.Scatter(x=x_estimated, y=y_estimated, mode='lines', name='Estimated Data'))

    fig.update_layout(title='Cosinor Analysis', xaxis_title='Time [hours]', yaxis_title='Value')

    st.plotly_chart(fig)

    amplitude = data[2]['amplitude']
    acrophase = data[2]['acrophase']
    mesor = data[2]['mesor']

    center_r = [0, amplitude]
    center_theta = [0, np.rad2deg(acrophase)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[amplitude],
        theta=[np.rad2deg(acrophase)],
        mode='markers',
        marker=dict(
            color='red',
            size=10
        ),
        name='Amplitude and Acrophase'
    ))

    fig.add_trace(go.Scatterpolar(
        r=center_r,
        theta=center_theta,
        mode='lines',
        line=dict(
            color='green',
            width=2
        ),
        name='Radius Line'
    ))

    hours, hours_deg = generate_polar_ticks(analysis_window_size)

    fig.update_layout(
        title=f'Cosinor Analysis - Window: {window_label}, Mesor: {mesor}',
        polar=dict(
            angularaxis=dict(
                tickmode='array',
                tickvals=hours_deg,
                ticktext=hours,
                direction='clockwise',
                rotation=0,
                thetaunit='degrees'
            ),
        )
    )

    st.plotly_chart(fig)

def all_dates_plot(results, analysis_window_size):
    """
    Plot the cosinor analysis results for all windows together.
    """
    fig = go.Figure()

    for key in results.keys():
        data = results[key]
        amplitude = data[2]['amplitude']
        acrophase = data[2]['acrophase']

        fig.add_trace(go.Scatterpolar(
            r=[amplitude],
            theta=[np.rad2deg(acrophase)],
            mode='markers',
            marker=dict(
                size=10
            ),
            name=key
        ))

    hours, hours_deg = generate_polar_ticks(analysis_window_size)

    fig.update_layout(
        title='Cosinor Analysis - All Windows',
        polar=dict(
            angularaxis=dict(
                tickmode='array',
                tickvals=hours_deg,
                ticktext=hours,
                direction='clockwise',
                rotation=0,
                thetaunit='degrees'
            ),
        )
    )

    st.plotly_chart(fig)

def quadrant_adjustment(theta, acrophase):
    """
    Adjust the acrophase based on the quadrant it falls into.
    """
    if 0 <= theta < (np.pi / 2):
        corrected_acrophase = np.rad2deg(acrophase)
    elif (np.pi / 2) <= theta < np.pi:
        corrected_acrophase = np.rad2deg(acrophase)
    elif np.pi <= theta < (3 * np.pi / 2):
        corrected_acrophase = 360 - np.rad2deg(acrophase)
    elif (3 * np.pi / 2) <= theta < (2 * np.pi):
        corrected_acrophase = 360 - np.rad2deg(acrophase)
    else:
        corrected_acrophase = acrophase % (2 * np.pi)
    return corrected_acrophase

def download_results(results, original_data):
    """
    Compile the cosinor analysis results into a DataFrame and provide a download option.
    """
    columns = ['window_label', 'amplitude','period','acrophase (rad)', 'corrected_acrophase (rad)',
                'corrected_acrophase (hours)', 'corrected_acrophase (degrees)',
                'mesor','AIC', 'BIC','p-value', 'p_reject', 'SNR', 'RSS',
                'resid_SE', 'ME','f-pvalue', 't-values const', 't-values x1',
                't-values x2','R-squared', 'R-squared adjusted', 'SSR', 'minutes_based']

    results_df = pl.DataFrame({col: [] for col in columns})

    for key in results.keys():
        model = results[key]
        cosinor_model = model[0]
        stats = model[1]
        params = model[2]
        original_data1 = original_data[original_data['test'] == key]

        peak_indices = params['peaks'] if len(params['peaks']) > 0 else [np.nan]
        theta = peak_indices[0]/params['period'] * 2 * np.pi
        corrected_acrophase_deg = quadrant_adjustment(theta, params['acrophase'])

        corrected_acrophase = np.deg2rad(corrected_acrophase_deg)

        total_hours = (corrected_acrophase_deg/360) * params['period'] * (1/60)
        hours = int(total_hours)
        minutes = int((total_hours - hours) * 60)
        corrected_acrophase_time = f"{hours:02d}:{minutes:02d}"

        cosinor_model_params = {
            'window_label': [key],
            'amplitude': [float(params['amplitude'])],
            'period': [float(params['period'])],
            'acrophase (rad)': [float(params['acrophase'])],
            'corrected_acrophase (rad)': [float(corrected_acrophase)],
            'corrected_acrophase (hours)': [corrected_acrophase_time],
            'corrected_acrophase (degrees)': [float(corrected_acrophase_deg)],
            'mesor': [float(params['mesor'])],
            'AIC': [float(cosinor_model.aic)],
            'BIC': [float(cosinor_model.bic)],
            'p-value': [float(stats['p'])],
            'p_reject': [bool(stats['p_reject'])],
            'SNR': [float(stats['SNR'])],
            'RSS': [float(stats['RSS'])],
            'resid_SE': [float(stats['resid_SE'])],
            'ME': [float(stats['ME'])],
            'f-pvalue': [float(cosinor_model.f_pvalue)],
            't-values const': [float(cosinor_model.tvalues[0])],
            't-values x1': [float(cosinor_model.tvalues[1])],
            't-values x2': [float(cosinor_model.tvalues[2])],
            'R-squared': [float(cosinor_model.rsquared)],
            'R-squared adjusted': [float(cosinor_model.rsquared_adj)],
            'SSR': [float(cosinor_model.ssr)],
            'minutes_based': [int(original_data1['y'].notna().sum())]
        }
        df = pl.DataFrame(cosinor_model_params, strict=False)
        results_df = pl.concat([results_df, df], how='vertical_relaxed')

    st.write("Download the results")

    show_results = st.checkbox("Show Results")

    if show_results:
        st.write(results_df.to_pandas())

    st.write("Download the results")

    file_name = st.text_input("Enter the file name", placeholder="cosinor_results.csv")

    csv = results_df.to_pandas().to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name=file_name,
        mime="text/csv"
    )

def sinusoidal_model(x, amplitude, frequency, phase, offset):
    """
    Define the sinusoidal model for interpolation.
    """
    return amplitude * np.sin(frequency * x + phase) + offset

def interpolate_data(data, method):
    """
    Interpolate missing data using the specified method.
    """
    data = pl.DataFrame(data)
    y = data["y"].to_numpy()
    x = np.arange(len(y))
    isnan = np.isnan(y)

    if np.any(isnan):
        x_valid = x[~isnan]
        y_valid = y[~isnan]

        if method == "sinusoidal":
            interpolated_values = np.interp(x, x_valid, y_valid)
            interpolated_values[~isnan] = y[~isnan]
            interpolated_values[isnan] = [(x+1)*np.median(y_valid) for x in np.cos(interpolated_values[isnan])]

        elif method == "sinusoidal (curve-fit)":
            initial_guess = [np.std(y_valid), 2 * np.pi / len(y_valid), 0, np.mean(y_valid)]
            try:
                params, _ = curve_fit(sinusoidal_model, x_valid, y_valid, p0=initial_guess)
                interpolated_values = y.copy()
                interpolated_values[isnan] = sinusoidal_model(x[isnan], *params)
            except RuntimeError:
                print("Curve fitting did not converge; consider refining the initial guess.")
                interpolated_values = y

        elif method == "polynomial":
            poly_coeff = np.polyfit(x_valid, y_valid, 2)
            poly_interp = np.polyval(poly_coeff, x)
            interpolated_values = y.copy()
            interpolated_values[isnan] = poly_interp[isnan]

        elif method == "spline-cubic":
            cubic_spline = CubicSpline(x_valid, y_valid)
            spline_interp = cubic_spline(x)
            interpolated_values = y.copy()
            interpolated_values[isnan] = spline_interp[isnan]

        elif method == "spline-quadratic":
            spline_quadratic = UnivariateSpline(x_valid, y_valid, k=2)
            spline_interp = spline_quadratic(x)
            interpolated_values = y.copy()
            interpolated_values[isnan] = spline_interp[isnan]

        data = data.with_columns(interpolated_y=pl.Series(interpolated_values)).to_pandas()

    return data

def main():
    st.title("Cosinor Analysis App")

    st.write("This app is built for testing cosinor parameters on your data.")
    st.write("We will use the \"subname Heart Rate and Steps and Sleep Aggregated.csv\" file for this analysis.")

    st.divider()

    st.write("Please upload the file")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.write("Data uploaded successfully")
            st.write(dataframe.head())

            required_columns = ["DateAndMinute", "BpmMean", "StepsInMinute", "not_in_israel", "is_dst_change"]
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            if missing_columns:
                st.error(f"The following required columns are missing from the uploaded file: {missing_columns}")
                return

            st.divider()

            st.write("Data is valid")

            st.write("Preprocessing the data")

            remove_not_in_IL = st.checkbox("Remove data when the subject wasn't in Israel?")

            remove_dst_change = st.checkbox("Remove data during DST change?")

            signal = st.selectbox("Select the signal to analyze", ["BpmMean", "StepsMean"])

            first_preprocess = first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal)

            st.write("Preprocessing done")

            show_preprocess = st.checkbox("Show preprocessed data")

            if show_preprocess:
                st.write(first_preprocess)

            st.write("Select the window size for analysis (in hours)")

            analysis_window_size = st.selectbox("Select the analysis window size (hours)", [24, 48, 72, 96, 120])

            st.write("Select the window shift size (in hours)")

            window_shift_size = st.selectbox("Select the window shift size (hours)", [12, 24])

            start_datetime = first_preprocess.select(pl.col("DateAndMinute").min()).item()
            end_datetime = first_preprocess.select(pl.col("DateAndMinute").max()).item()

            windows = generate_fixed_windows(start_datetime, end_datetime, analysis_window_size, window_shift_size)

            st.write(f"Generated {len(windows)} windows starting every {window_shift_size} hours with size {analysis_window_size} hours.")

            st.write("Select the window size for downsampling")

            window_size = st.selectbox("Select the window size", ["1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h"])

            win_size_int = {"1m": 1, "2m": 2, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "2h": 120}

            st.write("Select the tolerance for missing data per window")

            missing_tolerance = st.slider("Select the tolerance for missing data per window (percentage)", 0, 100, 10)

            st.write("Processing the data per window")

            interpolated = st.checkbox("Interpolate the missing data")

            if interpolated:
                st.write("Select the method for interpolation")

                method = st.selectbox("Select the method for interpolation", ["sinusoidal", "sinusoidal (curve-fit)", "polynomial", "spline-cubic", "spline-quadratic"])

            results = {}
            downsampled_data_list = []

            for window in windows:
                window_label = window['label']
                window_start = window['start']
                window_end = window['end']

                st.write(f"Processing window {window_label}")

                data_in_window = first_preprocess.filter(
                    (pl.col("DateAndMinute") >= window_start) & (pl.col("DateAndMinute") <= window_end)
                )

                total_points = data_in_window.shape[0]
                missing_points = data_in_window.filter(pl.col(signal).is_null()).shape[0]
                missing_percentage = (missing_points / total_points) * 100 if total_points > 0 else 100

                st.write(f"Missing data in window: {missing_percentage:.2f}%")

                if missing_percentage > missing_tolerance:
                    st.write(f"Skipping window due to high missing data.")
                    continue

                downsampled = downsample_signal(data_in_window, window_size, signal)

                if interpolated:
                    downsampled = interpolate_data(downsampled, method)

                period = (analysis_window_size * 60) / win_size_int[window_size]

                downsampled['x'] = np.arange(len(downsampled))
                downsampled['test'] = window_label

                cosinor_result = cosinor_analysis(downsampled, period)

                results[window_label] = cosinor_result[window_label]

                downsampled_data_list.append(downsampled)

            if downsampled_data_list:
                downsampled_data = pd.concat(downsampled_data_list, ignore_index=True)
            else:
                st.write("No windows to analyze after applying missing data tolerance.")
                return

            st.session_state.results = results
            st.session_state.downsampled_data = downsampled_data

            st.write("Cosinor Analysis done")

            plot = st.checkbox("Show plots")

            if plot:
                selected_window_label = st.selectbox("Select the window to plot", list(results.keys()))

                st.session_state.selected_window_label = selected_window_label

                window_size_selected = win_size_int[window_size]

                plot_cosinor(results, downsampled_data, window_size_selected, selected_window_label, analysis_window_size)

            show_all_windows = st.checkbox("Show all windows in a single plot")

            if show_all_windows:
                all_dates_plot(results, analysis_window_size)

            if results:
                st.write("Download the results")

                download = st.button("Download Results")

                if download:
                    download_results(results, downsampled_data)

        except Exception as e:
            st.write("Error occurred:")
            st.write(e)

if __name__ == "__main__":
    main()
