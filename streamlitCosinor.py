import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
from CosinorPy import file_parser, cosinor, cosinor1
from datetime import datetime, timedelta, date, time
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit

if 'analysed' not in st.session_state:
    st.session_state.analysed = False

if 'results' not in st.session_state:
    st.session_state.results = None

if 'selected_window_id' not in st.session_state:
    st.session_state.selected_window_id = None

def generate_time_windows(start_datetime, end_datetime, window_size_hours, window_shift_hours):
    windows = []
    current_start = start_datetime
    window_id = 0

    delta_window = timedelta(hours=window_size_hours)
    delta_shift = timedelta(hours=window_shift_hours)

    while current_start <= end_datetime:
        current_end = current_start + delta_window
        label = f"{current_start.strftime('%Y-%m-%d %H:%M:%S')} to {current_end.strftime('%Y-%m-%d %H:%M:%S')}"
        windows.append({'window_id': window_id, 'start': current_start, 'end': current_end, 'label': label})
        current_start += delta_shift
        window_id += 1

    return windows

def first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal):

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

def downsample_bpm_mean(df: pl.DataFrame, window_size: str, signal: str, tolerence: int, window_id: int) -> pd.DataFrame:

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
          .alias("downsampled")          
    ).select([
        pl.col("DateAndMinute"),                   
        pl.col("downsampled")               
    ]).with_row_count('Index')

    tolerence_minutes = (df.shape[0]) * tolerence / 100

    data_for_cosinor = (
        downsampled
        .filter(
            pl.col("downsampled").is_not_null()
        )
        .with_row_count('index')
        .select(
            x=pl.col('index'),
            y=pl.col('downsampled'),
            test=pl.lit(window_id).cast(pl.String())
        )
        .to_pandas()
    )

    return data_for_cosinor

def cosinor_analysis(data: pd.DataFrame, signal: str, period: int):

    if signal == "BpmMean":
        col = "BpmMean"
    elif signal == "StepsMean":
        col = "StepsInMinute"

    dates = data['test'].unique()

    results = {}

    for date in dates:
        data_for_date = data[data['test'] == date]
        index_range = np.arange(period)

        data_for_date = pl.DataFrame(data_for_date)

        data_for_date = data_for_date.drop_nulls()

        if 'interpolated_y' in data_for_date.columns:
            data_for_date = (
                data_for_date
                .with_columns(
                    y=pl.when(pl.col('y').is_null())
                        .then(pl.col('interpolated_y'))
                        .otherwise(pl.col('y'))
                )
                .to_pandas()
            )
        else:
            data_for_date = data_for_date.to_pandas()

        results[date] = cosinor.fit_me(data_for_date['x'], data_for_date['y'], n_components=1, period=period, plot=False, return_model=True, params_CI=True)

    return results

def generate_polar_ticks(analysis_window_size_hours):
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

def plot_cosinor(results, downsampled_data, window_size, window_id, analysis_window_size):

    # Get the result for the selected window
    data = results[window_id]
    # Safely get original data
    original_data_list = []
    for df in downsampled_data:
        if not df.empty and 'test' in df.columns:
            if df['test'].iloc[0] == str(window_id):
                original_data_list.append(df)
    if not original_data_list:
        st.error(f"No data found for window ID {window_id}")
        return
    original_data = pd.concat(original_data_list)

    # Get the first (and only) date in data
    date_selected = str(window_id)

    data = data[date_selected]

    length = len(original_data['x'])*2

    x_data = [(x * window_size) / 60 for x in original_data['x']]
    y_data = original_data['y']
    y_data_interpolated = original_data['interpolated_y'] if 'interpolated_y' in original_data.columns else None

    x_estimated = [(x * window_size) / 60 for x in data[3]][:500]
    y_estimated = data[4][:500]

    st.write(len(x_data), len(y_data), len(x_estimated), len(y_estimated))

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
        title=f'Cosinor Analysis - Window ID: {window_id}, Mesor: {mesor}',
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

# The rest of your code remains mostly unchanged
# Ensure to store 'results' and 'downsampled_data' in 'st.session_state' after analysis

def main():
    st.title("Cosinor Analysis App")

    st.write("This app is built for testing cosinor parameters on your data")

    st.write("We would use the \"subname Heart Rate and Steps and Sleep Aggregated.csv\" file for this analysis")

    st.divider()

    st.write("Please upload the file")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.write("Data uploaded successfully")
            st.write(dataframe.head())

            # Check for required columns
            required_columns = ["DateAndMinute", "BpmMean", "StepsInMinute", "not_in_israel", "is_dst_change"]
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            if missing_columns:
                st.error(f"The following required columns are missing from the uploaded file: {missing_columns}")
                return

            st.divider()

            st.write("Data is valid")

            st.write("Preprocessing the data")

            remove_not_in_IL = st.checkbox("Remove date that the subject wasn't in Israel?")

            remove_dst_change = st.checkbox("Remove date that the subject was in DST change?")

            signal = st.selectbox("Select the signal to analyze", ["BpmMean", "StepsMean"])

            first_preprocess = first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal)

            st.write("Preprocessing done")

            show_preprocess = st.checkbox("Show preprocessed data")

            if show_preprocess:
                st.write(first_preprocess)

            st.write("Select the analysis window size (hours)")

            analysis_window_size = st.selectbox("Select the analysis window size (hours)", [24, 48, 72, 96, 120])

            st.write("Select the window shift size (hours)")

            window_shift_size = st.selectbox("Select the window shift size (hours)", [12, 24])

            start_datetime = first_preprocess.select(pl.col("DateAndMinute").min()).item()
            end_datetime = first_preprocess.select(pl.col("DateAndMinute").max()).item()

            windows = generate_time_windows(start_datetime, end_datetime, analysis_window_size, window_shift_size)

            st.write(f"Generated {len(windows)} windows.")

            st.write("Select the window size for downsampling")

            window_size = st.selectbox("Select the window size", ["1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h"])

            win_size_int = {"1m": 1, "2m": 2, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "2h": 120}

            st.write("Select the tolerance for missing data")

            missing_tolerance = st.slider("Select the tolerance for missing data (percentage)", 0, 100, 10)

            st.write("Downsampling and processing the data per window")

            interpolated = st.checkbox("Interpolate the missing data")

            if interpolated:
                st.write("Select the method for interpolation")

                method = st.selectbox("Select the method for interpolation", ["sinosuidal", "sinosuidal (curve-fit)", "polynomial", "spline-cubic", "spline-quadratic"])

            if 'results' in st.session_state and 'downsampled_data' in st.session_state:
                results = st.session_state.results
                downsampled_data = st.session_state.downsampled_data
            else:
                results = {}
                downsampled_data = []

                for window in windows:
                    window_id = window['window_id']
                    window_start = window['start']
                    window_end = window['end']

                    st.write(f"Processing window {window_id}: {window_start} to {window_end}")

                    # Extract data in the window
                    data_in_window = first_preprocess.filter(
                        (pl.col("DateAndMinute") >= window_start) & (pl.col("DateAndMinute") < window_end)
                    )

                    # Downsample data in the window
                    downsampled = downsample_bpm_mean(data_in_window, window_size, signal, missing_tolerance, window_id)

                    # Perform interpolation if selected
                    if interpolated:
                        downsampled = interpolate_data(downsampled, method)

                    # Perform cosinor analysis
                    period = (analysis_window_size * 60) / win_size_int[window_size]
                    result = cosinor_analysis(downsampled, signal, period)

                    # Store results
                    results[window_id] = result
                    downsampled_data.append(downsampled)

                st.session_state.results = results
                st.session_state.downsampled_data = downsampled_data

            st.write("Cosinor Analysis done")

            plot = st.checkbox("Show plots")

            if plot:
                # Create a mapping from labels to window IDs
                window_labels = {window['label']: window['window_id'] for window in windows}

                selected_label = st.selectbox("Select the window to plot", list(window_labels.keys()))
                selected_window_id = window_labels[selected_label]

                st.session_state.selected_window_id = selected_window_id

                window_size_selected = win_size_int[window_size]

                plot_cosinor(results, downsampled_data, window_size_selected, selected_window_id, analysis_window_size)

            # Rest of your code...

        except Exception as e:
            st.write("Error uploading file")
            st.write(e)

if __name__ == "__main__":
    main()
