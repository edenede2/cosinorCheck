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

if 'selected_date' not in st.session_state:
    st.session_state.selected_date = None

if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = None

def first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal, period_size, shift_size, window_size, win_size_int, missing_tolerance, interpolated, interpolation_method):

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



    # start_date = df.select(pl.col("DateAndMinute").min()).item().date()
    # end_date = df.select(pl.col("DateAndMinute").max()).item().date()

    start_datetime = df.select(pl.col("DateAndMinute").min()).item()
    end_datetime = df.select(pl.col("DateAndMinute").max()).item()

    # start_datetime = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
    # end_datetime = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)


    windows = generate_fixed_windows(start_datetime, end_datetime, period_size, shift_size)

    st.write(f"Generated {len(windows)} windows for the analysis")

    results = {}
    download_data_list = []

    preprocessed_windows = []

    for window in windows:
        window_start = window['start']
        window_end = window['end']
        window_id = window['window_id']
        window_label = window['label']

        st.write(f"Processing window {window_id} - {window_label}")

        data_in_window = df.filter(
            pl.col("DateAndMinute") >= window_start,
            pl.col("DateAndMinute") <= window_end
        )

        total_points = data_in_window.shape[0]
        missing_points = data_in_window.filter(pl.col(col).is_null()).shape[0]
        missing_percentage = (missing_points / total_points) * 100 if total_points > 0 else 100

        # st.write(f"Total points: {total_points}, Missing points: {missing_points}, Missing percentage: {missing_percentage:.2f}%")

        if missing_percentage > missing_tolerance:
            # st.write(f"Skipping window {window_id} - {window_label} due to high missing data")
            continue

        downsampled = downsample_signal(data_in_window, window_size, col)

        if interpolated:
            data_in_window = interpolate_data(downsampled, interpolation_method)
        else:
            data_in_window = downsampled

        # period = period_size * 60 / win_size_int[window_size]

        window_df = pl.DataFrame({
            "test": [window_label] * data_in_window.shape[0],
            "x": np.arange(data_in_window.shape[0]),
            "y": data_in_window['downsampled'],
            "interpolated_y": data_in_window['interpolated_y'] if 'interpolated_y' in data_in_window.columns else None
        })

        preprocessed_windows.append(window_df)

    return preprocessed_windows



    # interval = '1m'

    # dates_df = pl.DataFrame(
    #     {
    #         "DateAndMinute": pl.datetime_range(start=start_datetime, end=end_datetime, interval=interval, eager=True)
    #     }
    # )


    # df = (
    #     dates_df
    #     .join(
    #         df,
    #         on='DateAndMinute',
    #         how='left'
    #     )
    #     .sort("DateAndMinute")
    # )



    # first_null = (
    #     df
    #     .with_columns(
    #         is_missing = pl.col(col).is_null(),
    #         date = pl.col("DateAndMinute").dt.date()
    #     )
    # )

    # first_signal = (
    #     first_null
    #     .with_columns(
    #         group_id = (pl.col("is_missing") == False).cum_sum().over("date")
    #     )
    # )



    # missing_runs_signal = first_signal.filter(pl.col("is_missing"))

    # missing_counts_signal = missing_runs_signal.group_by(["date", "group_id"]).agg(
    #     pl.count().alias("run_length")
    # )

    # max_missing_signal = missing_counts_signal.group_by("date").agg(
    #     pl.col("run_length").max().alias("max_consecutive_missing_signal")
    # )

    # first_with_max_missing = first_signal.join(
    #     max_missing_signal,
    #     on="date",
    #     how="left"
    # )




    # first_with_max_missing = (
    #     first_with_max_missing
    #     .with_columns(
    #         missing_per_date = pl.col("is_missing").sum().over("date"),
    #     )
    #     .with_columns(
    #         pl.col('max_consecutive_missing_signal').fill_null(strategy='zero'),
    #     )
    # )



    # first_with_max_missing = first_with_max_missing.drop(["group_id", "date"])


    # if remove_not_in_IL:
    #     first_with_max_missing = first_with_max_missing.filter(pl.col("not_in_israel") == False)

    # if remove_dst_change:
    #     first_with_max_missing = first_with_max_missing.filter(pl.col("is_dst_change") == False)

    # return first_with_max_missing


def downsample_signal(df: pl.DataFrame, window_size: str, signal: str) -> pl.DataFrame:
    if signal == "BpmMean":
        col = "BpmMean"
    elif signal == "StepsInMinute":
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
    ]
    )

    downsampled = grouped.with_columns(
        pl.when(pl.col("count_missing") > pl.col("count_non_missing"))
        .then(None)
        .otherwise(pl.col("mean"))
        .alias("downsampled")
    ).select([
        pl.col("DateAndMinute"),
        pl.col("downsampled")
    ])

    downsampled_df = downsampled.to_pandas()

    return downsampled_df



def generate_fixed_windows(start_datetime, end_datetime, period_size, shift_size):
    windows = []
    window_id = 0
    delta_window = timedelta(hours=period_size)
    delta_shift = timedelta(hours=shift_size)

    current_start = start_datetime.replace(minute=0, second=0, microsecond=0)
    reminder_hours = (current_start.hour % period_size)

    if reminder_hours != 0:
        current_start += timedelta(hours=shift_size - reminder_hours)

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



def downsample_bpm_mean(df: pl.DataFrame, window_size: str, signal: str, tolerence: int) -> pl.DataFrame:

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
    ]).with_row_index('5minuteIndex')


    

    downsampled = (
        df
        .join(
            downsampled,
            on='DateAndMinute',
            how='left'
        )
    )

    tolerence_minutes = 1440 * tolerence / 100

    data_for_cosinor = (
        downsampled
        .filter(
            pl.col("5minuteIndex").is_not_null()
        )
        .filter(
            pl.col("missing_per_date") <= tolerence_minutes,
            pl.col("max_consecutive_missing_signal") <= tolerence_minutes
        )
        .with_row_index('index')
        .with_columns(
            index=pl.col('index').cum_count().over(pl.col('DateAndMinute').dt.date())
        )
        .select(
            x=pl.col('index'),
            y=pl.col('downsampled'),
            test=pl.col('DateAndMinute').dt.date().cast(pl.String())
        )
        .to_pandas()
    )

    return data_for_cosinor


def cosinor_analysis(data: list, signal: str, period: int):


    if signal == "BpmMean":
        col = "BpmMean"
    elif signal == "StepsMean":
        col = "StepsInMinute"

    results = {}

    data = pl.concat(data, how='vertical_relaxed').to_pandas()

    labels = data['test'].unique()
    st.write(period)
    for label in labels:
        data_for_label = data[data['test'] == label]
        data_for_label = data_for_label.dropna(subset=['y'])

        if any(data_for_label['interpolated_y'].notna()):
            data_for_label['y'] = np.where(data_for_label['y'].isnull(), data_for_label['interpolated_y'], data_for_label['y'])

        results[label] = cosinor.fit_me(data_for_label['x'], data_for_label['y'], n_components=1, period=period, plot=False, return_model=True, params_CI=True)

    return results





    # dates = data['test'].unique()

    # results = {}

    # for date in dates:
    #     data_for_date = data[data['test'] == date]
    #     index_range = np.arange(period)

    #     # period = data['x'].max()

    #     data_for_date = pl.DataFrame(data_for_date)

    #     data_for_date = data_for_date.drop_nulls()

    #     if 'interpolated_y' in data_for_date.columns:
    #         data_for_date = (
    #             data_for_date
    #             .with_columns(
    #                 y=pl.when(pl.col('y').is_null())
    #                     .then(pl.col('interpolated_y'))
    #                     .otherwise(pl.col('y'))
    #             )
    #             .to_pandas()
    #         )
    #     else:
    #         data_for_date = data_for_date.to_pandas()



    #     results[date] = cosinor.fit_me(data_for_date['x'], data_for_date['y'], n_components=1, period=period, plot=False, return_model=True, params_CI=True)

    # return results

    
def plot_cosinor(data, original_data, window_size, date_selected, period, select_period_size):

    fig = go.Figure()

    original_data = pl.concat(original_data, how='vertical_relaxed').to_pandas()

    st.write(original_data)
    data = data[date_selected]
    original_data = original_data[original_data['test'] == date_selected]

    if "12:00:00" in date_selected:
        half_day = True
    else:
        half_day = False

    length = len(original_data['x'])*2

    x_data = [(x*window_size)/60 for x in original_data['x']]
    y_data = original_data['y']
    y_data_interpolated = original_data['interpolated_y'] if 'interpolated_y' in original_data.columns else None

    x_estimated = [(x*window_size)/60 for x in data[3]][:500]
    y_estimated = data[4][:500]

    st.write(len(x_data), len(y_data), len(x_estimated), len(y_estimated))

    # if half_day:
    #     x_data = [x + 12 if x < 12 else x - 12 for x in x_data]
    #     x_estimated = [x + 12 if x < 12 else x - 12 for x in x_estimated]


    if y_data_interpolated is not None:
        fig.add_trace(go.Scatter(x=x_data, y=y_data_interpolated, mode='markers', name='Interpolated Data'))

    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Original Data'))
    fig.add_trace(go.Scatter(x=x_estimated, y=y_estimated, mode='lines', name='Estimated Data'))
    

    if half_day:
        fig.update_layout(title='Cosinor Analysis', xaxis_title='Time [hours] (12-hour format)', yaxis_title='Value')
    else:
        fig.update_layout(title='Cosinor Analysis', xaxis_title='Time [hours]', yaxis_title='Value')

    st.plotly_chart(fig)

    theta = data[2]['peaks'][0]/data[2]['period'] * 2 * np.pi
    acrophase = quadrant_adjustment(theta, data[2]['acrophase'], radian=False)
    # corrected_acrophase = quadrant_adjustment(theta, data[2]['acrophase'])

    amplitude = data[2]['amplitude']
    # if half_day:
    #     acrophase = data[2]['acrophase']
    #     acrophase = np.abs(acrophase) + np.pi
    # else:
    #     acrophase = data[2]['acrophase']
    #     if acrophase < 0:
    #         acrophase = 2 * np.pi + acrophase
    #     else:
    #         acrophase = acrophase
    # acrophase = corrected_acrophase
    mesor = data[2]['mesor']


    center_r = [0, amplitude]

    st.write(f"abs acrophase: {acrophase}")

    center_theta = [0, acrophase]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[amplitude],
        theta=[acrophase],
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

    hours, hours_deg = generate_polarticks(period, select_period_size)

    # hours = ['00:00', '21:00', '18:00', '15:00', '12:00', '09:00', '06:00', '03:00']
    # hours_deg = [0, 45, 90, 135, 180, 225, 270, 315]



    fig.update_layout(
        title=f'Cosinor Analysis - Date: {date_selected}, Mesor: {mesor}',
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


def generate_polarticks(period, select_period_size):
    total_hours = period

    num_ticks = 12

    tick_interval = select_period_size / num_ticks

    hours = []
    hours_deg = []

    if select_period_size == 24:
        for i in range(num_ticks + 1):
            hour = i * tick_interval
            deg = (i * 360) / num_ticks
            hour_int = int(hour % 24)
            label = f"{hour_int:02d}:00"
            hours.append(label)
            hours_deg.append(deg)
    else:
        for i in range(num_ticks +1):
            hour = i * tick_interval
            deg = (i * 360) / num_ticks
            hour_int = int(hour % 24)
            day = int(hour // 24) + 1
            label = f"Day {day} - {hour_int:02d}:00"
            hours.append(label)
            hours_deg.append(deg)

    return hours, hours_deg

def all_dates_plot(results, original_data, window_size, period, select_period_size):
    fig = go.Figure()

    for key in results.keys():

        if "12:00:00" in key:
            half_day = True
        else:
            half_day = False
        # Extract parameters
        amplitude = results[key][2]['amplitude']
        acrophase = results[key][2]['acrophase']

        theta = results[key][2]['peaks'][0]/results[key][2]['period'] * 2 * np.pi
        acrophase = quadrant_adjustment(theta, acrophase,radian=True)

        # if half_day:
        #     acrophase = acrophase - np.pi
        
        
        # Extract confidence intervals
        ci_amplitude = results[key][2]['CI(amplitude)']
        ci_acrophase = results[key][2]['CI(acrophase)']

        # for i in range(len(ci_acrophase)):
        #     ci_acrophase[i] = quadrant_adjustment(theta, ci_acrophase[i], radian=True)
        
        if half_day:
            # ci_acrophase = [x - np.pi for x in ci_acrophase]
            ci_acrophase = [x + 180 if x < 0 else x - 180 for x in ci_acrophase]

        # st.write(f"CI Amplitude: {ci_amplitude}")
        st.write(f"CI Acrophase: {ci_acrophase}")
        # Plot the center point
        fig.add_trace(go.Scatterpolar(
            r=[amplitude],
            theta=[np.rad2deg(acrophase)],
            mode='markers',
            marker=dict(
                color='red',
                size=10
            ),
            name=key
        ))

        # Plot the radius line from the center to the point
        center_r = [0, amplitude]
        center_theta = [0, np.rad2deg(acrophase)]

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

        # Generate confidence ellipse points
        num_points = 100  # Number of points for smooth ellipse
        theta_range = np.linspace(0, 2 * np.pi, num_points)

        # Calculate the mean amplitude and acrophase
        mean_amplitude = (ci_amplitude[0] + ci_amplitude[1]) / 2
        mean_acrophase = (ci_acrophase[0] + ci_acrophase[1]) / 2

        # Calculate semi-axis lengths based on confidence intervals
        semi_amplitude = (ci_amplitude[1] - ci_amplitude[0]) / 2
        semi_acrophase = (ci_acrophase[1] - ci_acrophase[0]) / 2

        # st.write(f"Mean Amplitude: {mean_amplitude}")
        # st.write(f"Mean Acrophase: {mean_acrophase}")
        # st.write(f"Semi Amplitude: {semi_amplitude}")
        # st.write(f"Semi Acrophase: {semi_acrophase}")
        # Generate ellipse points in polar coordinates
        ellipse_r = mean_amplitude + semi_amplitude * np.cos(theta_range)
        ellipse_theta = mean_acrophase + semi_acrophase * np.sin(theta_range)
        ellipse_theta_deg = np.rad2deg(ellipse_theta)# Convert to degrees for plotting

        

        # Add ellipse trace
        fig.add_trace(go.Scatterpolar(
            r=ellipse_r,
            theta=ellipse_theta_deg,
            mode='lines',
            line=dict(
                color='blue',
                dash='dash',
                width=1
            ),
            name=f'{key} CI Ellipse'
        ))

    # period_hours = 24
    hours, hours_deg = generate_polarticks(period, select_period_size)

    # # Customize angular axis labels
    # hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    # hours_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    fig.update_layout(
        title='Cosinor Analysis - All Dates with Confidence Ellipse',
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

def quadrant_adjustment(thta, acrphs, radian=True):
    # Check which quadrant the acrophase falls into
    if 0 <= thta < (np.pi / 2):
        if radian:
            corrected_acrophase = acrphs
        else:
            # First quadrant: no correction needed
            corrected_acrophase = np.rad2deg(acrphs)
    elif (np.pi / 2) <= thta < np.pi:
        # Second quadrant: subtract a constant to realign
        if radian:
            corrected_acrophase = acrphs 
        else:
            corrected_acrophase =  np.rad2deg(acrphs)
    elif np.pi <= thta < (3 * np.pi / 2):
        # Third quadrant: make it negative
        if radian:
            corrected_acrophase = 2 * np.pi - acrphs
        else:
            corrected_acrophase = 360 - np.rad2deg(acrphs)
    elif (3 * np.pi / 2) <= thta < (2 * np.pi):
        if radian:
            corrected_acrophase = 2 * np.pi - acrphs
        else:
            # Fourth quadrant: shift to bring into biological range
            corrected_acrophase = 360 - np.rad2deg(acrphs)
    else:
        # If outside normal bounds, wrap it
        corrected_acrophase = acrphs % (2 * np.pi)

    return corrected_acrophase


def download_results(results, original_data, window_size, period, select_period_size):

    original_data = pl.concat(original_data, how='vertical_relaxed').to_pandas()
    
    columns = ['date', 'amplitude','period','acrophase (rad)', 'corrected_acrophase (rad)',
                'corrected_acrophase (hours)', 'corrected_acrophase (degrees)',
                'mesor','AIC', 'BIC','peaks','heights', 'troughs', 'trough_time', 
                'heights2','max_loc', 'period2', 'p-value', 'p_reject', 'SNR', 'RSS', 
                'resid_SE', 'ME','f-pvalue', 't-values const', 't-values x1',
                't-values x2','R-squared', 'R-squared adjusted', 'SSR', 'minutes_based']

    # Initialize an empty DataFrame with these columns
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

        trough_indices = params['troughs'][0] if len(params['troughs']) > 0 else [np.nan]
        
        trough_loc = trough_indices/params['period'] * period
    
        trough_hours = int(trough_loc)
        trough_minutes = int((trough_loc - trough_hours) * 60)

        if trough_hours > 24:
            trough_days = trough_hours // 24
            trough_hours = trough_hours % 24
            trough_time = f"{trough_days} day(s) {trough_hours:02d}:{trough_minutes:02d}"

        else:
            trough_time = f"{trough_hours:02d}:{trough_minutes:02d}"


        # convert the corrected acrophase degrees to time in format HH:MM
        hours = int((corrected_acrophase_deg/360) * 24)

        if hours > 24:
            days = hours // 24
            hours = hours % 24
            minutes = int((corrected_acrophase_deg/360) * 24 * 60) % 60
            corrected_acrophase_time = f"{days} day(s) {hours:02d}:{minutes:02d}"
        else:
            minutes = int((corrected_acrophase_deg/360) * 24 * 60) % 60
            corrected_acrophase_time = f"{hours:02d}:{minutes:02d}"
        # minutes = int((corrected_acrophase_deg/360) * 24 * 60) % 60
        # corrected_acrophase_time = f"{hours:02d}:{minutes:02d}"



        cosinor_model_params = {
            'date': [key],
            'amplitude': [float(params['amplitude'])],
            'period': [float(params['period'])],
            'acrophase (rad)': [float(params['acrophase'])],
            'corrected_acrophase (rad)': [float(corrected_acrophase)],
            'corrected_acrophase (hours)': [corrected_acrophase_time],
            'corrected_acrophase (degrees)': [float(corrected_acrophase_deg)],
            'mesor': [float(params['mesor'])],
            'AIC': [float(model[0].aic)],  # ensure floats
            'BIC': [float(model[0].bic)],
            'peaks': [str(params['peaks'])],  # Convert list to string
            'heights': [str(params['heights'])],  # Convert list to string
            'troughs': [str(params['troughs'])],
            'trough_time': [str(trough_time)],
            'heights2': [str(params['heights2'])],
            'max_loc': [float(params['max_loc'])],
            'period2': [float(params['period2'])],
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

    results_df

    st.write("Download the results")

    show_results = st.checkbox("Show Results")

    if show_results:
        st.write(results_df)

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
    return amplitude * np.sin(frequency * x + phase) + offset


def interpolate_data(data, method):
    data = pl.DataFrame(data)
    y = data["y"].to_numpy()
    x = np.arange(len(y))  
    isnan = np.isnan(y)  

    if np.any(isnan):
        x_valid = x[~isnan]
        y_valid = y[~isnan]

        if method == "sinosuidal":
            interpolated_values = np.interp(x, x_valid, y_valid)  
            interpolated_values[~isnan] = y[~isnan] 
            interpolated_values[isnan] = [(x+1)*np.median(y_valid) for x in np.cos(interpolated_values[isnan])]

        elif method == "sinosuidal (curve-fit)":
            # Initial guess for the parameters: amplitude, frequency, phase, and offset
            initial_guess = [np.std(y_valid), 2 * np.pi / len(y_valid), 0, np.mean(y_valid)]

            # Fit the model to the known (non-missing) data points
            try:
                params, _ = curve_fit(sinusoidal_model, x_valid, y_valid, p0=initial_guess)
                interpolated_values = y.copy()  # Start with original data
                interpolated_values[isnan] = sinusoidal_model(x[isnan], *params)  # Apply fitted model to missing
            except RuntimeError:
                print("Curve fitting did not converge; consider refining the initial guess.")
                interpolated_values = y  # Fallback to the original data if fitting fails


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

    st.write("This app built for testing cosinor parameters on your data")

    st.write("We would use the \"subname Heart Rate and Steps and Sleep Aggregated.csv\" file for this analysis")

    st.divider()

    st.write("Please upload the file")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)
            st.write("Data uploaded successfully")

            st.divider()


            st.write("Data is valid")

            st.write("Preprocessing the data")

            remove_not_in_IL = st.checkbox("Remove date that the subject was'nt in Israel?")

            remove_dst_change = st.checkbox("Remove date that the subject was in DST change?")

            signal = st.selectbox("Select the signal to analyze", ["BpmMean", "StepsMean"])


            st.write("Select the period size for cosinor analysis")

            select_period_size = st.selectbox("Select the period size (hours)", ["24", "48", "72", "96", "120", "144", "168"])

            select_period_size = int(select_period_size)

            st.write("Select the shift size for each period")

            select_shift_size = st.selectbox("Select the shift size (hours)", ["48","24", "12"])

            select_shift_size = int(select_shift_size)

            st.write("Select the window size for downsampling")

            window_size = st.selectbox("Select the window size", ["1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h"])

            win_size_int = {"1m": 1, "2m": 2, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "2h": 120}


            st.write("Select the tolerance for missing data")

            missing_tolerance = st.slider("Select the tolerance for missing data (percentage)", 0, 100, 10)


            interpolated = st.checkbox("Interpolate the missing data?")

            if interpolated:
                st.write("Select the method for interpolation")

                interpolation_method = st.selectbox("Select the method for interpolation", ["sinosuidal", "sinosuidal (curve-fit)", "polynomial", "spline-cubic", "spline-quadratic"])

            else:
                interpolation_method = None
            

            if st.button("Preprocess the data"):
                first_preprocess = first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal, select_period_size, select_shift_size, window_size, win_size_int, missing_tolerance, interpolated, interpolation_method)
                st.session_state.preprocessed = first_preprocess


            st.write("Preprocessing done")

            show_preprocess = st.checkbox("Show preprocessed data")

            if show_preprocess:
                st.write(st.session_state.preprocessed)

            
            st.divider()

            st.write("Run Cosinor Analysis")

            if st.session_state.preprocessed is not None and st.button("Run Cosinor Analysis"):
                st.session_state.analysed = True

            # if st.session_state.analysed:
                period = select_period_size * 60 / win_size_int[window_size]

                results = cosinor_analysis(st.session_state.preprocessed, signal, period)

                st.session_state.results = results

          

            if st.session_state.analysed:
                plot = st.checkbox("Show plots")

                if plot:
                    selected_date = st.selectbox("Select the date to plot", list(st.session_state.results.keys()))

                    st.session_state.selected_date = selected_date


                    window_size_selected = win_size_int[window_size]
                    period = select_period_size * 60 / win_size_int[window_size]
                    st.write(f"Period: {period}")
                    plot_cosinor(st.session_state.results, st.session_state.preprocessed, window_size_selected, selected_date, period, select_period_size)

                show_all_dates = st.checkbox("Show all dates")

                if show_all_dates:
                    period = select_period_size * 60 / win_size_int[window_size]
                    all_dates_plot(st.session_state.results, st.session_state.preprocessed, window_size_selected, period, select_period_size)

                st.write("Cosinor Analysis done")

                if st.session_state.results:
                    st.write("Download the results")

                    download = st.button("Download Results")
                    window_size_selected = win_size_int[window_size]

                    if download:
                        period = select_period_size * 60 / win_size_int[window_size]
                        download_results(st.session_state.results, st.session_state.preprocessed, window_size_selected, period, select_period_size)




                    
                



                st.write("Select the period for cosinor analysis")


        except Exception as e:
            st.write("Error uploading file")
            st.write(e)




if __name__ == "__main__":
    main()