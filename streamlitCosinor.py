import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
from CosinorPy import file_parser, cosinor, cosinor1
from datetime import datetime, timedelta, date, time
import plotly.graph_objects as go


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
            pl.col("DateAndMinute").str.strptime("%Y-%m-%d %H:%M:%S"),
        )
    )


    start_date = df.select(pl.col("DateAndMinute").min()).item().date()
    end_date = df.select(pl.col("DateAndMinute").max()).item().date()


    start_datetime = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
    end_datetime = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)

    interval = '1m'

    dates_df = pl.DataFrame(
        {
            "DateAndMinute": pl.datetime_range(start=start_datetime, end=end_datetime, interval=interval, eager=True)
        }
    )


    df = (
        dates_df
        .join(
            df,
            on='DateAndMinute',
            how='left'
        )
        .sort("DateAndMinute")
    )



    first_null = (
        df
        .with_columns(
            is_missing = pl.col(col).is_null(),
            date = pl.col("DateAndMinute").dt.date()
        )
    )

    first_signal = (
        first_null
        .with_columns(
            group_id = (pl.col("is_missing") == False).cum_sum().over("date")
        )
    )



    missing_runs_signal = first_signal.filter(pl.col("is_missing"))

    missing_counts_signal = missing_runs_signal.group_by(["date", "group_id"]).agg(
        pl.count().alias("run_length")
    )

    max_missing_signal = missing_counts_signal.group_by("date").agg(
        pl.col("run_length").max().alias("max_consecutive_missing_signal")
    )

    first_with_max_missing = first_signal.join(
        max_missing_signal,
        on="date",
        how="left"
    )




    first_with_max_missing = (
        first_with_max_missing
        .with_columns(
            missing_per_date = pl.col("is_missing").sum().over("date"),
        )
        .with_columns(
            pl.col('max_consecutive_missing').fill_null(strategy='zero'),
        )
    )



    first_with_max_missing = first_with_max_missing.drop(["group_id", "date"])


    if remove_not_in_IL:
        first_with_max_missing = first_with_max_missing.filter(pl.col("not_in_israel") == False)

    if remove_dst_change:
        first_with_max_missing = first_with_max_missing.filter(pl.col("is_dst_change") == False)

    return first_with_max_missing


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
            pl.col("max_consecutive_missing") <= tolerence_minutes
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


def cosinor_analysis(data: pd.DataFrame, signal: str):

    period = data['x'].max()

    if signal == "BpmMean":
        col = "BpmMean"
    elif signal == "StepsMean":
        col = "StepsInMinute"

    dates = data['test'].unique()

    results = {}

    for date in dates:
        data_for_date = data[data['test'] == date]

        results[date] = cosinor.fit_me(data_for_date['x'], data_for_date['y'], n_components=1, period=period, plot=False, return_model=True)

    return results

    
def plot_cosinor(data, plot_type, original_data, window_size, date_selected):

    if plot_type == "Cartesian":
        fig = go.Figure()

        length = len(original_data['x'])*2

        x_data = [(x*window_size)/60 for x in original_data['x']]
        y_data = original_data['y']

        x_estimated = [(x*window_size)/60 for x in data[3][:length]]
        y_estimated = data[4][:length]

        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='Original Data'))
        fig.add_trace(go.Scatter(x=x_estimated, y=y_estimated, mode='lines', name='Estimated Data'))

        fig.update_layout(title='Cosinor Analysis', xaxis_title='Time [hours]', yaxis_title='Value')

        st.plotly_chart(fig)

    elif plot_type == "Polar":

        amplitude = data[2]['amplitude']
        acrophase = data[2]['acrophase']
        mesor = data[2]['mesor']


        center_r = [0, amplitude]

        center_theta = [0, np.deg2rad(acrophase)]

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

        hours = ['00:00', '21:00', '18:00', '15:00', '12:00', '09:00', '06:00', '03:00']
        hours_deg = [0, 45, 90, 135, 180, 225, 270, 315]

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




def download_results(results, original_data):
    
    columns = ['date', 'amplitude','period','acrophase (rad)', 'acrophase (hours)',
                'acrophase (degrees)', 'mesor','AIC', 'BIC','peaks', 
            'heights', 'troughs', 'heights2', 'max_loc', 'period2', 
            'p-value', 'p_reject', 'SNR', 'RSS', 'resid_SE', 'ME', 
            'f-pvalue', 't-values const', 't-values x1', 't-values x2', 
            'R-squared', 'R-squared adjusted', 'SSR', 'minutes_based']

    # Initialize an empty DataFrame with these columns
    results_df = pl.DataFrame({col: [] for col in columns})


    for key in results.keys():
        model = results[key]
        cosinor_model = model[0]
        stats = model[1]
        params = model[2]
        original_data1 = original_data[original_data['test'] == key]
        cosinor_model_params = {
            'date': [datetime.strptime(key, '%Y-%m-%d')],
            'amplitude': [float(params['amplitude'])],
            'period': [float(params['period'])],
            'acrophase (rad)': [float(params['acrophase'])],
            'acrophase (hours)': [float(params['acrophase'] * 24 / (2 * np.pi))],
            'acrophase (degrees)': [float(params['acrophase'] * 180 / np.pi)],
            'mesor': [float(params['mesor'])],
            'AIC': [float(model[0].aic)],  # ensure floats
            'BIC': [float(model[0].bic)],
            'peaks': [str(params['peaks'])],  # Convert list to string
            'heights': [str(params['heights'])],  # Convert list to string
            'troughs': [str(params['troughs'])],
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

    file_name = st.text_input("Enter the file name", "cosinor_results.csv")

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name=file_name,
        mime="text/csv"
    )






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

            st.write("Please confirm that the file is valid before proceeding")

            valid = st.button("Confirm")

            if valid:
                st.write("Data is valid")

                st.write("Preprocessing the data")

                remove_not_in_IL = st.checkbox("Remove date that the subject was'nt in Israel?")

                remove_dst_change = st.checkbox("Remove date that the subject was in DST change?")

                signal = st.selectbox("Select the signal to analyze", ["BpmMean", "StepsMean"])

                first_preprocess = first_preprocess_step(dataframe, remove_not_in_IL, remove_dst_change, signal)

                st.write("Preprocessing done")

                show_preprocess = st.checkbox("Show preprocessed data")

                if show_preprocess:
                    st.write(first_preprocess)

                st.write("Select the window size for downsampling")

                window_size = st.selectbox("Select the window size", ["1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h"])

                win_size_int = {"1m": 1, "2m": 2, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "2h": 120}


                st.write("Select the tolerance for missing data")

                missing_tolerance = st.slider("Select the tolerance for missing data (percentage)", 0, 100, 10)

                st.write("Downsampling the data")

                downsampled = downsample_bpm_mean(first_preprocess, window_size, signal, missing_tolerance)

                st.write("Downsampling done")

                show_downsampled = st.checkbox("Show downsampled data")

                if show_downsampled:
                    st.write(downsampled)



                st.divider()

                st.write("Cosinor Analysis is ready to start")

                start_analysis = st.button("Start Cosinor Analysis")

                if start_analysis:
                    results = cosinor_analysis(downsampled, signal)

                    plot = st.checkbox("Show plots")

                    if plot:
                        selected_date = st.selectbox("Select the date to plot", results.keys())

                        selected_plot = st.selectbox("Select the plot type", ["Cartesian", "Polar"])

                        window_size_selected = win_size_int[window_size]

                        plot_cosinor(results[selected_date], selected_plot, downsampled, window_size_selected, selected_date)

                    st.write("Cosinor Analysis done")

                    if results:
                        st.write("Download the results")

                        download = st.button("Download Results")

                        if download:
                            download_results(results, downsampled)




                    
                



                st.write("Select the period for cosinor analysis")


        except Exception as e:
            st.write("Error uploading file")
            st.write(e)




if __name__ == "__main__":
    main()