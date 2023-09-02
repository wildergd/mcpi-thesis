from datetime import timedelta
import pandas as pd

def extract_data_for_days_measured(df: pd.DataFrame, days_measured: int) -> pd.DataFrame:
    df_grouped_days = df.groupby(pd.Grouper(freq='1D')).activity.count()
    dates_list = df_grouped_days[df_grouped_days >= 1440].index.to_list()[:days_measured + 1]
    from_date, to_date = dates_list[0], dates_list[-1]
    return df.loc[from_date:to_date - timedelta(minutes = 1)]

def remove_days_with_insufficient_data(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.groupby(pd.Grouper(freq='1D')).activity.agg(lambda x: x.eq(0).sum())
    dates = dfx[dfx >= 1400].index.date
    drop_dates_count = dates.shape[0]
    if drop_dates_count == 0:
        return df
    
    df_timestamps = pd.concat([
        pd.date_range(
            f'{date} 00:00:00',
            f'{date} 23:59:00',
            freq = 'T'
        ).to_frame(index = False, name = 'timestamp')
        for date in dates
    ])
    df_dropped = df.drop(index = df_timestamps['timestamp'])
    
    new_indexes = pd.date_range(
        df.index.min(),
        df.index.max() - timedelta(days = drop_dates_count),
        freq = 'T'
    )

    new_df = pd.DataFrame(
        data = {
            'date': new_indexes,
            'activity': df_dropped['activity']
        },
        index = new_indexes
    )
    
    new_df['date'] = pd.to_datetime(df['date'])
    new_df['date'] = new_df['date'].dt.date
    
    return new_df

def get_measured_days(df: pd.DataFrame, subject: int) -> pd.DataFrame:
    return df.loc[subject].days

def read_activity_dataset(path: str, days: int) -> pd.DataFrame:
    df_row = pd.read_csv(
        path,
        parse_dates=['timestamp', 'date'],
        index_col = 'timestamp'
    )
    
    # fill missing data
    df_row = df_row.reindex(
        pd.date_range(df_row.index.min(), df_row.index.max(), freq = 'T'),
        method = 'nearest'
    )
    
    # remove days with insufficient data
    df_row = remove_days_with_insufficient_data(df_row)
    
    max_days_measured = (df_row.index.max() - df_row.index.min()).days
    
    # extract data  only for the number of days measured
    return extract_data_for_days_measured(
        df_row,
        min(11, max(max_days_measured, days))
    )
    
def read_scores_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        index_col = 'number'
    )