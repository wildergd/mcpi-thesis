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

def read_activity_dataset(
    path: str,
    days: int,
    max_dates: int = 11,
) -> pd.DataFrame:
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
        min(max_dates, max(max_days_measured, days))
    )
    
def read_scores_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        index_col = 'number'
    )
    
def check_activity_variation(
    df: pd.DataFrame,
    column: str = 'activity',
    initial_state: int = 0
):
    std = df[column].std()
    mean = df[column].mean()
    prev_val = initial_state
    
    def check_variation_values(values):
        if values.shape[0] < 2:
            return initial_state
        
        nonlocal prev_val
        
        before, after = values[0], values[1]
        diff = after - before
        print(abs(diff))
        if  abs(diff) >= std:
            if diff < 0:
                prev_val = 0
                return 0
            if diff > 0:
                prev_val = 1
                return 1
        return prev_val
    
    return check_variation_values
