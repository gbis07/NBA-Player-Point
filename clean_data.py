import pandas as pd

df = pd.read_csv("nba_allstats.csv")
print(df.head())
print(df.columns)

columns_list = df.columns.tolist()
sseason_index = columns_list.index('Season')
shifted_columns = ['EXTRA_COLUMN_1'] + columns_list[:sseason_index - 1] + ['Season']
df.columns = shifted_columns
print(df.head())

df = df.drop(['EXTRA_COLUMN_1'], axis=1)
print('\n Dropped extra column')
print(df.head())

# currently_playing = df[df['Season'] == '2024-25']
# print('\nCurrently Playing')
# print(currently_playing)
# print(len(currently_playing))

# filtered_df = df[df['PLAYER'].isin(currently_playing['PLAYER'])]
# print(filtered_df.head())
# print(len(filtered_df)) 

# filtered_df.to_csv('cleaned_data.csv')
df.to_csv('cleaned_data.csv')