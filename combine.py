import pandas as pd
import os

# Set the folder path containing Excel files
folder_path = 'D:\\zyxthesis\\z2020\\zzzz'  # ← Change to your folder path, note the double backslashes

# Get all Excel files in the directory
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

# Check if there are any files
if not files:
    print("No Excel files found. Please check the folder path and file extensions (.xlsx/.xls only).")
    exit()

merged_df = None  # Initialize the merged DataFrame

for file in files:
    file_path = os.path.join(folder_path, file)
    print(f'Reading file: {file}')

    try:
        df = pd.read_excel(file_path, header=0)  # By default, read the first row as column headers
    except Exception as e:
        print(f'Error reading file {file}, skipped. Error message: {e}')
        continue

    if 'NAME' not in df.columns:
        print(f' File {file} does not contain a NAME column, skipped.')
        continue

    df = df.set_index('NAME')  # Set NAME column as index

    if merged_df is None:
        merged_df = df
    else:
        # Merge horizontally, keep all NAME values and all columns, fill missing data with NaN
        merged_df = merged_df.join(df, how='outer', lsuffix='', rsuffix=f'_{file}')

# Save the result
if merged_df is None:
    print("❗ No valid data found to merge. Please check the files.")
else:
    merged_df = merged_df.reset_index()  # Reset NAME as a normal column
    output_path = os.path.join(folder_path, 'merged_result1.xlsx')
    merged_df.to_excel(output_path, index=False)
    print(f'Merge completed. File saved as: {output_path}')