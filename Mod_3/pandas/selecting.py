import pandas as pd
data = {
    "Name": ["Aarav", "Diya", "Kabir", "Meera"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 78, 90],
    "City": ["Chennai", "Delhi", "Mumbai", "Pune"],
}
df = pd.DataFrame(data)
print(df)

# --- Selecting COLUMNS ---
print(df["Name"])                 # one column (returns a Series)
print(df[["Name", "Marks"]])      # multiple columns (note the double brackets -> a DataFrame)
# --- Selecting ROWS by position with .iloc ---
print(df.iloc[0])                 # the first row
print(df.iloc[0:2])               # the first two rows

# --- Selecting ROWS by label with .loc ---
print(df.loc[0])                  # row with index label 0
print(df.loc[0, "Name"])          # a single cell: row 0, column "Name"
