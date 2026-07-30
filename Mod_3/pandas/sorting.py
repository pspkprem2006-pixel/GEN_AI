import pandas as pd
data = {
    "Name": ["Aarav", "Diya", "Kabir", "Meera"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 78, 90],
    "City": ["Chennai", "Delhi", "Mumbai", "Pune"],
}
df = pd.DataFrame(data) 
print(df)
print(df.sort_values("Marks"))                      # ascending (low to high)
print(df.sort_values("Marks", ascending=False))     # descending (high to low)
