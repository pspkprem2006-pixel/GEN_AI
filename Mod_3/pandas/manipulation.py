import pandas as pd
data = {
    "Name": ["Aarav", "Diya", "Kabir", "Meera"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 78, 90],
    "City": ["Chennai", "Delhi", "Mumbai", "Pune"],
}
df = pd.DataFrame(data)
print(df)
df["Passed"] = df["Marks"] >= 40         
df["Grade"] = df["Marks"] / 10          
print(df)
# Apply a custom function to a column with .apply():
def grade_letter(mark):
    if mark >= 90: return "A"
    elif mark >= 75: return "B"
    else: return "C"

df["Letter"] = df["Marks"].apply(grade_letter)
print(df[["Name", "Marks", "Letter"]])
