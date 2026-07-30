import pandas as pd
data = {
    "Name": ["Aarav", "Diya", "Kabir", "Meera"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 78, 90],
    "City": ["Chennai", "Delhi", "Mumbai", "Pune"],
}
df = pd.DataFrame(data)
print(df)
print("===============")
high = df[df["Marks"] > 85]
print(high)
print("===============")
combc=df[(df["Marks"] > 80) & (df["Age"] < 21)] 
print(combc)