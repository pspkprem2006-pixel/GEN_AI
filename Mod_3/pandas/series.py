import pandas as pd

marks = pd.Series([85, 92, 78, 90], index=["Math", "Science", "English", "AI"])
print(marks)
print(marks["Math"])