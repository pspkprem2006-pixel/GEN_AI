import pandas as pd
df = pd.DataFrame({"Name": ["A", "B"], "Score": [90, 85]})


pd.DataFrame([{"Name": "A", "Score": 90}, {"Name": "B", "Score": 85}])

# Exporting your cleaned/analyzed data:
df.to_csv("cleaned.csv", index=False)     
df.to_excel("report.xlsx", index=False)   
df.to_json("data.json", orient="records")