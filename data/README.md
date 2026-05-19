# Data

Place the raw CSV files here before running experiments.

## Expected files

```
data/
  SQLA5(1).csv  ...  SQLA5(7).csv
  SQLA6(1).csv  ...  SQLA6(7).csv
  SQLA7(1).csv  ...  SQLA7(7).csv
  SQLA8(1).csv  ...  SQLA8(7).csv
  SQLA9(1).csv  ...  SQLA9(7).csv
  SQLB9(1).csv  ...  SQLB9(7).csv
```

- 6 sensor groups × 7 days = 42 files
- Each file contains time-series readings from anchor bolt load sensors (锚杆) and surrounding rock sensors (围岩)
- Sampling interval: 5 seconds
- Files are excluded from Git via `.gitignore` due to size (~60 MB total)
