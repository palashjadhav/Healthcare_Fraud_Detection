import sqlite3
import pandas as pd
con = sqlite3.connect('Healthcare.db')
provider_df = pd.read_csv("Train-1542865627584.csv")
inpatient_df = pd.read_csv("Train_Inpatientdata-1542865627584.csv")
outpatient_df = pd.read_csv("Train_Outpatientdata-1542865627584.csv")
beneficiary_df = pd.read_csv("Train_Beneficiarydata-1542865627584.csv")
provider_df.to_sql(name='Provider', con=con)
inpatient_df.to_sql(name='Inpatient', con=con)
outpatient_df.to_sql(name='Outpatient', con=con)
beneficiary_df.to_sql(name='Beneficiary', con=con)