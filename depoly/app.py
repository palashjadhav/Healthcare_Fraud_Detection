from flask import Flask, request, json, render_template

app = Flask(__name__)

import pickle
import pandas as pd
import numpy as np
import sqlite3


con = sqlite3.connect('Healthcare.db', check_same_thread=False)


with open('metadata/only_fraud_inpatient_diagnosis_code.pkl', 'rb') as file:
    only_fraud_inpatient_diagnosis_code = pickle.load(file)

with open('metadata/only_fraud_outpatient_diagnosis_code.pkl', 'rb') as file:
    only_fraud_outpatient_diagnosis_code = pickle.load(file)

with open('metadata/only_fraud_inpatient_procedure_code.pkl', 'rb') as file:
    only_fraud_inpatient_procedure_code = pickle.load(file)

with open('metadata/only_fraud_outpatient_procedure_code.pkl', 'rb') as file:
    only_fraud_outpatient_procedure_code = pickle.load(file)

with open('metadata/only_fraud_inpatient_physician.pkl', 'rb') as file:
    only_fraud_inpatient_physician = pickle.load(file)

with open('metadata/only_fraud_outpatient_physician.pkl', 'rb') as file:
    only_fraud_outpatient_physician = pickle.load(file)

with open('metadata/only_fraud_op_inpatient_physician.pkl', 'rb') as file:
    only_fraud_op_inpatient_physician = pickle.load(file)

with open('metadata/only_fraud_op_outpatient_physician.pkl', 'rb') as file:
    only_fraud_op_outpatient_physician = pickle.load(file)

with open('metadata/inpatient_county_columns.pkl', 'rb') as file:
    inpatient_county_columns = pickle.load(file)

with open('metadata/outpatient_county_columns.pkl', 'rb') as file:
    outpatient_county_columns = pickle.load(file)

with open('metadata/inpatient_county_pca.pkl', 'rb') as file:
    inpatient_county_pca = pickle.load(file)

with open('metadata/outpatient_county_pca.pkl', 'rb') as file:
    outpatient_county_pca = pickle.load(file)

with open('metadata/inpatient_diagnosis_code_count_columns.pkl', 'rb') as file:
    inpatient_diagnosis_code_count_columns = pickle.load(file)

with open('metadata/inpatient_diagnosis_code_prob.pkl', 'rb') as file:
    inpatient_diagnosis_code_prob = pickle.load(file)

with open('metadata/pca_inpatient_diagnosis_code.pkl', 'rb') as file:
    pca_inpatient_diagnosis_code = pickle.load(file)

with open('metadata/outpatient_diagnosis_code_count_columns.pkl', 'rb') as file:
    outpatient_diagnosis_code_count_columns = pickle.load(file)

with open('metadata/outpatient_diagnosis_code_prob.pkl', 'rb') as file:
    outpatient_diagnosis_code_prob = pickle.load(file)

with open('metadata/pca_outpatient_diagnosis_code.pkl', 'rb') as file:
    pca_outpatient_diagnosis_code = pickle.load(file)

with open('metadata/column_order_inpatient.pkl', 'rb') as file:
    column_order_inpatient = pickle.load(file)
with open('metadata/column_order_outpatient.pkl', 'rb') as file:
    column_order_outpatient = pickle.load(file)

with open('metadata/outpatient_model.pkl', 'rb') as file:
    outpatient_model = pickle.load(file)
with open('metadata/inpatient_model.pkl', 'rb') as file:
    inpatient_model = pickle.load(file)

with open('metadata/trans_outpatient.pkl', 'rb') as file:
    trans_outpatient = pickle.load(file)


@app.route('/healthcare')
def view():
    return render_template('pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    # data = json.loads(request.data)
    # provider_list = data['provider']
    # provider_str = "','".join(provider_list)
    provider_str = request.form['content']
    provider_str = provider_str.replace(",","','")
    inpatient_df = pd.read_sql("select * from Inpatient where Provider in ('"+ provider_str +"');", con, index_col='index')
    outpatient_df = pd.read_sql("select * from Outpatient where Provider in ('"+ provider_str +"');", con, index_col='index')
    disease = ['RenalDiseaseIndicator','ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 
                'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']
    final_inpatient = []
    final_outpatient= []
    if inpatient_df.shape[0] != 0:
        # Total Claim
        inpatient_claim_count_per_provider = inpatient_df.groupby("Provider").size().reset_index(name='Claim count').set_index('Provider')
        
        # Money Reimbursed
        inpatient_total_reimbursed_per_provider = inpatient_df.groupby('Provider')['InscClaimAmtReimbursed'].agg('sum')
        inpatient_input = pd.merge(inpatient_claim_count_per_provider,inpatient_total_reimbursed_per_provider, left_index=True, right_index=True)
        
        #Average Money Reimbursed
        inpatient_avg_reimbursed_per_provider = inpatient_df.groupby('Provider')['InscClaimAmtReimbursed'].agg('mean')
        inpatient_input = pd.merge(inpatient_input,inpatient_avg_reimbursed_per_provider, left_index=True, right_index=True)
        
        inpatient_input.rename(columns = {'InscClaimAmtReimbursed_x':'InscClaimAmtReimbursed_total'}, inplace = True)
        inpatient_input.rename(columns = {'InscClaimAmtReimbursed_y':'InscClaimAmtReimbursed_avg'}, inplace = True)
        
        # Total days admitted
        inpatient_df[['AdmissionDt','DischargeDt']] = inpatient_df[['AdmissionDt','DischargeDt']].apply(pd.to_datetime)
        inpatient_df['AdmitDays'] = (inpatient_df['DischargeDt'] - inpatient_df['AdmissionDt']).dt.days + 1
        inpatient_avg_admitdays_per_provider = inpatient_df.groupby('Provider')['AdmitDays'].agg('mean')
        inpatient_input = pd.merge(inpatient_input,inpatient_avg_admitdays_per_provider, left_index=True, right_index=True)
        
        # Claimend date and discharge date same
        inpatient_df[['ClaimStartDt','ClaimEndDt']] = inpatient_df[['ClaimStartDt','ClaimEndDt']].apply(pd.to_datetime)
        inpatient_df['IsClaimEndDtDischardeDt'] = inpatient_df['DischargeDt'] != inpatient_df['ClaimEndDt']
        inpatient_claim_discharge_date_per_provider = inpatient_df.groupby('Provider')['IsClaimEndDtDischardeDt'].agg('count')
        inpatient_input = pd.merge(inpatient_input,inpatient_claim_discharge_date_per_provider, left_index=True, right_index=True)
        
        # Fetch beneficiary
        beneficiary_df = pd.read_sql("select * from Beneficiary where BeneID in (select BeneID from Inpatient where Provider in ('"+ str(provider_str) +"'));", con, index_col='index')
        beneficiary_df["County"] = "C" + beneficiary_df["County"].astype(str)
        beneficiary_df["State"] = "S" + beneficiary_df["State"].astype(str)
        beneficiary_df["Gender"] = beneficiary_df["Gender"].replace(2,0)
        beneficiary_df["RenalDiseaseIndicator"] = beneficiary_df["RenalDiseaseIndicator"].replace('Y',1)

        beneficiary_df["ChronicCond_Alzheimer"] = beneficiary_df["ChronicCond_Alzheimer"].replace(2, 0)
        beneficiary_df["ChronicCond_Heartfailure"] = beneficiary_df["ChronicCond_Heartfailure"].replace(2, 0)
        beneficiary_df["ChronicCond_KidneyDisease"] = beneficiary_df["ChronicCond_KidneyDisease"].replace(2, 0)
        beneficiary_df["ChronicCond_Cancer"] = beneficiary_df["ChronicCond_Cancer"].replace(2, 0)
        beneficiary_df["ChronicCond_ObstrPulmonary"] = beneficiary_df["ChronicCond_ObstrPulmonary"].replace(2, 0)
        beneficiary_df["ChronicCond_Depression"] = beneficiary_df["ChronicCond_Depression"].replace(2, 0)
        beneficiary_df["ChronicCond_Diabetes"] = beneficiary_df["ChronicCond_Diabetes"].replace(2, 0)
        beneficiary_df["ChronicCond_IschemicHeart"] = beneficiary_df["ChronicCond_IschemicHeart"].replace(2, 0)
        beneficiary_df["ChronicCond_Osteoporasis"] = beneficiary_df["ChronicCond_Osteoporasis"].replace(2, 0)
        beneficiary_df["ChronicCond_rheumatoidarthritis"] = beneficiary_df["ChronicCond_rheumatoidarthritis"].replace(2, 0)
        beneficiary_df["ChronicCond_stroke"] = beneficiary_df["ChronicCond_stroke"].replace(2, 0)
        
        inpatient_beneficiary_detail = pd.merge(beneficiary_df, inpatient_df, how="inner", on='BeneID')
        
        # Gender count for each provider
        inpatient_gender = pd.crosstab(inpatient_beneficiary_detail['Provider'], inpatient_beneficiary_detail['Gender'], rownames=['Provider'], colnames=['Gender'])
        inpatient_gender.columns = ['Gender_0','Gender_1']
        inpatient_input = pd.merge(inpatient_input,inpatient_gender, left_index=True, right_index=True)
        
        # Race count for each provider
        inpatient_race = pd.crosstab(inpatient_beneficiary_detail['Provider'], inpatient_beneficiary_detail['Race'], rownames=['Provider'], colnames=['Race'])
        race_col = [1,2,3,5]
        for i in race_col:
            if not i in inpatient_race.columns:
                inpatient_race[i] = 0
        inpatient_race = inpatient_race[race_col]
        inpatient_race.columns = ['Race_1','Race_2','Race_3','Race_5']
        inpatient_input = pd.merge(inpatient_input,inpatient_race, left_index=True, right_index=True)
        
        # Age
        inpatient_beneficiary_detail['DOB'] = pd.to_datetime(inpatient_beneficiary_detail['DOB'] , format = '%Y-%m-%d')
        inpatient_beneficiary_detail['DOD'] = pd.to_datetime(inpatient_beneficiary_detail['DOD'],format = '%Y-%m-%d')
        inpatient_beneficiary_detail['Age'] = round(((inpatient_beneficiary_detail['DOD'] - inpatient_beneficiary_detail['DOB']).dt.days)/365)
        inpatient_beneficiary_detail["Age"].fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - inpatient_beneficiary_detail['DOB']).dt.days)/365),inplace=True)
        inpatient_avg_age = inpatient_beneficiary_detail.groupby('Provider')['Age'].agg('mean')
        inpatient_input = pd.merge(inpatient_input,inpatient_avg_age, left_index=True, right_index=True)
        
        
        # Fraud Diagnosis code
        inpatient_diagnosis_code = pd.DataFrame(np.vstack([inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_1"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_2"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_3"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_4"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_5"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_6"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_7"]],  inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_8"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_9"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_10"]]]), columns=["ClaimID","Provider","ClmDiagnosisCode"])
        inpatient_diagnosis_code = inpatient_diagnosis_code.dropna()
        inpatient_diagnosis_code['ClmDiagnosisCode'] = 'D' + inpatient_diagnosis_code['ClmDiagnosisCode'].astype(str) 
        inpatient_fraud_code_count = pd.crosstab(inpatient_diagnosis_code['Provider'], inpatient_diagnosis_code['ClmDiagnosisCode'], rownames=['Provider'], colnames=['ClmDiagnosisCode'])
        inpatient_fraud_code_count = inpatient_fraud_code_count.loc[:,inpatient_fraud_code_count.columns.isin(only_fraud_inpatient_diagnosis_code)]
        inpatient_fraud_code_count['only_fraud_code_count'] = inpatient_fraud_code_count.sum(axis=1)
        inpatient_input = pd.merge(inpatient_input,inpatient_fraud_code_count['only_fraud_code_count'], how='left', left_index=True, right_index=True)
        inpatient_input = inpatient_input.fillna(0)
        
        # Fraud Procedure code
        inpatient_procedure_code = pd.DataFrame(np.vstack([inpatient_df[["ClaimID","Provider","ClmProcedureCode_1"]], inpatient_df[["ClaimID","Provider","ClmProcedureCode_2"]], inpatient_df[["ClaimID","Provider","ClmProcedureCode_3"]], inpatient_df[["ClaimID","Provider","ClmProcedureCode_4"]], inpatient_df[["ClaimID","Provider","ClmProcedureCode_5"]], inpatient_df[["ClaimID","Provider","ClmProcedureCode_6"]]]), columns=["ClaimID","Provider","ClmProcedureCode"])
        inpatient_procedure_code = inpatient_procedure_code.dropna()
        inpatient_procedure_code['ClmProcedureCode'] = 'P' + inpatient_procedure_code['ClmProcedureCode'].astype(str)
        inpatient_procedure_code_count = pd.crosstab(inpatient_procedure_code['Provider'], inpatient_procedure_code['ClmProcedureCode'], rownames=['Provider'], colnames=['ClmProcedureCode'])
        inpatient_procedure_code_count = inpatient_procedure_code_count.loc[:,inpatient_procedure_code_count.columns.isin(only_fraud_inpatient_procedure_code)]
        inpatient_procedure_code_count['only_fraud_procedure_code_count'] = inpatient_procedure_code_count.sum(axis=1)
        inpatient_input = pd.merge(inpatient_input,inpatient_procedure_code_count['only_fraud_procedure_code_count'],how='left', left_index=True, right_index=True)
        inpatient_input = inpatient_input.fillna(0)
        
        
        # Fraud Physician Code
        inpatient_physician_count = pd.crosstab(inpatient_df['Provider'], inpatient_df['AttendingPhysician'], rownames=['Provider'], colnames=['AttendingPhysician'])
        inpatient_physician_count = inpatient_physician_count.loc[:,inpatient_physician_count.columns.isin(only_fraud_inpatient_physician)]
        inpatient_physician_count['only_fraud_physician_count'] = inpatient_physician_count.sum(axis=1)
        inpatient_input = pd.merge(inpatient_input,inpatient_physician_count['only_fraud_physician_count'],how='left', left_index=True, right_index=True)
        inpatient_input = inpatient_input.fillna(0)
        
        # Fraud operating Physician code
        inpatient_op_physician_count = pd.crosstab(inpatient_df['Provider'], inpatient_df['OperatingPhysician'], rownames=['Provider'], colnames=['OperatingPhysician'])
        inpatient_op_physician_count = inpatient_op_physician_count.loc[:,inpatient_op_physician_count.columns.isin(only_fraud_op_inpatient_physician)]
        inpatient_op_physician_count['only_fraud_op_physician_count'] = inpatient_op_physician_count.sum(axis=1)
        inpatient_input = pd.merge(inpatient_input,inpatient_op_physician_count['only_fraud_op_physician_count'],how='left', left_index=True, right_index=True)
        inpatient_input = inpatient_input.fillna(0)
        
        
        # Alive count
        inpatient_is_alive = inpatient_beneficiary_detail.groupby(['Provider','DOD'],dropna=False).size().reset_index(name='alive_count')
        inpatient_is_alive = inpatient_is_alive[inpatient_is_alive['DOD'].notnull()].set_index('Provider')[['alive_count']]
        inpatient_input = pd.merge(inpatient_input,inpatient_is_alive, on='Provider',how='left')
        inpatient_input = inpatient_input.fillna(0)
        
        
        # Number of doctor
        inpatient_total_doctor = pd.DataFrame(np.vstack([inpatient_df[["ClaimID","Provider","OperatingPhysician"]], inpatient_df[["ClaimID","Provider","AttendingPhysician"]], inpatient_df[["ClaimID","Provider","OtherPhysician"]]]), columns=["ClaimID","Provider","Doctor"])
        inpatient_total_doctor = inpatient_total_doctor.dropna()
        inpatient_doctor_count = inpatient_total_doctor.groupby('Provider')['Doctor'].unique()
        inpatient_doctor_count = inpatient_doctor_count.str.len()
        inpatient_input = pd.merge(inpatient_input,inpatient_doctor_count, left_index=True, right_index=True)
        inpatient_input = inpatient_input.fillna(0)
        
        # County
#         inpatient_county = pd.crosstab(inpatient_beneficiary_detail['Provider'], inpatient_beneficiary_detail['County'], rownames=['Provider'], colnames=['County'])
#         for i in inpatient_county_columns:
#             if not i in inpatient_county:
#                 inpatient_county[i] = 0
#         inpatient_county = inpatient_county[inpatient_county_columns]
        county_dict = dict()
        for i in inpatient_input.index:
            county_dict[i] = dict()
            for j in inpatient_county_columns:
                county_dict[i][j] = 0
        for k,l in dict(inpatient_beneficiary_detail.groupby('Provider')['County'].value_counts()).items():
            try:
                county_dict[k[0]][k[1]] = l
            except:
                pass
        inpatient_county = pd.DataFrame(county_dict).transpose()[inpatient_county_columns]
        
        array = inpatient_county_pca.transform(inpatient_county)
        inpatient_county_pca_data = pd.DataFrame(data = array,index = inpatient_county.index,columns=['County_'+str(i) for i in range(inpatient_county_pca.n_components_)])
        inpatient_input = pd.merge(inpatient_input,inpatient_county_pca_data, left_index=True, right_index=True)
        
        # Diagnosis Code
        inpatient_diagnosis_code = pd.DataFrame(np.vstack([inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_1"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_2"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_3"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_4"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_5"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_6"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_7"]],  inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_8"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_9"]], inpatient_df[["ClaimID","Provider","ClmDiagnosisCode_10"]]]), columns=["ClaimID","Provider","ClmDiagnosisCode"])
        inpatient_diagnosis_code = inpatient_diagnosis_code.dropna()
        inpatient_diagnosis_code['ClmDiagnosisCode'] = 'D' + inpatient_diagnosis_code['ClmDiagnosisCode'].astype(str)
#         inpatient_diagnosis_code_count = pd.crosstab(inpatient_diagnosis_code['Provider'], inpatient_diagnosis_code['ClmDiagnosisCode'], rownames=['Provider'], colnames=['ClmDiagnosisCode'])
#         for col in inpatient_diagnosis_code_count.columns.values:
#             try:
#                 inpatient_diagnosis_code_count[col] = inpatient_diagnosis_code_prob[col]
#             except:
#                 inpatient_diagnosis_code_count[col] = 0
#         for i in inpatient_diagnosis_code_count_columns:
#             if not i in inpatient_diagnosis_code_count:
#                 inpatient_diagnosis_code_count[i] = 0
#         inpatient_diagnosis_code_count = inpatient_diagnosis_code_count[inpatient_diagnosis_code_count_columns]
        
        diagnosis_dict = dict()
        for i in inpatient_input.index:
            diagnosis_dict[i] = dict()
            for j in inpatient_diagnosis_code_count_columns:
                diagnosis_dict[i][j] = 0
        for k,l in dict(inpatient_diagnosis_code.groupby('Provider')['ClmDiagnosisCode'].value_counts()).items():
            try:
                diagnosis_dict[k[0]][k[1]] = inpatient_diagnosis_code_prob[k[1]]
            except:
                pass
        inpatient_diagnosis_code_count = pd.DataFrame(diagnosis_dict).transpose()
        
        array = pca_inpatient_diagnosis_code.transform(inpatient_diagnosis_code_count)
        inpatient_diagnosis_code_count_pca = pd.DataFrame(data = array,index = inpatient_diagnosis_code_count.index,columns=['Diagnosis_'+str(i) for i in range(pca_inpatient_diagnosis_code.n_components_)])
        inpatient_input = pd.merge(inpatient_input,inpatient_diagnosis_code_count_pca, left_index=True, right_index=True)
        
        # Disease
#         disease = ['ChronicCond_Osteoporasis']
        for j in disease:
            temp_dict = dict()
            for key, value in dict(inpatient_beneficiary_detail.groupby('Provider')[j].value_counts()).items():
                
                try:
                    temp_dict[key[0]][j + '_'+str(key[1])] = value
                except:
                    temp_dict[key[0]] = dict()
                    temp_dict[key[0]][j + '_'+str(key[1])] = value
            temp_disease_pd = pd.DataFrame(temp_dict).transpose()
            temp_disease = [j+'_0', j+'_1']
            for i in temp_disease:
                if not i in temp_disease_pd:
                    temp_disease_pd[i] = 0
            temp_disease_pd = temp_disease_pd[temp_disease]
            temp_disease_pd.fillna(0,inplace=True)
            inpatient_input = pd.merge(inpatient_input,temp_disease_pd, left_index=True, right_index=True)
        
        inpatient_input = inpatient_input[column_order_inpatient]
        inpatient_result = inpatient_model.predict(inpatient_input)
        final_inpatient = []
        for i ,j in zip(inpatient_input.index,inpatient_result):
            if j == 0:
                j = False
            else:
                j = True
            final_inpatient.append([i,str(j)])
        # print("Inpatient Prediction")
        # print(final_inpatient)
        
    if outpatient_df.shape[0] != 0:
        # Total Claim
        outpatient_claim_count_per_provider = outpatient_df.groupby("Provider").size().reset_index(name='Claim count').set_index('Provider')
        
        # Money Reimbursed
        outpatient_total_reimbursed_per_provider = outpatient_df.groupby('Provider')['InscClaimAmtReimbursed'].agg('sum')
        outpatient_input = pd.merge(outpatient_claim_count_per_provider,outpatient_total_reimbursed_per_provider, left_index=True, right_index=True)
        
        # Average Money Reimbursed
        outpatient_avg_reimbursed_per_provider = outpatient_df.groupby('Provider')['InscClaimAmtReimbursed'].agg('mean')
        outpatient_input = pd.merge(outpatient_input,outpatient_avg_reimbursed_per_provider, left_index=True, right_index=True)
        
        outpatient_input.rename(columns = {'InscClaimAmtReimbursed_x':'InscClaimAmtReimbursed_total'}, inplace = True)
        outpatient_input.rename(columns = {'InscClaimAmtReimbursed_y':'InscClaimAmtReimbursed_avg'}, inplace = True)
        
        # Fetch beneficiary
        beneficiary_df = pd.read_sql("select * from Beneficiary where BeneID in (select BeneID from Outpatient where Provider in ('"+ str(provider_str) +"'));", con, index_col='index')
        beneficiary_df["County"] = "C" + beneficiary_df["County"].astype(str)
        beneficiary_df["State"] = "S" + beneficiary_df["State"].astype(str)
        beneficiary_df["Gender"] = beneficiary_df["Gender"].replace(2,0)
        beneficiary_df["RenalDiseaseIndicator"] = beneficiary_df["RenalDiseaseIndicator"].replace('Y',1)

        beneficiary_df["ChronicCond_Alzheimer"] = beneficiary_df["ChronicCond_Alzheimer"].replace(2, 0)
        beneficiary_df["ChronicCond_Heartfailure"] = beneficiary_df["ChronicCond_Heartfailure"].replace(2, 0)
        beneficiary_df["ChronicCond_KidneyDisease"] = beneficiary_df["ChronicCond_KidneyDisease"].replace(2, 0)
        beneficiary_df["ChronicCond_Cancer"] = beneficiary_df["ChronicCond_Cancer"].replace(2, 0)
        beneficiary_df["ChronicCond_ObstrPulmonary"] = beneficiary_df["ChronicCond_ObstrPulmonary"].replace(2, 0)
        beneficiary_df["ChronicCond_Depression"] = beneficiary_df["ChronicCond_Depression"].replace(2, 0)
        beneficiary_df["ChronicCond_Diabetes"] = beneficiary_df["ChronicCond_Diabetes"].replace(2, 0)
        beneficiary_df["ChronicCond_IschemicHeart"] = beneficiary_df["ChronicCond_IschemicHeart"].replace(2, 0)
        beneficiary_df["ChronicCond_Osteoporasis"] = beneficiary_df["ChronicCond_Osteoporasis"].replace(2, 0)
        beneficiary_df["ChronicCond_rheumatoidarthritis"] = beneficiary_df["ChronicCond_rheumatoidarthritis"].replace(2, 0)
        beneficiary_df["ChronicCond_stroke"] = beneficiary_df["ChronicCond_stroke"].replace(2, 0)
        
        outpatient_beneficiary_detail = pd.merge(beneficiary_df, outpatient_df, how="inner", on='BeneID')
        
        # Gender count for each provider
        outpatient_gender = pd.crosstab(outpatient_beneficiary_detail['Provider'], outpatient_beneficiary_detail['Gender'], rownames=['Provider'], colnames=['Gender'])
        outpatient_gender.columns = ['Gender_0','Gender_1']
        outpatient_input = pd.merge(outpatient_input,outpatient_gender, left_index=True, right_index=True)
        
        # Race count for each provider
        outpatient_race = pd.crosstab(outpatient_beneficiary_detail['Provider'], outpatient_beneficiary_detail['Race'], rownames=['Provider'], colnames=['Race'])
        race_col = [1,2,3,5]
        for i in race_col:
            if not i in outpatient_race.columns:
                outpatient_race[i] = 0
        outpatient_race = outpatient_race[race_col]
        outpatient_race.columns = ['Race_1','Race_2','Race_3','Race_5']
        outpatient_input = pd.merge(outpatient_input,outpatient_race, left_index=True, right_index=True)
        
        # Age
        outpatient_beneficiary_detail['DOB'] = pd.to_datetime(outpatient_beneficiary_detail['DOB'] , format = '%Y-%m-%d')
        outpatient_beneficiary_detail['DOD'] = pd.to_datetime(outpatient_beneficiary_detail['DOD'],format = '%Y-%m-%d')
        outpatient_beneficiary_detail['Age'] = round(((outpatient_beneficiary_detail['DOD'] - outpatient_beneficiary_detail['DOB']).dt.days)/365)
        outpatient_beneficiary_detail["Age"].fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - outpatient_beneficiary_detail['DOB']).dt.days)/365),inplace=True)
        outpatient_avg_age = outpatient_beneficiary_detail.groupby('Provider')['Age'].agg('mean')
        outpatient_input = pd.merge(outpatient_input,outpatient_avg_age, left_index=True, right_index=True)
        
        # Fraud Diagnosis code
        outpatient_diagnosis_code = pd.DataFrame(np.vstack([outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_1"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_2",]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_3"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_4"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_5"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_6"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_7"]],  outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_8"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_9"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_10"]]]), columns=["ClaimID","Provider","ClmDiagnosisCode"])
        outpatient_diagnosis_code = outpatient_diagnosis_code.dropna()
        outpatient_diagnosis_code['ClmDiagnosisCode'] = 'D' + outpatient_diagnosis_code['ClmDiagnosisCode'].astype(str)
        outpatient_fraud_code_count = pd.crosstab(outpatient_diagnosis_code['Provider'], outpatient_diagnosis_code['ClmDiagnosisCode'], rownames=['Provider'], colnames=['ClmDiagnosisCode'])
        outpatient_fraud_code_count = outpatient_fraud_code_count.loc[:,outpatient_fraud_code_count.columns.isin(only_fraud_outpatient_diagnosis_code)]
        outpatient_fraud_code_count['only_fraud_code_count'] = outpatient_fraud_code_count.sum(axis=1)
        outpatient_input = pd.merge(outpatient_input,outpatient_fraud_code_count['only_fraud_code_count'], how='left', left_index=True, right_index=True)
        outpatient_input = outpatient_input.fillna(0)
        
        # Fraud Procedure code
        outpatient_procedure_code = pd.DataFrame(np.vstack([outpatient_df[["ClaimID","Provider","ClmProcedureCode_1"]], outpatient_df[["ClaimID","Provider","ClmProcedureCode_2"]], outpatient_df[["ClaimID","Provider","ClmProcedureCode_3"]], outpatient_df[["ClaimID","Provider","ClmProcedureCode_4"]], outpatient_df[["ClaimID","Provider","ClmProcedureCode_5"]], outpatient_df[["ClaimID","Provider","ClmProcedureCode_6"]]]), columns=["ClaimID","Provider","ClmProcedureCode"])
        outpatient_procedure_code = outpatient_procedure_code.dropna()
        outpatient_procedure_code['ClmProcedureCode'] = 'P' + outpatient_procedure_code['ClmProcedureCode'].astype(str)
        outpatient_procedure_code_count = pd.crosstab(outpatient_procedure_code['Provider'], outpatient_procedure_code['ClmProcedureCode'], rownames=['Provider'], colnames=['ClmProcedureCode'])
        outpatient_procedure_code_count = outpatient_procedure_code_count.loc[:,outpatient_procedure_code_count.columns.isin(only_fraud_outpatient_procedure_code)]
        outpatient_procedure_code_count['only_fraud_procedure_code_count'] = outpatient_procedure_code_count.sum(axis=1)
        outpatient_input = pd.merge(outpatient_input,outpatient_procedure_code_count['only_fraud_procedure_code_count'], how='left', left_index=True, right_index=True)
        outpatient_input = outpatient_input.fillna(0)
        
        
        # Fraud Physician Code
        outpatient_physician_count = pd.crosstab(outpatient_df['Provider'], outpatient_df['AttendingPhysician'], rownames=['Provider'], colnames=['AttendingPhysician'])
        outpatient_physician_count = outpatient_physician_count.loc[:,outpatient_physician_count.columns.isin(only_fraud_outpatient_physician)]
        outpatient_physician_count['only_fraud_physician_count'] = outpatient_physician_count.sum(axis=1)
        outpatient_input = pd.merge(outpatient_input,outpatient_physician_count['only_fraud_physician_count'],how='left', left_index=True, right_index=True)
        outpatient_input = outpatient_input.fillna(0)
        
        # Fraud Operating Physician code
        outpatient_op_physician_count = pd.crosstab(outpatient_df['Provider'], outpatient_df['OperatingPhysician'], rownames=['Provider'], colnames=['OperatingPhysician'])
        outpatient_op_physician_count = outpatient_op_physician_count.loc[:,outpatient_op_physician_count.columns.isin(only_fraud_op_outpatient_physician)]
        outpatient_op_physician_count['only_fraud_op_physician_count'] = outpatient_op_physician_count.sum(axis=1)
        outpatient_input = pd.merge(outpatient_input,outpatient_op_physician_count['only_fraud_op_physician_count'],how='left', left_index=True, right_index=True)
        outpatient_input = outpatient_input.fillna(0)
        
        # Alive count
        outpatient_is_alive = outpatient_beneficiary_detail.groupby(['Provider','DOD'],dropna=False).size().reset_index(name='alive_count')
        outpatient_is_alive = outpatient_is_alive[outpatient_is_alive['DOD'].notnull()].set_index('Provider')[['alive_count']]
        outpatient_input = pd.merge(outpatient_input,outpatient_is_alive, on='Provider',how='left')
        outpatient_input = outpatient_input.fillna(0)
        
        # Number of Doctor
        outpatient_total_doctor = pd.DataFrame(np.vstack([outpatient_df[["ClaimID","Provider","OperatingPhysician"]], outpatient_df[["ClaimID","Provider","AttendingPhysician"]], outpatient_df[["ClaimID","Provider","OtherPhysician"]]]), columns=["ClaimID","Provider","Doctor"])
        outpatient_total_doctor = outpatient_total_doctor.dropna()
        outpatient_doctor_count = outpatient_total_doctor.groupby('Provider')['Doctor'].unique()
        outpatient_doctor_count = outpatient_doctor_count.str.len()
        outpatient_input = pd.merge(outpatient_input,outpatient_doctor_count, left_index=True, right_index=True)
        outpatient_input = outpatient_input.fillna(0)
        
        
        # County
#         outpatient_county = pd.crosstab(outpatient_beneficiary_detail['Provider'], outpatient_beneficiary_detail['County'], rownames=['Provider'], colnames=['County'])
#         for i in outpatient_county_columns:
#             if not i in outpatient_county:
#                 outpatient_county[i] = 0
#         outpatient_county = outpatient_county[outpatient_county_columns]

        # County Dict
        county_dict = dict()
        for i in outpatient_input.index:
            county_dict[i] = dict()
            for j in outpatient_county_columns:
                county_dict[i][j] = 0
        for k,l in dict(outpatient_beneficiary_detail.groupby('Provider')['County'].value_counts()).items():
            try:
                county_dict[k[0]][k[1]] = l
            except:
                pass
        outpatient_county = pd.DataFrame(county_dict).transpose()[outpatient_county_columns]
        
        array = outpatient_county_pca.transform(outpatient_county)
        outpatient_county_pca_data = pd.DataFrame(data = array,index = outpatient_county.index,columns=['County_'+str(i) for i in range(outpatient_county_pca.n_components_)])
        outpatient_input = pd.merge(outpatient_input,outpatient_county_pca_data, left_index=True, right_index=True)
        
        # Diagnosis Code
        outpatient_diagnosis_code = pd.DataFrame(np.vstack([outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_1"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_2"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_3"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_4"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_5"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_6"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_7"]],  outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_8"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_9"]], outpatient_df[["ClaimID","Provider","ClmDiagnosisCode_10"]]]), columns=["ClaimID","Provider","ClmDiagnosisCode"])
        outpatient_diagnosis_code = outpatient_diagnosis_code.dropna()
        outpatient_diagnosis_code['ClmDiagnosisCode'] = 'D' + outpatient_diagnosis_code['ClmDiagnosisCode'].astype(str)
#         outpatient_diagnosis_code_count = pd.crosstab(outpatient_diagnosis_code['Provider'], outpatient_diagnosis_code['ClmDiagnosisCode'], rownames=['Provider'], colnames=['ClmDiagnosisCode'])
#         for col in outpatient_diagnosis_code_count.columns.values:
#             try:
#                 outpatient_diagnosis_code_count[col] = outpatient_diagnosis_code_prob[col]
#             except:
#                 outpatient_diagnosis_code_count[col] = 0
#         for i in outpatient_diagnosis_code_count_columns:
#             if not i in outpatient_diagnosis_code_count:
#                 outpatient_diagnosis_code_count[i] = 0
#         outpatient_diagnosis_code_count = outpatient_diagnosis_code_count[outpatient_diagnosis_code_count_columns]
        
        # diagnosis dict
        diagnosis_dict = dict()
        for i in outpatient_input.index:
            diagnosis_dict[i] = dict()
            for j in outpatient_diagnosis_code_count_columns:
                diagnosis_dict[i][j] = 0
        for k,l in dict(outpatient_diagnosis_code.groupby('Provider')['ClmDiagnosisCode'].value_counts()).items():
            try:
                diagnosis_dict[k[0]][k[1]] = outpatient_diagnosis_code_prob[k[1]]
            except:
                pass
        outpatient_diagnosis_code_count = pd.DataFrame(diagnosis_dict).transpose()
        
        array = pca_outpatient_diagnosis_code.transform(outpatient_diagnosis_code_count)
        outpatient_diagnosis_code_count_pca = pd.DataFrame(data = array,index = outpatient_diagnosis_code_count.index,columns=['Diagnosis_'+str(i) for i in range(pca_outpatient_diagnosis_code.n_components_)])
        outpatient_input = pd.merge(outpatient_input,outpatient_diagnosis_code_count_pca, left_index=True, right_index=True)
        
        # Disease
        for j in disease:
            temp_dict = dict()
            for key, value in dict(outpatient_beneficiary_detail.groupby('Provider')[j].value_counts()).items():
                
                try:
                    temp_dict[key[0]][j + '_'+str(key[1])] = value
                except:
                    temp_dict[key[0]] = dict()
                    temp_dict[key[0]][j + '_'+str(key[1])] = value
            temp_disease_pd = pd.DataFrame(temp_dict).transpose()
            temp_disease = [j+'_0', j+'_1']
            for i in temp_disease:
                if not i in temp_disease_pd:
                    temp_disease_pd[i] = 0
            temp_disease_pd = temp_disease_pd[temp_disease]
            temp_disease_pd.fillna(0,inplace=True)
            outpatient_input = pd.merge(outpatient_input,temp_disease_pd, left_index=True, right_index=True)
        
        
        # for j in outpatient_input.columns:
        #     if not  j in  column_order_outpatient:
        #         print(j)
        outpatient_input = outpatient_input[column_order_outpatient]
        outpatient_final_sd = trans_outpatient.transform(outpatient_input)
        outpatient_final_sd = pd.DataFrame(data=outpatient_final_sd,columns=outpatient_input.columns.values)
        outpatient_result = outpatient_model.predict(outpatient_final_sd)
        final_outpatient = []
        for i ,j in zip(outpatient_input.index,outpatient_result):
            if j == 0:
                j = False
            else:
                j = True
            final_outpatient.append([i,str(j)])
        # print("Outpatient Prediction")
        # print(final_outpatient)
    return render_template('index.html', inpatient=final_inpatient, outpatient=final_outpatient)
    # return {"inpatient":final_inpatient, "outpatient":final_outpatient}

@app.route('/')
def test():
    return "Server is running"

if __name__ == "__main__":
    app.run()