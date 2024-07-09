import requests

url = "https://credit-default-prediction-x14o.onrender.com/predict"
headers = {"Content-Type": "application/json"}

# Replace the following dictionary with your actual data
data = {
    "AMT_INCOME_TOTAL": 0,
    "AMT_CREDIT": 0,
    "REGION_POPULATION_RELATIVE": 0,
    "DAYS_REGISTRATION": 0,
    "DAYS_BIRTH": 0,
    "DAYS_ID_PUBLISH": 0,
    "FLAG_WORK_PHONE": 0,
    "FLAG_PHONE": 0,
    "REGION_RATING_CLIENT_W_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0,
    "FLAG_DOCUMENT_3": 0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": 0,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Other",
    "NAME_FAMILY_STATUS": "Civil marriage",
    "OCCUPATION_TYPE": "Laborers",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "CREDIT_ACTIVE_Active_count_Bureau": 0,
    "CREDIT_ACTIVE_Closed_count_Bureau": 0,
    "DAYS_CREDIT_Bureau": 0,
    "AMT_INSTALMENT_mean_HCredit_installments": 0,
    "DAYS_INSTALMENT_mean_HCredit_installments": 0,
    "NUM_INSTALMENT_NUMBER_mean_HCredit_installments": 0,
    "NUM_INSTALMENT_VERSION_mean_HCredit_installments": 0,
    "NAME_CONTRACT_STATUS_Active_count_pos_cash": 0,
    "NAME_CONTRACT_STATUS_Completed_count_pos_cash": 0,
    "SK_DPD_DEF_pos_cash": 0,
    "NAME_CONTRACT_STATUS_Refused_count_HCredit_PApp": 0,
    "NAME_GOODS_CATEGORY_Other_count_HCredit_PApp": 0,
    "NAME_PORTFOLIO_Cash_count_HCredit_PApp": 0,
    "NAME_PRODUCT_TYPE_walk_in_count_HCredit_PApp": 0,
    "NAME_SELLER_INDUSTRY_Other_count_HCredit_PApp": 0,
    "NAME_YIELD_GROUP_high_count_HCredit_PApp": 0,
    "NAME_YIELD_GROUP_low_action_count_HCredit_PApp": 0,
    "AMT_CREDIT_HCredit_PApp": 0,
    "SELLERPLACE_AREA_HCredit_PApp": 0
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print("Prediction:", result)
else:
    print("Error:", response.status_code, response.text)