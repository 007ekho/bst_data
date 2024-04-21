# import joblib
# import streamlit as st
# import numpy as np
# import wget

# model_name = 'RF_Loan_model.joblib'
# file_url = "https://raw.githubusercontent.com/manifoldailearning/Complete-MLOps-BootCamp/main/Build-ML-App-Streamlit/RF_Loan_model.joblib"
# wget.download(file_url)
# model = joblib.load(model_name)

# def prediction(Gender,Married,Dependents,
#          Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
#          LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
#         if Gender == "Male":
#             Gender = 1
#         else:
#             Gender = 0

#         if Married == "Yes":
#             Married = 1
#         else:
#             Married = 0

#         if Education == "Graduate":
#             Education = 0
#         else:
#             Education = 1
        
#         if Self_Employed == "Yes":
#             Self_Employed = 1
#         else:
#             Self_Employed = 0

#         if Credit_History == "Outstanding Loan":
#             Credit_History = 1
#         else:
#             Credit_History = 0   
        
#         if Property_Area == "Rural":
#             Property_Area = 0
#         elif Property_Area == "Semi Urban":
#             Property_Area = 1  
#         else:
#             Property_Area = 2  
#         Total_Income =    np.log(ApplicantIncome + CoapplicantIncome)

#         prediction = model.predict([[Gender, Married, Dependents, Education, Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,Total_Income]])
#         print(print(prediction))

#         if prediction==0:
#             pred = "Rejected"

#         else:
#             pred = "Approved"
#         return pred        


# def main():
#     # Front end
#     st.title("Welcome to Loan Application")
#     st.header("Please enter your details to proceed with your loan Application")
#     Gender = st.selectbox("Gender",("Male","Female"))
#     Married = st.selectbox("Married",("Yes","No"))
#     Dependents = st.number_input("Number of Dependents")
#     Education = st.selectbox("Education",("Graduate","Not Graduate"))
#     Self_Employed = st.selectbox("Self Employed",("Yes","No"))
#     ApplicantIncome = st.number_input("Applicant Income")
#     CoapplicantIncome = st.number_input("Coapplicant Income")
#     LoanAmount = st.number_input("LoanAmount")
#     Loan_Amount_Term = st.number_input("Loan Amount Term")
#     Credit_History = st.selectbox("Credit History",("Outstanding Loan", "No Outstanding Loan"))
#     Property_Area = st.selectbox("Property Area",("Rural","Urban","Semi Urban"))

#     if st.button("Predict"):
#         result = prediction(Gender,Married,Dependents,
#          Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
#          LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
        
#         if result == "Approved":
#             st.success("Your loan Application is Approved")
#         else:
#             st.error("Your loan Application is Rejected")

# if __name__ == "__main__":
#     main()






#model 2

# import streamlit as st
# import joblib
# model_name = 'RF_loan_model.joblib'
# model = joblib.load(model_name)
# import numpy as np

# def prediction(Gender,Married, Dependents,
#                             Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
#                             LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
#     if Gender =="Male":
#         Gender =1
#     else: Gender = 0

#     if Married =="Yes":
#         Married =1
#     else: Married = 0

#     if Education =="Graduate":
#         Education =1
#     else: Education = 0

#     if Self_Employed =="Yes":
#         Self_Employed =1
#     else: Self_Employed = 0

#     if Credit_History =="Outstanding Loan":
#         Credit_History =1
#     else: Credit_History = 0

#     if Property_Area =="Rural":
#         Property_Area = 0
#     elif Property_Area =="Semi Urban":
#         Property_Area = 1
#     else:
#         Property_Area = 2

#     Total_Income = np.log(ApplicantIncome + CoapplicantIncome)

#     prediction = model.predict([[Gender, Married, Dependents, Education, Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,Total_Income]])
    
#     prediction_proba= model.predict_proba([[Gender, Married, Dependents, Education, Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,Total_Income]])

#     if prediction == 0:
#         pred = "Rejected"
#     else:
#         pred = "Approved"
    
#     return pred, prediction_proba


# def main():
#     st.title("welcome to teh loan application")
#     st.header("please enter your details")

#     Gender = st.selectbox("Gender",("Male","Female"))
#     Married = st.selectbox("Married",("Yes", "No"))
#     Dependents = st.number_input("Number of dependents")
#     Education = st.selectbox("Education",("Graduate","Not Graduate"))
#     Self_Employed = st.selectbox("Self Employed",("Yes", "No"))
#     ApplicantIncome = st.number_input("Applicat Income")
#     CoapplicantIncome = st.number_input("Coapplicant Income")
#     LoanAmount= st.number_input("LoanAmount")
#     Loan_Amount_Term= st.number_input("Loan Amount Term")
#     Credit_History= st.selectbox("Credit History",("Outstanding Loan", "No Outstanding Loan"))
#     Property_Area= st.selectbox("Property Area",("Rural","Urban","Semi Urban"))
    
#     if st.button("Predict"):
#         result,probas = prediction(Gender,Married, Dependents,
#                             Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
#                             LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
#         proba_rejected = probas[0][0]  # Probability of rejection
#         proba_approved = probas[0][1]  # Probability of approval

#         if result =="Approved":
#             st.write(f"Probability of Approval: {proba_approved * 100:.2f} %")
#             st.success("your loan application is approved")
            
            
#         else:
#             st.write(f"Probability of Rejection: {proba_rejected * 100:.2f}%")
#             st.error("Your loan Application is Rejected")
        
#         # proba_rejected = probas[0][0]  # Probability of rejection
#         # proba_approved = probas[0][1]  # Probability of approval

#         # st.write(f"Probability of Rejection: {proba_rejected:.2f}")
#         # st.write(f"Probability of Approval: {proba_approved:.2f}")
        

    
# if __name__ == "__main__":
#     main()






#model 3

import streamlit as st
import joblib
model_name = 'pred_model.joblib'
model = joblib.load(model_name)
import numpy as np

def prediction(id,hour,weekday,start_stop):
    if weekday =="Monday":
        weekday =1
    elif weekday =="Tuesday":
        weekday =2
    elif weekday =="Wednesday":
        weekday =3
    elif weekday =="Thurssday":
        weekday =4
    elif weekday =="Friday":
        weekday =5
    elif weekday =="Saturday":
        weekday =6
    else: weekday = 7

    
    prediction = model.predict([[id,hour,weekday,start_stop]])
    print(prediction)
    prediction_proba= model.predict_proba([[id,hour,weekday,start_stop]])
    print(prediction_proba)
    if prediction == 1:
        pred = "z1"
    elif prediction == 2:
        pred = "z2"
    elif prediction == 3:
        pred = "z3"
    elif prediction == 4:
        pred = "z4"
    elif prediction == 5:
        pred = "z5"
    elif prediction == 6:
        pred = "z6"
    
    else:
        pred = "z7"
    

    return pred, prediction_proba


def main():
    st.title("welcome to the z application")
    st.header("please enter your details")

    id = st.selectbox("id",(1000010, 1000030, 1000008, 1000014, 1000029, 1000005, 1000002,
       1000012, 1000011, 1000001, 1000018, 1000022, 1000007, 1000021,
       1000028, 1000004, 1000025, 1000023, 1000024, 1000006, 1000026,
       1000003, 1000016, 1000020, 1000015, 1000017, 1000013, 1000009,
       1000027, 1000019))
    hour = st.number_input("24 hours input")
    weekday= st.selectbox("weekday",("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
    start_stop= st.selectbox("start_stop",(1,2,3,4,5,6,7))
    
    if st.button("Predict"):
        result,probas = prediction(id,hour,weekday,start_stop)
        proba_z1 = probas[0][0]  
        proba_z2 = probas[0][1]  
        proba_z3 = probas[0][2]  
        proba_z4 = probas[0][3]  
        proba_z5 = probas[0][4]  
        proba_z6 = probas[0][5]  
        proba_z7 = probas[0][6]  
        if result =="z1":
            st.write(f"stop  {result}")
            st.write(f"Probability of z1: {proba_z1 * 100:.2f} %")
        
        elif result =="z2":
            st.write(f"stop  {result}")
            st.write(f"Probability of z2: {proba_z2 * 100:.2f} %")

        elif result =="z3":
            st.write(f"stop  {result}")
            st.write(f"Probability of z3: {proba_z3 * 100:.2f} %")

        elif result =="z4":
            st.write(f"stop  {result}")
            st.write(f"Probability of z4: {proba_z4 * 100:.2f} %")

        elif result =="z5":
            st.write(f"stop  {result}")
            st.write(f"Probability of z5: {proba_z5 * 100:.2f} %")  

        elif result =="z6":
            st.write(f"stop  {result}")
            st.write(f"Probability of z6: {proba_z6 * 100:.2f} %")
            
        else:
            st.write(f"stop  {result}")
            st.write(f"Probability of z7: {proba_z7 * 100:.2f}%")
            # st.error("Your loan Application is Rejected")
        
        # proba_rejected = probas[0][0]  # Probability of rejection
        # proba_approved = probas[0][1]  # Probability of approval

        # st.write(f"Probability of Rejection: {proba_rejected:.2f}")
        # st.write(f"Probability of Approval: {proba_approved:.2f}")
        

    
if __name__ == "__main__":
    main()
