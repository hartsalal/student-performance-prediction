import streamlit as st
import pandas as pd
import joblib

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Student Success Prediction",
    page_icon="🎓",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("student_performance_model.joblib")
scaler = joblib.load("student_scaler.joblib")

# ==========================
# HEADER
# ==========================
st.markdown("""
<h1 style='text-align:center;'>
🎓 Student Success Prediction System
</h1>

<h4 style='text-align:center; color:gray;'>
Machine Learning Based Academic Performance Analysis
</h4>
""", unsafe_allow_html=True)

st.divider()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("📚 Navigation")

menu = st.sidebar.radio(
    "Select Menu",
    [
        "🏠 Home",
        "📈 Data Insights",
        "🎯 Prediction Center",
        "🧠 Model Analysis",
        "👨‍💻 Project Info"
    ]
)

# ==========================
# HOME
# ==========================
if menu == "🏠 Home":

    st.subheader("Welcome")

    st.write("""
This application predicts student academic performance
using Machine Learning algorithms.
The best model obtained from training is Random Forest.
""")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(
            """
            📊 Dataset

            2392 Records
            """
        )

    with col2:
        st.success(
            """
            🎯 Accuracy

            91.23%
            """
        )

    with col3:
        st.warning(
            """
            📚 Features

            14 Variables
            """
        )

    with col4:
        st.error(
            """
            🤖 Best Model

            Random Forest
            """
        )

    st.divider()

    st.subheader("Project Objective")

    st.write("""
The objective of this project is to analyze student characteristics
and predict GradeClass using supervised machine learning techniques.
""")

# ==========================
# DATA INSIGHTS
# ==========================
elif menu == "📈 Data Insights":

    st.subheader("Dataset Insights")

    st.write("""
Key findings from exploratory data analysis:
""")

    st.markdown("""
### Most Influential Features

1. GPA
2. Absences
3. StudyTimeWeekly

### Correlation Findings

- GPA strongly influences GradeClass.
- High absences tend to reduce academic performance.
- More study time generally leads to better academic outcomes.
""")

    st.subheader("Top Influential Features")

    feature_df = pd.DataFrame({
        "Feature": [
            "GPA",
            "Absences",
            "StudyTimeWeekly"
        ],
        "Importance": [
        0.42,
        0.31,
        0.12
    ]
})

    st.bar_chart(
        feature_df.set_index("Feature")
)

# ==========================
# PREDICTION CENTER
# ==========================
elif menu == "🎯 Prediction Center":

    st.subheader("Student Academic Profile")

    with st.expander(
        "Fill Student Information",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.number_input(
                "Student ID",
                value=1000
            )

            age = st.number_input(
                "Age",
                min_value=15,
                max_value=25,
                value=18
            )

            gender_text = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            ethnicity = st.selectbox(
                "Ethnicity",
                [0, 1, 2, 3]
            )

            parental_education = st.selectbox(
                "Parental Education",
                [0, 1, 2, 3, 4]
            )

            study_time = st.number_input(
                "Study Time Weekly",
                min_value=0.0,
                value=10.0
            )

            absences = st.number_input(
                "Absences",
                min_value=0,
                value=5
            )

        with col2:

            tutoring_text = st.selectbox(
                "Tutoring",
                ["No", "Yes"]
            )

            parental_support = st.selectbox(
                "Parental Support",
                [0, 1, 2, 3, 4]
            )

            extracurricular_text = st.selectbox(
                "Extracurricular",
                ["No", "Yes"]
            )

            sports_text = st.selectbox(
                "Sports",
                ["No", "Yes"]
            )

            music_text = st.selectbox(
                "Music",
                ["No", "Yes"]
            )

            volunteering_text = st.selectbox(
                "Volunteering",
                ["No", "Yes"]
            )

            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=4.0,
                value=2.5
            )

    # ======================
    # ENCODING INPUT
    # ======================

    gender = 0 if gender_text == "Male" else 1

    tutoring = 1 if tutoring_text == "Yes" else 0

    extracurricular = (
        1 if extracurricular_text == "Yes" else 0
    )

    sports = (
        1 if sports_text == "Yes" else 0
    )

    music = (
        1 if music_text == "Yes" else 0
    )

    volunteering = (
        1 if volunteering_text == "Yes" else 0
    )

    if st.button("🚀 Predict Student Performance"):

        input_data = pd.DataFrame([[
            student_id,
            age,
            gender,
            ethnicity,
            parental_education,
            study_time,
            absences,
            tutoring,
            parental_support,
            extracurricular,
            sports,
            music,
            volunteering,
            gpa
        ]], columns=[
            'StudentID',
            'Age',
            'Gender',
            'Ethnicity',
            'ParentalEducation',
            'StudyTimeWeekly',
            'Absences',
            'Tutoring',
            'ParentalSupport',
            'Extracurricular',
            'Sports',
            'Music',
            'Volunteering',
            'GPA'
        ])

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )[0]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 4:
            st.success(
                "🌟 GradeClass 4 - Excellent Performance"
            )

        elif prediction == 3:
            st.success(
                "✅ GradeClass 3 - Good Performance"
            )

        elif prediction == 2:
            st.warning(
                "📘 GradeClass 2 - Average Performance"
            )

        elif prediction == 1:
            st.warning(
                "📙 GradeClass 1 - Below Average"
            )

        else:
            st.error(
                "📉 GradeClass 0 - Needs Improvement"
            )

# ==========================
# MODEL ANALYSIS
# ==========================
elif menu == "🧠 Model Analysis":

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Train Accuracy",
            "99.77%"
        )

    with col2:
        st.metric(
            "Test Accuracy",
            "91.23%"
        )

    st.divider()

    st.subheader("Top Influential Features")

    feature_df = pd.DataFrame({
        "Feature": [
            "GPA",
            "Absences",
            "StudyTimeWeekly"
        ],
        "Importance": [
            0.42,
            0.31,
            0.12
        ]
    })

    st.bar_chart(
        feature_df.set_index(
            "Feature"
        )
    )

    st.write("""
GPA, Absences, and StudyTimeWeekly were identified as
the most important variables affecting student performance.
""")

# ==========================
# PROJECT INFO
# ==========================
elif menu == "👨‍💻 Project Info":

    st.subheader("Project Information")

    st.markdown("""
### Student Success Prediction System

#### Machine Learning Algorithms
- Logistic Regression
- Decision Tree
- Random Forest

#### Deployment Tools
- Joblib
- Streamlit

#### Best Model
Random Forest

#### Final Accuracy
91.23%

#### Key Features
- GPA
- Absences
- StudyTimeWeekly
""")

    st.success(
        "Developed for Supervised Learning Final Project"
    )   