
import streamlit as st
import pandas as pd
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-font {
    font-size: 40px !important;
    font-weight: bold;
}

.metric-card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------

model = joblib.load("models/titanic_survival_model.pkl")

# -------------------------
# HEADER
# -------------------------

st.markdown(
    "<p class='big-font'>🚢 Titanic Survival Predictor</p>",
    unsafe_allow_html=True
)

st.markdown("""
Predict whether a passenger would have survived the Titanic disaster
using a Machine Learning model trained on historical passenger data.
""")

st.divider()

# -------------------------
# SIDEBAR INPUTS
# -------------------------

st.sidebar.header("⚙ Passenger Information")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.sidebar.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.sidebar.slider(
    "Age",
    1,
    80,
    25
)

sibsp = st.sidebar.number_input(
    "Siblings / Spouse",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.sidebar.number_input(
    "Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.sidebar.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.sidebar.selectbox(
    "Embarked Port",
    ["S", "C", "Q"]
)

family_size = sibsp + parch + 1

# -------------------------
# DASHBOARD
# -------------------------

left, right = st.columns([1, 1])

with left:

    st.subheader("👤 Passenger Profile")

    st.info(f"""
    **Passenger Class:** {pclass}

    **Gender:** {sex.title()}

    **Age:** {age}

    **Fare:** £{fare:.2f}

    **Family Size:** {family_size}

    **Embarked:** {embarked}
    """)

    st.subheader("📖 Feature Guide")

    st.markdown("""
    - **Pclass:** Travel class of passenger
    - **Fare:** Ticket price
    - **Embarked:** Port where passenger boarded
    - **Family Size:** Total family members onboard
    - **Gender & Age:** Strong survival indicators
    """)

with right:

    st.subheader("🤖 Prediction")

    if st.button(
        "Predict Survival",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked],
            "Family_Size": [family_size]
        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.metric(
            "Survival Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

        confidence = max(
            probability,
            1 - probability
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2%}"
        )

        if prediction == 1:

            st.success(
                f"✅ Passenger likely SURVIVES"
            )

            st.balloons()

        else:

            st.error(
                f"❌ Passenger likely DOES NOT SURVIVE"
            )

        # Session History

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Class": pclass,
            "Gender": sex,
            "Age": age,
            "Probability": round(
                probability * 100,
                2
            )
        })

# -------------------------
# MODEL PERFORMANCE
# -------------------------

st.divider()

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Accuracy",
        "82.12%"
    )

with c2:
    st.metric(
        "Dataset Size",
        "891 Rows"
    )

with c3:
    st.metric(
        "Model",
        "Random Forest"
    )

# -------------------------
# PREDICTION HISTORY
# -------------------------

if "history" in st.session_state:

    st.divider()

    st.subheader(
        "🕒 Prediction History"
    )

    st.dataframe(
        pd.DataFrame(
            st.session_state.history
        ),
        use_container_width=True
    )

# -------------------------
# TITANIC FACTS
# -------------------------

st.divider()

st.subheader("🚢 Titanic Facts")

st.markdown("""
- Titanic sank on **15 April 1912**
- Around **2,224 passengers and crew** were onboard
- Approximately **1,500 people lost their lives**
- Women and children had significantly higher survival rates
- Passenger class strongly influenced survival chances
""")

# -------------------------
# FOOTER
# -------------------------

st.divider()

st.caption(
    "Built by Ronit Kumar | Machine Learning Project | Streamlit + Scikit-Learn"
)
