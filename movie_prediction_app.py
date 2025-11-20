import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 
from collections import defaultdict
import warnings
import time
import os 
from sklearn.inspection import permutation_importance
# Suppress common Scikit-learn warnings in Streamlit for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION AND CONSTANTS ---

st.set_page_config(
    layout="wide", 
    page_title="Ultimate Global Box Office Predictor (Indian & Hollywood)",
    initial_sidebar_state="expanded"
)

# Define the final set of 10 core predictive features (all in Crores for numeric monetary columns)
FEATURE_COLUMNS = [
    'Production_Budget_INR_Cr', 'Marketing_Budget_INR_Cr', 
    'Release_Month', 
    'Popularity_Score', 
    'Runtime_Minutes', 
    'Star_Success_Factor', 
    'Genre', 'Primary_Language',
    'Diaspora_Strength_Score',
    'Origin' 
]
TARGET_COLUMNS = ['India_Gross_INR_Cr', 'International_Gross_INR_Cr']

# Conversion scales and caps
CRORE_SCALE = 10_000_000.0                # 1 Crore = 10,000,000 INR
MAX_REVENUE_CRORE = 50000.0                # cap (in Crores) for clipping model outputs
MAX_LOG_PREDICTION = np.log1p(MAX_REVENUE_CRORE)  # clipping on log1p(revenue_in_crores)

TEST_SIZE_RATIO = 0.15 

# --- UTILITY FUNCTIONS ---

def format_inr(value: float) -> str:
    """
    Format a full-rupee numeric value into a human readable string with ₹ and commas.
    Expects 'value' in full INR (not Crores).
    """
    if isinstance(value, str):
        return value
    try:
        return f'₹{int(round(value)):,}'
    except Exception:
        return str(value)

def format_inr_crore_to_rupees_string(crore_value: float) -> str:
    """
    Helper: convert value in Crores -> rupees string.
    """
    rupees = float(crore_value) * CRORE_SCALE
    return format_inr(rupees)

def make_dual_prediction(input_data_df: pd.DataFrame, model_name: str) -> dict:
    """
    Performs the dual prediction (India Gross and International Gross).
    Returns predictions in Crores (matching training units).
    """
    if DUAL_MODELS is None or model_name not in DUAL_MODELS:
        return {"India": 0.0, "International": 0.0, "Total": 0.0}
        
    models = DUAL_MODELS[model_name]

    # Models were trained on log1p(revenue_in_crores); clip to realistic log range
    india_log_pred = np.clip(models["India"].predict(input_data_df)[0], a_min=None, a_max=MAX_LOG_PREDICTION)
    int_log_pred = np.clip(models["International"].predict(input_data_df)[0], a_min=None, a_max=MAX_LOG_PREDICTION)
    
    # Convert back to Crores
    india_gross_cr = max(0.0, np.expm1(india_log_pred))
    int_gross_cr = max(0.0, np.expm1(int_log_pred))

    return {
        "India": india_gross_cr,
        "International": int_gross_cr,
        "Total": india_gross_cr + int_gross_cr
    }

@st.cache_data(show_spinner="Loading data from CSV...")
def load_box_office_data():
    """Reads the combined box office data from a CSV file (expects Crore units for budgets & targets)."""
    file_path = 'box_office_data.csv' 
    
    if not os.path.exists(file_path):
        st.error(f"Error: Data file '{file_path}' not found. Please run the data generation script or create the CSV.")
        return pd.DataFrame() 
        
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        required_cols = FEATURE_COLUMNS + TARGET_COLUMNS
        if not all(col in df.columns for col in required_cols):
            st.error("Error: The CSV file is missing required columns. Ensure you ran the latest data generation script.")
            return pd.DataFrame()
            
        # Ensure numeric columns are numeric
        for c in ['Production_Budget_INR_Cr','Marketing_Budget_INR_Cr','Popularity_Score','Runtime_Minutes','Star_Success_Factor','Diaspora_Strength_Score','India_Gross_INR_Cr','International_Gross_INR_Cr']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(inplace=True)
        return df[required_cols]

    except Exception as e:
        st.error(f"Error reading or processing CSV file: {e}")
        return pd.DataFrame() 

# --- DATA PREPROCESSING, AND TRAINING CORE ---

@st.cache_data(show_spinner="Training and validating three advanced models...")
def load_data_and_train_model():
    """Core function for data loading, feature engineering, and training all models."""
    start_time = time.time()

    df = load_box_office_data()
    if df.empty:
        raise Exception("Failed to load data from CSV.")
        
    N_SAMPLES = len(df)
    st.info(f"Loaded **{N_SAMPLES}** samples from CSV. ({100 * (1 - TEST_SIZE_RATIO):.0f}% Training, {100 * TEST_SIZE_RATIO:.0f}% Test)")
    
    X = df[FEATURE_COLUMNS]
    # Targets expressed in Crores (as in dataset)
    y_india = np.log1p(df['India_Gross_INR_Cr']) 
    y_int = np.log1p(df['International_Gross_INR_Cr'])
    
    X_train, X_test, y_india_train, y_india_test = train_test_split(X, y_india, test_size=TEST_SIZE_RATIO, random_state=42)
    _, _, y_int_train, y_int_test = train_test_split(X, y_int, test_size=TEST_SIZE_RATIO, random_state=42)
    
    # For MAE and RMSE calculation, we need actual test values in Crores
    y_india_test_actual = np.expm1(y_india_test)
    y_int_test_actual = np.expm1(y_int_test)

    numerical_features = ['Production_Budget_INR_Cr', 'Marketing_Budget_INR_Cr', 'Popularity_Score', 'Runtime_Minutes', 'Star_Success_Factor', 'Diaspora_Strength_Score'] 
    categorical_features = ['Release_Month', 'Genre', 'Primary_Language', 'Origin'] 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_features),
            ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
        ],
        remainder='passthrough'
    )

    model_definitions = {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=400, activation='relu', solver='adam', random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42, max_iter=300, max_depth=8) 
    }

    targets = {"India": (y_india_train, y_india_test, y_india_test_actual), 
               "International": (y_int_train, y_int_test, y_int_test_actual)}
    
    dual_models = defaultdict(dict)
    all_test_results = defaultdict(dict)
    raw_mae_comparison = defaultdict(dict) 

    for model_name, base_model in model_definitions.items():
        
        for target_name, (y_train, y_test, y_test_actual) in targets.items():
            
            # Reset the regressor for the next target training 
            if model_name == "Random Forest": current_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
            elif model_name == "Neural Network (MLP)": current_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=400, activation='relu', solver='adam', random_state=42)
            else: current_model = HistGradientBoostingRegressor(random_state=42, max_iter=300, max_depth=8)
                
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', current_model)])
            
            pipeline.fit(X_train, y_train)
            
            y_pred_log = pipeline.predict(X_test)
            y_pred_crore = np.expm1(y_pred_log)  # predictions in Crores

            r2 = r2_score(y_test, y_pred_log) 
            mae = mean_absolute_error(y_test_actual, y_pred_crore) 
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_crore))

            all_test_results[model_name][f"{target_name}_R2"] = r2
            all_test_results[model_name][f"{target_name}_MAE"] = mae
            all_test_results[model_name][f"{target_name}_RMSE"] = rmse
            dual_models[model_name][target_name] = pipeline

            raw_mae_comparison[target_name][model_name] = mae

    end_time = time.time()
    st.success(f"All SIX Dual-Target Models Trained and Validated Successfully in {end_time - start_time:.2f} seconds!")
    
    results_df = pd.DataFrame(all_test_results).T
    # Convert MAE/RMSE from Crores -> Rupees before formatting for display
    for col in results_df.columns:
        if 'MAE' in col or 'RMSE' in col:
            results_df[col] = results_df[col].apply(lambda x: format_inr(x * CRORE_SCALE))
    
    st.subheader("Model Performance Summary on **UNSEEN TEST DATA**")
    st.markdown("Metrics derived from the 15% validation set. **Lower MAE/RMSE is better.** (MAE/RMSE shown in ₹)")
    # Format R2 values as decimals
    st.dataframe(results_df.style.format({c: '{:.4f}' for c in results_df.columns if 'R2' in c}), use_container_width=True)

    # Return models (which expect Crore inputs) and raw MAE comparison (in Crores)
    return dual_models, raw_mae_comparison

# --- EXECUTION ---
try:
    DUAL_MODELS, RAW_MAE_COMPARISON = load_data_and_train_model()
    MODEL_OPTIONS = list(DUAL_MODELS.keys())
except Exception as e:
    st.error(f"FATAL ERROR during model loading/training: {e}")
    DUAL_MODELS = None
    MODEL_OPTIONS = []

# --- MAIN APP INTERFACE (REVISED UI/UX) ---

st.header("🌍 Global Box Office Predictor: Indian & Hollywood")
st.markdown("A robust, dual-target system using **advanced Machine Learning** to estimate domestic (India) and international gross revenue for both Indian and global productions.")

if DUAL_MODELS is None:
    st.stop()

# --- TABBED NAVIGATION ---

tab1, tab2, tab3 = st.tabs(["🚀 Generate Prediction", "🔬 Model Deep Dive", "📊 Performance Visualization"])

# ===============================================
# 1. PREDICTION TAB
# ===============================================

with tab1:
    st.subheader("1. Configure Movie Parameters and Model")
    
    col_input_1, col_input_2, col_input_3, col_predict = st.columns([1.2, 1, 1, 0.8])

    # --- INPUT COLUMN 1: Financial and Core ---
    with col_input_1:
        with st.container():
            st.markdown('**💸 Financial & Core Factors**')
            
            # Production Budget Slider (Max 1500 Cr) - slider returns value in Crores
            budget_cr = st.slider(
                '1. Production Budget (₹ Cr)',
                min_value=1, 
                max_value=1500,
                value=300,
                step=5, 
                format="₹%d Cr" 
            )
            # Keep the Crore value for model input
            budget = float(budget_cr)
            
            # Marketing Budget Slider (Max 500 Cr)
            marketing_budget_cr = st.slider(
                '2. Marketing Budget (₹ Cr)',
                min_value=1, 
                max_value=500, 
                value=100, 
                step=5,
                format="₹%d Cr" 
            )
            marketing_budget = float(marketing_budget_cr)
            
            # Origin Selector
            origin = st.selectbox(
                '3. Film Origin',
                options=['Indian', 'Hollywood'],
                index=0,
                help="Select the geographical origin of the production."
            )


    # --- INPUT COLUMN 2: Performance and Genre ---
    with col_input_2:
        with st.container():
            st.markdown('**🎬 Performance & Genre**')

            popularity = st.slider(
                '4. Anticipated Popularity Score (0-100)',
                min_value=1, max_value=100, value=75, step=1,
                help="Gauges pre-release buzz."
            )
            
            runtime = st.slider(
                '5. Runtime (in Minutes)',
                min_value=90, max_value=200, value=145, step=5
            )
            
            genre_options = ['Action (Masala)', 'Historical Epic', 'Rom-Com', 'Drama', 'Thriller', 
                             'Sci-Fi/Action', 'Fantasy/Adventure', 'Animated', 'Horror/Thriller', 'Superhero']
            genre = st.selectbox(
                '6. Primary Genre',
                options=genre_options,
                index=0 
            )

    # --- INPUT COLUMN 3: Human Factor & Timing ---
    with col_input_3:
        with st.container():
            st.markdown('**⭐ Talent & Global Reach**')

            star_factor = st.slider(
                '7. Star/Director Success Factor (1=Low, 10=High)',
                min_value=1, max_value=10, value=8, step=1,
                help="Historical box-office draw of key talent."
            )
            
            diaspora_score = st.slider(
                '8. Diaspora Strength Score (0-10)',
                min_value=0, max_value=10, value=5, step=1,
                help="Critical feature for International Gross (especially for Indian films)."
            )
            
            month_options = {
                1: '1 (Jan)', 2: '2 (Feb)', 3: '3 (Mar)', 4: '4 (Apr)',
                5: '5 (May)', 6: '6 (Jun)', 7: '7 (Jul)', 8: '8 (Aug)',
                9: '9 (Sep)', 10: '10 (Oct - Diwali)', 11: '11 (Nov)', 12: '12 (Dec - Christmas)'
            }
            
            selected_month_label = st.selectbox(
                '9. Release Month (Festival Impact)',
                options=list(month_options.values()),
                index=6 
            )
            try:
                 selected_month = int(selected_month_label.split(' ')[0])
            except ValueError:
                 selected_month = 7
                 
            
            # --- CONDITIONAL LANGUAGE LOGIC ---
            if origin == 'Hollywood':
                language_options = ['English']
                default_index = 0
            else: # 'Indian'
                # Full list for Indian films
                language_options = ['Hindi', 'Telugu', 'Tamil', 'Others', 'English']
                
                # Try to preserve a previous selection, but default to Hindi if selection is invalid
                try:
                    current_lang = st.session_state.get('primary_language', 'Hindi')
                    default_index = language_options.index(current_lang)
                except ValueError:
                    default_index = language_options.index('Hindi')


            language = st.selectbox(
                '10. Primary Language',
                options=language_options,
                index=default_index,
                key='primary_language', # Use a key to maintain state across reruns
                help="Language options change based on Film Origin."
            )


    # --- PREDICT COLUMN ---
    with col_predict:
        with st.container():
            st.markdown('**🧠 Model & Execution**')
            
            selected_model = st.radio(
                "Choose Algorithm:",
                options=MODEL_OPTIONS,
                index=0,
                label_visibility="visible"
            )
            
            st.markdown("<br>", unsafe_allow_html=True) 
            
            predict_button = st.button('GENERATE DUAL PREDICTION', type='primary', use_container_width=True)


    st.divider()
    
    st.subheader("2. Estimated Worldwide Box Office")
    
    # --- Prediction Logic and Output ---
    if predict_button:
        # Prepare the input data frame (CRORE units)
        input_data = pd.DataFrame({
            'Production_Budget_INR_Cr': [budget], 
            'Marketing_Budget_INR_Cr': [marketing_budget], 
            'Release_Month': [selected_month], 
            'Popularity_Score': [popularity],
            'Runtime_Minutes': [runtime], 
            'Star_Success_Factor': [star_factor],
            'Genre': [genre], 
            'Primary_Language': [language],
            'Diaspora_Strength_Score': [diaspora_score],
            'Origin': [origin] 
        })

        # Uncomment for debugging to confirm Crore units going in:
        # st.write("DEBUG - Model Input (Crores):", input_data)

        predictions_cr = make_dual_prediction(input_data, selected_model)  # returns Crores

        # Convert to rupees for display (UX requested)
        predictions_rupees = {k: v * CRORE_SCALE for k, v in predictions_cr.items()}

        # Use columns for a clean output layout
        col_total, col_india, col_int = st.columns([1.5, 1, 1])

        with col_total:
            # Highlight the Total Worldwide Gross prominently (display in ₹)
            st.metric(
                label="TOTAL WORLDWIDE GROSS",
                value=format_inr(predictions_rupees['Total']),
                delta=f"Predicted via {selected_model}",
                delta_color="normal"
            )
        
        with col_india:
             st.metric(
                label="India Gross (Domestic)",
                value=format_inr(predictions_rupees['India'])
            )

        with col_int:
            st.metric(
                label="International Gross (Overseas)",
                value=format_inr(predictions_rupees['International'])
            )
            
    else:
        st.info("Set your parameters above and click 'GENERATE DUAL PREDICTION'.")


# ===============================================
# 2. DEEP DIVE TAB
# ===============================================
with tab2:
    st.header("🔬 Model Architecture and Methodology")
    
    st.subheader("1. Dual Target Regression for Global Markets")
    st.markdown("""
    The core challenge is predicting two distinct, high-variance values across two different production systems: **India Gross** and **International Gross**. We now include the **Origin** feature, allowing the models to learn fundamentally different rules for how budgets and stars translate to revenue for 'Indian' vs. 'Hollywood' films.
    """)
    
    st.subheader("2. The 10 Critical Predictive Features")
    st.markdown("""
    The feature set now includes the **Origin** discriminator:
    * **Origin:** Categorical feature ('Indian' or 'Hollywood') that dictates the base revenue curve and market logic.
    * **Financial:** **Production Budget** and **Marketing Budget** (₹ Crore).
    * **Talent/Reach:** **Star/Director Success Factor** and **Diaspora Strength Score**.
    * **Timing/Type:** **Release Month**, **Primary Language**, and **Genre**.
    * **Audience Interest:** **Popularity Score** and **Runtime**.
    """)
    
    st.subheader("3. Data Preprocessing Pipeline")
    st.markdown("""
    The preprocessing remains crucial, with **One-Hot Encoding** applied to categorical features alongside numeric scaling. We continue to use $\mathbf{\log(1 + Revenue_{Crore})}$ for stable training. 
    """)

    st.subheader("4. Three Advanced Machine Learning Algorithms")
    st.markdown("""
    You can choose from the following validated models: **Random Forest Regressor**, **Gradient Boosting Regressor**, or **Neural Network (MLP Regressor)**.
    """)


# ===============================================
# 3. MODEL INFO & DOCUMENTATION (CLEAN VERSION)
# ===============================================
with tab3:
    st.header("📘 Model Documentation & User Guide")
    st.markdown(
        """
        This page provides a clear overview of the dataset, model pipeline, and how to
        interpret predictions from the **Global Box Office Predictor**.
        No visual charts are shown here to keep the interface clean and professional.
        """
    )

    # ----------------------------------------
    # 1. DATASET SUMMARY
    # ----------------------------------------
    st.markdown("## 1. Dataset Summary")

    try:
        df_sum = load_box_office_data()
        if df_sum.empty:
            st.info("Dataset not found. Ensure `box_office_data.csv` exists.")
        else:
            st.markdown(f"**Total Samples:** {len(df_sum):,}")

            st.markdown(
                """
                **Features Used (10):**  
                - Production_Budget_INR_Cr  
                - Marketing_Budget_INR_Cr  
                - Release_Month  
                - Popularity_Score  
                - Runtime_Minutes  
                - Star_Success_Factor  
                - Genre  
                - Primary_Language  
                - Diaspora_Strength_Score  
                - Origin  
                
                **Targets (2):**  
                - India_Gross_INR_Cr  
                - International_Gross_INR_Cr  
                """
            )

            # Show only summary of key numeric columns
            stats_cols = [
                "Production_Budget_INR_Cr",
                "Marketing_Budget_INR_Cr",
                "India_Gross_INR_Cr",
                "International_Gross_INR_Cr"
            ]
            st.subheader("🔹 Key Numeric Statistics (Crores)")
            st.dataframe(
                df_sum[stats_cols].describe().transpose().round(2),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Could not load dataset summary: {e}")

    st.markdown("---")

    # ----------------------------------------
    # 2. MODEL TRAINING OVERVIEW
    # ----------------------------------------
    st.markdown("## 2. Machine Learning Pipeline")

    st.markdown(
        """
        **Algorithms included:**
        - Random Forest Regressor  
        - HistGradientBoosting Regressor  
        - Neural Network (MLP Regressor)

        **Preprocessing:**
        - Numeric features → `StandardScaler`  
        - Categorical features → `OneHotEncoder(handle_unknown='ignore')`  
        - Targets → Log transform (`y = log1p(y)`)  
        - Train/Test split → **85% / 15%**

        **Why Log Transform?**  
        Film revenues are extremely skewed. `log1p()` stabilizes training and improves accuracy.
        """
    )

    st.subheader("Training Configuration")
    st.table(pd.DataFrame([
        {"Model": "Random Forest", "Hyperparameters": "n_estimators=300, max_depth=10"},
        {"Model": "Gradient Boosting", "Hyperparameters": "max_iter=300, max_depth=8"},
        {"Model": "Neural Network (MLP)", "Hyperparameters": "hidden_layer_sizes=(100,50), max_iter=400"}
    ]))

    st.markdown("---")

    # ----------------------------------------
    # 3. PREDICTION WORKFLOW
    # ----------------------------------------
    st.markdown("## 3. How Predictions Work")

    st.markdown(
        """
        Prediction steps:

        1. You provide the **10 core movie features** (budgets in Crores, metadata, factors).
        2. These inputs are preprocessed through standard scaling + one-hot encoding.
        3. The model predicts:
           - `log1p(India_Gross_Cr)`
           - `log1p(International_Gross_Cr)`
        4. Predictions are inverted using `expm1()` → giving gross in **Crores**.
        5. For display, the app converts Crores → full **INR** by multiplying by 10,000,000.

        **Important:** Model output represents best estimates based on training distribution.
        """
    )

    st.markdown("---")

    # ----------------------------------------
    # 4. INTERPRETING MODEL OUTPUTS
    # ----------------------------------------
    st.markdown("## 4. How to Interpret Predictions")

    st.markdown(
        """
        - **Budget Input Units:** Entered in **₹ Crore**.
        - **Displayed Units:** Output is shown in **full ₹ (Indian Rupees)** for clarity.
        - **Prediction Size:** Large totals can appear depending on:
          - High popularity score  
          - High star success factor  
          - High diaspora score (esp. international gross)  
          - Holiday release months  
        - **Model choice:**  
          - Use **Random Forest** for balanced, stable predictions.  
          - Use **Gradient Boosting** for accuracy if dataset is large.  
          - Use **MLP** for testing highly non-linear relationships.
        """
    )

    st.markdown("---")

    # ----------------------------------------
    # 5. SAMPLE PREDICTIONS (TABLE ONLY)
    # ----------------------------------------
    st.markdown("## 5. Example Predictions")

    try:
        if not df_sum.empty:
            example_df = df_sum.sample(min(5, len(df_sum)), random_state=1).reset_index(drop=True)
            rf = DUAL_MODELS.get("Random Forest", None)

            if rf:
                india_logs = rf["India"].predict(example_df[FEATURE_COLUMNS])
                int_logs = rf["International"].predict(example_df[FEATURE_COLUMNS])

                example_df["Pred_India_Cr"] = np.expm1(india_logs)
                example_df["Pred_Int_Cr"] = np.expm1(int_logs)
                example_df["Pred_Total_Cr"] = example_df["Pred_India_Cr"] + example_df["Pred_Int_Cr"]

                st.dataframe(
                    example_df[
                        [
                            "Production_Budget_INR_Cr",
                            "Marketing_Budget_INR_Cr",
                            "Genre",
                            "Origin",
                            "Pred_India_Cr",
                            "Pred_Int_Cr",
                            "Pred_Total_Cr"
                        ]
                    ].round(2),
                    use_container_width=True
                )
            else:
                st.info("Model not available to show predictions.")
    except Exception as e:
        st.error(f"Example predictions failed: {e}")

    st.markdown("---")

    # ----------------------------------------
    # 6. FAQ
    # ----------------------------------------
    st.markdown("## 6. Frequently Asked Questions")

    st.markdown(
        """
        **Q: Why no visual charts here?**  
        A: This tab focuses on documentation, clarity, and usage guidance.

        **Q: Are predictions perfectly accurate?**  
        A: No model is perfect. Use them as best estimates, not guarantees.

        **Q: What if I upload my own dataset?**  
        A: The app will automatically retrain on the new CSV.

        **Q: Why dual prediction models?**  
        A: India and International markets behave differently — separate models improve accuracy.
        """
    )

    st.markdown("---")

    # ----------------------------------------
    # 7. SUGGESTED IMPROVEMENTS (OPTIONAL)
    # ----------------------------------------
    st.markdown("## 7. Suggested Improvements")

    st.markdown(
        """
        - Add preset movie profiles (Indie / Mid-budget / Big Bollywood / Hollywood Franchise).  
        - Allow exporting predictions to CSV.  
        - Add an admin mode to re-upload training data.  
        - Add confidence intervals using model MAE ranges.
        """
    )