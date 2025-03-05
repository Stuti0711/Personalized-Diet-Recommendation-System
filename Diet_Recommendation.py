import pandas as pd
import numpy as np
import streamlit as st
import google.generativeai as genai

# Configure Google Gemini AI
genai.configure(api_key="AIzaSyDSTetT2Y7AXytGWNeCEicJ1YYS9LQ5QtQ")

# Load datase
food_data = pd.read_csv('updated_food_500_health_recommended.csv')
nutrition_data = pd.read_csv('updated_food_500_final_nutrition.csv')


# 🔹 Standardizing column names
food_data.columns = food_data.columns.str.strip().str.replace(' ', '_').str.lower()
nutrition_data.columns = nutrition_data.columns.str.strip().str.replace(' ', '_').str.lower()

# 🔹 Merge datasets
common_columns = list(set(food_data.columns) & set(nutrition_data.columns))
if not common_columns:
    st.error("No common columns found for merging. Check CSV files!")
    st.stop()

merged_data = pd.merge(food_data, nutrition_data, on=common_columns, how='outer')
merged_data.fillna("Data Not Available", inplace=True)

# 🔹 Assign meal categories
def assign_meal_type(food_item):
    food_item = food_item.lower()
    
    if any(word in food_item for word in ['poha', 'paratha', 'idli', 'dosa', 'cheela', 'oats', 'sandwich']):
        return 'Breakfast'
    elif any(word in food_item for word in ['dal', 'chawal', 'roti', 'sabzi', 'biryani', 'paneer', 'rajma', 'chole']):
        return 'Lunch'
    elif any(word in food_item for word in ['tandoori', 'matar paneer', 'khichdi', 'dal fry', 'naan', 'gravy', 'soup', 'stir fry']):
        return 'Dinner'
    else:
        return 'Anytime Snack'

merged_data['meal_type'] = merged_data['food_items'].apply(assign_meal_type)
merged_data = merged_data.drop_duplicates(subset=['food_items'], keep='first')

# 🔹 Assign Veg/Non-Veg category
def assign_veg_nonveg(food_item):
    food_item = food_item.lower()
    
    if any(word in food_item for word in ['chicken', 'fish', 'mutton', 'egg', 'keema', 'prawn', 'butter chicken']):
        return 'Non-Veg'
    else:
        return 'Veg'

merged_data['category'] = merged_data['food_items'].apply(assign_veg_nonveg)

# 🔹 Format numerical values correctly
def format_numbers(df):
    for col in ['calories', 'fats', 'proteins', 'carbohydrates']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.') if pd.notnull(x) else x)
    return df

merged_data = format_numbers(merged_data)

# 🎨 **Streamlit UI**
st.title("🥗 Diet Recommendation System")

# 🔹 User Input
gender = st.selectbox("Select Gender:", ["Male", "Female"], index=0)
weight = st.number_input("Enter Weight (kg):", min_value=20.0, max_value=200.0)
height_unit = st.selectbox("Select Height Unit:", ["Centimeters", "Feet & Inches"])

if height_unit == "Centimeters":
    height = st.number_input("Enter Height (cm):", min_value=50.0, max_value=250.0)
else:
    height_feet = st.number_input("Feet:", min_value=1, max_value=8)
    height_inches = st.number_input("Inches:", min_value=0, max_value=11)
    height = (height_feet * 30.48) + (height_inches * 2.54)

food_pref = st.radio("Food Preference:", ["Veg", "Non-Veg"])

# 🔹 BMI Calculation
def calculate_bmi_category(bmi, gender):
    if gender == "Male":
        return ("Severely Underweight" if bmi < 17 else
                "Underweight" if bmi < 20 else
                "Healthy" if bmi < 26 else
                "Overweight" if bmi < 31 else
                "Obese Class I" if bmi < 36 else
                "Obese Class II" if bmi < 41 else
                "Obese Class III")
    else:
        return ("Severely Underweight" if bmi < 16 else
                "Underweight" if bmi < 19 else
                "Healthy" if bmi < 25 else
                "Overweight" if bmi < 30 else
                "Obese Class I" if bmi < 35 else
                "Obese Class II" if bmi < 40 else
                "Obese Class III")

# 🔹 Get Recommendations
def get_meal_recommendations(meal_type, food_pref):
    meal_data = merged_data[(merged_data['meal_type'] == meal_type) & (merged_data['category'] == food_pref)]
    meal_data = meal_data.drop_duplicates(subset=['food_items'])  # Prevent repeated values

    return meal_data.sample(n=min(3, len(meal_data)), random_state=np.random.randint(0, 1000)) if not meal_data.empty else None

if st.button("Get Recommendation"):
    if height == 0:
        st.error("⚠️ Height cannot be zero!")
    else:
        bmi = weight / ((height / 100) ** 2)
        bmi_category = calculate_bmi_category(bmi, gender)

        st.subheader(f"Your BMI: {bmi:.2f} ({bmi_category})")

        for meal in ["Breakfast", "Lunch", "Dinner"]:
            st.subheader(f"🍽 {meal} Recommendations")
            meal_data = get_meal_recommendations(meal, food_pref)

            if meal_data is not None:
                st.table(meal_data[['food_items', 'calories', 'fats', 'proteins', 'carbohydrates']].reset_index(drop=True))
            else:
                st.warning(f"No recommendations found for {meal} with your selection.")

# 🔮 AI Diet Planner
def get_gemini_recommendation(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else response.candidates[0]['content']

if st.button("🤖 AI Diet Planner"):
    prompt = f"""
    I am a {gender} with a weight of {weight} kg and height of {height} cm.
    My BMI is {weight / ((height / 100) ** 2):.2f}, and I am categorized as {calculate_bmi_category(weight / ((height / 100) ** 2), gender)}.
    I prefer {food_pref} food.
    Suggest a meal plan with breakfast, lunch, and dinner that matches my dietary needs.
    """
    gemini_meal_plan = get_gemini_recommendation(prompt)
    st.subheader("🥗 AI-Generated Meal Plan")
    st.write(gemini_meal_plan)
