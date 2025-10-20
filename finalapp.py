
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Hospital Patient Analytics",
    page_icon="🏥",
    layout="wide"
)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('patients.csv')
    # Calculate length of stay
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    df['length_of_stay'] = (df['departure_date'] - df['arrival_date']).dt.days
    return df

# Load data
df = load_data()

# Title and description
st.title("🏥 Hospital Patient Analytics Dashboard")
st.markdown("""
This dashboard provides insights into patient satisfaction, service utilization, 
and hospital performance metrics.
""")

# Sidebar for filters
st.sidebar.header("🔍 Filter Data")

# Filter 1: Service type multi-select
services = st.sidebar.multiselect(
    "Select Service Types:",
    options=df['service'].unique(),
    default=df['service'].unique()
)

# Filter 2: Age range slider
age_range = st.sidebar.slider(
    "Select Age Range:",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Filter 3: Satisfaction score input
min_satisfaction = st.sidebar.number_input(
    "Minimum Satisfaction Score:",
    min_value=0,
    max_value=100,
    value=0
)

# Apply filters
filtered_df = df[
    (df['service'].isin(services)) &
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['satisfaction'] >= min_satisfaction)
]

# Display dataset info
st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)}")
st.sidebar.markdown(f"**Total Records:** {len(df)}")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Service Analysis", "👥 Patient Demographics", "🔍 Detailed Data"])

with tab1:
    st.header("Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_satisfaction = filtered_df['satisfaction'].mean()
        st.metric("Average Satisfaction", f"{avg_satisfaction:.1f}")
    
    with col2:
        avg_stay = filtered_df['length_of_stay'].mean()
        st.metric("Average Stay (days)", f"{avg_stay:.1f}")
    
    with col3:
        total_patients = len(filtered_df)
        st.metric("Total Patients", total_patients)
    
    with col4:
        most_common_service = filtered_df['service'].mode()[0]
        st.metric("Most Common Service", most_common_service)
    
    # Query 1: Service type vs satisfaction
    st.subheader("Query 1: How does patient satisfaction vary by service type?")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    service_satisfaction = filtered_df.groupby('service')['satisfaction'].mean().sort_values(ascending=False)
    
    bars = ax1.bar(service_satisfaction.index, service_satisfaction.values, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_xlabel('Service Type')
    ax1.set_ylabel('Average Satisfaction Score')
    ax1.set_title('Average Patient Satisfaction by Service Type')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    st.pyplot(fig1)
    
    # Insights
    st.info(f"""
    **Insight:** {service_satisfaction.index[0]} has the highest average satisfaction score 
    ({service_satisfaction.values[0]:.1f}), while {service_satisfaction.index[-1]} has the lowest 
    ({service_satisfaction.values[-1]:.1f}).
    """)

with tab2:
    st.header("Service Analysis")
    
    # Query 2: Length of stay vs satisfaction correlation
    st.subheader("Query 2: Is there a relationship between length of stay and patient satisfaction?")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter = ax2.scatter(filtered_df['length_of_stay'], filtered_df['satisfaction'], 
                        alpha=0.6, c=filtered_df['satisfaction'], cmap='viridis')
    ax2.set_xlabel('Length of Stay (days)')
    ax2.set_ylabel('Satisfaction Score')
    ax2.set_title('Length of Stay vs Patient Satisfaction')
    plt.colorbar(scatter, ax=ax2, label='Satisfaction Score')
    
    st.pyplot(fig2)
    
    # Calculate correlation
    correlation = filtered_df['length_of_stay'].corr(filtered_df['satisfaction'])
    st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    if correlation > 0:
        st.success("Positive correlation: Longer stays tend to have higher satisfaction scores.")
    elif correlation < 0:
        st.warning("Negative correlation: Longer stays tend to have lower satisfaction scores.")
    else:
        st.info("No clear correlation between length of stay and satisfaction.")

with tab3:
    st.header("Patient Demographics")
    
    # Age distribution histogram
    st.subheader("Age Distribution of Patients")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(filtered_df['age'], bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Number of Patients')
    ax3.set_title('Age Distribution of Patients')
    ax3.grid(True, alpha=0.3)
    
    st.pyplot(fig3)
    
    # Age groups analysis
    st.subheader("Satisfaction by Age Groups")
    
    # Create age groups
    filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                    bins=[0, 18, 35, 50, 65, 100],
                                    labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    age_group_satisfaction = filtered_df.groupby('age_group')['satisfaction'].mean()
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    age_group_satisfaction.plot(kind='line', marker='o', ax=ax4, color='#FF6B6B', linewidth=2)
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Average Satisfaction Score')
    ax4.set_title('Average Satisfaction by Age Group')
    ax4.grid(True, alpha=0.3)
    
    st.pyplot(fig4)

with tab4:
    st.header("Detailed Patient Data")
    
    # Search functionality
    search_term = st.text_input("Search patients by name:")
    
    if search_term:
        search_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False, na=False)]
    else:
        search_df = filtered_df
    
    # Display data table
    st.dataframe(
        search_df[['patient_id', 'name', 'age', 'service', 'length_of_stay', 'satisfaction']],
        use_container_width=True
    )
    
    # Download filtered data
    csv = search_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_patient_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("### 📊 Data Summary")
col1, col2 = st.columns(2)

with col1:
    st.write("**Service Distribution:**")
    service_counts = filtered_df['service'].value_counts()
    st.write(service_counts)

with col2:
    st.write("**Satisfaction Statistics:**")
    st.write(filtered_df['satisfaction'].describe())