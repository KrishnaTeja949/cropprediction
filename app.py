import streamlit as st
import pandas as pd
import plotly.express as px

# Simulated DataFrame (replace this with your actual data)
hardcoded_results = [
        {"Technique": "RFE", "Model": "Naive Bayes", "Accuracy": 98.82, "Precision": 98.87, "Recall": 98.82, "F1-Score": 98.82},
        {"Technique": "RFE", "Model": "Decision Tree", "Accuracy": 98.82, "Precision": 98.87, "Recall": 98.82, "F1-Score": 98.81},
        {"Technique": "RFE", "Model": "SVM", "Accuracy": 97.23, "Precision": 97.74, "Recall": 97.23, "F1-Score": 97.20},
        {"Technique": "RFE", "Model": "Random Forest", "Accuracy": 99.32, "Precision": 99.36, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "RFE", "Model": "KNN", "Accuracy": 97.45, "Precision": 97.64, "Recall": 97.45, "F1-Score": 97.46},

        {"Technique": "Boruta", "Model": "Naive Bayes", "Accuracy": 99.32, "Precision": 99.35, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "Boruta", "Model": "Decision Tree", "Accuracy": 98.55, "Precision": 98.63, "Recall": 98.55, "F1-Score": 98.54},
        {"Technique": "Boruta", "Model": "SVM", "Accuracy": 97.50, "Precision": 97.98, "Recall": 97.50, "F1-Score": 97.46},
        {"Technique": "Boruta", "Model": "Random Forest", "Accuracy": 99.32, "Precision": 99.36, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "Boruta", "Model": "KNN", "Accuracy": 97.95, "Precision": 98.12, "Recall": 97.95, "F1-Score": 97.95},

        {"Technique": "SMOTE", "Model": "Naive Bayes", "Accuracy": 99.32, "Precision": 99.35, "Recall": 99.32, "F1-Score": 99.32},
        {"Technique": "SMOTE", "Model": "Decision Tree", "Accuracy": 98.45, "Precision": 98.54, "Recall": 98.45, "F1-Score": 98.45},
        {"Technique": "SMOTE", "Model": "SVM", "Accuracy": 97.50, "Precision": 97.98, "Recall": 97.50, "F1-Score": 97.46},
        {"Technique": "SMOTE", "Model": "Random Forest", "Accuracy": 99.27, "Precision": 99.32, "Recall": 99.27, "F1-Score": 99.27},
        {"Technique": "SMOTE", "Model": "KNN", "Accuracy": 98.05, "Precision": 98.20, "Recall": 98.05, "F1-Score": 98.05},

        {"Technique": "ROSE", "Model": "Naive Bayes", "Accuracy": 99.36, "Precision": 99.40, "Recall": 99.36, "F1-Score": 99.36},
        {"Technique": "ROSE", "Model": "Decision Tree", "Accuracy": 98.68, "Precision": 98.74, "Recall": 98.68, "F1-Score": 98.68},
        {"Technique": "ROSE", "Model": "SVM", "Accuracy": 97.50, "Precision": 97.98, "Recall": 97.50, "F1-Score": 97.46},
        {"Technique": "ROSE", "Model": "Random Forest", "Accuracy": 99.36, "Precision": 99.40, "Recall": 99.36, "F1-Score": 99.36},
        {"Technique": "ROSE", "Model": "KNN", "Accuracy": 97.95, "Precision": 98.12, "Recall": 97.95, "F1-Score": 97.95}
    ]

results_df = pd.DataFrame(hardcoded_results)

# Melt the data for plotting
melted = results_df.melt(
    id_vars=["Technique", "Model"],
    value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
    var_name="Metric",
    value_name="Score"
)

# Check if melted data is valid
if melted.empty:
    st.error("The melted data is empty. Please check the input data.")
else:
    for tech in melted["Technique"].unique():
        tech_data = melted[melted["Technique"] == tech]

        st.subheader(f"📊 {tech} - Model Performance")

        # Create the plotly bar chart
        fig = px.bar(
            tech_data,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )

        # Update the layout to make the plot more professional
        fig.update_layout(
            height=480,
            font=dict(
                family="Segoe UI, sans-serif",
                size=14,
                color="#333"
            ),
            xaxis=dict(
                title="Model",
                titlefont=dict(size=16, family="Segoe UI, sans-serif"),
                tickfont=dict(size=14, family="Segoe UI, sans-serif"),
                showgrid=False
            ),
            yaxis=dict(
                title="Score (%)",
                titlefont=dict(size=16, family="Segoe UI, sans-serif"),
                tickfont=dict(size=14, family="Segoe UI, sans-serif"),
                showgrid=True,
                gridcolor="#eaeaea"
            ),
            legend=dict(
                title="Metric",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=30, t=50, b=40),
            plot_bgcolor='white',
            bargap=0.2  # Add space between bars for better visibility
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
