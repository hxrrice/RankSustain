
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the Streamlit app
st.title("RankSustain: Using Big Data and TOPSIS for Dynamic Sustainability Ranking and Evaluation")

# Instructions for the user
st.markdown("""
    **Welcome to RankSustain!**  
    This system uses the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method to rank alternatives based on sustainability criteria.  
    Enter your sustainability criteria and alternatives below to see the dynamic ranking results.
""")

# Upload data (CSV) for sustainability criteria and alternatives
uploaded_file = st.file_uploader("Upload your sustainability data (CSV format)", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the CSV into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully!")
    st.write(data)

    # Process the data to extract criteria and alternatives
    criteria = data.columns[1:].tolist()  # Assuming first column is alternatives
    alternatives = data['Alternative'].tolist()

    # Input weights for the criteria
    st.sidebar.subheader("Input Weights for Criteria")
    weights = []
    for criterion in criteria:
        weight = st.sidebar.slider(f"Weight for {criterion}", 0.0, 1.0, 0.1)
        weights.append(weight)
    total_weight = sum(weights)

    # Normalize the weights to sum up to 1
    normalized_weights = [w / total_weight for w in weights]
    st.sidebar.write("Normalized Weights:", normalized_weights)

    # Normalization of the data (TOPSIS Step 1)
    norm_data = data.iloc[:, 1:].apply(lambda x: x / np.sqrt((x**2).sum()), axis=0)
    st.write("Normalized Data (Step 1):")
    st.write(norm_data)

    # Multiply the normalized data by the weights (TOPSIS Step 2)
    weighted_data = norm_data * normalized_weights
    st.write("Weighted Normalized Data (Step 2):")
    st.write(weighted_data)

    # Calculate the Ideal and Negative Ideal Solutions (TOPSIS Step 3)
    ideal_solution = weighted_data.max()
    negative_ideal_solution = weighted_data.min()

    st.write("Ideal Solution (Step 3):")
    st.write(ideal_solution)
    st.write("Negative Ideal Solution (Step 3):")
    st.write(negative_ideal_solution)

    # Calculate the Euclidean distance for each alternative (TOPSIS Step 4)
    distance_ideal = np.sqrt(((weighted_data - ideal_solution)**2).sum(axis=1))
    distance_negative_ideal = np.sqrt(((weighted_data - negative_ideal_solution)**2).sum(axis=1))

    st.write("Distance to Ideal Solution (Step 4):")
    st.write(distance_ideal)
    st.write("Distance to Negative Ideal Solution (Step 4):")
    st.write(distance_negative_ideal)

    # Calculate the TOPSIS Score and Rank (TOPSIS Step 5)
    topsis_scores = distance_negative_ideal / (distance_ideal + distance_negative_ideal)
    ranked_alternatives = pd.DataFrame({
        'Alternative': alternatives,
        'TOPSIS Score': topsis_scores,
    }).sort_values(by='TOPSIS Score', ascending=False)

    st.write("TOPSIS Scores and Ranking (Step 5):")
    st.write(ranked_alternatives)

    # Plot the ranking
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TOPSIS Score', y='Alternative', data=ranked_alternatives, palette='viridis')
    plt.title("TOPSIS Ranking of Alternatives")
    st.pyplot()

else:
    st.write("Please upload a CSV file to continue.")
