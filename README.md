# Mall Customer Segmentation using K-Means

This project performs customer segmentation analysis on a shopping mall dataset using the **K-Means clustering algorithm**.  
The goal is to identify distinct customer groups based on their **annual income** and **spending behavior**, enabling data-driven marketing strategies.

---

##  Dataset
- Source: Mall Customers Dataset
- Features used:
  - Annual Income (k$)
  - Spending Score (1–100)
- No missing values detected

---

##  Methodology
1. Data loading and exploration
2. Feature selection
3. Optimal cluster selection using the **Elbow Method**
4. K-Means clustering (k = 5)
5. Segment-level statistical analysis
6. Data visualization and interpretation

---

## 📈 Visualizations
- Elbow Method plot to determine the optimal number of clusters
- Scatter plot showing customer segments and cluster centroids

---

##  Customer Segments
- **Standard Customers:** Medium income and spending
- **VIP Customers:** High income and high spending
- **Careful Customers:** High income, low spending
- **Potential Customers:** Low income, high spending
- **Low-Value Customers:** Low income and low spending

---

## Technologies Used
- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

##  How to Run
```bash
python mall.py
