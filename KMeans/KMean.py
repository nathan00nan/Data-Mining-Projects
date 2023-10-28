import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\pumpk\Desktop\Nathan's File\DOCUMENTS\My Notes\Projects\KMeans.csv")
# print(data.head())
# print(data.isnull().sum())
data = data.dropna()

clustering_data = data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
for i in clustering_data.columns:MinMaxScaler(i)

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(clustering_data)
data["CREDIT_CARD_SEGMENTS"] = clusters

data["CREDIT_CARD_SEGMENTS"] = data["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"})

PLOT = go.Figure()
for i in list(data["CREDIT_CARD_SEGMENTS"].unique()):
    PLOT.add_trace(go.Scatter3d(x=data[data["CREDIT_CARD_SEGMENTS"] == i]['BALANCE'],
                                y=data[data["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z=data[data["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],
                                mode='markers', marker_size=6, marker_line_width=1,
                                name=str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                   scene=dict(xaxis=dict(title='BALANCE', titlefont_color='black'),
                              yaxis=dict(title='PURCHASES', titlefont_color='black'),
                              zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')),
                   font=dict(family="Gilroy", color='black', size=12))

