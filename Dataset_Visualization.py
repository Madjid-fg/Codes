import plotly.express as px

fig = px.line(data, x=data.index, y="AEP_MW", title="Time Series of AEP_MW")
fig.show()
