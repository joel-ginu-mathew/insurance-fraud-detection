import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import plotly.express as px



df = pd.read_csv("insurance_claims.csv")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <h1>Insurance Claims </h1>
    <p>view  <a href='/dashboard'>dashboard</a> .</p>
    <p>view  <a href='/predictor'>predictor</a> .</p>

    """
@app.get("/predictor", response_class=HTMLResponse)
def read_root():
    return """
    <h1>predictor </h1>
    <p> </p>
    <p>Go to <a href='/'>home</a>.</p>
     """
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    # Chart 1: Total claim amount by incident type - Bar chart
    chart1 = px.bar(
        df.groupby("incident_type")["total_claim_amount"].sum().reset_index(),
        x="incident_type",
        y="total_claim_amount",
        title="Total Claim Amount by Incident Type"
    )
    
    # Chart 2: Number of claims by incident severity - Pie chart
    severity_counts = df["incident_severity"].value_counts().reset_index()
    severity_counts.columns = ["incident_severity", "count"]
    chart2 = px.pie(
        severity_counts,
        values="count",
        names="incident_severity",
        title="Claims Distribution by Incident Severity"
    )
    
    # Chart 3: Average claim amount by policy state
    avg_claim_by_state = df.groupby("policy_state")["total_claim_amount"].mean().reset_index()
    chart3 = px.choropleth(
        avg_claim_by_state,
        locations="policy_state",
        locationmode="USA-states",
        color="total_claim_amount",
        color_continuous_scale="Viridis",
        scope="usa",
        title="Average Claim Amount by Policy State"
    )
    
    # Combine the charts into one HTML string for the dashboard page
    dashboard_html = f"""
    <html>
    <head>
        <title>Insurance Claims Dashboard</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
    </head>
    <body>
        <h1>Insurance Claims Dashboard</h1>
        <div>{chart1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div>{chart2.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div>{chart3.to_html(full_html=False, include_plotlyjs=False)}</div>
        <p>Go to <a href='/'>home</a>.</p>


    </body>
    </html>
    """

    return HTMLResponse(content=dashboard_html)

