import streamlit as st
from sqlalchemy import create_engine, MetaData, text
from langchain.chat_models import ChatOpenAI
import networkx

# Replace with your actual MySQL credentials
DATABASE_URL = "mysql+pymysql://USER:Password@localhost:3306/Database"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)


DB_HOST = 'localhost'
DB_USER = ''
DB_PASSWORD = ''
DB_NAME = ''


# Initialize the GPT-4 model using ChatOpenAI from LangChain
chat_openai = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key='')

# Initialize the NetworkX graph
kg = networkx.DiGraph()

# Store conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []



# SQL Query Execution Function
def execute_sql_query(engine, query):
    """
    Execute the generated SQL query and return the results.
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            # Use row._mapping to ensure compatibility
            result_set = [dict(row._mapping) for row in result]  # Convert the result to a list of dictionaries
        return result_set
    except Exception as e:
        return str(e)  # Return the error if any occurs


# Function to get sample data from a table
def get_sample_data(engine, table_name, limit=5):
    """
    Retrieve sample data from a given table.
    """
    query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
    with engine.connect() as connection:
        result = connection.execute(query)
        # Use row._mapping to ensure compatibility
        sample_data = [dict(row._mapping) for row in result]
    
    return sample_data

sql_keywords = {'SUM', 'AS', 'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'JOIN', 'TIMESTAMPDIFF', 'ON', 'LIMIT', 'DESC', 'ASC', 'COUNT', 'AVG', 'MAX', 'MIN', 'WITH', 'DISTINCT', 'STR_TO_DATE', 'WHEN', 'CROSS', 'GROUP BY', 'CASE'}

import re
# Function to parse SQL query
def parse_sql_query(sql_query, schema_info):
    used_elements = {
        "schemas": {"mysql"},
        "databases": {"telecom_data"},
        "tables": set(),
        "columns": set()
    }

    # Normalize the query: remove newlines and extra spaces
    sql_query = ' '.join(sql_query.split())

    # Extract tables from FROM, JOIN, and other clauses
    table_pattern = r'(?:FROM|JOIN|CROSS JOIN)\s+`?(\w+)`?'
    tables = re.findall(table_pattern, sql_query, re.IGNORECASE)

    # Validate the extracted tables against schema_info
    valid_tables = set()
    invalid_tables = set()
    for table in tables:
        if table in schema_info:
            valid_tables.add(table)
        else:
            invalid_tables.add(table)
    
  
    used_elements["tables"].update(valid_tables)

    # Extract columns from the SELECT clause
    column_pattern = r'`(\w+)`|(?<=\.)`?(\w+)`?|\b(\w+)\b'
    potential_columns = re.findall(column_pattern, sql_query)

    for match in potential_columns:
        column = next((col for col in match if col), None)
        if column and column.lower() not in sql_keywords:
            # Check if the column exists in any of the valid tables in the schema
            for table in valid_tables:
                if column in schema_info.get(table, []):
                    used_elements["columns"].add(column)
                    break

    return used_elements

import plotly.express as px
import plotly.io as pio

def plot_data_visualization(df, chart_type):
    chart_type = chart_type.lower()
    fig = None

    # Convert all columns to string type except the numeric ones to preserve full numbers
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            df[col] = df[col].apply(lambda x: '{:.0f}'.format(x))  # Format to avoid scientific notation

    # Let the user select the x and y axes from the dataframe columns
    x_axis = st.selectbox('Select X-axis', df.columns)
    y_axis = None
    if chart_type in ["line", "bar", "scatter"]:
        y_axis = st.selectbox('Select Y-axis', df.columns)

    if chart_type == "line":
        fig = px.line(df, x=x_axis, y=y_axis)
    elif chart_type == "bar":
        fig = px.bar(df, x=x_axis, y=y_axis)
    elif chart_type == "pie":
        names = st.selectbox('Select "Names" for Pie Chart', df.columns)
        values = st.selectbox('Select "Values" for Pie Chart', df.columns)
        fig = px.pie(df, names=names, values=values)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis)

    if fig:
        # Disable tick abbreviations for both x and y axes
        fig.update_layout(
            xaxis_tickformat="none",
            yaxis_tickformat="none",
            xaxis=dict(tickmode='linear', tick0=0),
            yaxis=dict(tickmode='linear', tick0=0)
        )
        st.plotly_chart(fig)
    else:
        st.error("Unable to create the selected chart type with the given data.")



def get_schema_info(host, user, password, database)  # Truncate very long files
