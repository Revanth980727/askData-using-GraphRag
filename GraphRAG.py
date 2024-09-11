import streamlit as st
from sqlalchemy import create_engine, MetaData, text
from langchain.chat_models import ChatOpenAI
import networkx as nx

# Replace with your actual MySQL credentials
DATABASE_URL = "mysql+pymysql://root:Password@localhost:3306/Database"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)


DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Password'
DB_NAME = 'Database'


# Initialize the GPT-4 model using ChatOpenAI from LangChain
chat_openai = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key='API_KEY')

# Initialize the NetworkX graph
kg = nx.DiGraph()

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
            result_set = [dict(row) for row in result]  # Convert the result to a list of dictionaries
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
        sample_data = [dict(row) for row in result]
    
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

def plot_data_visualization(df, chart_type):
    chart_type = chart_type.lower()
    fig = None

    # Convert all columns to string type
    df = df.astype(str)

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
        st.plotly_chart(fig)
    else:
        st.error("Unable to create the selected chart type with the given data.")


def get_schema_info(host, user, password, database):
    import mysql.connector
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    
    cursor = conn.cursor()
    query = """
    SELECT TABLE_NAME, COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = %s;
    """
    
    cursor.execute(query, (database,))
    results = cursor.fetchall()
    
    schema_info = {}
    for table, column in results:
        if table not in schema_info:
            schema_info[table] = []
        schema_info[table].append(column)
    
    cursor.close()
    conn.close()
    
    return schema_info

# Function to retrieve full context with sample data
def retrieve_full_context_with_datatypes_and_samples(graph, engine):
    """
    Retrieve context from all tables in the knowledge graph, including columns, data types, relationships, 
    and sample data from each table.
    """
    context = "Database schema with column data types, relationships, and sample data:\n"
    
    for table_name, table_data in graph.nodes(data=True):
        context += f"Table: {table_name}\n"
        
        # Include columns and their data types
        for column in table_data['columns']:
            column_type = table_data.get(f'col_{column}', {}).get('type', 'Unknown')
            context += f"  - Column: {column}, Type: {column_type}\n"
        
        # Add relationships to other tables
        related_tables = list(graph.successors(table_name))
        if related_tables:
            context += f"  Related tables: {', '.join(related_tables)}\n"
        
        # Include sample data for the table
        sample_data = get_sample_data(engine, table_name)
        if sample_data:
            context += "  Sample Data:\n"
            for row in sample_data:
                context += f"    {row}\n"
    
    return context

def summarize_history(history):
    """
    Summarize the conversation history to reduce the token length.
    """
    summarized_history = ""
    max_entries = 3  # Limit to the last 3 questions
    for i, (user_question, sql_query, query_result) in enumerate(history[-max_entries:]):
        summarized_history += f"Previous Question {i+1}: {user_question}\n"
        summarized_history += f"Generated SQL Query {i+1}: {sql_query}\n"
        summarized_history += f"Summarized Query Result {i+1}: {len(query_result)} rows returned\n"
    return summarized_history




# Function to generate SQL query with conversation history and example queries for better understanding
def generate_detailed_sql_query_with_history(question, context, chat_model, history):
    """
    Generate an SQL query using GPT-4 (ChatOpenAI) that ensures a detailed response is returned,
    considering conversation history, detailed column data types, and sample data, and example queries.
    """
    # Format the conversation history for the prompt
    history_prompt = ""
    for i, (user_question, sql_query, query_result) in enumerate(history):
        history_prompt += f"User Question {i+1}: {user_question}\n"
        history_prompt += f"Generated SQL Query {i+1}: {sql_query}\n"
        history_prompt += f"Query Result {i+1}: {query_result}\n"


    summarized_history = summarize_history(history)

    # Create a prompt with full context and history
    prompt = f"""Previous Conversation History:
    {summarized_history}

    
    Context: {context}
    
    Based on the above conversation history, generate an SQL query for the following question:
    
    New Question: {question}
    
    SQL Query:"""
    
    # Use the chat model (GPT-4) to generate the SQL query
    response = chat_model.predict(prompt)
    
    return response

# Function to summarize SQL query results using GPT-4
def summarize_query_results(query_results, chat_model):
    """
    Generate a natural language summary of the SQL query results using GPT-4.
    """
    if not query_results:
        return "No results found."
    
    # Format the result into a string
    formatted_results = "Query Results:\n"
    for i, row in enumerate(query_results):
        formatted_results += f"Row {i + 1}: {row}\n"
    
    prompt = f"""
    Based on the following SQL query results, generate a natural language summary. Summarize key insights such as the total number of records, any aggregate metrics (e.g., average, total), or notable patterns:

    {formatted_results}
    
    Natural Language Summary:
    """
    
    # Generate the summary using GPT-4
    response = chat_model.predict(prompt)
    
    return response


# Function to manage the conversation history (keep only the last 3 questions)
def manage_conversation_history(history, new_question, sql_query, query_result):
    """
    Manage the conversation history by keeping only the last 3 questions in the history.
    """
    # If the history already contains 3 or more entries, remove the oldest one
    if len(history) >= 3:
        history.pop(0)  # Remove the oldest question
    
    # Add the new question, generated SQL query, and result to the history
    history.append((new_question, sql_query, query_result))

    return history


# When creating the knowledge graph
for table_name, table in metadata.tables.items():
    # Add each table as a node
    kg.add_node(table_name, label='table', columns=list(table.columns.keys()))

    # Add column metadata, including data type
    for column in table.columns:
        kg.nodes[table_name][f'col_{column.name}'] = {
            'type': str(column.type),  # Include column data type
            'primary_key': column.primary_key,
            'nullable': column.nullable
        }

    # Add edges based on foreign key relationships
    for column in table.columns:
        if column.foreign_keys:
            for fk in column.foreign_keys:
                related_table = fk.column.table.name
                kg.add_edge(related_table, table_name, relationship='foreign_key')

import pandas as pd
import matplotlib.pyplot as plt

# Streamlit Interface
st.title("DataInsightsBot")

tabs = st.tabs(["Query & Result", "Data Lineage Graph", "Data Visualization"])


with tabs[0]:

        # Display the conversation history in a scrollable container at the top
    with st.container():
        st.subheader("Conversation History:")
        for i, entry in enumerate(st.session_state.conversation_history):
            if len(entry) == 2:
                user_question, sql_query = entry
                st.write(f"**User Question {i+1}:** {user_question}")
                st.write(f"**Generated SQL Query {i+1}:** {sql_query}")
            elif len(entry) == 3:
                user_question, sql_query, query_result = entry
                st.write(f"**User Question {i+1}:** {user_question}")
                st.write(f"**Generated SQL Query {i+1}:** {sql_query}")
                st.write(f"**Query Result {i+1}:**")
                st.dataframe(pd.DataFrame(query_result))

    # Create a container at the bottom for the search bar and the "Clear History" button
    st.write("###")
    st.write("###")
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Placing the search bar and the "Clear History" button in a row layout
    with st.container():
        col1, col2 = st.columns([8, 2])  # Adjust the ratio to give the search bar more space
        
        with col1:
            # User inputs a new question
            question = st.text_input("Ask a new question or follow-up question:", key="input_box")

        with col2:
            # Clear History Button
            if st.button("Clear History"):
                st.session_state.conversation_history = []
                st.success("Conversation history cleared! You can now ask a new question.")

    # Ensure schema_info is loaded
    if 'schema_info' not in st.session_state:
        st.session_state['schema_info'] = get_schema_info(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

    schema_info = st.session_state['schema_info']
        
    # Step 1: Show the full schema (context) for all tables with data types and sample data
    context = retrieve_full_context_with_datatypes_and_samples(kg, engine)

    # Step 2: Generate the SQL query using GPT-4 with conversation history if a question is asked
    if question:
        sql_query = generate_detailed_sql_query_with_history(
            question, context, chat_openai, st.session_state.conversation_history)

        st.write(f"Generated SQL Query: {sql_query}")

        # Step 4: Execute the generated SQL query
        query_results = execute_sql_query(engine, sql_query)

        # Step 5: Display the query results
        if isinstance(query_results, str):
            st.error(f"Error executing query: {query_results}")
        else:
            st.write("Query Results:")
            df = pd.DataFrame(query_results)
            st.dataframe(df)
            # Parse the SQL query to extract tables and columns
            used_elements = parse_sql_query(sql_query, schema_info)
            st.session_state['used_elements'] = used_elements  # Save used elements for the lineage graph
            print(used_elements)
            # Save query results and column names in session state for visualization
            st.session_state['result'] = query_results
            st.session_state['columns'] = df.columns.tolist()


            # Store the question, generated SQL query, and query results in conversation history
            st.session_state.conversation_history.append((question, sql_query, query_results))

with tabs[1]:
    st.subheader("Data Lineage Graph")
    if 'used_elements' in st.session_state:
        used_elements = st.session_state['used_elements']
        print(used_elements)
        G = nx.DiGraph()

        # Add table nodes and edges between table and columns
        for table in used_elements["tables"]:
            G.add_node(table, label=table, type='table')

            # Add column nodes and edges only between table and column
            for column in used_elements["columns"]:
                if column in schema_info.get(table, []):
                    G.add_node(column, label=column, type='column')
                    G.add_edge(table, column)  # Edge only from table to column

        pos = {}

        # Set position for tables, centered horizontally
        pos_y = 0
        table_pos = {}
        pos_x = -len(used_elements["tables"]) * 2

        for node, data in G.nodes(data=True):
            if data['type'] == 'table':
                pos[node] = (pos_x, pos_y)
                table_pos[node] = pos_x
                pos_x += 5  # Spread tables horizontally

        # Set position for columns horizontally under each table
        pos_y -= 2  # Move down the y-axis for columns
        for node, data in G.nodes(data=True):
            if data['type'] == 'column':
                table = list(G.predecessors(node))[0]
                pos[node] = (table_pos[table], pos_y)
                table_pos[table] += 2

        plt.figure(figsize=(12, 6))

        # Draw the edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v in G.edges()],
            edge_color='lightcoral', arrows=False, width=1.0, alpha=0.6
        )

        # Draw the labels
        for node, (x, y) in pos.items():
            if G.nodes[node]['type'] == 'table':
                plt.text(x, y, G.nodes[node]['label'], ha="center", va="center",
                         fontsize=9, fontweight="bold")
                plt.text(x, y + 0.1, "Table", ha="center", va="center",
                         fontsize=9, fontweight="bold")
            elif G.nodes[node]['type'] == 'column':
                plt.text(x, y, G.nodes[node]['label'], ha="center", va="center",
                         fontsize=9, fontweight="bold")
                plt.text(x, y + 0.1, "Column", ha="center", va="center",
                         fontsize=9, fontweight="bold")

        # Add title and adjust layout
        plt.title("Database Lineage Flow", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Render the plot in Streamlit
        st.pyplot(plt)
    else:
        st.info("Submit a query in the 'Query & Result' tab to view the data lineage graph.")

with tabs[2]:
    st.subheader("Data Visualization")
    st.info("Select a query in the 'Query & Result' tab to view data visualization.")

    # Check if result and columns are available in session state
    if 'result' in st.session_state and 'columns' in st.session_state:
        result = st.session_state['result']
        columns = st.session_state['columns']
        df = pd.DataFrame(result, columns=columns)

        # Select chart type and plot the data
        chart_type = st.selectbox("Select Chart Type", options=["line", "bar", "pie", "scatter"])
        plot_data_visualization(df, chart_type)
    else:
        st.warning("No query results available for visualization.")
