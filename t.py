# import asyncio
# from fpdf import FPDF
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import io
# from langchain_community.llms import Ollama
# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# # Assuming you have a DataFrame `data` to generate the plots
# # Example dummy data (replace with actual data)
# data = pd.read_csv('0D.csv',nrows=5000)
# # Initialize the Ollama model
# llm = Ollama(model="llama3.2:1b-instruct-q4_K_S")

# # Prompt configuration
# data_description = """
# The dataset contains vibration testing measurements. Key columns include:
# - 'Measured_RPM': The measured RPM values during the test.
# - 'Vibration_1', 'Vibration_2', 'Vibration_3': Vibration measurements at three different sensor positions.
# The objective is to analyze the relationship between RPM and vibration measurements.
# """

# # Define prompt with data context
# prompt = f"""
# You are analyzing vibration testing data. The dataset contains columns such as Measured_RPM, Vibration_1, Vibration_2, and Vibration_3.
# The task is to generate insights from this data, focusing on the relationships between RPM and the vibration measurements.

# Here is a description of the data:
# {data_description}

# Your task is to write an insightful report on how these measurements relate to each other.
# """
# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content=prompt),
#     MessagesPlaceholder(variable_name="chat_history"),
# ])
# chat_history = []
# chain = prompt_template | llm

# # Function to create the PDF
# def create_pdf(text, filename):
#     pdf = FPDF()

#     # Add the first page with report content
#     pdf.add_page()
#     pdf.set_font("Arial", style="B", size=16)
#     pdf.cell(200, 10, txt="Vibration Testing Report", ln=True, align='C')
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, text)

#     # Add space for the charts
#     pdf.add_page()
#     return pdf

# # Function to add plots to the PDF
# def add_plot_to_pdf(pdf, plot_func):
#     # Generate plot image in memory
#     buf = io.BytesIO()
#     plot_func(buf)
#     buf.seek(0)
#     pdf.add_page()
#     pdf.image(buf, x=10, y=10, w=190)
#     return pdf

# # Plot functions
# def plot_measured_rpm_vs_vibration_1(buf):
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=data['Measured_RPM'], y=data['Vibration_1'])
#     plt.title('Measured_RPM vs Vibration_1')
#     plt.xlabel('Measured_RPM')
#     plt.ylabel('Vibration_1')
#     plt.tight_layout()
#     plt.savefig(buf, format='png')
#     plt.close()

# def plot_measured_rpm_vs_vibration_2(buf):
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=data['Measured_RPM'], y=data['Vibration_2'])
#     plt.title('Measured_RPM vs Vibration_2')
#     plt.xlabel('Measured_RPM')
#     plt.ylabel('Vibration_2')
#     plt.tight_layout()
#     plt.savefig(buf, format='png')
#     plt.close()

# def plot_measured_rpm_vs_vibration_3(buf):
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=data['Measured_RPM'], y=data['Vibration_3'])
#     plt.title('Measured_RPM vs Vibration_3')
#     plt.xlabel('Measured_RPM')
#     plt.ylabel('Vibration_3')
#     plt.tight_layout()
#     plt.savefig(buf, format='png')
#     plt.close()

# def plot_pair_plot(buf):
#     plt.figure(figsize=(10, 8))
#     sns.pairplot(data[['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']])
#     plt.suptitle('Pair Plot of All Columns', y=1.02)
#     plt.tight_layout()
#     plt.savefig(buf, format='png')
#     plt.close()

# # Main function
# async def main():
#     # Initialize the response stream
#     response_stream = chain.astream({
#         "input": prompt,
#         "chat_history": chat_history
#     })

#     # Gather the text from the stream
#     generated_text = ""
#     async for response in response_stream:
#         print(response, end='')
#         generated_text += response  # Response is a plain string

#     # Create the PDF with the generated text and plots
#     pdf = create_pdf(generated_text, "vibration_report1.pdf")

#     # Add the plots to the PDF
#     pdf = add_plot_to_pdf(pdf, plot_measured_rpm_vs_vibration_1)
#     pdf = add_plot_to_pdf(pdf, plot_measured_rpm_vs_vibration_2)
#     pdf = add_plot_to_pdf(pdf, plot_measured_rpm_vs_vibration_3)
#     pdf = add_plot_to_pdf(pdf, plot_pair_plot)

#     # Save the PDF
#     pdf.output("vibration_report1.pdf")
#     print(f"PDF saved as vibration_report1.pdf")

# # Run the main function
# if __name__ == "__main__":
#     asyncio.run(main())



import asyncio
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Assuming you have a DataFrame `data` to generate the plots
# Example dummy data (replace with actual data)
data = pd.read_csv('0D.csv', nrows=5000)
df = pd.read_csv('vibrations.csv')
# Initialize the Ollama model
llm = Ollama(model="llama3.2:1b-instruct-q4_K_S")

# Prompt configuration
data_description = """
The dataset contains vibration testing measurements. Key columns include:
- 'Measured_RPM': The measured RPM values during the test.
- 'Vibration': Vibration measurements at a single sensor position.
The objective is to analyze the vibration measurements.
"""

# Define prompt with data context
prompt = f"""
You are analyzing vibration testing data. The dataset contains columns such as Measured_RPM and Vibration.
The task is to generate insights from this data, focusing on the vibration measurements.

Here is a description of the data:
{data_description}

Your task is to write an insightful report on how the vibration measurements behave.
"""
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=prompt),
    MessagesPlaceholder(variable_name="chat_history"),
])
chat_history = []
chain = prompt_template | llm

# Function to create the PDF
def create_pdf(text, filename):
    pdf = FPDF()

    # Add the first page with report content
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Vibration Testing Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)

    # Add space for the charts
    pdf.add_page()
    return pdf

# Function to add plots to the PDF
def add_plot_to_pdf(pdf, plot_func):
    # Generate plot image in memory
    buf = io.BytesIO()
    plot_func(buf)
    buf.seek(0)
    pdf.add_page()
    pdf.image(buf, x=10, y=10, w=190)
    return pdf

# Plot functions
def plot_vibration(buf):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['mm'], kde=True, color='blue')
    plt.title('Vibration Distribution')
    plt.xlabel('Vibration (mm/s)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

def plot_pair_plot(buf):
    plt.figure(figsize=(10, 8))
    sns.pairplot(data[['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']])
    plt.suptitle('Pair Plot of All Columns', y=1.02)
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

# Main function
async def main():
    # Initialize the response stream
    response_stream = chain.astream({
        "input": prompt,
        "chat_history": chat_history
    })

    # Gather the text from the stream
    generated_text = ""
    async for response in response_stream:
        print(response, end='')
        generated_text += response  # Response is a plain string

    # Create the PDF with the generated text and plots
    pdf = create_pdf(generated_text, "vibration_report1.pdf")

    # Add the plots to the PDF
    pdf = add_plot_to_pdf(pdf, plot_vibration)
    pdf = add_plot_to_pdf(pdf, plot_pair_plot)

    # Save the PDF
    pdf.output("vibration_report2.pdf")
    print(f"PDF saved as vibration_report2.pdf")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

