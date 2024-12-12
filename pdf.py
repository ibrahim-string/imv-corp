from fpdf import FPDF
import asyncio
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize the Ollama model
llm = Ollama(model="llama3.2:1b-instruct-q4_K_S")

# Prompt configuration
prompt = "Generate the content for the first page of a report on vibration testing."
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=prompt),
    MessagesPlaceholder(variable_name="chat_history"),
])
chat_history = []
chain = prompt_template | llm


# Function to create the PDF
def create_pdf(text, filename):
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Add a title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Vibration Testing Report", ln=True, align='C')

    # Add content
    pdf.set_font("Arial", size=12)
    pdf.ln(10)  # Line break
    pdf.multi_cell(0, 10, text)  # Multi-cell for text wrapping

    # Save the PDF
    pdf.output(filename)
    print(f"PDF saved as {filename}")


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
        generated_text += response  # Response is a plain string

    # Create the PDF with the generated text
    create_pdf(generated_text, "vibration_report1.pdf")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
