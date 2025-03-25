import os
import re
import time
import zipfile
import asyncio
import logging
import io

import pandas as pd
import requests
import fitz  # PyMuPDF

import gradio as gr
from openai import OpenAI
import urllib3

from browser_use import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Constants and configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_DOWNLOAD_DIR = r"./downloads/Valid_PDS_PDFs"
ZIP_OUTPUT_PATH = r"./downloads/Valid_PDS_PDFs.zip"
OUTPUT_EXCEL_PATH = r"./downloads/processed.xlsx"
MAX_STEPS = 5
SAVE_INTERVAL = 5  # Save progress every 5 rows

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables!")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure download directory exists
if not os.path.exists(PDF_DOWNLOAD_DIR):
    os.makedirs(PDF_DOWNLOAD_DIR)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Setup logging: capture logs in a StringIO stream for the UI.
log_stream = io.StringIO()
logger = logging.getLogger("browser_use")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter("%(asctime)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def map_columns(df):
    expected_columns = ["APIR code", "Product name", "PDS date", "Web link"]
    mapped_columns = {col: expected_columns[i] for i, col in enumerate(df.columns[:4])}
    return df.rename(columns=mapped_columns)

def extract_pdf_text_first_page(url):
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/pdf", "Referer": url}
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            if response.status_code != 200 or 'application/pdf' not in response.headers.get('Content-Type', ''):
                logger.info(f"Attempt {attempt + 1}: Invalid response - Status {response.status_code}")
                continue
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                text = doc[0].get_text() if len(doc) > 0 else ""
                logger.info(f"Text extracted from {url}: {text[:50]}...")
                return text
        except Exception as e:
            logger.info(f"Attempt {attempt + 1}: PDF extraction failed for {url} - {e}")
    return ""

def extract_pds_date(text):
    date_pattern = r'(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})'
    match = re.search(date_pattern, text, re.IGNORECASE)
    if match:
        day, month, year = match.groups()
        return f"{int(day)} {month} {year}"
    return ""

async def search_with_browser_use(product_name, apir_code, custom_prompt):
    """
    Returns a tuple: (pdf_url_or_message, steps_taken)
    """
    # Use the user-provided custom_prompt if available, otherwise fallback
    prompt = custom_prompt if custom_prompt.strip() else (
        f"Start by searching Google for '{product_name}'. From the Google results, "
        "select the most relevant official fund website (prioritizing domains that match or include "
        "parts of '{product_name}' or words like 'fund', 'investment', 'financial', and avoiding "
        "aggregators like Morningstar). Go to that website. On the site, search for a PDF link "
        "specifically labeled 'Product Disclosure Statement' or 'PDS', prioritizing exact matches. "
        "If the link isn’t visible, navigate up to 5 menu levels deep, clicking on menu options, "
        "buttons, or links that seem relevant (e.g., 'About', 'Invest', 'Resources') to find sections "
        "titled 'Product Disclosure Statement' or 'PDS'. If a login page is encountered, stop and "
        "return 'Not found | Login required'. Click on the link to open the PDF if found. "
        "Return the URL of the PDF prefixed with 'URL: ', or return 'Not found' with a brief reason "
        "(e.g., 'No PDS PDF found', 'Site navigation failed')."
    )
    try:
        agent = Agent(task=prompt, llm=llm)
        result = await agent.run(max_steps=MAX_STEPS)
        steps_taken = agent.state.n_steps
        final_content = None
        if hasattr(result, 'all_results') and result.all_results:
            final_content = result.all_results[-1].extracted_content
        elif hasattr(result, 'final_result'):
            final_result = result.final_result
            final_content = final_result() if callable(final_result) else final_result

        if not final_content or not isinstance(final_content, str):
            return f"Not found | Invalid result content: {type(final_content)}", steps_taken

        url_match = re.search(r'URL\s*:\s*(https?://[^\s]+)', final_content, re.IGNORECASE)
        if url_match:
            return url_match.group(1), steps_taken

        return f"Not found | No URL in result: {final_content}", steps_taken

    except Exception as e:
        return f"Not found | browser-use failed: {str(e)}", 0

def download_pdf_file(url, product_name):
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        file_path = os.path.join(PDF_DOWNLOAD_DIR, f"{product_name} PDS.pdf")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded PDF for '{product_name}' to {file_path}")
        return file_path
    else:
        logger.info(f"Failed to download PDF from {url}")
        return None

def process_row(row, custom_prompt):
    product_name = row['Product name'].strip()
    apir_code = str(row['APIR code']).strip() if pd.notna(row['APIR code']) else None

    pdf_url_or_message, steps_taken = asyncio.run(search_with_browser_use(product_name, apir_code, custom_prompt))
    logger.info(f"Processing '{product_name}': {pdf_url_or_message}")

    if pdf_url_or_message.startswith("Not found"):
        if "|" in pdf_url_or_message:
            parts = pdf_url_or_message.split("|", 1)
            web_link = parts[0].strip()
            reason = parts[1].strip()
        else:
            web_link = "Not found"
            reason = pdf_url_or_message
        return web_link, 0, reason, "", steps_taken

    web_link = pdf_url_or_message
    text = extract_pdf_text_first_page(web_link)
    if not text:
        return web_link, 0, "No text extracted", "", steps_taken

    pds_date = extract_pds_date(text)
    logger.info(f"Extracted PDS date for '{product_name}': {pds_date}")
    return web_link, 100, "", pds_date, steps_taken

def process_excel(file_obj, custom_prompt):
    # Read uploaded Excel into a DataFrame
    data = pd.read_excel(file_obj.name)
    data = map_columns(data)
    
    # Prepare new columns for output
    data['Web link'] = ""
    data['Validity Score'] = 0
    data['Validation Reason'] = ""
    data['PDS date'] = ""
    data['Steps Taken'] = 0

    downloaded_files = []

    for index, row in data.iterrows():
        if pd.isna(row['Product name']):
            continue

        logger.info(f"\nProcessing row {index + 1}...")
        result, score, reason, pds_date, steps_taken = process_row(row, custom_prompt)
        data.at[index, 'Web link'] = result
        data.at[index, 'Validity Score'] = score
        data.at[index, 'Validation Reason'] = reason
        data.at[index, 'PDS date'] = pds_date
        data.at[index, 'Steps Taken'] = steps_taken

        if score == 100 and not result.startswith("Not found"):
            logger.info(f"Attempting to download PDF for '{row['Product name']}'")
            pdf_file_path = download_pdf_file(result, row['Product name'])
            if pdf_file_path:
                downloaded_files.append(pdf_file_path)

        if (index + 1) % SAVE_INTERVAL == 0 or (index + 1) == len(data):
            data.to_excel(OUTPUT_EXCEL_PATH, index=False)
            logger.info(f"Saved progress to {OUTPUT_EXCEL_PATH} at row {index + 1}")

        time.sleep(0.5)

    data.to_excel(OUTPUT_EXCEL_PATH, index=False)
    logger.info("Processing complete! Excel file saved.")

    if downloaded_files:
        with zipfile.ZipFile(ZIP_OUTPUT_PATH, 'w') as zipf:
            for file in downloaded_files:
                zipf.write(file, arcname=os.path.basename(file))
        logger.info(f"All valid PDFs have been zipped into {ZIP_OUTPUT_PATH}")
    else:
        logger.info("No valid PDFs downloaded to zip.")

    return OUTPUT_EXCEL_PATH, ZIP_OUTPUT_PATH, log_stream.getvalue()

def run_processing(file_obj, custom_prompt):
    # Reset log stream for each run
    log_stream.truncate(0)
    log_stream.seek(0)
    try:
        excel_path, zip_path, logs = process_excel(file_obj, custom_prompt)
        return excel_path, zip_path, logs
    except Exception as e:
        return "", "", f"Processing failed: {e}"

# Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# PDS Processing with Browser Use and Gradio")
    
    with gr.Row():
        file_input = gr.File(label="Upload Excel File", file_types=[".xlsx"])
        prompt_input = gr.Textbox(label="Custom Search Prompt (optional)", placeholder="Enter custom search prompt here...", lines=3)
    
    run_button = gr.Button("Run Processing")
    
    with gr.Row():
        excel_output = gr.Textbox(label="Processed Excel File Path", interactive=False)
        zip_output = gr.Textbox(label="ZIP File Path", interactive=False)
    
    log_output = gr.Textbox(label="Processing Logs (Browser Use Steps)", interactive=False, lines=15)
    
    run_button.click(fn=run_processing, inputs=[file_input, prompt_input], outputs=[excel_output, zip_output, log_output])

if __name__ == "__main__":
    demo.launch()
