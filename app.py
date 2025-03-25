import gradio as gr
import os
from main import run_processing

def process_file(uploaded_file_path, custom_prompt):
    if not uploaded_file_path:
        return "No file uploaded.", None, None, None

    # Call run_processing from main.py. This function should return:
    # - processed Excel file path
    # - ZIP file path (if any)
    # - logs (as a string)
    excel_path, zip_path, logs = run_processing(uploaded_file_path, custom_prompt)
    status = "Processing complete! Download your files below."
    return status, excel_path, zip_path, logs

demo = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(type="filepath", label="Upload your Excel file here"),
        gr.Textbox(label="Custom Search Prompt (optional)", 
                   placeholder="Enter custom search prompt here...", 
                   lines=3)
    ],
    outputs=[
        gr.Textbox(label="Processing Status"),
        gr.File(label="Processed Excel File"),
        gr.File(label="ZIP File (if available)"),
        gr.Textbox(label="Logs (Browser Use Steps)", lines=15)
    ],
    title="PDS Scraper & Validator",
    description="Upload an Excel file, run the scraper, and download the results. Logs display browser_use steps."
)

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("PORT", 7860)), server_name="0.0.0.0")
