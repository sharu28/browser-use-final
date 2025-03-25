import gradio as gr
import os

from main import run_processing

DEFAULT_PROMPTS = [
    "Search Google for the product name, find an official site, and locate a PDS PDF.",
    "Search Google for the product name plus 'Product Disclosure Statement', open the first official PDF link.",
    "Search for the product name on the official fund manager website, then find the PDS link."
]

def process_file(uploaded_file_path, default_prompt, custom_prompt):
    if not uploaded_file_path:
        return "No file uploaded.", None, None, None

    # Combine default prompt and custom prompt
    combined_prompt = default_prompt
    if custom_prompt.strip():
        combined_prompt += "\n" + custom_prompt.strip()

    excel_path, zip_path, logs = run_processing(uploaded_file_path, combined_prompt)
    status = "Processing complete! Download your files below."
    return status, excel_path, zip_path, logs

with gr.Blocks() as demo:
    gr.Markdown("# PDS Scraper & Validator")

    with gr.Accordion("Instructions", open=False):
        gr.Markdown(
            """
            ## Step-by-Step Instructions for Preparing the PDS Table

            **1. Copy the PDS Table**  
            - Open the unedited SoA (Statement of Advice).  
            - Find the Product Disclosure Statement (PDS) table.  
            - (Only works with RI and Consultum SoAs for now.)

            **2. Paste into Excel**  
            - Open a new Excel sheet.  
            - Right-click where you want to paste.  
            - Choose “Match Destination Formatting” under Paste Options.  
              This ensures only plain text is pasted without any extra formatting.

            **3. Format the Excel**  
            - Rearrange the columns in the following order:  
              - Column A: APIR Code  
              - Column B: Product  
              - Column C: PDS Date  
              - Column D: URL  
            - Column names may differ depending on the dealer group – that’s okay!  
              Just make sure the column order is correct.

            **4. Clean Up (Optional but Recommended)**  
            - Delete empty rows  
            - Remove section headers (e.g., rows with headings like "Superannuation", "Managed Investments").

            **5. Save and Upload**
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                type="filepath",
                label="Upload your Excel file here"
            )
            default_prompt_dropdown = gr.Dropdown(
                choices=DEFAULT_PROMPTS,
                value=DEFAULT_PROMPTS[0],
                label="Default Search Prompt"
            )
            custom_prompt_input = gr.Textbox(
                label="Custom Search Prompt (optional)",
                placeholder="Enter additional instructions here...",
                lines=3
            )
            run_button = gr.Button("Submit")
        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="Logs (Browser Use Steps)",
                lines=20,
                interactive=False
            )

    with gr.Row():
        status_output = gr.Textbox(
            label="Processing Status",
            interactive=False,
            lines=1
        )

    with gr.Row():
        excel_output = gr.File(label="Processed Excel File")
        zip_output = gr.File(label="ZIP File (if available)")

    run_button.click(
        fn=process_file, 
        inputs=[file_input, default_prompt_dropdown, custom_prompt_input],
        outputs=[status_output, excel_output, zip_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("PORT", 7860)), server_name="0.0.0.0")
