"""
Gradio application entry point for the Agentic RAG system.
Provides a UI for document upload and financial QA.
"""

import gradio as gr
import yaml
from typing import Tuple, List, Any

from agents.qa_agent import QAAgent

def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config("configs/agent_config.yaml")
agent = QAAgent(config)

def process_query(file_objs: List[str], query: str) -> Tuple[str, str]:
    """
    Process the user query using the initialized LangGraph agent.
    
    Args:
        file_objs: List of uploaded file paths for document ingestion.
        query: User's financial question.
        
    Returns:
        Tuple containing the generated answer and formatted citations.
    """
    if not query.strip():
        return "Please enter a valid question.", ""
        
    try:
        # Invoke the LangGraph agent
        response = agent.invoke({"question": query})
        
        answer = response.get("answer", "No answer generated.")
        citations = response.get("citations", [])
        
        # Format citations
        cited_text = "\n\n".join(
            [f"[{i+1}] {cite.get('source', 'Unknown')}: {cite.get('text', '')}" 
             for i, cite in enumerate(citations)]
        )
        
        return answer, cited_text
        
    except Exception as e:
        return f"Error processing query: {str(e)}", ""

def build_ui() -> gr.Blocks:
    """Constructs the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 📈 Agentic RAG for Financial Document QA")
        gr.Markdown("Upload financial reports (10-K, 10-Q) and ask complex analytical questions. The agent provides fully cited answers.")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Financial Documents (PDF)", file_count="multiple")
                query_input = gr.Textbox(
                    label="Your Question", 
                    placeholder="e.g., What are the key risk factors mentioned for FY2023?",
                    lines=3
                )
                submit_btn = gr.Button("Analyze", variant="primary")
                
            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Generated Analysis", lines=10, interactive=False)
                citation_output = gr.Textbox(label="Source Citations", lines=8, interactive=False)
                
        submit_btn.click(
            fn=process_query,
            inputs=[file_input, query_input],
            outputs=[answer_output, citation_output]
        )
        
    return interface

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
