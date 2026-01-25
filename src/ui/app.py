"""
Research Agent UI

Gradio-based interface for the research assistant.
"""

import gradio as gr
from typing import Optional


def create_app(agent=None):
    """
    Create the Gradio application.
    
    Args:
        agent: ResearchAgent instance (optional for testing UI)
        
    Returns:
        Gradio Blocks app
    """
    
    with gr.Blocks(
        title="Research Assistant",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🔬 Research Assistant
        
        Social sciences research helper with autonomous knowledge building.
        
        **Capabilities:**
        - Literature review and paper discovery
        - Paper summarization  
        - Web search for grey literature
        - Data analysis
        """)
        
        with gr.Tab("💬 Research Chat"):
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Ask me about your research topic..."
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="What are the key theories in urban anthropology?",
                    label="Your question",
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Chat")
                
            with gr.Accordion("⚙️ Settings", open=False):
                search_depth = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Max external results per source"
                )
                auto_ingest = gr.Checkbox(
                    label="Automatically add high-quality sources to knowledge base",
                    value=False
                )
        
        with gr.Tab("📚 Knowledge Base"):
            gr.Markdown("## Your Research Library")
            
            with gr.Row():
                kb_stats = gr.JSON(
                    label="Statistics",
                    value={
                        "total_papers": 0,
                        "total_notes": 0,
                        "total_web_sources": 0
                    }
                )
                refresh_btn = gr.Button("🔄 Refresh")
            
            gr.Markdown("### Add Papers")
            
            with gr.Row():
                upload_pdf = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("Process & Add", variant="primary")
            
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload PDFs to add to your knowledge base"
            )
            
            gr.Markdown("### Browse Papers")
            papers_table = gr.Dataframe(
                headers=["Title", "Year", "Authors", "Added"],
                label="Papers in Knowledge Base"
            )
        
        with gr.Tab("📊 Data Analysis"):
            gr.Markdown("## Analyze Your Data")
            
            data_input = gr.File(
                label="Upload CSV or Excel file",
                file_types=[".csv", ".xlsx", ".xls"]
            )
            
            analysis_type = gr.Radio(
                choices=[
                    "Descriptive Statistics",
                    "Correlation Analysis",
                    "Frequency Analysis",
                    "Custom Query"
                ],
                label="Analysis Type",
                value="Descriptive Statistics"
            )
            
            custom_query = gr.Textbox(
                label="Custom analysis request",
                placeholder="e.g., 'Show me the distribution of ages by region'",
                visible=True
            )
            
            analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Row():
                analysis_output = gr.Markdown(label="Results")
            
            analysis_plot = gr.Plot(label="Visualization")
        
        # Event handlers
        def respond(message, history):
            """Handle chat messages."""
            if agent is None:
                # Demo mode
                response = f"[Demo mode] You asked: {message}\n\nThe agent is not loaded. Run with a real agent to get responses."
            else:
                # Real mode - would be async
                response = "Agent response would appear here"
            
            history.append((message, response))
            return "", history
        
        def refresh_stats():
            """Refresh knowledge base statistics."""
            # TODO: Get real stats from vector store
            return {
                "total_papers": 0,
                "total_notes": 0,
                "total_web_sources": 0
            }
        
        # Wire up events
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], outputs=[chatbot])
        refresh_btn.click(refresh_stats, outputs=[kb_stats])
    
    return app


def launch_app(
    agent=None,
    port: int = 7860,
    share: bool = False
):
    """
    Launch the Gradio app.
    
    Args:
        agent: ResearchAgent instance
        port: Port to run on
        share: Create public link
    """
    app = create_app(agent)
    app.launch(
        server_port=port,
        share=share,
        show_error=True
    )


if __name__ == "__main__":
    # Launch in demo mode
    print("Launching in demo mode (no agent loaded)...")
    launch_app()
