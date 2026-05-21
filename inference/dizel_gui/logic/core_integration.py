"""
dizel_gui/logic/core_integration.py

Handles setting up the new Ecosystem Core modules (Memory, Orchestrator, Tools)
and providing the glue layer between the UI / ChatManager and the backend Agents.
"""

from core.memory import MemorySystem
from core.orchestration import Orchestrator
from core.protocol import ToolRegistry, ToolDispatcher
from core.agents.base_agent import AgentResult


def setup_core_ecosystem(chat_manager) -> tuple[MemorySystem, ToolRegistry, ToolDispatcher, Orchestrator]:
    """
    Instantiates the core ecosystem and registers tools and agent handlers.
    Returns: (memory, tools, dispatcher, orchestrator)
    """
    memory = MemorySystem()
    tools = ToolRegistry()
    dispatcher = ToolDispatcher(tools)
    orchestrator = Orchestrator()

    # ---------------------------------------------------------
    # Register Tools
    # ---------------------------------------------------------
    from core.tools.web_search import search_web
    tools.register(
        name="web_search",
        handler=search_web,
        description="Search the web for real-time information. Returns a string summary of search results.",
        input_schema={"query": "The search query string"}
    )
    
    # ---------------------------------------------------------
    # Register Agent Handlers
    # ---------------------------------------------------------
    
    # 1. Dizel (LLM Generator)
    # Since Dizel streams, the handler here is mostly a placeholder/formatter 
    # for the Orchestrator's internal usage. The actual UI streaming is triggered separately.
    def dizel_handler(task) -> dict:
        # In the context of orchestration, dizel just returns the processed input text 
        # or acknowledges receipt, because actual generation is handled by the UI.
        return {"text": task.input.get("user_message", "")}
        
    orchestrator.register_agent("dizel", dizel_handler)
    
    # 2. Lily (File Parsing)
    def lily_handler(task) -> dict:
        from core.agents.lily_agent import LilyAgent
        files = task.context.get("files", [])
        if not files:
            return {"error": "No files provided"}
        
        # Process first file for simplicity, or iterate
        res = LilyAgent().process(files[0])
        return {"text": res.to_context_block()}
        
    orchestrator.register_agent("lily", lily_handler)
    
    # 3. Dict (Image Processing)
    def dict_handler(task) -> dict:
        from core.agents.dict_agent import DictAgent
        files = task.context.get("files", [])
        if not files:
            return {"error": "No images provided"}
            
        res = DictAgent().process(files[0])
        return {"text": res.to_context_block()}
        
    orchestrator.register_agent("dict", dict_handler)

    return memory, tools, dispatcher, orchestrator
