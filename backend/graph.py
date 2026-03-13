"""
Graph Module — LangGraph state graph with router, tool nodes, and subtool routing.

ARCHITECTURE:
  router → (conditional) → tool_node
  
  For search_agent:
    search_agent → subtool_router → (conditional) → subtool_node → END
                                                  ↘ END (if subtool=none)
  
  For all other tools:
    tool_node → END

KEY DESIGN:
  Each subtool (summarizer, mcq_generator, etc.) can be reached from TWO paths:
    1. Directly from router (primary tool)
    2. From subtool_router (as a sub-agent of search_agent)
  
  To avoid LangGraph's "duplicate edge" error, we use WRAPPER NODES for subtools.
  - "summarizer" is the primary node (router → summarizer → END)
  - "sub_summarizer" is the subtool wrapper (subtool_router → sub_summarizer → END)
  Both use the same ToolNode([summarizer_func]) internally.
"""
import hashlib
import json
import re

from typing import TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_tool: str
    subtool: str


def build_graph(tools, llm):
    """Build and compile the LangGraph agent graph."""

    tool_map = {t.name: t for t in tools}

    # Tools that can also be used as subtools from search_agent
    SUBTOOL_NAMES = ["summarizer", "mcq_generator", "notes_maker", "exam_prep_agent", "concept_explainer"]

    def route_agent(state: AgentState) -> AgentState:
        """LLM-based intent classification router."""
        if not state["messages"]:
            raise ValueError("No messages found in state")

        query_text = state["messages"][-1].content

        classification_prompt = (
            f"You are a query classifier for a multi-agent RAG system. "
            f"Analyze the user's query and choose the BEST PRIMARY tool.\n\n"
            f"Available Tools:\n"
            f"- search_agent: web search, internet, google, latest news, external research, browse, lookup\n"
            f"- summarizer: summarize, summary, overview, condense, brief, key highlights\n"
            f"- mcq_generator: mcq, mcqs, multiple choice, questions, quiz, generate questions, practice test\n"
            f"- notes_maker: notes, make notes, key notes, revision notes, study notes, outline\n"
            f"- exam_prep_agent: prepare exam, study plan, revision plan, exam strategy, probable questions\n"
            f"- concept_explainer: explain, what is, define, how does it work, breakdown, clarify\n"
            f"- chat_agent: greetings, hello, hi, general chat, casual conversation, thanks, bye, "
            f"or any query that doesn't clearly fit the other tools\n\n"
            f"Query: '{query_text}'\n\n"
            f"Respond with ONLY the tool name (e.g., 'summarizer')."
        )

        try:
            classification_response = llm.invoke(classification_prompt)
            tool_name = getattr(classification_response, "content", str(classification_response)).strip().lower()

            # Clean up LLM output — sometimes it returns extra text
            # Extract just the tool name using regex
            tool_name = re.sub(r'[^a-z_]', '', tool_name.split('\n')[0].strip())

            # Map to exact tool name
            if "search" in tool_name or "web" in tool_name:
                tool_name = "search_agent"
            elif "summariz" in tool_name:
                tool_name = "summarizer"
            elif "mcq" in tool_name or "question" in tool_name:
                tool_name = "mcq_generator"
            elif "notes" in tool_name:
                tool_name = "notes_maker"
            elif "exam" in tool_name or "prep" in tool_name:
                tool_name = "exam_prep_agent"
            elif "concept" in tool_name or "explain" in tool_name:
                tool_name = "concept_explainer"
            elif "chat" in tool_name:
                tool_name = "chat_agent"
            else:
                tool_name = "chat_agent"
        except Exception:
            # Keyword fallback
            ql = query_text.lower()
            if any(w in ql for w in ["web", "search", "internet", "online", "google", "latest", "browse"]):
                tool_name = "search_agent"
            elif any(w in ql for w in ["summarize", "summary", "overview", "condense", "brief"]):
                tool_name = "summarizer"
            elif any(w in ql for w in ["mcq", "mcqs", "multiple choice", "question", "quiz"]):
                tool_name = "mcq_generator"
            elif any(w in ql for w in ["notes", "make notes", "key notes", "revision notes"]):
                tool_name = "notes_maker"
            elif any(w in ql for w in ["prepare", "exam", "study plan", "probable"]):
                tool_name = "exam_prep_agent"
            elif any(w in ql for w in ["explain", "what is", "define", "how does"]):
                tool_name = "concept_explainer"
            else:
                tool_name = "chat_agent"

        tool_call = {
            "name": tool_name,
            "args": {"query": query_text},
            "id": f"call_{tool_name}_{hashlib.md5(query_text.lower().encode()).hexdigest()[:8]}",
        }
        ai_message = AIMessage(content="", tool_calls=[tool_call])

        return {
            "messages": state["messages"] + [ai_message],
            "next_tool": tool_name,
            "subtool": "none",
        }

    def route_subtool(state: AgentState) -> AgentState:
        """Route search_agent output to a subtool if needed."""
        if state["next_tool"] != "search_agent":
            return {"messages": state["messages"], "subtool": "none"}

        # Find the search_agent ToolMessage
        tool_msgs = [
            m for m in state["messages"]
            if isinstance(m, ToolMessage) and m.tool_call_id.startswith("call_search_agent")
        ]
        if not tool_msgs:
            return {"messages": state["messages"], "subtool": "none"}

        result_content = tool_msgs[-1].content

        # ToolNode serializes dict returns as JSON strings
        result = None
        if isinstance(result_content, str):
            try:
                parsed = json.loads(result_content)
                if isinstance(parsed, dict):
                    result = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        elif isinstance(result_content, dict):
            result = result_content

        if result is None:
            return {"messages": state["messages"], "subtool": "none"}

        subtool = result.get("subtool", "none")
        search_content = result.get("content", "")

        if not subtool or subtool == "none" or not search_content:
            return {"messages": state["messages"], "subtool": "none"}

        if subtool not in SUBTOOL_NAMES:
            return {"messages": state["messages"], "subtool": "none"}

        query = state["messages"][0].content
        tool_call = {
            "name": subtool,
            "args": {"query": query, "context": search_content},
            "id": f"call_sub_{subtool}_{hashlib.md5(query.encode()).hexdigest()[:8]}",
        }
        ai_message = AIMessage(content="", tool_calls=[tool_call])

        return {
            "messages": state["messages"] + [ai_message],
            "subtool": subtool,
        }

    # ==================== Build the graph ====================
    graph = StateGraph(AgentState)

    # Router node
    graph.add_node("router", RunnableLambda(route_agent))

    # Primary tool nodes (used when the router selects them directly)
    for tool_func in tools:
        graph.add_node(tool_func.name, ToolNode([tool_func]))

    # Subtool wrapper nodes (used when subtool_router activates them after search_agent)
    # These are separate nodes to avoid LangGraph's "duplicate edge" error
    for sname in SUBTOOL_NAMES:
        if sname in tool_map:
            graph.add_node(f"sub_{sname}", ToolNode([tool_map[sname]]))

    # Subtool router node
    graph.add_node("subtool_router", RunnableLambda(route_subtool))

    # ---- Edges ----

    # Router → tool nodes (conditional)
    graph.add_conditional_edges(
        "router",
        lambda state: state["next_tool"],
        {t.name: t.name for t in tools},
    )

    # All primary tools except search_agent → END
    for tool_func in tools:
        if tool_func.name != "search_agent":
            graph.add_edge(tool_func.name, END)

    # search_agent → subtool_router
    graph.add_edge("search_agent", "subtool_router")

    # subtool_router → sub_* wrapper nodes or END (conditional)
    subtool_edges = {sname: f"sub_{sname}" for sname in SUBTOOL_NAMES}
    subtool_edges["none"] = END

    graph.add_conditional_edges(
        "subtool_router",
        lambda state: state.get("subtool", "none"),
        subtool_edges,
    )

    # All sub_* wrapper nodes → END
    for sname in SUBTOOL_NAMES:
        graph.add_edge(f"sub_{sname}", END)

    graph.set_entry_point("router")
    return graph.compile()
