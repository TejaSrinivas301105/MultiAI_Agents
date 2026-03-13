"""
Agents Module — All 7 LangChain tool-based agents for the Multi-Agentic RAG system.
"""
import re
import os
import json
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper


def get_combined_context(retriever, query: str, max_chunks: int = 10) -> str:
    """Retrieve and format document chunks for a query."""
    if retriever is None:
        return ""
    docs = retriever.invoke(query)
    if not docs:
        return ""
    return "\n\n".join([
        f"[{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs[:max_chunks]
    ])


def create_tools(retriever, llm):
    """Create all agent tools bound to the given retriever and LLM."""

    @tool
    def summarizer(query: str, context: str = "") -> str:
        """
        Summarize the topic in the user's query using ONLY the retrieved document chunks or provided context.
        Args:
            query: The user's instruction (may include 'in X lines').
            context: Optional external context (e.g., from web search).
        Returns:
            A concise paragraph-level summary based strictly on the provided context.
        """
        length_match = re.search(r'in\s+(\d+)\s+lines', query.lower())
        desired_lines = int(length_match.group(1)) if length_match else 10

        combined_content = context if context else get_combined_context(retriever, query)
        if not combined_content:
            return f"No relevant content found in the documents for '{query}'."

        prompt = (
            f"You are an expert academic summarizer. Your task is to create a precise, informative summary.\n\n"
            f"RULES:\n"
            f"- Use ONLY the provided context below. Do NOT add outside knowledge.\n"
            f"- Write approximately {desired_lines} lines in clear paragraph form.\n"
            f"- Include simple examples where they help clarify concepts.\n"
            f"- Use proper academic language but keep it accessible.\n"
            f"- Highlight key terms in bold using **term** format.\n\n"
            f"CONTEXT:\n{combined_content}\n\n"
            f"QUERY: {query}\n\n"
            f"Provide your summary now:"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @tool
    def mcq_generator(query: str, context: str = "") -> str:
        """
        Generates multiple-choice questions (MCQs) based on the content with specified count and topic.
        The agent intelligently determines the topic from the query and generates relevant MCQs.
        Args:
            query: The user's instruction specifying topic and optionally count.
            context: Optional external context.
        Returns:
            Formatted MCQs with options, correct answers, and explanations.
        """
        count_match = re.search(r'(\d+)\s*(?:mcqs?|questions?)', query.lower())
        desired_count = int(count_match.group(1)) if count_match else 10

        combined_content = context if context else get_combined_context(retriever, query)
        if not combined_content:
            return f"No relevant content found in the documents for '{query}'."

        prompt = (
            f"You are an expert MCQ question paper setter for university-level examinations.\n\n"
            f"TASK: Generate exactly {desired_count} multiple-choice questions on the topic derived from the query.\n\n"
            f"QUERY: '{query}'\n\n"
            f"RULES:\n"
            f"- Analyze the query to identify the SPECIFIC TOPIC the user wants MCQs on.\n"
            f"- Generate questions ONLY from the provided context below.\n"
            f"- Questions should range from conceptual understanding to application-based.\n"
            f"- Each question must have exactly 4 options (a, b, c, d).\n"
            f"- Provide the correct answer and a 2-line explanation.\n"
            f"- If the context mentions figures, circuits, or diagrams, include ASCII art representations.\n"
            f"- Number questions sequentially.\n\n"
            f"FORMAT (follow EXACTLY):\n"
            f"Question 1: [question text]\n\n"
            f"options\n"
            f"a)[option A]\n"
            f"b)[option B]\n"
            f"c)[option C]\n"
            f"d)[option D]\n\n"
            f"correct answer: [letter])[full option text]\n\n"
            f"explanation:\n"
            f"[Line 1 of explanation]\n"
            f"[Line 2 of explanation]\n\n\n"
            f"CONTEXT:\n{combined_content}\n\n"
            f"Generate {desired_count} MCQs now:"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @tool
    def notes_maker(query: str, context: str = "") -> str:
        """
        Create concise notes ONLY from the retrieved context using side headings with examples.
        Args:
            query: The user's instruction (may include 'in X lines').
            context: Optional external context.
        Returns:
            Short paragraphs grouped by side headings like ## Key Concept, ## Formula, etc.
        """
        length_match = re.search(r'in\s+(\d+)\s+lines', query.lower())
        desired_lines = int(length_match.group(1)) if length_match else 20

        combined_content = context if context else get_combined_context(retriever, query)
        if not combined_content:
            return f"No relevant content found in the documents for '{query}'."

        prompt = (
            f"You are an expert note-maker specializing in creating efficient study notes for competitive exams "
            f"(JEE, NEET, UPSC, university semester exams, government job exams).\n\n"
            f"TASK: Create concise notes (~{desired_lines} lines) strictly from the provided content.\n\n"
            f"RULES:\n"
            f"- Use side headings to organize: ## Key Concept, ## Formula, ## Important Fact, "
            f"## Mnemonic, ## Revision Point, ## Exam Tip, ## Example\n"
            f"- Write in SHORT PARAGRAPHS (not bullet points).\n"
            f"- Include SIMPLE EXAMPLES wherever they help clarify a concept.\n"
            f"- Use text-based tables or diagrams for clarity when appropriate.\n"
            f"- Highlight important terms with **bold**.\n"
            f"- Include numerical values and formulas in LaTeX format where applicable.\n\n"
            f"CONTEXT:\n{combined_content}\n\n"
            f"QUERY: {query}\n\n"
            f"Create your notes now:"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @tool
    def exam_prep_agent(query: str, context: str = "") -> str:
        """
        Generate probable exam questions with difficulty grading and a study plan from context.
        Produces exactly 5 questions: 2 Easy, 2 Medium, 1 Tough.
        Args:
            query: The user's instruction.
            context: Optional external context.
        Returns:
            A structured study plan with 5 difficulty-graded probable exam questions.
        """
        combined_content = context if context else get_combined_context(retriever, query)
        if not combined_content:
            return f"No relevant content found in the documents for '{query}'."

        prompt = (
            f"You are an expert academic exam preparation coach and question paper predictor.\n\n"
            f"TASK: Based on the provided study material, do the following:\n\n"
            f"## Part 1: Probable Exam Questions\n"
            f"Generate exactly 5 probable exam questions that are most likely to appear in the exam.\n"
            f"Categorize them by difficulty:\n\n"
            f"### 🟢 Easy (2 Questions)\n"
            f"- These should test basic recall and fundamental understanding.\n"
            f"- Format: 'Q1 [Easy]: [question]' followed by a brief model answer outline.\n\n"
            f"### 🟡 Medium (2 Questions)\n"
            f"- These should test application and analytical thinking.\n"
            f"- Format: 'Q3 [Medium]: [question]' followed by a brief model answer outline.\n\n"
            f"### 🔴 Tough (1 Question)\n"
            f"- This should test deep understanding, synthesis, or multi-step problem solving.\n"
            f"- Format: 'Q5 [Tough]: [question]' followed by a brief model answer outline.\n\n"
            f"## Part 2: Study Plan & Revision Strategy\n"
            f"- List the key topics to revise, priority-ordered.\n"
            f"- Suggest time allocation for each topic.\n"
            f"- Include last-minute revision tips.\n\n"
            f"CONTEXT:\n{combined_content}\n\n"
            f"QUERY: {query}\n\n"
            f"Generate the exam preparation material now:"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @tool
    def concept_explainer(query: str, context: str = "") -> str:
        """
        Explain the requested concept in simple terms using ONLY the retrieved context.
        Automatically determines the topic from the query and provides a detailed explanation.
        Args:
            query: The user's instruction or question (e.g., 'Explain overloading in Java').
            context: Optional external context.
        Returns:
            A clear, detailed explanation of the concept.
        """
        combined_content = context if context else get_combined_context(retriever, query)
        if not combined_content:
            return f"No relevant content found in the documents for '{query}'."

        match = re.search(r'(\d+)\s*lines?', query, re.IGNORECASE)
        if match:
            line_instruction = f"Answer strictly in {int(match.group(1))} clear and concise lines."
        else:
            line_instruction = "Give a thorough, detailed explanation."

        prompt = (
            f"You are an expert teacher who explains complex concepts in simple, easy-to-understand terms.\n\n"
            f"TASK: Explain the concept asked about in the query below.\n\n"
            f"RULES:\n"
            f"- Use ONLY the provided context. Do NOT use outside knowledge.\n"
            f"- Start with a clear definition.\n"
            f"- Provide real-world analogies where helpful.\n"
            f"- Include code examples or formulas if relevant to the topic.\n"
            f"- Use **bold** for key terms.\n"
            f"- {line_instruction}\n\n"
            f"CONTEXT:\n{combined_content}\n\n"
            f"QUERY: {query}\n\n"
            f"Explain now:"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @tool
    def search_agent(query: str) -> str:
        """
        Use Google Custom Search to fetch up-to-date web information and answer the query.
        Searches the web, extracts content from up to 10 websites, and determines if a subtool
        is needed for further processing.
        Args:
            query: The user's web search question.
        Returns:
            A JSON string with 'content', 'subtool', and 'sources' keys.
        """
        search = GoogleSearchAPIWrapper(
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            google_cse_id=os.environ.get("GOOGLE_CSE_ID", ""),
        )
        try:
            results = search.results(query, num_results=5)
            full_contents = []
            sources = []
            for res in results:
                title = res.get("title", "No title")
                link = res.get("link", "No link")
                sources.append({"title": title, "link": link})
                try:
                    resp = requests.get(link, timeout=3)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)[:3000]
                    full_contents.append(f"Title: {title}\nLink: {link}\nContent: {text}")
                except Exception as fetch_e:
                    full_contents.append(
                        f"Title: {title}\nLink: {link}\n"
                        f"Snippet: {res.get('snippet', 'No snippet')}\n"
                        f"Error fetching full content: {str(fetch_e)}"
                    )

            web_text = "\n\n".join(full_contents)

            prompt = (
                f"Query: {query}\n\n"
                f"Based on this online content, extract and organize the relevant information.\n\n"
                f"Instructions:\n"
                f"- Extract and display the complete relevant information from each website.\n"
                f"- Organize by source if multiple.\n"
                f"- Include mathematical equations in LaTeX format where present.\n"
                f"- If diagrams are described, represent them as ASCII art.\n"
                f"- Do not generate MCQs or summaries here; just extract and organize.\n\n"
                f"IMPORTANT: At the very end, analyze the FULL query '{query}' and determine the subtool:\n"
                f"- 'summarizer' if query asks for summary/overview/condense\n"
                f"- 'mcq_generator' if query asks for MCQs/questions/quiz\n"
                f"- 'notes_maker' if query asks for notes/key points\n"
                f"- 'exam_prep_agent' if query asks for exam prep/study plan\n"
                f"- 'concept_explainer' if query asks for explanation/definition\n"
                f"- 'none' if no specific processing is needed\n\n"
                f"End your response with EXACTLY this line:\n"
                f"Final Subtool Decision: [subtool_name]\n\n"
                f"Content:\n{web_text}"
            )
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))

            # Extract subtool decision
            subtool_match = re.search(r'Final Subtool Decision:\s*\[?(\w+)\]?', content, re.IGNORECASE)
            subtool = subtool_match.group(1).lower() if subtool_match else None

            if not subtool:
                ql = query.lower()
                keywords = {
                    "summarizer": ["summarize", "summary", "overview", "condense", "brief"],
                    "mcq_generator": ["mcq", "mcqs", "question", "questions", "quiz"],
                    "notes_maker": ["notes", "make notes", "key points", "revision"],
                    "exam_prep_agent": ["prepare", "exam", "study plan"],
                    "concept_explainer": ["explain", "what is", "define", "how does"],
                }
                subtool = "none"
                for tool_name_key, kws in keywords.items():
                    if any(kw in ql for kw in kws):
                        subtool = tool_name_key
                        break

            content = re.sub(r'Final Subtool Decision:.*', '', content, flags=re.IGNORECASE | re.DOTALL).strip()

            # Return as JSON string to avoid ToolNode serialization issues
            return json.dumps({"content": content, "subtool": subtool, "sources": sources})

        except Exception as e:
            return json.dumps({
                "content": f"Error during search: {str(e)}",
                "subtool": "none",
                "sources": [],
            })

    @tool
    def chat_agent(query: str) -> str:
        """
        Free-form chat with the LLM. No RAG retrieval — direct conversation.
        Use this for general questions, greetings, or when the user just wants to talk.
        Args:
            query: The user's message.
        Returns:
            The LLM's response.
        """
        prompt = (
            f"You are a friendly, knowledgeable AI study assistant for college students.\n"
            f"Respond helpfully, concisely, and in a supportive tone.\n"
            f"If the user asks a factual question, provide accurate information.\n"
            f"If the user greets you, greet them back warmly.\n\n"
            f"User: {query}\n\n"
            f"Assistant:"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))

    return [summarizer, mcq_generator, notes_maker, exam_prep_agent, concept_explainer, search_agent, chat_agent]
