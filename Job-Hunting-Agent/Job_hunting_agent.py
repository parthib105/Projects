import os
import PyPDF2
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.string import StrOutputParser
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0.6,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
# Initialize the Tavily tool
tavily_tool = TavilySearch(max_results=5, api_key=os.getenv("TAVILY_API_KEY"))

# --- Define Agent State ---
class AgentState(TypedDict):
    resume_path: str
    resume_text: str
    search_queries: List[str]
    job_listings: List[Dict] 
    ranked_jobs: str

# --- Define Agent Nodes ---
def parse_resume(state: AgentState) -> Dict[str, str]:
    """Parses the resume PDF to extract text."""
    print("---NODE: PARSING RESUME---")
    path = state.get("resume_path")
    try: 
        with open(path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text: str = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            if not text:
                raise ValueError("PDF content could not be extracted or is empty!")
        print("--- Done parsing resume ✅ ---")
        return {"resume_text": text}
    except FileNotFoundError:
        raise FileNotFoundError(f"Resume file not found at path: {path}")

def generate_search_queries(state: AgentState) -> Dict[str, Any]:
    """Generates job search queries using Gemini based on resume text."""
    print("---NODE: GENERATING SEARCH QUERIES---")
    resume_text = state.get("resume_text")
    prompt = ChatPromptTemplate.from_template(
        """Based on the following resume text, generate 10-15 relevant and diverse job search queries (e.g., 'Senior Python Developer', 'Machine Learning Engineer remote').
        Return them as a comma-separated list.

        Resume:
        {resume}"""
    )
    chain = prompt | llm | StrOutputParser()
    queries_str = chain.invoke({"resume": resume_text})
    queries_list = [q.strip() for q in queries_str.split(',')]
    print("--- Done generating search queries ✅ ---")
    return {"search_queries": queries_list}

def search_for_jobs(state: AgentState) -> Dict[str, Any]:
    """Searches for jobs online using Tavily Search."""
    print("---NODE: SEARCHING FOR JOBS---")
    queries = state.get("search_queries")
    all_results: List[Dict] = []
    
    for query in queries:
        print(f"Searching for: '{query} job opening'")
        try:
            # Tavily Search returns results in a different format
            search_results = tavily_tool.invoke({"query": f"{query} job opening"})
            
            # Debug: Print the type and structure of results
            print(f"Search results type: {type(search_results)}")
            
            # Handle different possible return formats from Tavily
            results_to_process = []
            
            if isinstance(search_results, dict):
                # If it's a dict, look for 'results' key
                if 'results' in search_results:
                    results_to_process = search_results['results']
                elif 'content' in search_results and 'url' in search_results:
                    # If it's a single result in dict format
                    results_to_process = [search_results]
                else:
                    # Try to use the dict as is
                    results_to_process = [search_results]
            elif isinstance(search_results, list):
                # If it's already a list
                results_to_process = search_results
            else:
                # If it's a string or other format, wrap it
                results_to_process = [{"content": str(search_results), "url": "N/A"}]
            
            # Process the results
            for res in results_to_process:
                if isinstance(res, dict):
                    content = res.get('content', res.get('snippet', str(res)))
                    url = res.get('url', res.get('link', 'N/A'))
                    title = res.get('title', 'N/A')
                    
                    # Check if it's a job-related result
                    if content and isinstance(content, str):
                        content_lower = content.lower()
                        if any(keyword in content_lower for keyword in [
                            "responsibilities", "qualifications", "requirements", 
                            "description", "apply", "job", "position", "role", 
                            "career", "hiring", "employment", "work", "salary"
                        ]):
                            all_results.append({
                                "content": content,
                                "url": url,
                                "title": title
                            })
                elif isinstance(res, str):
                    # If result is a string, treat it as content
                    if any(keyword in res.lower() for keyword in [
                        "job", "position", "role", "career", "hiring", "employment"
                    ]):
                        all_results.append({
                            "content": res,
                            "url": "N/A",
                            "title": "N/A"
                        })
                        
        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            continue
    
    # If no results found, create some fallback results
    if not all_results:
        print("No results found from Tavily, creating fallback results...")
        # You might want to add fallback logic here or modify the search strategy
        fallback_results = [
            {
                "content": "Entry level machine learning position with focus on Python and data analysis. Requirements include programming skills and analytical thinking.",
                "url": "https://example.com/job1",
                "title": "Machine Learning Intern"
            },
            {
                "content": "Software engineering internship opportunity for students with C++ and algorithm experience. Great learning environment.",
                "url": "https://example.com/job2", 
                "title": "Software Engineering Intern"
            }
        ]
        all_results = fallback_results
        print("Using fallback results for demonstration purposes")
    
    print(f"--- Collected {len(all_results)} valid job listings ✅ ---")
    return {"job_listings": all_results}

def filter_and_rank_jobs(state: AgentState) -> Dict[str, str]:
    """Uses Gemini to filter and rank the found job listings against the resume."""
    print("---NODE: FILTERING AND RANKING JOBS---")
    resume_text = state.get("resume_text")
    listings = state.get("job_listings")
    
    formatted_listings = ""
    for i, job in enumerate(listings):
        if isinstance(job, dict):
            content = job.get('content', 'No description available.')
            url = job.get('url', 'No URL available.')
            title = job.get('title', 'No title available.')
        else:
            content = str(job)
            url = 'No URL available.'
            title = 'No title available.'
        formatted_listings += f"--- Job {i+1} ---\nTitle: {title}\nURL: {url}\nDescription: {content}\n\n"

    prompt = ChatPromptTemplate.from_template(
        """You are an expert career assistant. Based on the provided resume, analyze the following job listings.
        Filter out irrelevant listings and rank the top 5 most suitable jobs.
        For each ranked job, provide the job title, its URL, and a brief (2-3 sentences) explanation of why it's a good match.

        Resume:
        {resume}

        Job Listings:
        {listings}
        """
    )
    chain = prompt | llm | StrOutputParser()
    ranked_list = chain.invoke({"resume": resume_text, "listings": formatted_listings})
    print("--- Done ranking jobs ✅ ---")
    return {"ranked_jobs": ranked_list}

# --- Define the graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("parse_resume", parse_resume)
workflow.add_node("generate_queries", generate_search_queries)
workflow.add_node("search_jobs", search_for_jobs)
workflow.add_node("rank_jobs", filter_and_rank_jobs)

# Define edges (the flow of control)
workflow.set_entry_point("parse_resume")
workflow.add_edge("parse_resume", "generate_queries")
workflow.add_edge("generate_queries", "search_jobs")
workflow.add_edge("search_jobs", "rank_jobs")
workflow.add_edge("rank_jobs", END)

# Compile the graph into a runnable app
app = workflow.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    # Using a raw string (r"...") or forward slashes is safer for file paths on Windows
    resume_file = r"D:\Programming\pythonProjetcs\myProjects\Parthib_CV_for_ML.pdf"
    inputs = {"resume_path": resume_file}
    
    final_state = app.invoke(inputs)

    print("\n\n---JOB SEARCH COMPLETE--- ✅")
    print(final_state['ranked_jobs'])