# Job Search Agent

An intelligent job search agent that analyzes your resume and automatically finds relevant job opportunities using AI-powered search and ranking.

## Features

- **Resume Analysis**: Automatically extracts and analyzes text from PDF resumes
- **Smart Query Generation**: Uses Google's Gemini AI to generate diverse, relevant job search queries based on your resume
- **Automated Job Search**: Searches for job listings using Tavily Search API
- **Intelligent Ranking**: Filters and ranks job opportunities based on resume compatibility
- **Multi-Agent Workflow**: Uses LangGraph for orchestrated, state-based processing

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini AI)
- Tavily API Key (for web search)

## Project Structure

```
Job-Hunting-Agent/
├── job_search_agent.py    # Main application script
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md            # This file
└── your_resume.pdf      # Your resume file
```

## Installation

1. Clone the repository or download the script
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project directory with your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd job-search-agent
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Add Your Resume**:
   Place your PDF resume in the project directory

4. **Update Script**:
   Edit the resume path in `job_search_agent.py`:
   ```python
   resume_file = r"path/to/your/resume.pdf"
   ```

5. **Run the Agent**:
   ```bash
   python job_search_agent.py
   ```

## How It Works

The agent follows a structured workflow with four main stages:

### 1. Resume Parsing
- Extracts text content from PDF resume using PyPDF2
- Handles errors for missing or unreadable files

### 2. Query Generation
- Analyzes resume content using Google's Gemini 2.5 Pro model
- Generates 10-15 diverse, relevant job search queries
- Examples: "Senior Python Developer", "Machine Learning Engineer remote"

### 3. Job Search
- Searches for job listings using Tavily Search API
- Processes multiple query formats and result structures
- Filters results for job-related content
- Includes fallback mechanism for demonstration purposes

### 4. Ranking and Filtering
- Uses Gemini AI to analyze job compatibility with resume
- Filters out irrelevant listings
- Ranks top 5 most suitable opportunities
- Provides detailed explanations for each recommendation

## Output

The agent provides:
- Job title and URL for each recommended position
- Brief explanation (2-3 sentences) of why each job is a good match
- Ranked list of top 5 most suitable opportunities

## Configuration

### LLM Settings
- **Model**: Gemini 2.5 Pro
- **Temperature**: 0.6 (balanced creativity and consistency)

### Search Settings
- **Max Results**: 5 per query
- **Query Format**: "{query} job opening"

## Error Handling

The agent includes robust error handling for:
- Missing or corrupted PDF files
- API connection issues
- Malformed search results
- Empty or invalid resume content

## Limitations

- Only supports PDF resume format
- Requires active internet connection for API calls
- Search results depend on Tavily's available job listings
- May require API rate limit considerations for large-scale usage

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the agent's functionality.

## License

This project is provided as-is for educational and personal use.

## API Keys Setup

### Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Generative AI API
4. Create credentials (API Key)
5. Add the key to your `.env` file

### Tavily API Key
1. Visit [Tavily](https://tavily.com/)
2. Sign up for an account
3. Generate an API key from your dashboard
4. Add the key to your `.env` file

## Troubleshooting

### Common Issues
- **PDF not readable**: Ensure the PDF contains selectable text (not scanned images)
- **API errors**: Verify your API keys are correct and have sufficient quotas
- **No results**: Check your internet connection and API key validity
- **Path errors**: Use raw strings (`r"path"`) or forward slashes for file paths

### Debug Information
The script includes detailed logging for each stage:
- Resume parsing status
- Query generation results
- Search progress and results count
- Ranking completion status