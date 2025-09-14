# Buddy

## AI-Powered Support Agent using LangGraph & LLMs
### Abstract
This project presents the design and development of an AI-powered support agent tailored for ITcompanies to handle both employee and customer queries in a unified and efficient manner. Unlike general-purpose assistants such as ChatGPT, Gemini, or Perplexity, this system is domain-specific, ensuring secure, company-focused interactions. The agent is built using LangGraph, enabling multi-step reasoning, dynamic workflows, and seamless integration with enterprise systems. Powered by Large Language Models (LLMs), the agent can understand natural language, retrieve company-specific knowledge, and provide accurate responses in real-time. It supports employees by answering HR, payroll, and IT-related queries while simultaneously assisting customers with product support, service requests, and troubleshooting. The uniqueness of this system lies in its ability to act as a bridge between employees, customers, and IT systems, unlike general AI assistants that provide generic knowledge. By integrating contextual company data and workflows, the agent ensures reliable, secure, and personalized support, thereby reducing human workload, increasing response accuracy, and improving overall organizational efficiency.

ollama pull mistral     # ~4.1 GB (fast, very good general-purpose)
ollama pull gemma:2b    # ~1.4 GB (much smaller, good for quick testing)
ollama pull phi3        # ~2.2 GB (small, optimized for reasoning)
ollama pull llama3.1:8b # ~4.7 GB (the one you are downloading now)
ollama pull phi3:mini   # ~1.1 GB
ollama pull qwen2:0.5b  # ~0.5 GB (very small, runs almost anywhere)
