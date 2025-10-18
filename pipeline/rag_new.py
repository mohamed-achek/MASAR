"""
RAG Module - Answer generation with LLM

This module handles:
- Assembling retrieved context
- Building prompts for LLM
- Generating answers with citations
- Post-processing responses
- Handling different question types
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ============================================================================
# CONTEXT ASSEMBLER
# ============================================================================

class ContextAssembler:
    """Assemble retrieved results into LLM context."""
    
    def __init__(self, max_context_tokens: int = 4000):
        """
        Initialize context assembler.
        
        Args:
            max_context_tokens: Maximum tokens for context
        """
        self.max_context_tokens = max_context_tokens
    
    def assemble(
        self,
        results: List[Dict[str, Any]],
        include_metadata: bool = True,
        include_citations: bool = True
    ) -> str:
        """
        Assemble results into formatted context.
        
        Args:
            results: List of retrieval results
            include_metadata: Include metadata in context
            include_citations: Add citation markers
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Citation marker
            citation = f"[{i}]" if include_citations else ""
            
            # Extract text and metadata
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            result_type = result.get('type', 'chunk')
            
            # Format section
            section = f"{citation} "
            
            # Add metadata header
            if include_metadata:
                meta_obj = metadata.get('metadata', metadata)
                section_title = metadata.get('section_title', 'N/A')
                university = meta_obj.get('university_id', 'N/A')
                program = meta_obj.get('program', 'N/A')
                
                section += f"[{university} - {program} - {section_title}]\n"
            
            # Add text
            section += text
            
            # Add table HTML if available
            if result_type == 'table' and 'table_html' in metadata:
                section += f"\n\nTable:\n{metadata['table_html']}"
            
            context_parts.append(section)
        
        # Join all parts
        context = "\n\n---\n\n".join(context_parts)
        
        # TODO: Truncate to max_context_tokens if needed
        # For now, return as is
        return context
    
    def create_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Create citation references.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            meta_obj = metadata.get('metadata', metadata)
            
            citation = {
                "id": i,
                "type": result.get('type', 'chunk'),
                "section": metadata.get('section_title', 'Unknown'),
                "university_id": meta_obj.get('university_id', 'Unknown'),
                "program": meta_obj.get('program', 'Unknown'),
                "year": meta_obj.get('year', 'Unknown'),
                "source_file": meta_obj.get('source_file', 'Unknown')
            }
            
            citations.append(citation)
        
        return citations


# ============================================================================
# PROMPT BUILDER
# ============================================================================

class PromptBuilder:
    """Build prompts for different question types."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize prompt builder.
        
        Args:
            language: Language for prompts ('en' or 'ar')
        """
        self.language = language
    
    def build_qa_prompt(
        self,
        question: str,
        context: str,
        instructions: Optional[str] = None
    ) -> str:
        """
        Build question-answering prompt.
        
        Args:
            question: User question
            context: Retrieved context
            instructions: Additional instructions
            
        Returns:
            Formatted prompt
        """
        if self.language == "ar":
            prompt = f"""أنت مساعد أكاديمي متخصص في الإجابة على الأسئلة حول المناهج الجامعية.

استخدم المعلومات التالية للإجابة على السؤال. إذا لم تجد الإجابة في السياق المقدم، قل ذلك بوضوح.

السياق:
{context}

السؤال: {question}

تعليمات:
- أجب بدقة بناءً على المعلومات المقدمة
- استشهد بالمصادر باستخدام الأرقام [1], [2], إلخ
- إذا كانت المعلومات غير كافية، اذكر ذلك
- كن موجزاً ومباشراً

الإجابة:"""
        else:
            prompt = f"""You are an academic assistant specialized in answering questions about university curricula.

Use the following information to answer the question. If you cannot find the answer in the provided context, clearly state so.

Context:
{context}

Question: {question}

Instructions:
- Answer accurately based on the provided information
- Cite sources using numbers [1], [2], etc.
- If information is insufficient, mention it
- Be concise and direct

Answer:"""
        
        if instructions:
            prompt = prompt.replace("Instructions:", f"Instructions:\n{instructions}\n-")
        
        return prompt
    
    def build_comparison_prompt(
        self,
        question: str,
        context: str
    ) -> str:
        """Build prompt for comparison questions."""
        if self.language == "ar":
            prompt = f"""أنت مساعد أكاديمي متخصص في مقارنة المناهج الجامعية.

قارن بين البرامج أو المؤسسات بناءً على المعلومات التالية.

السياق:
{context}

السؤال: {question}

قدم مقارنة منظمة مع الاستشهاد بالمصادر.

الإجابة:"""
        else:
            prompt = f"""You are an academic assistant specialized in comparing university curricula.

Compare programs or institutions based on the following information.

Context:
{context}

Question: {question}

Provide a structured comparison with source citations.

Answer:"""
        
        return prompt
    
    def build_summarization_prompt(
        self,
        topic: str,
        context: str
    ) -> str:
        """Build prompt for summarization."""
        if self.language == "ar":
            prompt = f"""لخص المعلومات التالية حول: {topic}

السياق:
{context}

قدم ملخصاً شاملاً ومنظماً.

الملخص:"""
        else:
            prompt = f"""Summarize the following information about: {topic}

Context:
{context}

Provide a comprehensive and structured summary.

Summary:"""
        
        return prompt


# ============================================================================
# LLM INTERFACE
# ============================================================================

class LLMInterface:
    """Interface for different LLM providers."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize LLM interface.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'ollama')
            model: Model name
            api_key: API key (if required)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize client
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            self.client = openai.OpenAI(api_key=api_key)
            
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif provider == "ollama":
            # Use local Ollama server
            self.client = None
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        print(f"✅ LLM initialized: {provider} - {model}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif self.provider == "ollama":
            # Use Ollama API
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
            )
            return response.json()['response']
        
        return ""


# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline for question answering."""
    
    def __init__(
        self,
        retriever,
        llm: LLMInterface,
        prompt_builder: Optional[PromptBuilder] = None,
        context_assembler: Optional[ContextAssembler] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: HybridRetriever instance
            llm: LLM interface
            prompt_builder: Prompt builder (optional)
            context_assembler: Context assembler (optional)
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.context_assembler = context_assembler or ContextAssembler()
    
    def answer_question(
        self,
        question: str,
        k: int = 5,
        metadata_filters: Optional[Dict] = None,
        question_type: str = "qa",
        return_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            k: Number of context chunks
            metadata_filters: Metadata filters for retrieval
            question_type: Type of question ('qa', 'comparison', 'summary')
            return_citations: Include citation references
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Retrieve relevant context
        print(f"🔍 Retrieving context for: {question}")
        results = self.retriever.hybrid_search(
            query=question,
            k=k,
            metadata_filters=metadata_filters,
            rerank=True
        )
        
        if not results:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "citations": [],
                "num_sources": 0
            }
        
        # Assemble context
        context = self.context_assembler.assemble(results, include_citations=True)
        
        # Build prompt
        if question_type == "qa":
            prompt = self.prompt_builder.build_qa_prompt(question, context)
        elif question_type == "comparison":
            prompt = self.prompt_builder.build_comparison_prompt(question, context)
        elif question_type == "summary":
            prompt = self.prompt_builder.build_summarization_prompt(question, context)
        else:
            prompt = self.prompt_builder.build_qa_prompt(question, context)
        
        # Generate answer
        print("🤖 Generating answer...")
        answer = self.llm.generate(prompt)
        
        # Create citations
        citations = self.context_assembler.create_citations(results) if return_citations else []
        
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "num_sources": len(results),
            "metadata_filters": metadata_filters
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for RAG testing."""
    import argparse
    import sys
    from pathlib import Path
    
    # Add pipeline directory to path if not already there
    pipeline_dir = Path(__file__).parent
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    
    from retrieve import QueryEncoder, Reranker, HybridRetriever
    
    parser = argparse.ArgumentParser(description="RAG question answering")
    parser.add_argument("--index-dir", type=Path, required=True, help="Index directory")
    parser.add_argument("--embeddings-dir", type=Path, required=True, help="Embeddings directory")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="llama3.2", help="LLM model")
    parser.add_argument("--api-key", help="API key (if needed)")
    parser.add_argument("--k", type=int, default=5, help="Number of context chunks")
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing RAG pipeline...")
    encoder = QueryEncoder()
    reranker = Reranker()
    
    retriever = HybridRetriever(
        index_dir=args.index_dir,
        embeddings_dir=args.embeddings_dir,
        encoder=encoder,
        reranker=reranker
    )
    
    llm = LLMInterface(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key
    )
    
    rag = RAGPipeline(retriever=retriever, llm=llm)
    
    # Answer question
    result = rag.answer_question(question=args.question, k=args.k)
    
    # Display result
    print("\n" + "=" * 80)
    print(f"Question: {result['question']}")
    print("=" * 80)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n📚 Sources: {result['num_sources']}")
    
    if result['citations']:
        print("\nCitations:")
        for citation in result['citations']:
            print(f"  [{citation['id']}] {citation['university_id']} - {citation['program']} - {citation['section']}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
