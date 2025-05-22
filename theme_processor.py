"""
Theme identification module using shared vector database.

This module identifies common themes across documents by finding the most 
semantically similar chunks to the user's query, using the shared vector_db
module for efficient retrieval.
"""

import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
import vector_db
from dotenv import load_dotenv

load_dotenv()

# Rebuild ChatOpenAI model for compatibility
ChatOpenAI.model_rebuild()

def identify_themes(query, document_results, top_n=5):
    """
    Identify common themes across document query results using semantic search.
    
    Args:
        query (str): The original query that was posed
        document_results (list): List of results from individual documents
        top_n (int): Number of top themes to identify
            
    Returns:
        dict: Synthesized response with identified themes and citations
    """
    # Handle cases with no documents
    if not document_results:
        return {
            'themes': [],
            'synthesized_answer': "No documents were processed for this query."
        }
    
    # Handle the case with only one document
    if len(document_results) == 1:
        return {
            'themes': [{
                'name': 'Single Document Analysis',
                'description': document_results[0]['answer'],
                'supporting_documents': [document_results[0]['doc_id']],
                'citations': [document_results[0]['citation']]
            }],
            'synthesized_answer': document_results[0]['answer']
        }
    
    # Get list of document IDs
    doc_ids = [result['doc_id'] for result in document_results]
    
    # Create a multi-document vector database
    multi_doc_db = vector_db.create_multi_document_vector_db(doc_ids)
    
    if not multi_doc_db:
        return {
            'themes': [],
            'synthesized_answer': "Error: Could not create vector database for theme identification."
        }
    
    # Find the top N most relevant chunks to the query
    top_results = vector_db.query_vector_db(multi_doc_db, query, k=top_n)
    
    # Build a map of document information
    doc_info_map = {}
    for result in document_results:
        doc_info_map[result['doc_id']] = {
            'filename': result['filename'],
            'answer': result['answer']
        }
    
    # Create themes from top results
    themes = []
    for i, result in enumerate(top_results):
        content = result.page_content
        metadata = result.metadata
        doc_id = metadata['doc_id']
        citation = metadata['citation']
        
        # Extract first sentence for theme name (up to 50 chars)
        first_sentence_match = re.search(r'(.*?[.!?])', content)
        if first_sentence_match:
            theme_name = first_sentence_match.group(1)
            if len(theme_name) > 50:
                theme_name = theme_name[:47] + "..."
        else:
            theme_name = content[:50] + "..." if len(content) > 50 else content
        
        # Add this as a theme
        themes.append({
            'name': f"Theme {i+1}: {theme_name}",
            'description': content,
            'supporting_documents': [doc_id],
            'citations': [citation]
        })
    
    # Find additional supporting documents for each theme
    for theme in themes:
        theme_content = theme['description']
        
        # Search for similar content to this theme
        similar_results = vector_db.query_vector_db(multi_doc_db, theme_content, k=10)
        
        # Skip the first result (which is the theme itself)
        for similar_doc in similar_results[1:]:
            similar_doc_id = similar_doc.metadata['doc_id']
            similar_citation = similar_doc.metadata['citation']
            
            # If this document is not already supporting this theme
            if similar_doc_id not in theme['supporting_documents']:
                # Add it as a supporting document
                theme['supporting_documents'].append(similar_doc_id)
                theme['citations'].append(similar_citation)
    
    # Create a synthesized answer using modern LangChain patterns
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")  # Fixed model name
    output_parser = StrOutputParser()
    
    # Create a proper prompt template using modern LangChain
    synthesis_prompt = PromptTemplate(
        template="""User Query: {query}

I have identified the following key themes from the documents:

{themes_text}

Please create a comprehensive, well-organized answer that:
1. Directly addresses the user's query
2. Synthesizes the information from these themes
3. Is clear, concise, and well-structured

Your synthesized answer:""",
        input_variables=["query", "themes_text"]
    )
    
    # Create the chain using modern LangChain patterns (Option 2)
    synthesis_chain = LLMChain(
        llm=llm,
        prompt=synthesis_prompt,
        output_parser=output_parser
    )
    
    # Format themes for the prompt
    themes_text = ""
    for i, theme in enumerate(themes):
        themes_text += f"Theme {i+1}: {theme['name']}\n"
        themes_text += f"Description: {theme['description']}\n"
        themes_text += f"Citations: {', '.join(theme['citations'])}\n\n"
    
    # Generate the synthesis using proper chain invocation (Option 2)
    try:
        synthesis_response = synthesis_chain.invoke({
            "query": query,
            "themes_text": themes_text
        })
        
        # Handle different response formats
        if isinstance(synthesis_response, dict):
            synthesized_answer = synthesis_response.get('text', str(synthesis_response))
        else:
            synthesized_answer = str(synthesis_response)
            
    except Exception as e:
        print(f"Error in synthesis chain: {str(e)}")
        # Fallback synthesis without LLM
        synthesized_answer = f"Analysis of {len(themes)} themes found across {len(doc_ids)} documents for query: '{query}'"
    
    return {
        'themes': themes,
        'synthesized_answer': synthesized_answer
    }


# Alternative implementation using the most modern LangChain patterns
class ModernThemeProcessor:
    """Modern theme processor using latest LangChain patterns"""
    
    def __init__(self, model="gpt-4o-mini-2024-07-18", temperature=0):
        """Initialize the theme processor"""
        self.llm = ChatOpenAI(temperature=temperature, model=model)
        self.output_parser = StrOutputParser()
        self.synthesis_chain = self._create_synthesis_chain()
    
    def _create_synthesis_chain(self):
        """Create the synthesis chain using ChatPromptTemplate"""
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst specializing in theme identification and synthesis.

Your task is to:
1. Analyze the provided document themes and content
2. Create a comprehensive synthesis that directly addresses the user's query
3. Ensure the response is well-structured and coherent
4. Maintain accuracy and provide clear reasoning

Guidelines:
- Focus on directly answering the user's query
- Synthesize information across all themes
- Be clear, concise, and well-organized
- Maintain factual accuracy"""),
            
            ("human", """User Query: {query}

I have identified the following key themes from the documents:

{themes_text}

Please create a comprehensive, well-organized answer that:
1. Directly addresses the user's query
2. Synthesizes the information from these themes
3. Is clear, concise, and well-structured

Your synthesized answer:""")
        ])
        
        # Create the chain using pipe operator (most modern approach)
        chain = synthesis_prompt | self.llm | self.output_parser
        return chain
    
    def identify_themes_modern(self, query, document_results, top_n=5):
        """
        Modern theme identification using latest LangChain patterns
        """
        # Handle edge cases (same as original function)
        if not document_results:
            return {
                'themes': [],
                'synthesized_answer': "No documents were processed for this query."
            }
        
        if len(document_results) == 1:
            return {
                'themes': [{
                    'name': 'Single Document Analysis',
                    'description': document_results[0]['answer'],
                    'supporting_documents': [document_results[0]['doc_id']],
                    'citations': [document_results[0]['citation']]
                }],
                'synthesized_answer': document_results[0]['answer']
            }
        
        # Get themes using vector search (same logic as original)
        doc_ids = [result['doc_id'] for result in document_results]
        multi_doc_db = vector_db.create_multi_document_vector_db(doc_ids)
        
        if not multi_doc_db:
            return {
                'themes': [],
                'synthesized_answer': "Error: Could not create vector database for theme identification."
            }
        
        top_results = vector_db.query_vector_db(multi_doc_db, query, k=top_n)
        
        # Create themes (same logic as original)
        themes = []
        for i, result in enumerate(top_results):
            content = result.page_content
            metadata = result.metadata
            doc_id = metadata['doc_id']
            citation = metadata['citation']
            
            # Extract theme name
            first_sentence_match = re.search(r'(.*?[.!?])', content)
            if first_sentence_match:
                theme_name = first_sentence_match.group(1)
                if len(theme_name) > 50:
                    theme_name = theme_name[:47] + "..."
            else:
                theme_name = content[:50] + "..." if len(content) > 50 else content
            
            themes.append({
                'name': f"Theme {i+1}: {theme_name}",
                'description': content,
                'supporting_documents': [doc_id],
                'citations': [citation]
            })
        
        # Find supporting documents (same logic as original)
        for theme in themes:
            theme_content = theme['description']
            similar_results = vector_db.query_vector_db(multi_doc_db, theme_content, k=10)
            
            for similar_doc in similar_results[1:]:
                similar_doc_id = similar_doc.metadata['doc_id']
                similar_citation = similar_doc.metadata['citation']
                
                if similar_doc_id not in theme['supporting_documents']:
                    theme['supporting_documents'].append(similar_doc_id)
                    theme['citations'].append(similar_citation)
        
        # Format themes for synthesis
        themes_text = ""
        for i, theme in enumerate(themes):
            themes_text += f"Theme {i+1}: {theme['name']}\n"
            themes_text += f"Description: {theme['description']}\n"
            themes_text += f"Citations: {', '.join(theme['citations'])}\n\n"
        
        # Generate synthesis using modern chain
        try:
            synthesis_response = self.synthesis_chain.invoke({
                "query": query,
                "themes_text": themes_text
            })
            
        except Exception as e:
            print(f"Error in modern synthesis chain: {str(e)}")
            synthesis_response = f"Analysis of {len(themes)} themes found across {len(doc_ids)} documents for query: '{query}'"
        
        return {
            'themes': themes,
            'synthesized_answer': synthesis_response
        }


# Helper function to use the modern processor
def identify_themes_with_modern_processor(query, document_results, top_n=5):
    """
    Wrapper function to use the modern theme processor
    """
    processor = ModernThemeProcessor()
    return processor.identify_themes_modern(query, document_results, top_n)