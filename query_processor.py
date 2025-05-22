"""
Query processing module that handles document queries using OpenAI and our shared vector database.

This module processes natural language queries against documents by using
the vector_db module for retrieval and OpenAI for answer generation.
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import vector_db
from dotenv import load_dotenv

load_dotenv()

# Rebuild ChatOpenAI for compatibility
ChatOpenAI.model_rebuild()

def query_document(query, content_path, doc_id):
    """
    Process a query against a document and return the answer with citation.
    
    Args:
        query (str): The natural language query to process
        content_path (str): Path to the document content file
        doc_id (str): Document identifier
        
    Returns:
        dict: Query result with answer, citation, and confidence score
    """
    try:
        print(f"Processing query for document {doc_id}: '{query}'")
        
        # Create a vector database for this document
        doc_vector_db = vector_db.create_document_vector_db(doc_id)
        
        if not doc_vector_db:
            return {
                'answer': f"Error: Could not process document {doc_id}",
                'citation': 'Error',
                'confidence': 0
            }
        
        # Get relevant documents
        relevant_docs = vector_db.query_vector_db(doc_vector_db, query, k=3)
        
        if relevant_docs:
            # Use modern LangChain approach for question answering
            # Initialize the LLM
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
            output_parser = StrOutputParser()
            
            # Create a prompt template for answering with citations
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that provides accurate information based on the provided document sections.
Answer the user's query based ONLY on the information in the document sections provided.
If the information to answer the query is not in the sections, say "I cannot find information about this in the document."
Do not make up or infer information that is not explicitly stated in the document sections.
Keep your answer concise and to the point."""),
                ("human", "User Query: {query}\n\nRelevant document sections:\n{context}")
            ])
            
            # Extract the document contents into a context string
            context = "\n\n".join([f"Document section (from {doc.metadata.get('citation', 'Unknown location')}):\n{doc.page_content}" 
                                 for doc in relevant_docs])
            
            # Create context function
            def get_context(inputs):
                return context
            
            # Create the chain using modern LangChain patterns
            chain = (
                {
                    "context": RunnableLambda(get_context), 
                    "query": RunnablePassthrough()
                }
                | prompt
                | llm
                | output_parser
            )
            
            # Run the chain
            try:
                response = chain.invoke(query)
                answer = response
                
                # Collect citations from all relevant documents
                citations = []
                for doc in relevant_docs:
                    citation = doc.metadata.get('citation', 'Citation not available')
                    if citation not in citations:
                        citations.append(citation)
                
                citation_text = "; ".join(citations)
                
                print(f"Successfully generated answer for query")
                
                return {
                    'answer': answer,
                    'citation': citation_text,
                    'confidence': 0.95
                }
                
            except Exception as chain_error:
                print(f"Error in chain execution: {str(chain_error)}")
                # Fallback to simple concatenation
                context_simple = "\n".join([doc.page_content for doc in relevant_docs])
                return {
                    'answer': f"Found relevant information: {context_simple[:500]}...",
                    'citation': "; ".join([doc.metadata.get('citation', 'Unknown') for doc in relevant_docs]),
                    'confidence': 0.7
                }
                
        else:
            print("No relevant information found in the document")
            return {
                'answer': 'No relevant information found in this document.',
                'citation': 'N/A',
                'confidence': 0
            }
            
    except Exception as e:
        print(f"Error processing query for document {doc_id}: {str(e)}")
        return {
            'answer': f"Error processing query: {str(e)}",
            'citation': 'Error',
            'confidence': 0
        }


# Alternative implementation using simpler approach
def query_document_simple(query, content_path, doc_id):
    """
    Simplified query processing that avoids complex chain structures
    """
    try:
        print(f"Processing query for document {doc_id}: '{query}'")
        
        # Create a vector database for this document
        doc_vector_db = vector_db.create_document_vector_db(doc_id)
        
        if not doc_vector_db:
            return {
                'answer': f"Error: Could not process document {doc_id}",
                'citation': 'Error',
                'confidence': 0
            }
        
        # Get relevant documents
        relevant_docs = vector_db.query_vector_db(doc_vector_db, query, k=3)
        
        if relevant_docs:
            # Initialize the LLM
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
            
            # Create context from relevant documents
            context = "\n\n".join([f"Section from {doc.metadata.get('citation', 'Unknown location')}:\n{doc.page_content}" 
                                 for doc in relevant_docs])
            
            # Create a simple prompt
            prompt_text = f"""You are a helpful assistant that provides accurate information based on document sections.

User Query: {query}

Relevant document sections:
{context}

Please answer the query based ONLY on the information provided above. If the information is not available, say so clearly.

Answer:"""
            
            # Get response directly from LLM
            try:
                from langchain_core.messages import HumanMessage
                message = HumanMessage(content=prompt_text)
                response = llm.invoke([message])
                answer = response.content
                
                # Collect citations
                citations = []
                for doc in relevant_docs:
                    citation = doc.metadata.get('citation', 'Citation not available')
                    if citation not in citations:
                        citations.append(citation)
                
                citation_text = "; ".join(citations)
                
                return {
                    'answer': answer,
                    'citation': citation_text,
                    'confidence': 0.95
                }
                
            except Exception as llm_error:
                print(f"Error with LLM call: {str(llm_error)}")
                # Return the context as fallback
                return {
                    'answer': f"Found relevant information in the document: {context[:300]}...",
                    'citation': "; ".join([doc.metadata.get('citation', 'Unknown') for doc in relevant_docs]),
                    'confidence': 0.5
                }
        else:
            return {
                'answer': 'No relevant information found in this document.',
                'citation': 'N/A',
                'confidence': 0
            }
            
    except Exception as e:
        print(f"Error processing query for document {doc_id}: {str(e)}")
        return {
            'answer': f"Error processing query: {str(e)}",
            'citation': 'Error',
            'confidence': 0
        }
    

    