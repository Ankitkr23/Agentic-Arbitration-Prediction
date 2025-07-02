import os
import json
from sec_edgar_downloader import Downloader
from huggingface_hub import login # Changed from notebook_login for script usage
from smolagents import Tool, HfApiModel, ToolCallingAgent
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# --- Configuration ---
# Set your Hugging Face API token here.
# IMPORTANT: For production, consider loading this from an environment variable
# (e.g., os.getenv("HF_TOKEN")) rather than hardcoding it.
# Some models (like Llama-3.1-70B-Instruct) may require a Hugging Face Pro subscription.
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE" # Replace with your actual token

# Login to Hugging Face Hub programmatically if HF_TOKEN is provided
if HF_TOKEN and HF_TOKEN != "YOUR_HUGGING_FACE_TOKEN_HERE":
    try:
        login(token=HF_TOKEN)
        print("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        print(f"Error logging into Hugging Face Hub: {e}. Please ensure your token is valid.")
else:
    print("Hugging Face token not provided or is placeholder. Some models may not work.")

#########################################################################################################################
# 1. Data Fetcher: Currently using SEC filings as an example, replace with your own data source
########################################################################################################################
class SECDataFetcher:
    """
    A class to download SEC filings using the sec_edgar_downloader library.
    This serves as an example data source; it can be replaced with other
    data fetching mechanisms for legal documents.
    """
    def __init__(self, storage_dir: str = "./sec_filings"):
        """
        Initializes the SECDataFetcher.

        Args:
            storage_dir (str): The directory where SEC filings will be downloaded.
        """
        self.downloader = Downloader(
            email_address="jjbigdub@gmail.com", # Replace with your actual email
            company_name="FAC-IITK",            # Replace with your actual company name
            download_folder=storage_dir
        )
        self.storage_dir = storage_dir

    def fetch_filings(self, cik: str, form_type: str = "10-Q") -> list:
        """
        Fetches SEC filings for a given CIK and form type.

        Args:
            cik (str): The Central Index Key (CIK) of the company.
            form_type (str): The type of SEC filing (e.g., "10-Q", "10-K").

        Returns:
            list: A list of file paths to the downloaded filings.
        """
        print(f"Fetching {form_type} filings for CIK: {cik}...")
        self.downloader.get(form_type, cik, limit=1) # Downloads the latest filing
        
        # Construct the path to the downloaded filing
        filings_found = []
        parent_folder = os.path.join(self.storage_dir, "sec-edgar-filings", cik, form_type)
        if os.path.exists(parent_folder):
            for root, dirs, files in os.walk(parent_folder):
                for file in files:
                    if file.endswith(".txt"): # Assuming filings are in .txt format
                        filings_found.append(os.path.join(root, file))
        return filings_found

##################################################################################################################################################
# 2. Extraction Agent: Preprocess the extracted data, can make multiple preprocessing functions depending on the structure of the fetched data
#################################################################################################################################################
class TabularDataAgent(Tool):
    """
    An agent (Tool) designed to extract relevant numerical financial data from
    unstructured SEC 10-Q filings using an LLM.
    """
    name = "tabular_data_extractor"
    description = (
        "Extracts and intelligently identifies relevant numerical financial data from an SEC 10-Q filing. "
        "If multiple distinct tabular datasets exist within the document, output them as a JSON array; "
        "each element must be an object with two keys: 'table' (a Markdown formatted table with two columns, "
        "'Financial Metric' and 'Value') and 'context' (detailed excerpts or explanation of where the numbers were found). "
        "Do not include any commentary outside of the JSON object."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "The file path of the SEC 10-Q filing document."
        }
    }
    output_type = "string"

    def __init__(self, model: HfApiModel, **kwargs):
        """
        Initializes the TabularDataAgent with an LLM model.

        Args:
            model (HfApiModel): The Hugging Face API model to use for extraction.
        """
        super().__init__(**kwargs)
        self.model = model

    def forward(self, file_path: str) -> str:
        """
        Reads the content of a file and uses the LLM to extract tabular financial data.

        Args:
            file_path (str): The path to the SEC 10-Q filing document.

        Returns:
            str: A JSON string containing the extracted tables and their contexts.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return json.dumps([{"table": "", "context": f"Error reading file: {e}"}])
        
        prompt = (
            "You are an expert financial analyst and data scientist. "
            "You are given the full text of an unstructured SEC 10-Q filing that may have numerical values scattered randomly. "
            "Your task is to intelligently identify which numerical values are relevant to the financial statements, "
            "capture the context (e.g., surrounding sentences or paragraphs) for each value, and output the results as a JSON array. "
            "Each element of the array must be an object with exactly two keys: 'table' and 'context'. "
            "The 'table' must be a Markdown formatted table with two columns: 'Financial Metric' and 'Value'. "
            "If only one dataset is found, output it as an array with a single object. Do not include any extra commentary.\n\n"
            "SEC 10-Q Filing Content:\n"
            f"{content}\n"
        )
        prompt_message = {"role": "user", "content": prompt}
        
        # Call the LLM to get the extraction result
        raw_result = self.model([prompt_message])
        
        # The raw_result from HfApiModel is a ChatMessage object,
        # we need to extract the content.
        return raw_result.content if hasattr(raw_result, 'content') else str(raw_result)

#############################################################
# 3. Evaluation Agent for Tabular Data Format
############################################################
class EvaluateAgent(Tool):
    """
    An agent (Tool) to validate and correct the JSON format of extracted tabular data.
    Ensures the output conforms to the expected structure for downstream processing.
    """
    name = "evaluate_tabular_data"
    description = (
        "Evaluates and corrects the format of a JSON output from a tabular data extraction. "
        "Ensure that the output is a JSON array of objects, where each object has exactly two keys: "
        "'table' and 'context'. Return a corrected JSON string if necessary."
    )
    inputs = {
        "extraction_output": {
            "type": "string",
            "description": "The JSON string output from the extraction tool."
        }
    }
    output_type = "string"

    def __init__(self, model: HfApiModel, **kwargs):
        """
        Initializes the EvaluateAgent with an LLM model.

        Args:
            model (HfApiModel): The Hugging Face API model to use for evaluation.
        """
        super().__init__(**kwargs)
        self.model = model

    def forward(self, extraction_output: str) -> str:
        """
        Validates and potentially corrects the format of the input JSON string.

        Args:
            extraction_output (str): The JSON string to be evaluated.

        Returns:
            str: The validated or corrected JSON string.
        """
        prompt = (
            "You are an expert in data formatting. Validate the following JSON data to ensure that it is a JSON array "
            "of objects, where each object has exactly two keys: 'table' and 'context'. If it is not correctly formatted, "
            "return a corrected JSON string. Otherwise, return the input unchanged.\n\n"
            f"Input JSON:\n{extraction_output}\n"
        )
        prompt_message = {"role": "user", "content": prompt}
        
        # Corrected: Call the LLM using self.model
        result = self.model([prompt_message])
        
        # The raw_result from HfApiModel is a ChatMessage object,
        # we need to extract the content.
        result_content = result.content if hasattr(result, 'content') else str(result)

        try:
            # Attempt to parse the result to confirm it's valid JSON
            json.loads(result_content)
            return result_content
        except json.JSONDecodeError:
            # If parsing fails, return the original extraction output
            # or handle the error more gracefully (e.g., log it)
            print(f"Warning: Evaluation agent returned invalid JSON. Original output: {extraction_output}")
            return extraction_output
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}. Original output: {extraction_output}")
            return extraction_output

#############################################################
# 4. Query Tool to Search the Vector Database
############################################################
class QueryVectorDBTool(Tool):
    """
    A tool to query a FAISS vector database and retrieve relevant documents
    based on a natural language query.
    """
    name = "query_vector_db"
    description = (
        "Queries the vector database to retrieve stored financial data. "
        "Input a natural language query and return the most relevant documents, each containing a Markdown table and detailed context."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "A natural language query to search the vector database."
        }
    }
    output_type = "string"

    def __init__(self, vectordb: FAISS, **kwargs):
        """
        Initializes the QueryVectorDBTool with a FAISS vector database instance.

        Args:
            vectordb (FAISS): The FAISS vector database to query.
        """
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        """
        Performs a similarity search on the vector database.

        Args:
            query (str): The natural language query.

        Returns:
            str: A formatted string of the most relevant documents found.
        """
        docs = self.vectordb.similarity_search(query, k=5) # Retrieve top 5 similar documents
        results = []
        for i, doc in enumerate(docs):
            results.append(
                f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}):\n{doc.page_content}\n"
            )
        return "\n".join(results)

# --- Main Execution Flow ---
if __name__ == '__main__':
    # Initialize data fetcher
    sec_fetcher = SECDataFetcher(storage_dir="./sec_filings")

    # Define CIK for Apple Inc. (example)
    cik = "0000320193"
    
    # Fetch SEC filings
    filings = sec_fetcher.fetch_filings(cik=cik, form_type="10-Q")

    if not filings:
        print(f"No 10-Q filings found for CIK {cik} in {sec_fetcher.storage_dir}. Exiting.")
    else:
        print(f"Found {len(filings)} filing(s) for CIK {cik}.")

        # Initialize the Hugging Face API model
        # Ensure your HF_TOKEN is correctly set for models requiring authentication/subscription
        model = HfApiModel("meta-llama/Llama-3.1-70B-Instruct", hf_token=HF_TOKEN)

        # Initialize extraction and evaluation tools
        extraction_tool = TabularDataAgent(model=model)
        evaluation_tool = EvaluateAgent(model=model)

        documents = []
        for filing in filings:
            print(f"\nProcessing filing: {filing}")
            # Perform extraction
            extraction_result = extraction_tool.forward(filing)
            
            # Evaluate and correct the extraction format
            evaluated_result = evaluation_tool.forward(extraction_result)
            
            try:
                # Attempt to parse the evaluated result
                extraction_data = json.loads(evaluated_result)
                if not isinstance(extraction_data, list):
                    # Ensure it's a list, even if only one object is returned
                    extraction_data = [extraction_data]
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from evaluation for {filing}: {e}. Raw result: {evaluated_result}")
                # Fallback: store the raw result if JSON parsing fails
                extraction_data = [{"table": "Parsing Error", "context": f"Failed to parse JSON: {e}. Raw content: {evaluated_result}"}]
            except Exception as e:
                print(f"An unexpected error occurred during parsing for {filing}: {e}. Raw result: {evaluated_result}")
                extraction_data = [{"table": "Error", "context": f"Unexpected error: {e}. Raw content: {evaluated_result}"}]

            for dataset in extraction_data:
                table = dataset.get("table", "N/A")
                context_detail = dataset.get("context", "N/A")
                combined_content = f"Table:\n{table}\n\nDetailed Context:\n{context_detail}"
                
                # Create a Document object for Langchain
                doc = Document(page_content=combined_content, metadata={"source": filing, "cik": cik})
                documents.append(doc)
                print(f"Stored dataset from filing: {filing}\n{'-'*40}\n{combined_content}\n{'='*40}\n")

        if not documents:
            print("No documents were successfully processed and stored. Vector database will be empty.")
        else:
            # Initialize embedding model
            embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

            # Create FAISS vector database from documents
            print("Creating FAISS vector database...")
            vectordb = FAISS.from_documents(
                documents=documents,
                embedding=embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            print("Vector database created.")

            # Initialize query tool and agent
            query_tool = QueryVectorDBTool(vectordb=vectordb)
            agent = ToolCallingAgent(tools=[query_tool], model=model)

            # Example query
            query = "Show me financial metrics related to revenue and net income."
            print(f"\nRunning query: '{query}'")
            query_result = agent.run(query)
            print("Query Result:\n", query_result)

