# Retrieval-Augmented-Generation-RAG-using-langchain-Streamlit-and-Mistral-7b-Instruct-
Develop a lightweight, real-time Retrieval-Augmented Generation (RAG) system that allows
users to upload multiple unrelated multi-page PDF documents and extract relevant highly
information based on a query.
The System must ensure that:
. Each document is treated as an independent knowledge source - responses should be per
document, with no cross-document contamination.
. The extracted responses are contextualized, showing where in the document the answer
is found.
. The System works across structured and unstructured PDFs, handling invoices, reports,
legal papers and contracts.
Aiden Al
. It efficiently processes large PDFs, prioritizing important sections rather than blingh
scanning all pages.
. Implement fast indexing or caching for already-uploaded PDFs so that subsequent
searches don't reprocess everything from scratch.
Expected Output:
. Evaluation based on retrieval accuracy, response quality, and efficiency.
. A QA system that retrieves the most relevant and accurate information while flagging
contradictory data
Brownie Points:
. Summarization of Extracted Data, i.e. Instead of pulling raw values, the system can
generate a short, human-readable summary of extracted data.
. Confidence Score, i.e. The system should provide confidence scores for extracted
results based on text clarity, and contextual relevance.
