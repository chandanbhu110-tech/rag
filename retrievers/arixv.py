import arxiv
from langchain_core.documents import Document

client = arxiv.Client()

search = arxiv.Search(
    query="large language models",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

docs = []

for result in client.results(search):
    docs.append(
        Document(
            page_content=result.summary,
            metadata={
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "url": result.entry_id
            }
        )
    )

for i, doc in enumerate(docs):

    print(f"\nResult {i + 1}")
    print("Title:", doc.metadata["title"])
    print("Authors:", doc.metadata["authors"])
    print("Summary:", doc.page_content[:500])