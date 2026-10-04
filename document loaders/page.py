from langchain_community.document_loaders import WebBaseLoader

url="https://www.flipkart.com/motorola-edge-70-fusion-pantone-country-air-512-gb/p/itmde5ddcbb7c261?pid=MOBHNSA6ZRCZPWZE&param=3882&BU=Mobile&pageUID=1790398342684"

data=WebBaseLoader(url)

docs=data.load()

print(docs[0].page_content)