from bs4 import BeautifulSoup
import requests
# Also requires html5lib packages

def get_text_from_url(url):
    r1 = requests.get(url)
    page_content = r1.content
    soup1 = BeautifulSoup(page_content, 'html5lib')
    page_news = soup1.find_all('div', class_='zn-body__paragraph')
    if len(page_news) !=0:
        news = ""
        for i in range(len(page_news)):
            pieces = page_news[i].get_text()
            news += pieces
    return news