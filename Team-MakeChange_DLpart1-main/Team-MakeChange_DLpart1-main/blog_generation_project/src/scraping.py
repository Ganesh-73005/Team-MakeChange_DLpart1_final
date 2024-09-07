from bs4 import BeautifulSoup
import requests
import csv
import os
import re
for count in range(4000):
    
    url = 'https://www.wikihow.com/Special:Randomizer'
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    article_title = soup.find('title').text.strip()
    print(article_title+" "+str(count))
    subheadings = []
    paragraphs = []
    steps = soup.find_all('div', { 'class': 'step'})
    for step in steps:
        subheading_element = step.find('b')
        if(subheading_element is not None):
            subheading_text = subheading_element.text.strip().replace('\n','')
            subheading_text = subheading_text.encode('ascii', errors='ignore').decode()
            subheading_text = re.sub(r'', '', subheading_text)
            subheadings.append(subheading_text)
            subheading_element.extract()
        for span_tag in step.find_all('span'):
            span_tag.extract()
        paragraph_text = step.text.strip().replace('\n','').replace('','')
        paragraph_text = paragraph_text.encode('ascii', errors='ignore').decode()
        paragraph_text = re.sub(r'','', paragraph_text)
        paragraphs.append(paragraph_text)
    file_path = './working/wikiHow.csv'
    file_exists = os.path.exists(file_path)
    if len(subheadings):
       os.makedirs(os.path.dirname(file_path), exist_ok=True)
       with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(['Article Title', 'Subheading', 'Paragraph'])
        
        for i in range(len(subheadings)):
            writer.writerow([article_title, subheadings[i], paragraphs[i]])