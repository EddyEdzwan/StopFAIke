from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import requests
import urllib.request
import time

def scrape_politifact(start_page, end_page, verbose = True):

    '''Function to scrape Politifact's list of latest fact-checks - https://www.politifact.com/factchecks/list/
    over given pages between start_page and end_page.
    
    Returns a DataFrame of
    (i) the fact-check's author,
    (ii) date,
    (iii) the statement made,
    (iv) source of the statement
    (v) labels (as-is) given by Politifact
    
    Set verbose=False to not print the print statements'''

    authors = []
    dates = []
    statements = []
    sources = []
    targets = []

    for page_number in range(start_page, end_page+1):
    
        page_num = str(page_number)
        URL = 'https://www.politifact.com/factchecks/list/?page='+page_num
        webpage = requests.get(URL)
        
        #time.sleep(3)
        soup = BeautifulSoup(webpage.text, "html.parser") #Parse the text from the website
        
        #Get the tags and it's class
        statement_footer =  soup.find_all('footer',attrs={'class':'m-statement__footer'})  #Get the tag and it's class
        statement_quote = soup.find_all('div', attrs={'class':'m-statement__quote'}) #Get the tag and it's class
        statement_meta = soup.find_all('div', attrs={'class':'m-statement__meta'})#Get the tag and it's class
        target = soup.find_all('div', attrs={'class':'m-statement__meter'}) #Get the tag and it's class
        
        if verbose:
            print(f"#### Scraping page: {page_number} ####")
        
        #loop through the footer class m-statement__footer to get the date and author
        for i in statement_footer:
            link1 = i.text.strip()
            name_and_date = link1.split()
            if len(name_and_date) < 7:
                full_name = np.nan
                date = np.nan
            else: 
                first_name = name_and_date[1]
                last_name = name_and_date[2]
                full_name = first_name + ' ' + last_name
                month = name_and_date[4]
                day = name_and_date[5]
                year = name_and_date[6]
                date = month + ' ' + day + ' ' + year
            dates.append(date)
            authors.append(full_name)
        
        #Loop through the div m-statement__quote to get the link
        for i in statement_quote:
            link2 = i.find_all('a')
            statements.append(link2[0].text.strip())
        
        #Loop through the div m-statement__meta to get the source
        for i in statement_meta:
            link3 = i.find_all('a') #Source
            source_text = link3[0].text.strip()
            sources.append(source_text)
            
        #Loop through the target or the div m-statement__meter to get the facts about the statement (True or False)
        for i in target:
            fact = i.find('div', attrs={'class':'c-image'}).find('img').get('alt')
            targets.append(fact)

    return pd.DataFrame({'author': authors,
    'statement': statements, 'source': sources,
    'date' : dates, 'target': targets})

def scrape_poynter(start_page, end_page, verbose=True):

    '''Function to scrape Poynter's list of latest COVID related fact-checks and its related explanation why it is labelled as such
    - https://www.poynter.org/ifcn-covid-19-misinformation/page/
    over given pages between start_page and end_page.
    
    Returns two DataFrames:
    DataFrame 1
    (i) the fact-check's source,
    (ii) date,
    (iii) the country where the source containing the statement made,
    (iv) the statement,
    (v) labels (as-is) given by Poynter

    DataFrame2
    (i) the corresponding statement's explanation why it is labelled as such 
    
    Set verbose=False to not print the print statements'''
    
    #initialising the empty list of information to collect
    source_list = []
    date_list = []
    country_list = []
    label_list = []
    title_list = []
    true_explanation_list = []

    #Looping over the pages
    for page_num in range(start_page-1, end_page):
        
        URL = 'https://www.poynter.org/ifcn-covid-19-misinformation/page/' + str(page_num+1) #append the page number to complete the URL
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'}
        
        #Add delay to request for explanation page, if required
        # time.sleep(2)

        webpage = requests.get(URL, headers=headers)  #Make a request to the website

        #Print statement for iteration
        if verbose:
            print('-'*50)
            print(f'Extracting information from page number {page_num+1}')
        
        webpage.raise_for_status() #Raise error if page unavailable
        
        soup = BeautifulSoup(webpage.content, "html.parser") #Parse the text from the website

        # Get source, date and country
        source_date_country_list = soup.find_all('p',attrs={'class':'entry-content__text'})

        for id_, item in enumerate(source_date_country_list):
            if id_%2 == 0:
                source_list.append(item.text.replace('Fact-Checked by: ', ''))
            else:
                date_list.append(item.text[:10])
                country_list.append(item.text[13:])

        #Get labels and title
        source_title_list = soup.find_all('h2', attrs={'class':'entry-title'})

        for item in source_title_list:
            label = item.find('span').text
            label_list.append(label.replace(':', ''))

            title = item.text.replace(label, '')
            title_list.append(title.strip())

        #Get link to explanation (for true)
        explanation_pages = soup.find_all('a', attrs={'class':'button entry-content__button entry-content__button--smaller'})

        for id_, item in enumerate(explanation_pages):
            article_url = item['href']

            #Add delay to request for explanation page
#             time.sleep(2)

            article_requests = requests.get(article_url, headers=headers)

            #Add print statement for iteration
            if verbose:
                print(f'Extracting true explanation for item {id_+1} from page number {page_num+1}')

            article_requests.raise_for_status()

            article_soup = BeautifulSoup(article_requests.content, "html.parser") #Parse the text from the website

            true_explanation = article_soup.find('p', attrs={'class':'entry-content__text entry-content__text--explanation'}).text. \
                 replace('Explanation: ', '')

            true_explanation_list.append(true_explanation)

    return pd.DataFrame(dict([('source', source_list), ('date', date_list), ('country', country_list),
      ('label',label_list),('title',title_list)])), pd.DataFrame(dict([('true_explanation',true_explanation_list)]))
