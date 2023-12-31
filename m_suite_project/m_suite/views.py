from django.shortcuts import render, redirect
import pandas as pd
from django.http import HttpResponse
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from django.http import JsonResponse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from django.http import FileResponse
import csv
from django.shortcuts import render, redirect
from django.contrib import messages
from io import StringIO
from io import StringIO
from django.http import HttpResponse
from .models import keyword_count_data


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse

# def index(request):
#     return render(request,"index.html",{})

def website_keyword(request):
    return render(request,"website_keyword.html",{})
 
extracted_links = []
    
def extract_links(request):
    output_url = "" 
    
    if request.method == "POST":
        output_url = request.POST.get('title[0]')
        additional_urls = [request.POST.get(f'title[{i}]') for i in range(1, 10)]
        all_inputs = [output_url] + additional_urls

    def extractlinks(url):
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            response.raise_for_status()
            


            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all anchor tags (<a>) that contain 'href' attribute
            links = soup.find_all('a', href=True)

            # Extract and normalize the URLs related to the input URL domain
            extracted_links = set()
            base_url = urlparse(url).scheme + '://' + urlparse(url).netloc  # Get base URL
            for link in links:
                href = link.get('href')
                normalized_url = urljoin(base_url, href)  # Normalize the URL
                # Check if the normalized URL belongs to the same domain as the input URL
                if urlparse(normalized_url).netloc == urlparse(url).netloc:
                    extracted_links.add(normalized_url)

            return list(extracted_links)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []
    
    all_urls = []
    for i in all_inputs:
        all_urls.extend(extractlinks(i))
    print(all_urls)

    
    
    
    def extract_keywords(url):
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            
            response.raise_for_status()

            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content from the page
            text = soup.get_text()

            # Tokenize the text by splitting it into words
            words = re.findall(r'\w+', text.lower())

            # Remove stopwords, one-letter, one-digit words, prepositions, and all numbers
            filtered_words = [word for word in words if word not in stopwords.words("english") and len(word) > 1 and not word.isdigit() and word not in 
                            ["a", "an", "the", "in", "on", "at", "to", "us", "day", "back", "contact", "cookies","cookie","help","menu"]]

            # Create a Counter to count word frequencies
            word_counter = Counter(filtered_words)

            return word_counter

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return Counter()
    keyword_data = []
    for url in all_urls:
        print(f"Loading URL: {url}")
        word_counter = extract_keywords(url)

        # Sort keywords by count in descending order
        sorted_keywords = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

        # Get the top 20 keywords with counts
        top_keywords = sorted_keywords[:30]

        # Append data to the keyword_data list
        for keyword, count in top_keywords:
            keyword_data.append([url, keyword, count])

    # Create a DataFrame for keywords
    keyword_df = pd.DataFrame(keyword_data, columns=["URL", "Keyword", "Count"])
    
    keyword_df = keyword_df.groupby('Keyword').agg({'Count': 'sum','URL': lambda x: x.mode().iloc[0] if not x.mode().empty else None
})
    keyword_df['Count'] = keyword_df['Count'].astype('int')
    # Reset the index and sort by 'Count'
    keyword_df = keyword_df.reset_index().sort_values(by='Count',ascending =False)
    request.session['keyword_df'] = keyword_df.to_dict(orient='records')

    
    
    print(keyword_df)
    
    keyword_list = keyword_df.iloc[0:9,:].to_dict(orient='records')
    keyword_list_bar = keyword_df.to_dict(orient='records')


    def generate_wordcloud_image(keyword_df):
        # Create a dictionary from the DataFrame for WordCloud input
        word_dict = dict(zip(keyword_df['Keyword'], keyword_df['Count']))
        # Check if the word_dict is empty
        if not word_dict:
            # Handle the case where there are no words
            print("No words to plot in the word cloud.")
            return None
        
        meta_mask = np.array(Image.open('assets/images/globe.png'))
       # meta_mask = np.array(Image.open('E:/enfection/internal_product/M_suite/M_Suite_p/assets/images/meta.png'))

        # Generate the WordCloudv 
        #wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)
        wordcloud = WordCloud(background_color = 'white',margin=10 , mask=meta_mask, contour_width = 2, colormap = 'BuPu_r',contour_color = 'white').generate_from_frequencies(word_dict)

        # Save the WordCloud image to a BytesIO object
        image_stream = BytesIO()
        wordcloud.to_image().save(image_stream, format='PNG')
        image_stream.seek(0)

        # Encode the image in base64
        image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
        return image_base64
    
  
    
    

    # Generate the WordCloud image and get the base64 encoding
    wordcloud_image = generate_wordcloud_image(keyword_df)

    

    # Store keyword_df in the session
    for index, row in keyword_df.iterrows():
        keyword_data_instance, created = keyword_count_data.objects.get_or_create(Keyword=row['Keyword'], defaults={'Count': row['Count'], 'Url': row['URL']})
        keyword_data_instance.Count = row['Count']
        keyword_data_instance.Url = row['URL']
        keyword_data_instance.save()
    # Optionally, you might want to save the instance to the database
    # keyword_data_instance.save()

    #KeywordCountData.objects.bulk_create([KeywordCountData(**data) for data in keyword_df])
    #keyword_data_instance = KeywordCountData.objects.create(Keyword=keyword_df['Keyword'], Count=keyword_df['Count'])

    return render(request,"website_keyword.html",{'given_url': output_url, 'keyword_list': keyword_list, 'keyword_list_bar':keyword_list,
                                        'wordcloud_image': wordcloud_image, 'keyword_df': keyword_df}
    )    
    #return render(request,"index.html",{'given_url':output_url,'keyword_list': keyword_list})
    #return None

from django.http import HttpResponse
import base64





def download_dataset(request):
    # Retrieve keyword_df from the session
    keyword_df = request.session.get('keyword_df', [])

    # Check if keyword_df is empty
    if not keyword_df:
        messages.error(request, 'No data to download. Please perform the extraction first.')
        return redirect('website_keyword')  # Redirect to the index view

    # Convert the data back to a DataFrame
    keyword_df = pd.DataFrame(keyword_df)

    # Create an in-memory CSV file
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)

    # Write the header
    writer.writerow(['Keyword', 'Count'])

    # Write the data
    for index, row in keyword_df.iterrows():
        writer.writerow([row['Keyword'], row['Count']])

    # Create an HttpResponse and set the headers
    response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="dataset.csv"'

    return response

def overview(request):
    return render(request,"overview.html",{})

def sentiment_analysis(request):
    return render(request,"sentiment_analysis.html",{})

def brand_authority(request):
    return render(request,"brand_authority.html",{})


def brand_personality(request):
    return render(request,"brand_personality.html",{})


def profile_analyzer(request):
    return render(request,"profile_analyzer.html",{})


def website_keyword(request):
    return render(request,"website_keyword.html",{})


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

def proceed_yt_url(request):
    # Set Chrome options to disable notifications
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2
    })

    # Initialize the Chrome WebDriver with the configured options
    driver = webdriver.Chrome(options=chrome_options)

    yt_url = "" 
    
    if request.method == "POST":
        yt_url = request.POST.get('yt_url')
        
        urls = [yt_url]
        print(urls)
        
        for url in urls:
            # Open the YouTube channel URL
            driver.get(url)

            # Get the channel name
            channel_name_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="channel-name"]')))
            channel_name = channel_name_element.text
        
            pro_pic = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, ' //*[@id="img"]')))
            image = pro_pic.get_attribute("src")
            details = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, ' //*[@id="content"]')))
            details.click()
        
            time.sleep(3)
            
            about = driver.find_elements(By.XPATH, '//*[@id="contents"]')
            text = about[-1].text

            lines = text.strip().split('\n')
            join_date_line = lines[-3].strip() if lines else "Joined date not found"
            join_date = join_date_line.replace("Joined ", "") if "Joined" in join_date_line else "Join date not found."

            # Assuming 'text' contains the information about subscribers
            pattern2_1 = r"(\d+)K subscribers"
            pattern2_2 = r"(\d+)M subscribers"

            
            # Extract the number of videos
            pattern3 = r"(\d+) videos"
            match = re.search(pattern3, text)
            num_videos = int(match.group(1)) if match else "Number of videos not found."

            
            # Extract the number of views
            pattern4 = r"([\d,.]+) views"
            match_views = re.search(pattern4, text)
            num_views = int(match_views.group(1).replace(",", "")) if match_views else "Number of views not found."


            
            # Search for the pattern in the text
            lines = text.strip().split('\n')
            country = lines[-2].strip() if lines else "Country not found"



            pattern2_1 = r"([\d.]+)K subscribers"
            pattern2_2 = r"([\d.]+)M subscribers"
            
            combined_pattern = f"{pattern2_1}|{pattern2_2}"
            match = re.search(combined_pattern, text)
            
            if match:
                if 'K' in match.group(0):
                    subscribers_str = match.group(1)
                    subscribers = int(float(subscribers_str) * 1000)
                elif 'M' in match.group(0):
                    subscribers_str = match.group(2)
                    subscribers = int(float(subscribers_str) * 1000000)
            else:
                subscribers = None

            avg_n_v = int(num_views/num_videos)
            
            # # Print or store the extracted information
            # print("Channel Name:", channel_name)
            # print("Join Date:", join_date)
            # print("Number of Subscribers:", subscribers)
            # print("Number of Videos:", num_videos)
            # print("Number of Views:", num_views)
            # print("Country:", country)
            # #print("Text:", text)
            # print("-" * 50)
       
        
    return render(request,"sentiment_analysis.html",
                  {'image':image,
                   'yt_url':yt_url,
                   'channel_name':channel_name,
                   'join_date':join_date,
                   'subscribers':subscribers,
                   'num_videos':num_videos,
                   'num_views':num_views,
                   'avg_n_v':avg_n_v,
                   'country':country})

    
