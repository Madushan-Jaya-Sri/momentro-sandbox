# Django-related imports
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Image processing
import base64
from io import BytesIO
from PIL import Image

# Web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# Natural Language Processing
import re
from nltk.corpus import stopwords
from collections import Counter

# Data visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# CSV handling
import csv
from io import StringIO

# Selenium for web automation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException ,StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException,ElementNotInteractableException
from selenium.webdriver.common.keys import Keys

# Miscellaneous
import string
import pickle
import emoji
import time


from .models import keyword_count_data,youtube_comments

# def index(request):
#     return render(request,"index.html",{})

def website_keyword(request):
    return render(request,"website_keyword.html",{})
 
extracted_links = []
    
    
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




def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def convert_emojis_to_words(text): 
    # Convert emojis to words using emoji.demojize
    text_with_emojis_as_words = emoji.demojize(text, delimiters=(' ', ' '))

    return text_with_emojis_as_words


with open('./savedModels/model/model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('./savedModels/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()
vocab = pd.read_csv('./savedModels/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()



from nltk.stem import PorterStemmer
ps = PorterStemmer()

def preprocessing(sentences):
    preprocessed_sentences = []

    for text in sentences:
        data = pd.DataFrame([text], columns=['Full_text'])
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        data["Full_text"] = data['Full_text'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
        data['Full_text'] = data['Full_text'].apply(convert_emojis_to_words)
        data["Full_text"] = data["Full_text"].apply(remove_punctuations)
        data["Full_text"] = data['Full_text'].str.replace('\d+', '', regex=True)
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
        preprocessed_sentences.append(data["Full_text"].iloc[0])

    return preprocessed_sentences
def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    
    return vectorized_lst_new
# def get_prediction(vectorized_text):
#     vectorized_text = vectorized_text.reshape(1, -1)
#     prediction = model.predict(vectorized_text)
#     if prediction == 1:
#         return 'positive'
#     else:
#         return 'negative'


# Define the thresholds for categorization
negative_threshold = 0.4
positive_threshold = 0.7

# Categorize the results
def categorize(probability):
    if probability < negative_threshold:
        return 'negative'
    elif negative_threshold <= probability < positive_threshold:
        return 'neutral'
    else:
        return 'positive'
    
def get_prediction(vectorized_text):
    vectorized_text = vectorized_text.reshape(1, -1)
    prediction_score = model.predict_proba(vectorized_text)
    return prediction_score


# sentences = ['amazing product', 'not expected with the quality', 'great experience']
# preprocessed_sentences = preprocessing(sentences)

# vectorized_sentences = vectorizer(preprocessed_sentences, tokens)

# predictions = [get_prediction(vectorized_sentence) for vectorized_sentence in vectorized_sentences]

# # Create a DataFrame
# output_df = pd.DataFrame({'Sentence': sentences, 'Sentiment': predictions})

# # Print the DataFrame
# print(output_df)



def proceed_yt_url(request):
    
    channel_names = []
    video_titles = []
    
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
       
       
        for url in urls:
            # Open the YouTube channel URL
            driver.get(url)

            # Get the channel name
            channel_name_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="channel-name"]')))
            channel_name = channel_name_element.text

            video = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="tabsContent"]/yt-tab-group-shape/div[1]/yt-tab-shape[2]/div[1]')))
            video.click()

            
            
            # Scroll down the page multiple times (adjust the count as needed)
            for _ in range(100):
                driver.execute_script("window.scrollBy(0, 500)")
                time.sleep(0.2)
                

            
            # Scrape video titles
            titles = driver.find_elements(By.XPATH, '//*[@id="video-title"]')
            for title in titles:
                channel_names.append(channel_name)
                video_titles.append(title.text)

        # Create a DataFrame from the scraped data
        data = {'Channel Name': channel_names, 'Video_Title': video_titles}
        df = pd.DataFrame(data)
        
        df_yt_titles = df.iloc[0:5,1:].to_dict(orient='records')



        # Create a DataFrame to store the data
        data = []


        # Loop through the videos and collect comments

        for video in titles:
            try:
                #video_title = video.text
                action = ActionChains(driver)
                action.move_to_element(video).click().perform()
            
                time.sleep(1)
                
                
                
                for _ in range(2):
                    # Scroll down by a fixed amount (e.g., 500 pixels)
                    driver.execute_script("window.scrollBy(0, 500);")
                    time.sleep(3)
                # driver.execute_script("window.scrollBy(0, 1000);")
                # time.sleep(4)
                
                comment_count = driver.find_element(By.XPATH,'/html/body/ytd-app/div[1]/ytd-page-manager/ytd-watch-flexy/div[5]/div[1]/div/div[2]/ytd-comments/ytd-item-section-renderer/div[1]/ytd-comments-header-renderer/div[1]/h2/yt-formatted-string/span[1]')
                string_with_comma = comment_count.text

                # Check if the string contains a comma
                if ',' in string_with_comma:
                    string_without_comma = string_with_comma.replace(',', '')
                    no_comments = int(string_without_comma)
                else:
                    no_comments = int(string_with_comma)
            
                if no_comments == 0:
                     for _ in range(1):
                        # Scroll down by a fixed amount (e.g., 500 pixels)
                        driver.execute_script("window.scrollBy(0, 500);")
                        time.sleep(0.5)
                    
                
                else:
                
                    for _ in range(int(no_comments/6)):
                        # Scroll down by a fixed amount (e.g., 500 pixels)
                        driver.execute_script("window.scrollBy(0, 500);")
                        time.sleep(0.5)
                    
                    
                    comments = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'style-scope ytd-expander')))
            
                    #comments = driver.find_elements(By.CLASS_NAME, 'style-scope ytd-expander')
                    emojis = driver.find_elements(By.XPATH, '//*[@id="content-text"]/img')
                    
            
                    for emoji_img in emojis:
                        emoji = emoji_img.get_attribute('alt')
                        driver.execute_script("arguments[0].innerHTML = arguments[1];", emoji_img, emoji)
                        text_with_emojis = driver.find_element(By.XPATH, '//*[@id="content-text"]/img').text
                        
                    # Process comments and emojis (if needed)
                    video_comments = []
                    for comment in comments:
                        video_comments.append(comment.text)
            
                    # Store comments in your data structure (e.g., a list)
                    for comment in video_comments:
                        data.append([channel_name, comment])

                
                #time.sleep(3)


        
            
            except ElementClickInterceptedException:
                    print(f"ElementClickInterceptedException for video: {'video_title'}. Skipping...")
                    
            except StaleElementReferenceException:
                    print(f"ElementClickInterceptedException for video: {'video_title'}. Skipping...")

            except TimeoutException:
                print("")

            except NoSuchElementException:
                print("")
                
            except ElementNotInteractableException:
                print("")
                
            except NoSuchWindowException:
                print("")
                
                
            # Go back to the channel page
            driver.back()

            
                




        #driver.close()

    df_comments = pd.DataFrame(data, columns=['Channel_Name', 'Comments'])
   
    request.session['df_comments'] = df_comments.to_dict(orient='records')
        # Create a DataFrame from the collected data
    
    df_yt_comments = df_comments.iloc[0:5,:].to_dict(orient='records')

    # Store keyword_df in the session
    for index, row in df_comments.iterrows():
        youtube_comments_instance, created = youtube_comments.objects.get_or_create(Channel_Name=row['Channel_Name'], Comments=row['Comments'])

        youtube_comments_instance.save()


    # # Create instances for each comment and save them
    # for index, row in df_comments.iterrows():
    #     youtube_comments_instance = youtube_comments(Channel_Name=row['Channel_Name'], Comments=row['Comments'])
    #     youtube_comments_instance.save()
   
    df_yt_comments = df_comments.iloc[0:5,:].to_dict(orient='records')
    # Print the DataFrame
    print(df_comments)
    
    
    






    def sentiment_model(df_comments):
        # Retrieve df_comments from the session
        rows = []
        rows_2 = []
        
        sentences = df_comments['Comments']
        
        for sentence in sentences:
            if sentence.strip():  # Check if the sentence is not empty or contains only whitespace
                preprocessed_sentence = preprocessing([sentence])
                #logging.info(f'Preprocessed Text : {preprocessed_sentence}')

                vectorized_sentence = vectorizer(preprocessed_sentence, tokens)
                #logging.info(f'Vectorized Text : {vectorized_sentence}')

                prediction_scores = get_prediction(vectorized_sentence)

                prediction = categorize(prediction_scores[0, 1])
                # Get the top 5 maximum numbers

                
                #logging.info(f'Prediction : {prediction}')
    
                rows.append({'Sentence': sentence, 'Sentiment': prediction})
                rows_2.append({'Sentence': sentence, 'prediction_scores': prediction_scores})

 
        output_df = pd.DataFrame(rows)
        output_df_2 = pd.DataFrame(rows_2)
        
        print(output_df_2)
        output_df_2['score'] = output_df_2['prediction_scores'].apply(lambda x: x[0][1])
        output_df_2 = output_df_2.drop('prediction_scores', axis=1)


        print(output_df_2)
        df_sorted_p = output_df_2.sort_values(by='score', ascending=False)
        filtered_df = df_sorted_p[df_sorted_p['score'] > positive_threshold]
        
        top_positive_comments=[]
        top_negative_comments=[]
        
        if not filtered_df.empty:
            top_positive_comments = filtered_df['Sentence'].head(5)
            print(top_positive_comments)
        else:
            print("No positive comments with score greater than 0.7 found.")
        
        df_sorted_n= output_df_2.sort_values(by='score', ascending=True)
        
        filtered_df = df_sorted_n[df_sorted_n['score'] < negative_threshold]
        
        if not filtered_df.empty:
            top_negative_comments = filtered_df['Sentence'].head(5)
            print(top_negative_comments)
        else:
            print("No negative comments with score less than 0.4 found.")
        
        
        return output_df,top_positive_comments,top_negative_comments

    # keyword_df = request.session.get('keyword_df', [])
    
    df_comments_yt,top_positive_comments,top_negative_comments = sentiment_model(df_comments)
    
    df_comments_yt = df_comments_yt['Sentiment'].value_counts().reset_index()
    
    print(df_comments_yt)
    # Assuming your dataframe is named df_comments_yt_counts
    # Initialize counts
    p_count, neg_count, neu_count = 0, 0, 0
    
    sentiments = ['positive', 'negative', 'neutral']
    # Loop through sentiments and update counts
    for sentiment in sentiments:
        if sentiment in df_comments_yt['Sentiment'].values:
            count = df_comments_yt.loc[df_comments_yt['Sentiment'] == sentiment, 'count'].iloc[0]
            if sentiment == 'positive':
                p_count = count
            elif sentiment == 'negative':
                neg_count = count
            elif sentiment == 'neutral':
                neu_count = count


    # If you want to convert the counts to integers, you can do that:
    p_count = int(p_count)
    neg_count = int(neg_count)
    neu_count = int(neu_count)

    

    print(p_count)
    print(neg_count)
    print(neu_count)

    top_positive_comments_df = pd.DataFrame(top_positive_comments)
    top_negative_comments_df = pd.DataFrame(top_negative_comments) 
    
    # Assuming you have a variable named top_negative_comments
    # Assuming you have a variable named top_negative_comments
    print(top_positive_comments_df.reset_index(drop=True, inplace=True))

    # Initialize variables
    tp1 = ''
    tp2 = ''
    tp3 = ''
    tp4 = ''
    tp5 = ''

    # Checking if the DataFrame is not empty before proceeding
    if not top_positive_comments_df.empty:
        # Assigning values to variables if not empty
        if len(top_positive_comments_df['Sentence']) > 0:
            tp1 = top_positive_comments_df['Sentence'][0]
        if len(top_positive_comments_df['Sentence']) > 1:
            tp2 = top_positive_comments_df['Sentence'][1]
        if len(top_positive_comments_df['Sentence']) > 2:
            tp3 = top_positive_comments_df['Sentence'][2]
        if len(top_positive_comments_df['Sentence']) > 3:
            tp4 = top_positive_comments_df['Sentence'][3]
        if len(top_positive_comments_df['Sentence']) > 4:
            tp5 = top_positive_comments_df['Sentence'][4]

        # Printing the assigned variables
        print(tp1)
        print(tp2)
        print(tp3)
        print(tp4)
        print(tp5)
    else:
        print("top_positive_comments_df is empty. Doing nothing.")

    # Resetting the index and modifying the DataFrame in-place
    top_negative_comments_df.reset_index(drop=True, inplace=True)

    # Initialize variables
    tn1 = ''
    tn2 = ''
    tn3 = ''
    tn4 = ''
    tn5 = ''

    # Checking if the DataFrame is not empty before proceeding
    if not top_negative_comments_df.empty:
        # Assigning values to variables if not empty
        if len(top_negative_comments_df['Sentence']) > 0:
            tn1 = top_negative_comments_df['Sentence'][0]
        if len(top_negative_comments_df['Sentence']) > 1:
            tn2 = top_negative_comments_df['Sentence'][1]
        if len(top_negative_comments_df['Sentence']) > 2:
            tn3 = top_negative_comments_df['Sentence'][2]
        if len(top_negative_comments_df['Sentence']) > 3:
            tn4 = top_negative_comments_df['Sentence'][3]
        if len(top_negative_comments_df['Sentence']) > 4:
            tn5 = top_negative_comments_df['Sentence'][4]

        # Printing the assigned variables
        print(tn1)
        print(tn2)
        print(tn3)
        print(tn4)
        print(tn5)
    else:
        print("top_negative_comments_df is empty. Doing nothing.")

        
    
    return render(request,"sentiment_analysis.html",
                  {'image':image,
                   'df_yt_titles':df_yt_titles,
                   'yt_url':yt_url,
                   'df_yt_comments':df_yt_comments,
                   'channel_name':channel_name,
                   'join_date':join_date,
                   'subscribers':subscribers,
                   'num_videos':num_videos,
                   'num_views':num_views,
                   'avg_n_v':avg_n_v,
                   'country':country,
                   'p_count':p_count,
                   'neg_count':neg_count,
                   'neu_count':neu_count,
                   'tp1':tp1,'tp2':tp2,'tp3':tp3,'tp4':tp4,'tp5':tp5,
                   'tn1':tn1,'tn2':tn2,'tn3':tn3,'tn4':tn4,'tn5':tn5,                
                   })

    


import pandas as pd
from io import BytesIO

def download_yt_comments(request):
    # Retrieve df_comments from the session
    df_comments = request.session.get('df_comments', [])

    # Check if df_comments is empty
    if not df_comments:
        messages.error(request, 'No data to download. Please perform the extraction first.')
        return redirect('website_keyword')  # Redirect to the index view

    # Convert the data back to a DataFrame
    df_comments = pd.DataFrame(df_comments)

    # Create an in-memory Excel file
    excel_buffer = BytesIO()

    # Write the DataFrame to an Excel file
    df_comments.to_excel(excel_buffer, index=False, sheet_name='YT_Comments')

    # Create an HttpResponse and set the headers
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=yt_comments.xlsx'

    # Write the Excel file content to the response
    excel_buffer.seek(0)
    response.write(excel_buffer.read())

    return response
   
from django.shortcuts import render
from .models import keyword_count_data, youtube_comments

def history(request):
    # Fetch data from the keyword_count_data and youtube_comments tables
    keyword_data = keyword_count_data.objects.all()
    yt_comments_data = youtube_comments.objects.all()

    # Convert querysets to lists or dictionaries based on your requirement
    keyword_data_list = list(keyword_data.values())
    yt_comments_data_list = list(yt_comments_data.values())

    # Pass the data to the template
    return render(request, "history.html", {'keyword_data': keyword_data_list, 'yt_comments_data': yt_comments_data_list})