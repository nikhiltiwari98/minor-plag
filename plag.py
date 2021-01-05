from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import re
import math
from requests import get 
import urllib
from bs4 import BeautifulSoup 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('stopwords') if it is first time uncomment it.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/new' )
def new():
    return render_template('index.html')

@app.route('/cosineSimilarity',methods=['POST'])

def cosineSimilarity():
    universalSetOfUniqueWords = []
    matchPercentage = 0

    ####################################################################################################
    inputQuery = request.form.get('plagtext')

    lowercaseQuery = inputQuery.lower()
    queryWordList = re.sub("[^\w]", " ",lowercaseQuery).split()			#Replace punctuation by space and split
	# queryWordList = map(str, queryWordList)					#This was causing divide by zero error
    for word in queryWordList:
        if word not in universalSetOfUniqueWords:
            universalSetOfUniqueWords.append(word)

	####################################################################################################
    database1=request.form.get('orgtext')
    yash=""
    # print(universalSetOfUniqueWords[0:20])
    # print(queryWordList[0:20])
    if(not database1):
            keyword = inputQuery
            url = "https://google.com/search?q="+keyword
            html = get(url).text

            
            soup = BeautifulSoup(html,features="lxml")
            temp="""
            class="kCrYT"
            """            
            links_with_text = []
          
            anurag=[]
            links_with_text=soup.find_all('a')
            for links in links_with_text:
                if 'href' in links.attrs:
                    anurag.append((links.attrs['href']))
                    # print(str(links.attrs['href'])+"\n")
            anurag_temp=[]
            for i in anurag:
                if "/url?q=" in i:
                    anurag_temp.append(i)  
            plag_result_links=[]
            for i in anurag_temp:
                first=i.find("q")
                last=i.find("&")
                plag_result_links.append(i[first:last])
            yash=plag_result_links[0]
            sarkari=plag_result_links[1]
            print(yash[2:])
            print(sarkari[2:])
            

            

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            database1 = '\n'.join(chunk for chunk in chunks if chunk)

    database1 = database1.lower()

    databaseWordList = re.sub("[^\w]", " ",database1).split()	#Replace punctuation by space and split
	# databaseWordList = map(str, databaseWordList)			#And this also leads to divide by zero error

    for word in databaseWordList:
	    if word not in universalSetOfUniqueWords:
		    universalSetOfUniqueWords.append(word)

	####################################################################################################

    queryTF = []
    databaseTF = []

    for word in universalSetOfUniqueWords:
	    queryTfCounter = 0
	    databaseTfCounter = 0

	    for word2 in queryWordList:
		    if word == word2:
			    queryTfCounter += 1
	    queryTF.append(queryTfCounter)

	    for word2 in databaseWordList:
		    if word == word2:
			    databaseTfCounter += 1
	    databaseTF.append(databaseTfCounter)




    ls1=[]
    for i in universalSetOfUniqueWords:

        word_count = universalSetOfUniqueWords.count(i)  # Pythons count function, count()
        ls1.append((i,word_count))       


    dict_1 = dict(ls1)


    ls2=[]
    for i in queryWordList:

        word_count = queryWordList.count(i)  # Pythons count function, count()
        ls2.append((i,word_count))       


    dict_2 = dict(ls2)

    # vectorizer_1 = CountVectorizer()
    # # tokenize and build vocab
    # vectorizer_1.fit(universalSetOfUniqueWords)
    # vectorizer_2 = CountVectorizer()
    # # tokenize and build vocab
    # vectorizer_2.fit(queryWordList)


    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    for i in dict_1:
        X1.append(dict_1[i])
        Y1.append(i)
    for j in dict_2:
        X2.append(dict_2[j])
        Y2.append(j)
    print(dict_1)
    print(dict_2)
    # plt.scatter(X1, Y1)
    # plt.show()
    # # # print(plt.show())
    # plt.scatter(X2, Y2)
    # plt.show()

    # print(plt.show())

    dotProduct = 0
    for i in range (len(queryTF)):
	    dotProduct += queryTF[i]*databaseTF[i]

    queryVectorMagnitude = 0
    for i in range (len(queryTF)):
	    queryVectorMagnitude += queryTF[i]**2
    queryVectorMagnitude = math.sqrt(queryVectorMagnitude)

    databaseVectorMagnitude = 0
    for i in range (len(databaseTF)):
	    databaseVectorMagnitude += databaseTF[i]**2
    databaseVectorMagnitude = math.sqrt(databaseVectorMagnitude)

    matchPercentage = (float)(dotProduct / (queryVectorMagnitude * databaseVectorMagnitude))*100

    output = matchPercentage
    output=round(output,2)
   
    
    if (yash!=""):
            return render_template('index2.html', plag_meter='Plagiarism Match: {}%'.format(output), link='{}'.format(yash[2:]));        
    else :   
            return render_template('index2.html', plag_meter='Plagiarism Match: {}%'.format(output));    

if __name__ == "__main__":
    app.run(debug=True)