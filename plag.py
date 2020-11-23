from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import re
import math
from requests import get  #pip3 install requests
import urllib
from bs4 import BeautifulSoup #pip3 install BeautifulSoup4
app = Flask(__name__)

q = ""

@app.route('/')
def home():
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
    if(not database1):
            keyword = inputQuery
            url = "https://google.com/search?q="+keyword
            html = get(url).text

            
            soup = BeautifulSoup(html,features="lxml")
            temp="""
            class="kCrYT"
            """
            links_with_text = []
            # for a in soup.find_all(temp, href=True):
            #     print(a)
            #     if a.text: 
            #         links_with_text.append(a['href'])
            # print(links_with_text)
            anurag=[]
            links_with_text=soup.find_all('a')
            for links in links_with_text:
                if 'href' in links.attrs:
                    anurag.append(str(links.attrs['href']))
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

    """
    print queryWordList
    print
    print databaseWordList


    print queryTF
    print
    print databaseTF
    """


    output = matchPercentage
    output=round(output,2)
    return render_template('index.html', plag_meter='Plagiarism Match: {}%'.format(output))






@app.route('/cosineSimilarity_api',methods=['POST'])
def cosineSimilarity_api():
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
    fd = open("database1.txt", "r")
    database1 = fd.read().lower()

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

    """
    print queryWordList
    print
    print databaseWordList


    print queryTF
    print
    print databaseTF
    """


    output = matchPercentage
    output=round(output,2)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)