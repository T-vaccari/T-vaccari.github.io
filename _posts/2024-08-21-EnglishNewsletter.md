---
layout: post
title: "Building a Daily Vocabulary Newsletter with Python, Google Docs, and Google Sheets"
categories: jekyll update
---

In today’s post, I’d like to share a simple app I developed to improve my English vocabulary. This app is a daily newsletter that sends me and to other subscribers  vocabulary words, and the twist is that it uses Google Docs as the database for storing the vocabulary words and their meanings, while Google Sheets is used to manage user subscriptions and track the words I've already studied.

I decided to create this tool because I’m actively learning English and wanted an automated way to refresh and expand my vocabulary. Since I have experience with Python and APIs, I took the opportunity to integrate Google Docs and Google Sheets APIs to store and manage vocabulary words and user data.

You can see the code in my [github repo](https://github.com/T-vaccari/VocabularyNewsletter)

## Overview of the app

Core Components:

1. Google Docs: Serves as the vocabulary database, containing a list of words and their meanings.
2. Google Sheets: Manages subscriber information and tracks vocabulary word appearances.
3. Email Automation: Sends out daily emails with a curated list of vocabulary words.

Here’s how everything fits together:

- Google Docs stores the vocabulary words along with their meanings, formatted like this:
word ||| meaning.
- Google Sheets maintains a list of recipients, along with metadata such as which document they are pulling their words from and whether they should receive an email on any given day.
- The app pulls a random set of words from Google Docs, formats them into an email, and sends it to the user.

## Why using Google Suit and relative APIs

There are many reasons I chose Google Docs and Google Sheets for this app:

- Google Docs: It allows me to easily add, update, and manage vocabulary terms in a simple, familiar text environment.
- Google Sheets: This is a natural choice for tracking user data, such as which words have already been sent, email preferences, and other metadata.
- Google API Integration: Google provides powerful APIs that allow seamless interaction with both Docs and Sheets, making it easy to read, write, and update data programmatically.

## How It Works

### 1. Fetching Recipients and Their Vocabulary Database from Google Sheets

The Google Sheet must be structured as follows: Email, DOCUMENT ID, SHEET ID, and a flag to receive the newsletter.

![Sheet](/assets/images/IMG_123.png)

The function `start_vocab_app()` retrieves recipient details, including their vocabulary document and tracking sheet IDs. It then uses the provided Google Sheets API to extract this data and determine which subscribers should receive the newsletter.

```python
def start_vocab_app():
  SHEET_ID = ""  # Insert the SHEET ID used for the database
  SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
  credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
  service = build('sheets', 'v4', credentials=credentials)
  SHEET_RANGE = ""  # Range to read data in the SHEET
  sheet = service.spreadsheets()
  result = sheet.values().get(spreadsheetId=SHEET_ID, range=SHEET_RANGE).execute()
  values = result.get('values', [])
  recipients = dict()
  for item in values[1:]:
    recipients[item[0]] = [thing for thing in item[1:]]

  for email, features in recipients.items():
    try:
      if (features[-1]).lower() == "no":
        words = read_google_doc(features[0])
        email_body = create_email_body(words)
        send_email(email, email_body)
      
        if features[1] != '.':
          counting_words(words, features[1])

        else:
            continue
      
    except Exception as e:
      logging.error(f"Errore per il destinatario {email}: {e}")
      print(f"Errore con {email}. Passo al destinatario successivo.")
      continue
```

### 2. Fetching Vocabulary from Google Docs

The app reads vocabulary words from a specific Google Doc using the Google Docs API. Each line in the document is formatted as a word and its meaning separated by \|\|\| (I used it for parsing pursuits).Here is how it mus look like to work with the script :

![DOC](/assets/images/IMG_124.png)

Here’s the part of the code that reads the words and returns the one that are selected randomly :

```python
def read_google_doc(DOCUMENT_ID):
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    wordstosend = 
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('docs', 'v1', credentials=credentials)

    doc = service.documents().get(documentId=DOCUMENT_ID).execute()
    content = doc.get('body').get('content')

    text = ""
    for element in content:
        if 'paragraph' in element:
            elements = element.get('paragraph').get('elements')
            for elem in elements:
                text_run = elem.get('textRun')
                if text_run:
                    text += text_run.get('content')

    lines = text.strip().split('\n')
    term_list = []
    
    for line in lines:
        if '|||' in line:
            term, meaning = line.split('|||')
            term_list.append([term.strip(), meaning.strip()])

    number = min(len(term_list), wordstosend)
    random.shuffle(term_list)
    return random.sample(term_list, number)


```

### 3. Tracking Logs in Google Sheet

We maintain a log of each word that is sent through the newsletter to ensure accurate tracking and to see what words the subscribers has already learned. Here's the code :

``` python
def counting_words(words, SHEET_ID):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=credentials)
    SHEET_RANGE = ""
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SHEET_ID, range=SHEET_RANGE).execute()
    values = result.get('values', [])
    my_dict = {row[0]: row for row in values[1:]}  

    modified_values = [values[0]]  
    english_terms = list()

    for terms in words:
        english_terms.append(terms[0])
    
    for row in values[1:]:
        if row[0] in english_terms:
            row[1] = str(int(row[1]) + 1)
        modified_values.append(row)

    for word in english_terms:
        if word not in my_dict:
            modified_values.append([word, '1'])

    body = {'values': modified_values}

    result = service.spreadsheets().values().update(
        spreadsheetId=SHEET_ID,
        range=SHEET_RANGE,
        valueInputOption="RAW",
        body=body
    ).execute()


```

### 4. Creating the body of the email

This function generates an HTML email with vocabulary words and their meanings. It formats each word-meaning pair into a styled, centered HTML layout using basic CSS for a clean appearance. The email includes a header, a list of words for the day. The function returns the email content as an HTML string, ready for sending.

```python
def create_email_body(words):
    word_html = "".join(
        f"<div style='text-align: center; margin: 10px 0; font-size: 24px;'><strong>{word_pair[0]} : {word_pair[1]}</strong></div>"
        for word_pair in words
    )

    return f"""
    <html>
    <head>
      <style>
        body {{
          font-family: Arial, sans-serif;
          background-color: #f9f9f9;
          padding: 20px;
        }}
        h1 {{
          color: #4A90E2;
          text-align: center;
          margin-bottom: 10px;
        }}
        h2 {{
          color: #333;
          text-align: center;
          margin-bottom: 20px;
        }}
        p {{
          font-size: 16px;
          line-height: 1.5;
        }}
        .container {{
          max-width: 600px;
          margin: 0 auto;
          background-color: #fff;
          padding: 30px;
          border-radius: 10px;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }}
        .footer {{
          text-align: center;
          margin-top: 20px;
          font-size: 14px;
          color: #888;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Daily Vocabulary</h1>
        <h2>Here are the words for today!</h2>
        {word_html}
        <div class="footer">
          <p>Keep Learning!</p>
          <p style='font-style: italic;'>Mail sent automatically. Please do not respond.</p>
        </div>
      </div>
    </body>
    </html>
    """

```

### 5. Sending the Email

Once the words are fetched and the progress tracked, the app creates an HTML email with the vocabulary words and sends it to the recipient using SMTP:

```python
def send_email(EMAIL_DESTINATION, email_body):
    msg = MIMEMultipart()
    msg['Subject'] = 'Daily Vocabulary'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_DESTINATION
    msg.attach(MIMEText(email_body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f'Email inviata a {EMAIL_DESTINATION}')
    except Exception as e:
        
        logging.error(f"Errore nell'invio dell'email a {EMAIL_DESTINATION}: {e}")
        print(f"Errore nell'invio dell'email a {EMAIL_DESTINATION}: {e}")

```

## Conclusion

Building this daily vocabulary newsletter has been a rewarding project that combines Python programming with Google APIs to automate and enhance language learning. By integrating Google Docs and Google Sheets, the application efficiently manages vocabulary data and user subscriptions, making the learning process more streamlined and interactive.

This tool not only helps me expand my English vocabulary but also serves as a practical example of how APIs can be used to automate tasks and improve productivity. Whether you're learning a new language or looking for ways to automate repetitive tasks, this project demonstrates the power of leveraging technology to create personalized solutions.

If you found this article helpful, I encourage you to experiment with similar integrations and explore the vast possibilities of API-driven applications. Feel free to reach out with any questions or share your own experiences with building automated tools.
