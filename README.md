# AhaBot
## Self Attention based knowledge space aggregator bot 


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)


## HOW TO RUN:
- Place the `label_encoder.pickle`, `chatbot_model.h5` and `tokenizer.pickle` in the root folder.
- run `pip install requirements.txt`
- then run `python manage.py runserver` to start the server. 
- simply open `index.html` in interface folder. You can now interact with our chatbot :)


## FEATURES:
- Our ChatBot allows you to create diverse knowledge base merely from a list of FAQs and automatically handles user requests in real time.
- The model is intelligent and doesn't require hardcoded values. 
- It is equipped with deep transformer neural networks to create a knowledge space for each question to model similar meaning situations. This is the core feature of our chatbot as this not only augments the data, it also allows you to handle a diverse set of requests.
- You can either the pre-trained model `chatbot_model.h5` or retrain the model on your custom FAQs.

## Sample Output:
![alt text](https://github.com/ahsanali2000/ahabot/blob/master/sample%20images/1.jpeg)
![alt text](https://github.com/ahsanali2000/ahabot/blob/master/sample%20images/2.jpeg)
![alt text](https://github.com/ahsanali2000/ahabot/blob/master/sample%20images/3.jpeg)

## About Code:
- We are using neural networks, deep transformer neural networks and BERT embeddings in our model.
- The model runs for `350` Epochs and gives `95%+` accuracy with `< 0.05` categorical loss.
- `vanilla javascript` was used for front end.
- `Django` was used for backend.
- The model contains `16000+` trainable parameters and uses the power of tensorflow to train on it.
- `After the modle is done training, the encoding files and model weights are saved. You don't have to go through the hassle to retrain it every time.


## CURRENT FAQs for bot's knowledege space:

The bot needs to be supplied a list of FAQs to operate on, by default the following FAQs with questions and answers are supplied:
- What is your name ?
- Where do you live ?
- How are you doing ?
- Can you help me ?
- I need to create a new account
- I want to submit a complaint
- How do I cancel my premium membership
- What is your pricing plan
- Do you offer a free trial account ?
- How do i cancel my free trial ?
- I forgot my password
- I want to reset my account
- I want to delete my account
- I want to talk to an agent
- I had been overcharged on the latest bill
- I need a copy of my credentials
- I can't download the backup
- how to change the profile picture ?
- website is loading very slow
- How do i implement the amazon web services AWS ?
- I want to improve website's SEO

## NOTE: 
The program is dynamic and would work on any set of questions/answers supplied..
- ✨Magic ✨
