
from django.urls import path
from chatbot.views import TalkToBot

urlpatterns = [
    path('ask', TalkToBot.as_view())
]
