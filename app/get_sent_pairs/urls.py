from django.urls import path
from .views import GetSentPairsView

urlpatterns = [
    path("getSentPairs", GetSentPairsView.as_view(), name="getSentPairs")
]