from django.urls import path
from .views import TermMatchView

urlpatterns = [
    path("termMatch2table", TermMatchView.as_view(), name="termMatch2table")
]