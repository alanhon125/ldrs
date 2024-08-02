from django.urls import path
from .views import TermExtractView

urlpatterns = [
    path("docparse2table", TermExtractView.as_view(), name="docparse2table")
]