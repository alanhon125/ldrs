from django.urls import path
from .views import DocParseView

urlpatterns = [
    path("docparse2json", DocParseView.as_view(), name="docparse2json")
]