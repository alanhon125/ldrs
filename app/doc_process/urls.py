from django.urls import path
from .views import DocProcessView

urlpatterns = [
    path("docparseTermExtract", DocProcessView.as_view(), name="docparseTermExtract")
]