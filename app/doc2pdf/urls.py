from django.urls import path
from .views import DocConvertView

urlpatterns = [
    path("convertDoc2pdf", DocConvertView.as_view(), name="convertDoc2pdf")
]