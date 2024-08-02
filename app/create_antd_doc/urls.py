from django.urls import path
from .views import CreateAntdDocView

urlpatterns = [
    path("createAntdDoc", CreateAntdDocView.as_view(), name="createAntdDoc")
]