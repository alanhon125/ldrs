from django.urls import path
from .views import GetModelFinetuneView

urlpatterns = [
    path("getModelFinetune", GetModelFinetuneView.as_view(), name="getModelFinetune")
]