from django.urls import path
from .views import QueryClauseIdView

urlpatterns = [
    path("queryClauseId", QueryClauseIdView.as_view(), name="queryClauseId")
]