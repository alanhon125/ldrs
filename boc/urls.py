"""
URL configuration for boc project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from drf_yasg import openapi
from config import ANALYTICS_ADDR, LOG_FILEPATH
import os

schema_view = get_schema_view(
   openapi.Info(
      title="LDRS Analytics API",
      default_version='v1',
      description="This API exposes endpoints to use the analytics for loan document review system (LDRS).",
      contact=openapi.Contact(name="Alan Hon",email="alanhon@astri.org")
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
   url=f"http://{ANALYTICS_ADDR}/api/"
   # server=openapi.Server(url="http://10.6.55.12:8000/api/",description="Server URL in Development Environment."),
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/",include("app.upload_doc.urls")),
    path("api/",include("app.doc2pdf.urls")),
    path("api/",include("app.docparse.urls")),
    path("api/",include("app.term_extract.urls")),
    path("api/",include("app.doc_process.urls")),
    path("api/",include("app.term_match.urls")),
    path("api/",include("app.create_antd_doc.urls")),
    path("api/",include("app.query_clause_identifier.urls")),
    path("api/",include("app.analytics.urls")),
    path("api/",include("app.health_check.urls"))
]

urlpatterns += [
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    re_path(r'^swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    re_path(r'^redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api-docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]

if not os.path.exists(LOG_FILEPATH):
    try:
        os.makedirs(os.path.dirname(LOG_FILEPATH),exist_ok=True)
    except:
        pass
    open(LOG_FILEPATH,'a')