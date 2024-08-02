from app.analytics.models import Analytics
from rest_framework import serializers

class AnalyticsSerializer(serializers.ModelSerializer):
     class Meta:
         model = Analytics
         fields = '__all__'