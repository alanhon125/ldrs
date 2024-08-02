from app.health_check.models import HealthCheck
from rest_framework import serializers

class HealthCheckSerializer(serializers.ModelSerializer):
     class Meta:
         model = HealthCheck
         fields = '__all__'