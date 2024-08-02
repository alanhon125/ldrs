from app.doc_process.models import DocProcess
from rest_framework import serializers

class DocProcessSerializer(serializers.ModelSerializer):
     class Meta:
         model = DocProcess
         fields = '__all__'