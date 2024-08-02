from app.doc2pdf.models import DocConvert
from rest_framework import serializers

class DocConvertSerializer(serializers.ModelSerializer):
     class Meta:
         model = DocConvert
         fields = '__all__'