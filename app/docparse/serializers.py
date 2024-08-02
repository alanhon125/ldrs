from app.docparse.models import DocParse
from rest_framework import serializers

class DocParseSerializer(serializers.ModelSerializer):
     class Meta:
         model = DocParse
         fields = '__all__'