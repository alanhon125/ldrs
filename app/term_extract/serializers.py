from app.term_extract.models import TermExtract
from rest_framework import serializers

class TermExtractSerializer(serializers.ModelSerializer):
     class Meta:
         model = TermExtract
         fields = '__all__'