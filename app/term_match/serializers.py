from app.term_match.models import TermMatch
from rest_framework import serializers

class TermMatchSerializer(serializers.ModelSerializer):
     class Meta:
         model = TermMatch
         fields = '__all__'