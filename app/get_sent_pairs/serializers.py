from app.get_sent_pairs.models import GetSentPairs
from rest_framework import serializers

class GetSentPairsSerializer(serializers.ModelSerializer):
     class Meta:
         model = GetSentPairs
         fields = '__all__'