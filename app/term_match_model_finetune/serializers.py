from app.term_match_model_finetune.models import GetModelFinetune
from rest_framework import serializers

class GetModelFinetuneSerializer(serializers.ModelSerializer):
     class Meta:
         model = GetModelFinetune
         fields = '__all__'