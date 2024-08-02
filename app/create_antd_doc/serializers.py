from app.create_antd_doc.models import CreateAntdDoc
from rest_framework import serializers

class CreateAntdDocSerializer(serializers.ModelSerializer):
     class Meta:
         model = CreateAntdDoc
         fields = '__all__'