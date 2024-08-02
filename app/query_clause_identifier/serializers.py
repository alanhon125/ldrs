from app.query_clause_identifier.models import QueryClauseId
from rest_framework import serializers

class QueryClauseIdSerializer(serializers.ModelSerializer):
     class Meta:
         model = QueryClauseId
         fields = '__all__'