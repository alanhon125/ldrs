from django.db import models
from django.db.models import AutoField, TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

# Create your models here.
class QueryClauseId(models.Model):
    # id = AutoField(auto_created=True, primary_key=True)
    id = IntegerField(primary_key=True)
    taskId = IntegerField(blank=True, null=True, db_index=True)
    pageId = IntegerField(blank=True, null=True, db_index=True)
    bbox = JSONField(max_length=255, blank=True, null=True, db_index=True)
    width = IntegerField(blank=True, null=True, db_index=True)
    height = IntegerField(blank=True, null=True, db_index=True)
    manualContent = CharField(max_length=255, blank=False, null=True, db_index=True)