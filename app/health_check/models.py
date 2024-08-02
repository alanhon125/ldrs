from django.db import models
from django.db.models import AutoField, TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

# Create your models here.
class HealthCheck(models.Model):
    id = AutoField(auto_created=True, primary_key=True)
    configurationStatus = CharField(max_length=255, blank=True, null=True, db_index=True)
    documentParsingStatus = CharField(max_length=255, blank=True, null=True, db_index=True)
    featuresExtractionStatus = CharField(max_length=255, blank=True, null=True, db_index=True)
    termMatchingStatus = CharField(max_length=255, blank=True, null=True, db_index=True)