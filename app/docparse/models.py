from django.db import models
from django.db.models import AutoField, TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

# Create your models here.
class DocParse(models.Model):
    # id = AutoField(auto_created=True, primary_key=True)
    id = IntegerField(primary_key=True)
    createUser = CharField(max_length=255, blank=True, null=True, db_index=True)
    updateUser = CharField(max_length=255, blank=True, null=True, db_index=True)
    fileName = CharField(max_length=255, blank=False, null=True, db_index=True)
    filePath = CharField(max_length=255, blank=False, null=True, db_index=True)
    fileType = CharField(max_length=32, blank=False, null=True, db_index=True)