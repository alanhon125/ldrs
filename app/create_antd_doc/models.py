from django.db import models
from django.db.models import AutoField, TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

# Create your models here.
class CreateAntdDoc(models.Model):
    # id = AutoField(auto_created=True, primary_key=True)
    taskId = IntegerField(primary_key=True)
