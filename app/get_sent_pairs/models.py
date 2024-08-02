from django.db import models
from django.db.models import IntegerField, CharField

# Create your models here.
class GetSentPairs(models.Model):
    task_id = IntegerField(blank=False, null=False, help_text='task ID')
    update_start_date = CharField(max_length=16, blank=True, null=True, db_index=True)
    update_end_date = CharField(max_length=16, blank=True, null=True, db_index=True)

    