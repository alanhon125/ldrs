from django.db import models
from django.db.models import AutoField, TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

# Create your models here.
class DocProcess(models.Model):
    # id = AutoField(auto_created=True, primary_key=True)
    id = IntegerField(primary_key=True, help_text='document ID (created automatically when file uploaded)')
    createUser = CharField(max_length=255, blank=True, null=True, db_index=True, help_text='username of task creator')
    updateUser = CharField(max_length=255, blank=True, null=True, db_index=True, help_text='username of task updater')
    fileName = CharField(max_length=255, blank=False, null=True, db_index=True, help_text='the filename of document to be converted')
    filePath = CharField(max_length=255, blank=False, null=True, db_index=True, help_text='the absolute file path of document to be converted')
    fileType = CharField(max_length=32, blank=False, null=True, db_index=True, help_text='document type, either FA for facility agreement or TS for term sheet')
    convertTo = CharField(max_length=16, blank=True, null=True, db_index=True, help_text='LibreOffice converter is used. It only supports file conversion from .doc to .pdf, .html, .txt, input value must be one of the following: html, pdf, txt, docx')
    outputFolder = CharField(max_length=255, blank=True, null=True, db_index=True, help_text='the desired output folder path')