# Generated by Django 4.2.3 on 2024-02-20 08:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("health_check", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="healthcheck",
            name="configurationStatus",
            field=models.CharField(
                blank=True, db_index=True, max_length=255, null=True
            ),
        ),
        migrations.AddField(
            model_name="healthcheck",
            name="documentParsingStatus",
            field=models.CharField(
                blank=True, db_index=True, max_length=255, null=True
            ),
        ),
        migrations.AddField(
            model_name="healthcheck",
            name="featuresExtractionStatus",
            field=models.CharField(
                blank=True, db_index=True, max_length=255, null=True
            ),
        ),
        migrations.AddField(
            model_name="healthcheck",
            name="termMatchingStatus",
            field=models.CharField(
                blank=True, db_index=True, max_length=255, null=True
            ),
        ),
    ]
