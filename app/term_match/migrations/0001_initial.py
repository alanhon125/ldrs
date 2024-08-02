# Generated by Django 4.2.3 on 2023-11-30 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="TermMatch",
            fields=[
                ("id", models.IntegerField(primary_key=True, serialize=False)),
                (
                    "createUser",
                    models.CharField(
                        blank=True, db_index=True, max_length=255, null=True
                    ),
                ),
                (
                    "updateUser",
                    models.CharField(
                        blank=True, db_index=True, max_length=255, null=True
                    ),
                ),
                (
                    "fileName",
                    models.CharField(db_index=True, max_length=255, null=True),
                ),
                (
                    "filePath",
                    models.CharField(db_index=True, max_length=255, null=True),
                ),
                ("fileType", models.CharField(db_index=True, max_length=32, null=True)),
            ],
        ),
    ]
