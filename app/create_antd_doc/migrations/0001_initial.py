# Generated by Django 4.2.3 on 2023-11-30 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="CreateAntdDoc",
            fields=[
                ("taskId", models.IntegerField(primary_key=True, serialize=False)),
            ],
        ),
    ]
