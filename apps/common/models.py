from django.db import models
from rest_framework import serializers


# Create your models here.


class BaseModel(models.Model):
    auto_uid = models.AutoField(db_column='auto_uid', verbose_name='自增id', primary_key=True)
    base_key = models.CharField(db_column='base_key', verbose_name='基础key', max_length=255, unique=True)
    base_value = models.CharField(db_column='base_value', verbose_name='基础value', max_length=255, blank=True,
                                  null=True)
    create_time = models.DateTimeField(db_column='create_time', verbose_name='创建时间', auto_now_add=True)
    update_time = models.DateTimeField(db_column='update_time', verbose_name='更新时间', auto_now=True)
    
    class Meta:
        db_table = 't_base_model'


class BaseModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = BaseModel
        fields = ['auto_uid', 'base_key', 'base_value', 'create_time', 'update_time']
