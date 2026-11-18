from django.core.management import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "检查表中数据是否包含预设"
    
    def handle(self, *args, **options):
        try:
            
            if 't_base_model' not in connection.introspection.table_names():
                self.stderr.write(f"表不存在,无法预存数据")
                return
            
            from apps.common.models import BaseModel, BaseType
            objects = [
                {
                    'base_key': 'example_key',
                    'base_value': 'example_value',
                    'base_desc': 'example_desc',
                    'base_type': BaseType.TEXT,
                }
            ]
            
            for object_data in objects:
                o, created = BaseModel.objects.get_or_create(
                    base_key=object_data['base_key'],
                    defaults=object_data,
                )
                if created:
                    self.stdout.write(f"已预存 {o.base_key}")
                else:
                    self.stdout.write(f"已存在 {o.base_key}")
            
            self.stdout.write(f"已完成数据预存")
        except Exception as e:
            self.stderr.write(f"预存数据过程中发生错误: {str(e)}")
