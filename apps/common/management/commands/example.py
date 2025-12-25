from django.core.management import BaseCommand


class Command(BaseCommand):
    help = 'command 示例'
    
    def handle(self, *args, **options):
        self.stdout.write(f"这里你可以实现不同的效果.")
