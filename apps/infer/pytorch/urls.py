from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('resnet18',views.resnet18,name='resnet18'),  # 暂时注释掉不存在的视图
]