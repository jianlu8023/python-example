from django.urls import include, path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('yolo/', include('apps.infer.yolo.urls')),
    path("pytorch/",include('apps.infer.pytorch.urls')),
]
