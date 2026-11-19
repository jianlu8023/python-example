# Django使用整理

## 创建django项目

* tips: 在cwd=django 以创建 名为 django_demo 的 django项目

```shell
# cwd=django 
django-admin startproject django_demo
# result: 会在django文件夹生成 django_demo 项目 django/django_demo


```

## 创建app

* tips: 在cwd=django/django_demo 创建 app

```shell
# 创建文件夹
mkdir -p  apps/common
# 创建app
django-admin startapp common apps/common
```