from django.urls import path
from myapp import views

urlpatterns = [
    path('',views.index),
    path('about/',views.about),
    path('hello/',views.hello),
    path('projects/',views.projects),
    path('tasks/',views.tasks),
     path('create_task/',views.create_task),
]