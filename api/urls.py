from django import views
from django.urls import path
from .views import list_projects, upload_project, delete_file, delete_project, generate_meta, upload_meta
from .views import list_logs,quick_crosstab,get_meta_titles,new_crosstab

urlpatterns = [
    path("projects/", list_projects),
    path("projects/upload/", upload_project),
    path("projects/<str:project_id>/delete/<path:file_name>/", delete_file),
        path("projects/<str:project_id>/delete/", delete_project),
            path("projects/<str:project_id>/generate_meta/", generate_meta),  # 👈 generate
    path("projects/<str:project_id>/upload_meta/", upload_meta),      # 👈 upload
        path("logs/", list_logs),   # GET /api/logs/
            path("projects/<str:project_id>/quick_crosstab/", quick_crosstab),
           path("projects/<str:project_id>/meta_titles/", get_meta_titles, name="meta_titles"),
           path("projects/<str:project_id>/new_crosstab/", new_crosstab, name="new_crosstab"),


            
]
