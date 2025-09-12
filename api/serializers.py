from rest_framework import serializers
from .models import Project

class ProjectSerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True)
    project_name = serializers.CharField()
    files = serializers.ListField(      # ✅ changed from file → files
        child=serializers.CharField(),
        read_only=True
    )
    created_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        project = Project(**validated_data)
        project.save()
        return project
