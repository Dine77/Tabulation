# api/models.py
import mongoengine as me
from datetime import datetime

class Project(me.Document):
    project_name = me.StringField(required=True, max_length=255, unique=True)
    files = me.ListField(me.StringField())
    meta_file = me.StringField()   # 👈 store path to meta Excel
    created_at = me.DateTimeField(default=datetime.utcnow)

    meta = {"collection": "projects"}

class ActivityLog(me.Document):
    user = me.StringField(required=True)   # later replace with auth user ID
    action = me.StringField(required=True) # e.g. "upload", "download", "delete"
    project_name = me.StringField()
    file_name = me.StringField()
    details = me.StringField()             # optional extra info
    timestamp = me.DateTimeField(default=datetime.utcnow)

    meta = {"collection": "activity_logs"}
