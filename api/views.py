from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.http import JsonResponse
import os
from datetime import datetime
from .models import Project,ActivityLog
from .serializers import ProjectSerializer
import pyreadstat
import pandas as pd
from .utils import crosstab_open_ended, log_activity,load_project_meta, crosstab_multi_response_with_topbreak,crosstab_numeric_with_topbreak,crosstab_open_ended_with_topbreak,crosstab_single_response_grid_with_topbreak
import math
from .utils import crosstab_single_choice, crosstab_multi_response,crosstab_numeric,crosstab_single_response_grid,crosstab_nps,crosstab_single_choice_with_topbreak



# Helper to load meta Excel as DataFrame
@api_view(["GET"])
def get_meta_titles(request, project_id):
    project = Project.objects(id=project_id).first()
    if not project or not project.meta_file:
        return Response({"error": "Meta Excel not found"}, status=400)

    meta_path = os.path.join(settings.MEDIA_ROOT, project.meta_file)
    meta_df = pd.read_excel(meta_path)

    titles = []
    seen = set()   # ✅ track duplicates

    for _, row in meta_df.iterrows():
        qtype = str(row["Question_Type"]).upper()
        table_title = str(row["Table_Title"])

        if qtype in ["MR", "SRG"]:
            grp = str(row.get("Var_Grp", "")).strip()
            key = f"{grp}::{table_title}"
            if key not in seen:
                seen.add(key)
                titles.append(table_title)
        else:
            if table_title not in seen:
                seen.add(table_title)
                titles.append(table_title)

    return JsonResponse(titles, safe=False)

@api_view(["GET"])
def list_projects(request):
    projects = Project.objects.order_by("-created_at")
    serializer = ProjectSerializer(projects, many=True)
    return Response(serializer.data)

@api_view(["GET"])
def generate_meta(request, project_id):
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    force = request.query_params.get("force") == "true"

    # If meta exists and not forcing → return existing file
    if project.meta_file and not force:
        return Response({"meta_file": project.meta_file, "message": "Meta file already exists"})

    if not project.files or len(project.files) == 0:
        return Response({"error": "No data files in project"}, status=400)

    # For now, pick first file (later can choose specific file)
    file_path = os.path.join(settings.MEDIA_ROOT, project.files[0])

    if file_path.endswith(".sav"):
        import pyreadstat
        import pandas as pd

        df, meta = pyreadstat.read_sav(file_path, metadataonly=True)

        rows = []
        for var in meta.column_names:
            rows.append({
                "Var_Name": var,
                "Var_Label": "",
                "Table_Title": meta.column_labels[meta.column_names.index(var)] if meta.column_labels else "",
                "Question_Type": "",
                "Var_Grp": "",
                "Base_Title": "",
                "Add_Question_Type": ""
            })

        df_meta = pd.DataFrame(rows)

        project_folder = os.path.join(settings.MEDIA_ROOT, project.project_name)
        os.makedirs(project_folder, exist_ok=True)

        meta_file_path = os.path.join(project_folder, "meta.xlsx")
        df_meta.to_excel(meta_file_path, index=False)

        relative_path = os.path.relpath(meta_file_path, settings.MEDIA_ROOT).replace("\\", "/")
        project.meta_file = relative_path
        project.save()
            
        # log activity
        # Log regenerate event
        log_activity(
            user=request.user.username if request.user.is_authenticated else "guest",
            action="regenerate_meta" if force else "generate_meta",
            project_name=project.project_name,
            file_name="meta.xlsx"
        )

        return Response({"meta_file": relative_path, "message": "Meta file generated (forced)"})
    else:
        return Response({"error": "Unsupported file type for metadata"}, status=400)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_project(request):
    project_name = request.data.get("project_name")
    file = request.FILES.get("file")

    if not project_name or not file:
        return Response({"error": "Project name and file required"}, status=400)

    # Create folder
    project_folder = os.path.join(settings.MEDIA_ROOT, project_name)
    os.makedirs(project_folder, exist_ok=True)

    # Save file
    file_path = os.path.join(project_folder, file.name)
    with open(file_path, "wb+") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

    relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT).replace("\\", "/")

    # Find project by name or create new
    project = Project.objects(project_name=project_name).first()
    if not project:
        project = Project(project_name=project_name, files=[], created_at=datetime.utcnow())

    if relative_path not in project.files:  # avoid duplicate file entries
        project.files.append(relative_path)

    project.save()

        # Log activity
    log_activity(
        user=request.user.username if request.user.is_authenticated else "guest",
        action="upload Project file",
        project_name=project_name,
        file_name=file.name
    )

    serializer = ProjectSerializer(project)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["DELETE"])
def delete_file(request, project_id, file_name):
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    # Remove file from disk
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from DB list
    project.files = [f for f in project.files if f != file_name]
    project.save()

        # Log delete file
    log_activity(
        user=request.user.username if request.user.is_authenticated else "guest",
        action="delete_file",
        project_name=project.project_name,
        file_name=file_name
    )

    return Response({"message": "File deleted successfully"})

@api_view(["DELETE"])
def delete_project(request, project_id):
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    # Delete all files in project folder
    project_folder = os.path.join(settings.MEDIA_ROOT, project.project_name)
    if os.path.exists(project_folder):
        import shutil
        shutil.rmtree(project_folder)  # removes folder + files
     # Log delete project
    log_activity(
        user=request.user.username if request.user.is_authenticated else "guest",
        action="delete_project",
        project_name=project.project_name
    )

    # Delete project from MongoDB
    project.delete()



    return Response({"message": f"Project '{project.project_name}' deleted successfully"})


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_meta(request, project_id):
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    file = request.FILES.get("meta_file")
    if not file:
        return Response({"error": "No file uploaded"}, status=400)

    project_folder = os.path.join(settings.MEDIA_ROOT, project.project_name)
    os.makedirs(project_folder, exist_ok=True)

    meta_file_path = os.path.join(project_folder, "meta.xlsx")
    with open(meta_file_path, "wb+") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

    relative_path = os.path.relpath(meta_file_path, settings.MEDIA_ROOT).replace("\\", "/")
    project.meta_file = relative_path
    project.save()
        # Log meta upload
    log_activity(
        user=request.user.username if request.user.is_authenticated else "guest",
        action="upload_meta",
        project_name=project.project_name,
        file_name="meta.xlsx"
    )

    return Response({"meta_file": relative_path, "message": "Meta file uploaded successfully"})

# // log_activity
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_project(request):
    project_name = request.data.get("project_name")
    file = request.FILES.get("file")

    if not project_name or not file:
        return Response({"error": "Project name and file required"}, status=400)

    # Create project folder
    project_folder = os.path.join(settings.MEDIA_ROOT, project_name)
    os.makedirs(project_folder, exist_ok=True)

    # Save file to disk
    file_path = os.path.join(project_folder, file.name)
    with open(file_path, "wb+") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

    relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT).replace("\\", "/")

    # Find or create project
    project = Project.objects(project_name=project_name).first()
    if not project:
        project = Project(
            project_name=project_name,
            created_at=datetime.utcnow(),
            files=[]
        )

    # Append file if not already stored
    if relative_path not in project.files:
        project.files.append(relative_path)

    project.save()  # ✅ project is guaranteed to exist here

    # Log activity
    log_activity(
        user=request.user.username if request.user.is_authenticated else "guest",
        action="upload Project file Sav",
        project_name=project_name,
        file_name=file.name
    )

    # Serialize response
    serializer = ProjectSerializer(project)
    return Response(serializer.data, status=status.HTTP_201_CREATED)

# log activity view
@api_view(["GET"])
def list_logs(request):
    logs = ActivityLog.objects.order_by("-timestamp").limit(100)  # last 100
    return Response([
        {
            "user": log.user,
            "action": log.action,
            "project": log.project_name,
            "file": log.file_name,
            "details": log.details,
            "timestamp": log.timestamp
        }
        for log in logs
    ])



@api_view(["GET"])
def quick_crosstab(request, project_id):
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    if not project.files or len(project.files) == 0:
        return Response({"error": "No data files found"}, status=400)

    # Use first SAV file
    sav_path = os.path.join(settings.MEDIA_ROOT, project.files[0])
    df, meta = pyreadstat.read_sav(sav_path)

    # Load meta Excel
    if not project.meta_file:
        return Response({"error": "Meta Excel not found"}, status=400)
    meta_path = os.path.join(settings.MEDIA_ROOT, project.meta_file)
    meta_df = pd.read_excel(meta_path)

    output = []

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return 0
        return obj

    # --- Single Choice (SC / SR) ---
    sc_rows = meta_df[meta_df["Question_Type"].str.upper().isin(["SC", "SR"])]
    for _, row in sc_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        add_type = str(row.get("Add_Question_Type", "")).strip().upper()

        if var_name in df.columns:
            if add_type == "NPS":
                table_data = crosstab_nps(df, var_name)   # 🔹 NPS function
                qtype = "NPS"
            else:
                value_labels = meta.variable_value_labels.get(var_name, {})
                table_data = crosstab_single_choice(df, var_name, value_labels)
                qtype = "SC"

            output.append({
                "question": table_title,
                "var_name": var_name,
                "type": qtype,
                "data": table_data,
                "add_type": add_type
            })


    # --- Multi Response (MR) ---
    mr_groups = meta_df[meta_df["Question_Type"].str.upper() == "MR"]["Var_Grp"].unique()
    for grp in mr_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        table_data = crosstab_multi_response(df, grp, group_rows, meta)
        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "MR",
            "data": table_data
        })

    # --- Numeric (NR) ---
    nr_rows = meta_df[meta_df["Question_Type"].str.upper() == "NR"]
    for _, row in nr_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        if var_name in df.columns:
            table_data = crosstab_numeric(df, var_name)
            output.append({
                "question": table_title,
                "var_name": var_name,
                "type": "NR",
                "data": table_data
            })

    # --- Open Ended (OE) ---
    oe_rows = meta_df[meta_df["Question_Type"].str.upper() == "OE"]
    for _, row in oe_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        if var_name in df.columns:
            table_data = crosstab_open_ended(df, var_name)
            output.append({
                "question": table_title,
                "var_name": var_name,
                "type": "OE",
                "data": table_data
            })

    # --- Single Response Grid (SRG) ---
    srg_groups = meta_df[meta_df["Question_Type"].str.upper() == "SRG"]["Var_Grp"].unique()
    for grp in srg_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        table_data = crosstab_single_response_grid(df, grp, group_rows, meta)
        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "SRG",
            "data": table_data
        })

    return Response(clean_for_json(output))

# Helper to get var name from title
def get_var_name_from_title(meta_df, table_title):
    row = meta_df[meta_df["Table_Title"] == table_title]
    if not row.empty:
        return str(row.iloc[0]["Var_Name"]).strip()
    return None



# New Crosstab view 
@api_view(["GET"])
def new_crosstab(request, project_id):
    topbreak_title = request.GET.get("topbreak")   # user-selected Table_Title
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    # Load SAV + meta
    sav_path = os.path.join(settings.MEDIA_ROOT, project.files[0])
    df, meta = pyreadstat.read_sav(sav_path)
    meta_path = os.path.join(settings.MEDIA_ROOT, project.meta_file)
    meta_df = pd.read_excel(meta_path)

    output = []

    # --- Single Choice ---
    sc_rows = meta_df[meta_df["Question_Type"].str.upper().isin(["SC", "SR"])]
    for _, row in sc_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        value_labels = meta.variable_value_labels.get(var_name, {})

        if topbreak_title:
            table_data = crosstab_single_choice_with_topbreak(
                df, var_name, value_labels, topbreak_title, meta, meta_df
            )
        else:
            table_data = crosstab_single_choice(df, var_name, value_labels)

        output.append({
            "question": table_title,
            "var_name": var_name,
            "type": "SC",
            "data": table_data
        })

    # --- Multi Response ---
    mr_groups = meta_df[meta_df["Question_Type"].str.upper() == "MR"]["Var_Grp"].unique()
    for grp in mr_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]

        if topbreak_title:
            table_data = crosstab_multi_response_with_topbreak(
                df, grp, group_rows, meta, topbreak_title, meta_df
            )
        else:
            table_data = crosstab_multi_response(df, grp, group_rows, meta)

        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "MR",
            "data": table_data
        })

    # --- Numeric ---
    nr_rows = meta_df[meta_df["Question_Type"].str.upper() == "NR"]
    for _, row in nr_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])

        if topbreak_title:
            table_data = crosstab_numeric_with_topbreak(df, var_name, topbreak_title, meta, meta_df)
        else:
            table_data = crosstab_numeric(df, var_name)

        output.append({
            "question": table_title,
            "var_name": var_name,
            "type": "NR",
            "data": table_data
        })

    # --- Open Ended ---
    # oe_rows = meta_df[meta_df["Question_Type"].str.upper() == "OE"]
    # for _, row in oe_rows.iterrows():
    #     var_name = str(row["Var_Name"]).strip()
    #     table_title = str(row["Table_Title"])

    #     if topbreak_title:
    #         table_data = crosstab_open_ended_with_topbreak(df, var_name, topbreak_title, meta, meta_df)
    #     else:
    #         table_data = crosstab_open_ended(df, var_name)

    #     output.append({
    #         "question": table_title,
    #         "var_name": var_name,
    #         "type": "OE",
    #         "data": table_data
    #     })

    # --- Single Response Grid ---
    srg_groups = meta_df[meta_df["Question_Type"].str.upper() == "SRG"]["Var_Grp"].unique()
    for grp in srg_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]

        if topbreak_title:
            table_data = crosstab_single_response_grid_with_topbreak(
                df, grp, group_rows, meta, topbreak_title, meta_df
            )
        else:
            table_data = crosstab_single_response_grid(df, grp, group_rows, meta)

        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "SRG",
            "data": table_data
        })

    return Response(output)
