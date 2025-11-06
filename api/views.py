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
from .utils import log_activity,load_project_meta, crosstab_multi_response_with_topbreak,crosstab_numeric_with_topbreak,crosstab_single_response_grid_with_topbreak
import math
from .utils import crosstab_single_choice, crosstab_multi_response,crosstab_numeric,crosstab_single_response_grid,crosstab_nps,crosstab_single_choice_with_topbreak,crosstab_multi_response_grid,crosstab_multi_response_grid_with_topbreak
from .utils import build_numeric_grid,build_numeric_grid_topbreak
from .utils import build_wordcloud_data

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

    # For now, pick first file
    file_path = os.path.join(settings.MEDIA_ROOT, project.files[0])

    # Create project folder
    project_folder = os.path.join(settings.MEDIA_ROOT, project.project_name)
    os.makedirs(project_folder, exist_ok=True)
    meta_file_path = os.path.join(project_folder, "meta.xlsx")

    # ---------- CASE 1: SPSS (.sav) ----------
    if file_path.endswith(".sav"):
        df, meta = pyreadstat.read_sav(file_path, metadataonly=True)

        # Build Sheet1 (variables metadata)
        rows = []
        for var in meta.column_names:
            rows.append({
                "Var_Name": var,
                "Var_Label": "",
                "Table_Title": meta.column_labels[meta.column_names.index(var)] if meta.column_labels else "",
                "Question_Type": "",
                "Var_Grp": "",
                "Base_Title": "",
                "Add_Question_Type": "",
                "Sortorder": ""
            })
        df_meta = pd.DataFrame(rows)

        # Build Sheet2 (value labels)
        valuelabel_rows = []
        for var, mapping in meta.variable_value_labels.items():
            first = True
            for val, label in mapping.items():
                valuelabel_rows.append({
                    "variable": var if first else "",
                    "value": val,
                    "label": label
                })
                first = False

        if valuelabel_rows:
            df_value_labels = pd.DataFrame(valuelabel_rows)
        else:
            # Always create empty sheet if no labels
            df_value_labels = pd.DataFrame(columns=["variable", "value", "label"])

        # Write both sheets into meta.xlsx
        with pd.ExcelWriter(meta_file_path, engine="openpyxl") as writer:
            df_meta.to_excel(writer, index=False, sheet_name="metadata")
            df_value_labels.to_excel(writer, index=False, sheet_name="value_label")


        # ---------- CASE 2: Excel (.xlsx) ----------
    elif file_path.endswith(".xlsx"):
        xls = pd.ExcelFile(file_path)

        # --- Read variable_label sheet ---
        if "variable_label" not in xls.sheet_names:
            return Response({"error": "Excel must contain a 'variable_label' sheet"}, status=400)

        variable_label_df = pd.read_excel(xls, "variable_label")

        # Normalize column names (lowercase + strip spaces)
        variable_label_df.columns = [c.strip().lower() for c in variable_label_df.columns]

        if not {"variable", "label"}.issubset(variable_label_df.columns):
            return Response({"error": "variable_label sheet must contain 'variable' and 'label' columns"}, status=400)

        # Build Sheet1 (variables metadata, same format as SAV)
        rows = []
        for _, row in variable_label_df.iterrows():
            rows.append({
                "Var_Name": row["variable"],
                "Var_Label": "",
                "Table_Title": row["label"],
                "Question_Type": "",
                "Var_Grp": "",
                "Base_Title": "",
                "Add_Question_Type": "",
                "Sortorder": "",
            })
            df_meta = pd.DataFrame(rows)

        # --- Read value_label sheet (optional) ---
        if "value_label" in xls.sheet_names:
            df_value_labels = pd.read_excel(xls, "value_label")
            # Normalize headers
            df_value_labels.columns = [c.strip().lower() for c in df_value_labels.columns]
        else:
            # Create empty structure if missing
            df_value_labels = pd.DataFrame(columns=["variable", "value", "label"])

        # --- Write both sheets ---
        with pd.ExcelWriter(meta_file_path, engine="openpyxl") as writer:
            df_meta.to_excel(writer, index=False, sheet_name="metadata")
            df_value_labels.to_excel(writer, index=False, sheet_name="value_label")


    else:
        return Response({"error": "Unsupported file type for metadata"}, status=400)

    # Save relative path in DB
    relative_path = os.path.relpath(meta_file_path, settings.MEDIA_ROOT).replace("\\", "/")
    project.meta_file = relative_path
    project.save()

    # Log activity
    log_activity(
        user=request.user.username if request.user.is_authenticated else "guest",
        action="regenerate_meta" if force else "generate_meta",
        project_name=project.project_name,
        file_name="meta.xlsx"
    )

    return Response({"meta_file": relative_path, "message": "Meta file generated"})

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

# // files upload view
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
        action="upload Project file",
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

from pprint import pprint

@api_view(["GET"])
def quick_crosstab(request, project_id):
    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    if not project.files or len(project.files) == 0:
        return Response({"error": "No data files found"}, status=400)

    # Use first SAV file
    data_file = os.path.join(settings.MEDIA_ROOT, project.files[0])

    ext = os.path.splitext(data_file)[1].lower()
    if ext == ".sav":
        df, _ = pyreadstat.read_sav(data_file)   # only df needed, ignore meta
    elif ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(data_file)
        if "data" not in xls.sheet_names:
            return Response({"error": "Excel must contain a 'data' sheet"}, status=400)
        df = pd.read_excel(xls, "data")
    else:
        return Response({"error": f"Unsupported file type: {ext}"}, status=400)

    # print(type(meta))
    # pprint(type(vars(meta)))

    # Load meta Excel
    if not project.meta_file:
        return Response({"error": "Meta Excel not found"}, status=400)
    meta_path = os.path.join(settings.MEDIA_ROOT, project.meta_file)
    # Load both sheets
    xls = pd.ExcelFile(meta_path)
    meta_df = pd.read_excel(xls, "metadata")

    if "value_label" in xls.sheet_names:
        value_df = pd.read_excel(xls, "value_label")
        # normalize column names
        value_df.columns = [c.strip().lower() for c in value_df.columns]

        # Forward fill variable column (in case blank rows after first variable)
        if "variable" in value_df.columns:
            value_df["variable"] = value_df["variable"].ffill()

        # Build dict → { "Q1": {1: "Male", 2: "Female"}, "Q2": {1: "Yes", 2: "No"} }
        value_label_dict = {}
        for var, group in value_df.groupby("variable"):
            mapping = dict(zip(group["value"], group["label"]))
            value_label_dict[var] = mapping
    else:
        value_label_dict = {}

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
        sortorder =row.get("Sortorder", "")

        if var_name in df.columns:
            if add_type == "NPS":
                table_data = crosstab_nps(df, var_name)   # 🔹 NPS function
                qtype = "NPS"
            else:
                value_labels = value_label_dict.get(var_name, {})
                table_data = crosstab_single_choice(df, var_name, value_labels,add_type)
                qtype = "SC"

            output.append({
                "question": table_title,
                "var_name": var_name,
                "type": qtype,
                "data": table_data,
                "add_type": add_type,
                "sortorder":sortorder
            })


    # --- Multi Response (MR) ---
    mr_groups = meta_df[meta_df["Question_Type"].str.upper() == "MR"]["Var_Grp"].unique()
    for grp in mr_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        table_data = crosstab_multi_response(df, grp, group_rows, value_label_dict)
        sortorder =group_rows.iloc[0]["Sortorder"]
        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "MR",
            "data": table_data,
            "sortorder":sortorder
        })

    # --- Numeric (NR) ---
    nr_rows = meta_df[meta_df["Question_Type"].str.upper() == "NR"]
    for _, row in nr_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        sortorder =row["Sortorder"]
        if var_name in df.columns:
            table_data = crosstab_numeric(df, var_name)
            output.append({
                "question": table_title,
                "var_name": var_name,
                "type": "NR",
                "data": table_data,
            "sortorder":sortorder
            })

    # --- Open Ended (OE) ---
    # oe_rows = meta_df[meta_df["Question_Type"].str.upper() == "OE"]
    # for _, row in oe_rows.iterrows():
    #     var_name = str(row["Var_Name"]).strip()
    #     table_title = str(row["Table_Title"])
    #     if var_name in df.columns:
    #         table_data = crosstab_open_ended(df, var_name)
    #         output.append({
    #             "question": table_title,
    #             "var_name": var_name,
    #             "type": "OE",
    #             "data": table_data
    #         })

    # --- Single Response Grid (SRG) ---
    srg_groups = meta_df[meta_df["Question_Type"].str.upper() == "SRG"]["Var_Grp"].unique()
    for grp in srg_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        table_data = crosstab_single_response_grid(df, grp, group_rows, value_label_dict)
        sortorder =group_rows.iloc[0]["Sortorder"]
        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "SRG",
            "data": table_data,
            "sortorder":sortorder
        })

        # --- Multi Response Grid (MRG) ---
    mrg_groups = meta_df[meta_df["Question_Type"].str.upper() == "MRG"]["Var_Grp"].unique()
    for grp in mrg_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        table_data = crosstab_multi_response_grid(df, grp, group_rows, value_label_dict)
        sortorder =group_rows.iloc[0]["Sortorder"]
        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "MRG",
            "data": table_data,
            "sortorder":sortorder
        })

    # --- Numeric Grid (NG) ---
    ng_groups = meta_df[meta_df["Question_Type"].str.upper() == "NRG"]["Var_Grp"].unique()
    for grp in ng_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        table_data = build_numeric_grid(df, grp, group_rows["Var_Name"], meta_df)
        sortorder =group_rows.iloc[0]["Sortorder"]
        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "NRG",
            "data": {
                "matrix": [table_data]
            },
            "sortorder":sortorder
        })
    

      # --- Open-ended Question Type ---
    # oe_groups = meta_df[meta_df["Question_Type"].str.upper() == "OE"]["Var_Name"].unique()

    # for var in oe_groups:
    #     table_data = build_wordcloud_data(df, var)
    #     output.append(table_data)

    oe_groups = meta_df[meta_df["Question_Type"].str.upper() == "OE"]["Var_Name"].unique()

    for var in oe_groups:
        # ✅ Get the question text (Table_Title)
        table_title = meta_df.loc[meta_df["Var_Name"] == var, "Table_Title"].iloc[0]

        # ✅ Pass both var and title
        sortorder =meta_df.loc[meta_df["Var_Name"] == var, "Sortorder"].iloc[0]
        table_data = build_wordcloud_data(df, var, table_title,sortorder)

        output.append(table_data)


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
    sig_param = request.GET.get("sig", "95")  # default "95"
    sig_level = "None" if sig_param == "None" else int(sig_param)

    project = Project.objects(id=project_id).first()
    if not project:
        return Response({"error": "Project not found"}, status=404)

    if not project.files or len(project.files) == 0:
        return Response({"error": "No data files found"}, status=400)

    # Use first SAV file
    data_file = os.path.join(settings.MEDIA_ROOT, project.files[0])

    ext = os.path.splitext(data_file)[1].lower()
    if ext == ".sav":
        df, _ = pyreadstat.read_sav(data_file)   # only df needed, ignore meta
    elif ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(data_file)
        if "data" not in xls.sheet_names:
            return Response({"error": "Excel must contain a 'data' sheet"}, status=400)
        df = pd.read_excel(xls, "data")
    else:
        return Response({"error": f"Unsupported file type: {ext}"}, status=400)

    # print(type(meta))
    # pprint(type(vars(meta)))

    # Load meta Excel
    if not project.meta_file:
        return Response({"error": "Meta Excel not found"}, status=400)
    meta_path = os.path.join(settings.MEDIA_ROOT, project.meta_file)
    # Load both sheets
    xls = pd.ExcelFile(meta_path)
    meta_df = pd.read_excel(xls, "metadata")

    if "value_label" in xls.sheet_names:
        value_df = pd.read_excel(xls, "value_label")
        # normalize column names
        value_df.columns = [c.strip().lower() for c in value_df.columns]

        # Forward fill variable column (in case blank rows after first variable)
        if "variable" in value_df.columns:
            value_df["variable"] = value_df["variable"].ffill()

        # Build dict → { "Q1": {1: "Male", 2: "Female"}, "Q2": {1: "Yes", 2: "No"} }
        value_label_dict = {}
        for var, group in value_df.groupby("variable"):
            mapping = dict(zip(group["value"], group["label"]))
            value_label_dict[var] = mapping
    else:
        value_label_dict = {}

    output = []

    # --- Single Choice ---
    sc_rows = meta_df[meta_df["Question_Type"].str.upper().isin(["SC", "SR"])]
    for _, row in sc_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        sortorder = row["Sortorder"]
        value_labels =value_label_dict.get(var_name, {})

        if topbreak_title:
            table_data = crosstab_single_choice_with_topbreak(
                df, var_name, value_labels, topbreak_title, value_label_dict, meta_df, sig_level=sig_level)
        else:
            table_data = crosstab_single_choice(df, var_name, value_labels)

        output.append({
            "question": table_title,
            "var_name": var_name,
            "type": "SC",
            "data": table_data,
            "crosstab_type":"new",
            "sortorder":sortorder
        })

    # --- Multi Response ---
    mr_groups = meta_df[meta_df["Question_Type"].str.upper() == "MR"]["Var_Grp"].unique()
    for grp in mr_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        sortorder = group_rows.iloc[0]["Sortorder"]

        if topbreak_title:
            table_data = crosstab_multi_response_with_topbreak(
                df, grp, group_rows, value_label_dict, topbreak_title, meta_df, sig_level=sig_level
            )
        else:
            table_data = crosstab_multi_response(df, grp, group_rows, value_label_dict)

        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "MR",
            "data": table_data,
            "crosstab_type":"new",
            "sortorder":sortorder
        })

    # --- Numeric ---
    nr_rows = meta_df[meta_df["Question_Type"].str.upper() == "NR"]
    for _, row in nr_rows.iterrows():
        var_name = str(row["Var_Name"]).strip()
        table_title = str(row["Table_Title"])
        sortorder = row["Sortorder"]

        if topbreak_title:
            table_data = crosstab_numeric_with_topbreak(df, var_name, topbreak_title, value_label_dict, meta_df,sig_level=sig_level)
        else:
            table_data = crosstab_numeric(df, var_name)

        output.append({
            "question": table_title,
            "var_name": var_name,
            "type": "NR",
            "data": table_data,
            "crosstab_type":"new",
            "sortorder":sortorder
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
        sortorder = group_rows.iloc[0]["Sortorder"]

        if topbreak_title:
            table_data = crosstab_single_response_grid_with_topbreak(
                df, grp, group_rows, value_label_dict, topbreak_title, meta_df,sig_level=sig_level
            )
        else:
            table_data = crosstab_single_response_grid(df, grp, group_rows, value_label_dict)

        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "SRG",
            "data": table_data,
            "crosstab_type":"new",
            "sortorder":sortorder
        })

    # --- Multi Response Grid ---
    mrg_groups = meta_df[meta_df["Question_Type"].str.upper() == "MRG"]["Var_Grp"].unique()
    for grp in mrg_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]
        table_title = group_rows.iloc[0]["Table_Title"]
        sortorder = group_rows.iloc[0]["Sortorder"]

        if topbreak_title:
            table_data = crosstab_multi_response_grid_with_topbreak(
                df, grp, group_rows, value_label_dict, meta_df, topbreak_title,sig_level=sig_level
            )
        else:
            table_data = crosstab_multi_response_grid(
                df, grp, group_rows, value_label_dict
            )

        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "MRG",
            "data": table_data,
            "crosstab_type": "new",
            "sortorder":sortorder
        })    


    # --- Numeric Grid with Topbreak (multi-table view) ---
    ngtb_groups = meta_df[meta_df["Question_Type"].str.upper() == "NRG"]["Var_Grp"].unique()

    for grp in ngtb_groups:
        group_rows = meta_df[meta_df["Var_Grp"] == grp]["Var_Name"]
        table_title = meta_df.loc[meta_df["Var_Grp"] == grp, "Table_Title"].iloc[0]
        sortorder = meta_df.loc[meta_df["Var_Grp"] == grp, "Sortorder"].iloc[0]
        table_data = build_numeric_grid_topbreak(
            df,
            grp,
            group_rows,
            topbreak_title,   # passed from payload
            meta_df,
            value_label_dict,
            sig_level=sig_level
        )

        output.append({
            "question": table_title,
            "var_group": grp,
            "type": "NRG",
            "data": {
                "matrix": table_data
            },
            "crosstab_type": "new",
            "sortorder":sortorder
        })
    



    return Response(output)
