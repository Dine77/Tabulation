import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function ProjectList() {
    const navigate = useNavigate();
  const [projects, setProjects] = useState([]);
  const [openModal, setOpenModal] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [file, setFile] = useState(null);
  const [selectedProject, setSelectedProject] = useState(null);


  // Fetch projects
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/projects/")
      .then((res) => res.json())
      .then((data) => setProjects(data))
      .catch((err) => console.error("API Error:", err));
  }, []);

  // Handle file upload
  // Handle file upload
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!projectName || !file) {
      alert("Project name and file are required");
      return;
    }

    const formData = new FormData();
    formData.append("project_name", projectName);
    formData.append("file", file); // 👈 must match "file" key in Django

    fetch("http://127.0.0.1:8000/api/projects/upload/", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((newProject) => {
        console.log("Upload success:", newProject);
        setProjects((prevProjects) => {
          const exists = prevProjects.find((p) => p.id === newProject.id);
          if (exists) {
            // Update existing project
            return prevProjects.map((p) =>
              p.id === newProject.id ? newProject : p
            );
          } else {
            // Add new project
            return [newProject, ...prevProjects];
          }
        });

        // reset modal + form
        setOpenModal(false);
        setProjectName("");
        setFile(null);
      })
      .catch((err) => console.error("Upload error:", err));
  };


  const handleDeleteFile = (projectId, fileName) => {
    fetch(`http://127.0.0.1:8000/api/projects/${projectId}/delete/${fileName}/`, {
      method: "DELETE",
    })
      .then((res) => res.json())
      .then((msg) => {
        console.log(msg);
        setProjects((prev) =>
          prev.map((p) =>
            p.id === projectId
              ? { ...p, files: p.files.filter((f) => f !== fileName) }
              : p
          )
        );
      })
      .catch((err) => console.error("Delete error:", err));
  };

  const handleDeleteProject = (projectId) => {
    if (!window.confirm("Are you sure you want to delete this project?")) return;

    fetch(`http://127.0.0.1:8000/api/projects/${projectId}/delete/`, {
      method: "DELETE",
    })
      .then((res) => res.json())
      .then((msg) => {
        console.log(msg);
        setProjects((prev) => prev.filter((p) => p.id !== projectId));
      })
      .catch((err) => console.error("Delete project error:", err));
  };


  // Download / Generate meta file
  const handleDownloadMeta = (projectId) => {
    fetch(`http://127.0.0.1:8000/api/projects/${projectId}/generate_meta/`)
      .then((res) => res.json())
      .then((data) => {
        if (data.meta_file) {
          window.open(`http://127.0.0.1:8000/media/${data.meta_file}`, "_blank");
        } else {
          alert(data.error || "Failed to generate meta file");
        }
      })
      .catch((err) => console.error("Meta download error:", err));
  };

  // Upload meta file
  const handleUploadMeta = (projectId, file) => {
    const formData = new FormData();
    formData.append("meta_file", file);

    fetch(`http://127.0.0.1:8000/api/projects/${projectId}/upload_meta/`, {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Meta uploaded:", data);
        alert("Meta file uploaded successfully!");
      })
      .catch((err) => console.error("Meta upload error:", err));
  };

  const handleRegenerateMeta = (projectId) => {
    fetch(`http://127.0.0.1:8000/api/projects/${projectId}/generate_meta/?force=true`)
      .then((res) => res.json())
      .then((data) => {
        if (data.meta_file) {
          alert("Meta file regenerated successfully!");
          window.open(`http://127.0.0.1:8000/media/${data.meta_file}`, "_blank");
        } else {
          alert(data.error || "Failed to regenerate meta file");
        }
      })
      .catch((err) => console.error("Meta regenerate error:", err));
  };


  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">My Projects</h1>
        <button
          onClick={() => setOpenModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700"
        >
          + Add New Project
        </button>
      </div>

      {/* Project list */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {projects.length > 0 ? (
          projects.map((proj) => (
            <div
              key={proj.id}
              className="p-4 border rounded-lg shadow hover:shadow-lg transition"
            >
              <h2 className="text-xl font-semibold flex justify-between items-center">
                {proj.project_name}
                <button
                  onClick={() => navigate(`/projects/${proj.id}/crosstabs`)}  // 👈 redirect
                  className="ml-3 px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  View Crosstabs
                </button>
                <button
                  onClick={() => handleDeleteProject(proj.id)}
                  className="ml-3 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
                >
                  Delete Project
                </button>
              </h2>

              <p className="text-gray-600 text-sm mt-1">Files:</p>
              <ul className="list-disc list-inside text-gray-700">
                {proj.files && proj.files.length > 0 ? (
                  proj.files.map((file, idx) => (
                    <li key={idx}>
                      <a
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setSelectedProject(proj.id); // 👈 mark this project as selected
                        }}
                        className="text-blue-600 hover:underline"
                      >
                        {file.split("/").pop()}
                      </a>
                      <button
                        onClick={() => handleDeleteFile(proj.id, file)}
                        className="ml-2 text-red-500 hover:text-red-700"
                      >
                        Delete
                      </button>
                    </li>
                  ))
                ) : (
                  <li className="italic text-gray-500">No files yet</li>
                )}
              </ul>

              {/* Meta Excel Actions 👇 inside project card */}
              {selectedProject === proj.id && (
              <div className="mt-3 space-x-2">
                <button
                  onClick={() => handleDownloadMeta(proj.id)}
                  className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Download Meta Excel
                </button>

                <button
                  onClick={() => handleRegenerateMeta(proj.id)}
                  className="px-3 py-1 bg-yellow-600 text-white rounded hover:bg-yellow-700"
                >
                  Regenerate Meta Excel
                </button>

                <label className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 cursor-pointer">
                  Upload Meta Excel
                  <input
                    type="file"
                    className="hidden"
                    accept=".xlsx"
                    onChange={(e) => handleUploadMeta(proj.id, e.target.files[0])}
                  />
                </label>

                {proj.meta_file && (
                  <p className="text-sm mt-1 text-gray-700">
                    Current Meta:{" "}
                    <a
                      href={`http://127.0.0.1:8000/media/${proj.meta_file}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline"
                    >
                      {proj.meta_file.split("/").pop()}
                    </a>
                  </p>
                )}
              </div>
            )}
            </div>
          ))
        ) : (
          <p className="text-gray-500">No projects yet</p>
        )}
      </div>

      {/* Modal */}
      {openModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-white rounded-lg p-6 w-96 shadow-lg">
            <h2 className="text-xl font-bold mb-4">Add New Project</h2>
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                placeholder="Project Name"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                className="w-full mb-3 px-3 py-2 border rounded-lg"
              />
              <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
                className="w-full mb-3"
                accept=".sav"
              />
              <div className="flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setOpenModal(false)}
                  className="px-4 py-2 bg-gray-400 text-white rounded-lg hover:bg-gray-500"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Upload
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProjectList;
