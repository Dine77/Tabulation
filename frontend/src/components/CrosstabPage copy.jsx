import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { saveAs } from "file-saver";
import * as XLSX from "xlsx";

function CrosstabPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [tables, setTables] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [menuOpen, setMenuOpen] = useState(null);
  const [sortConfig, setSortConfig] = useState({
    colIndex: null,
    direction: "none",
  });

  const runQuickCrosstab = () => {
    setLoading(true);
    setError(null);

    fetch(`http://127.0.0.1:8000/api/projects/${id}/quick_crosstab/`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to generate crosstab");
        return res.json();
      })
      .then((data) => {
        setTables(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error running crosstab:", err);
        setError("Something went wrong while generating crosstabs.");
        setLoading(false);
      });
  };

  // Export to Excel
  const handleExportExcel = (table) => {
    const rows = table.data.Total.rows.map((r) => ({
      Label: r.label,
      Count: r.count,
      Percentage: r.pct + "%",
    }));

    rows.unshift({
      Label: "Base",
      Count: table.data.Total.base,
      Percentage: "100%",
    });

    const ws = XLSX.utils.json_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Crosstab");

    const excelBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    const fileData = new Blob([excelBuffer], {
      type: "application/octet-stream",
    });
    saveAs(fileData, `${table.question}_crosstab.xlsx`);
  };

  // Export to CSV
  const handleExportCSV = (table) => {
    const rows = [["Label", "Count", "Percentage"]];
    rows.push(["Base", table.data.Total.base, "100%"]);
    table.data.Total.rows.forEach((r) => {
      rows.push([r.label, r.count, r.pct + "%"]);
    });

    const csvContent = rows.map((e) => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    saveAs(blob, `${table.question}_crosstab.csv`);
  };
  // Sorting function
  const handleSort = (colIdx) => {
    setSortConfig((prev) => {
      if (prev.colIndex === colIdx) {
        if (prev.direction === "desc")
          return { colIndex: colIdx, direction: "asc" };
        if (prev.direction === "asc")
          return { colIndex: colIdx, direction: "none" }; // reset
        return { colIndex: colIdx, direction: "desc" };
      }
      return { colIndex: colIdx, direction: "desc" };
    });
  };

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Project Crosstabs</h1>
        <button
          onClick={() => navigate("/")}
          className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Back to Projects
        </button>
      </div>

      {/* Run Quick Crosstab */}
      <button
        onClick={runQuickCrosstab}
        disabled={loading}
        className={`px-4 py-2 rounded-lg mb-6 ${
          loading
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 text-white"
        }`}
      >
        {loading ? "Generating..." : "Run Quick Crosstab"}
      </button>

      {loading && (
        <div className="text-center text-gray-600 font-medium my-4">
          ⏳ Please wait, generating crosstabs...
        </div>
      )}

      {error && (
        <div className="text-center text-red-500 font-medium my-4">{error}</div>
      )}

      {/* Tables */}
      <div className="space-y-6">
        {tables.map((table, idx) => (
          <div key={idx} className="bg-white border rounded-lg shadow">
            {/* Question Header */}
            <div className="flex justify-between items-center px-4 py-2 border-b bg-gray-50">
              <h2 className="text-green-700 font-semibold">{table.question}</h2>
              <div className="flex items-center gap-2 relative">
                <button
                  onClick={() => handleExportExcel(table)}
                  className="text-sm text-blue-600 hover:underline"
                >
                  Export
                </button>
                <button
                  onClick={() => setMenuOpen(menuOpen === idx ? null : idx)}
                  className="px-2 py-1 rounded hover:bg-gray-200"
                >
                  ⋮
                </button>
                {menuOpen === idx && (
                  <div className="absolute right-0 top-8 bg-white border rounded shadow-lg z-10">
                    <button
                      onClick={() => {
                        handleExportCSV(table);
                        setMenuOpen(null);
                      }}
                      className="block w-full text-left px-4 py-2 hover:bg-gray-100"
                    >
                      Export to CSV
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Crosstab Table */}
            {table.type === "SRG" ? (
              <table className="min-w-full border text-xs">
                <thead className="bg-gray-200">
                  <tr>
                    <th className="border px-2 py-1 text-left">Scale</th>
                    {table.data.Total.columns.map((col, cIdx) => (
                      <th
                        key={cIdx}
                        onClick={() => handleSort(cIdx)}
                        className={`border px-2 py-1 text-center cursor-pointer 
    hover:bg-gray-300 
    ${
      sortConfig.colIndex === cIdx && sortConfig.direction !== "none"
        ? "text-blue-600 font-bold hover:underline"
        : "text-gray-800"
    }`}
                      >
                        {col}
                        {/* {sortConfig.colIndex === cIdx && (
                          <span className="ml-1 text-xs">
                            {sortConfig.direction === "desc" ? "↓" : sortConfig.direction === "asc" ? "↑" : "↺"}
                          </span>
                        )} */}
                      </th>
                    ))}
                  </tr>
                </thead>

                {(() => {
                  let rows = [...table.data.Total.rows];

                  // fixed rows
                  const fixedTop = rows.filter((r) => r.label === "Total");
                  const fixedBottom = rows.filter((r) =>
                    ["Mean", "Median", "StdDev"].includes(r.label)
                  );
                  let sortableRows = rows.filter(
                    (r) =>
                      !["Total", "Mean", "Median", "StdDev"].includes(r.label)
                  );

                  if (
                    sortConfig.colIndex !== null &&
                    sortConfig.direction !== "none"
                  ) {
                    sortableRows = sortableRows.sort((a, b) => {
                      const valA = a.cells?.[sortConfig.colIndex]?.count || 0;
                      const valB = b.cells?.[sortConfig.colIndex]?.count || 0;
                      return sortConfig.direction === "desc"
                        ? valB - valA
                        : valA - valB;
                    });
                  }

                  // recombine: meta order when "none"
                  rows = [...fixedTop, ...sortableRows, ...fixedBottom];

                  // --- 4. Render ---
                  return (
                    <tbody>
                      {rows.map((row, rIdx) => (
                        <tr key={rIdx} className={rIdx % 2 === 0 ? "" : ""}>
                          <td className="border px-2 py-1 font-semibold">
                            {row.label}
                          </td>
                          {row.cells.map((cell, idx) => (
                            <td
                              key={idx}
                              className="border px-2 py-1 text-center"
                            >
                              {cell.pct ? (
                                <div className="flex flex-col items-center">
                                  <span>{cell.pct}</span>
                                  <span>{cell.count}</span>
                                </div>
                              ) : (
                                <span>{cell.count}</span>
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  );
                })()}
              </table>
            ) : (
              <table className="min-w-full text-sm">
                <thead>
                  <tr>
                    <th className="border px-3 py-2 text-left"></th>
                    <th className="border px-3 py-2 text-center">Qualified</th>
                  </tr>
                  <tr className="bg-gray-300">
                    <td className="border px-3 py-1 text-l font-bold">Total</td>
                    <td className="border px-3 py-1 text-center text-xs">
                      <div className="flex flex-col items-center">
                        <span className="text-gray-800 text-xs">100%</span>
                        <span className="text-gray-600 text-xs">
                          {table.data.Total.base}
                        </span>
                      </div>
                    </td>
                  </tr>
                </thead>
                <tbody>
                  {table.data.Total.rows.map((row, rIdx) => (
                    <tr key={rIdx} className={rIdx % 2 === 0 ? "" : ""}>
                      <td className="border px-3 py-2">{row.label}</td>
                      <td className="border px-3 py-2 text-center">
                        <div className="flex flex-col items-center">
                          <span className="text-gray-800 text-xs">
                            {row.pct} {table.type === "NR" ? "" : "%"}
                          </span>
                          <span className="text-gray-600 text-xs">
                            {row.count}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default CrosstabPage;
