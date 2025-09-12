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
  const [sortConfigs, setSortConfigs] = useState({});

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

  // Utility: apply same sorting used in UI
  const getSortedRows = (table, tableId) => {
    const sortConfig = sortConfigs[tableId] || {
      colIndex: null,
      direction: "none",
    };
    const hasMultipleColumns = table.type === "SRG";

    let rows = [...table.data.Total.rows];
    const fixedTop = rows.filter((r) => r.label === "Total");
    const fixedBottom = rows.filter((r) =>
      ["Mean", "Median", "StdDev"].includes(r.label)
    );
    let sortableRows = rows.filter(
      (r) => !["Total", "Mean", "Median", "StdDev"].includes(r.label)
    );

    if (sortConfig.colIndex !== null && sortConfig.direction !== "none") {
      sortableRows = sortableRows.sort((a, b) => {
        if (hasMultipleColumns) {
          const valA = a.cells?.[sortConfig.colIndex]?.count || 0;
          const valB = b.cells?.[sortConfig.colIndex]?.count || 0;
          return sortConfig.direction === "desc" ? valB - valA : valA - valB;
        }
        return sortConfig.direction === "desc"
          ? b.count - a.count
          : a.count - b.count;
      });
    }

    return [...fixedTop, ...sortableRows, ...fixedBottom];
  };

  // --- Export to Excel ---
  const handleExportExcel = (table, tableId) => {
    // --- Apply sorting first ---
    const sortedRows = getSortedRows(table, tableId);

    let countRows = [];
    let percentRows = [];

    if (table.type === "SRG") {
      // Headers
      countRows.push(["Label", ...table.data.Total.columns]);
      percentRows.push(["Label", ...table.data.Total.columns]);

      // Base row
      countRows.push([
        "Base",
        ...table.data.Total.columns.map(() => table.data.Total.base),
      ]);
      percentRows.push(["Base", ...table.data.Total.columns.map(() => "100%")]);

      // Sorted rows
      sortedRows.forEach((r) => {
        const countRow = [r.label];
        const pctRow = [r.label];

        r.cells.forEach((cell) => {
          if (["Mean", "Median", "StdDev"].includes(r.label)) {
            countRow.push(cell.count?.toFixed(2) || "");
            pctRow.push(""); // no %
          } else {
            countRow.push(cell.count);
            pctRow.push(cell.pct !== undefined ? cell.pct : "");
          }
        });

        countRows.push(countRow);
        percentRows.push(pctRow);
      });
    } else {
      // SC / MR / NR
      countRows.push(["Label", "Count"]);
      percentRows.push(["Label", "Percentage"]);

      countRows.push(["Base", table.data.Total.base]);
      percentRows.push(["Base", "100%"]);

      sortedRows.forEach((r) => {
        countRows.push([r.label, r.count]);
        percentRows.push([r.label, r.pct !== undefined ? r.pct : ""]);
      });
    }

    // --- Build Excel workbook ---
    const wb = XLSX.utils.book_new();

    const wsCount = XLSX.utils.aoa_to_sheet(countRows);
    XLSX.utils.book_append_sheet(wb, wsCount, "Count");

    const wsPercent = XLSX.utils.aoa_to_sheet(percentRows);
    XLSX.utils.book_append_sheet(wb, wsPercent, "Percent");

    // Save file
    const excelBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    const fileData = new Blob([excelBuffer], {
      type: "application/octet-stream",
    });
    saveAs(fileData, `${table.question}_crosstab.xlsx`);
  };

  // Export to CSV
  const handleExportCSV = (table) => {
    let rows = [];

    if (table.type === "SRG") {
      rows.push([
        "Label",
        ...table.data.Total.columns.flatMap((col) => [
          col + " Count",
          col + "",
        ]),
      ]);

      table.data.Total.rows.forEach((r) => {
        const row = [r.label];
        r.cells.forEach((cell) => {
          if (["Mean", "Median", "StdDev"].includes(r.label)) {
            // Only export the numeric value (count), leave % empty
            row.push(cell.count);
            row.push("");
          } else {
            row.push(cell.count);
            row.push(cell.pct !== undefined ? cell.pct : "");
          }
        });
        rows.push(row);
      });
    } else {
      rows.push(["Label", "Count", "Percentage"]);
      table.data.Total.rows.forEach((r) => {
        rows.push([r.label, r.count, r.pct + "%"]);
      });
    }

    const csvContent = rows.map((e) => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    saveAs(blob, `${table.question}_crosstab.csv`);
  };

  // Sorting function
  const handleSort = (tableId, colIdx) => {
    setSortConfigs((prev) => {
      const prevConfig = prev[tableId] || { colIndex: null, direction: "none" };

      if (prevConfig.colIndex === colIdx) {
        if (prevConfig.direction === "desc")
          return { ...prev, [tableId]: { colIndex: colIdx, direction: "asc" } };
        if (prevConfig.direction === "asc")
          return {
            ...prev,
            [tableId]: { colIndex: colIdx, direction: "none" },
          }; // reset
        return { ...prev, [tableId]: { colIndex: colIdx, direction: "desc" } };
      }
      return { ...prev, [tableId]: { colIndex: colIdx, direction: "desc" } };
    });
  };

  // Generic table renderer (works for SC, MR, NR, SRG)
  const renderTable = (table, tableId) => {
    const hasMultipleColumns = table.type === "SRG";
    const columns = hasMultipleColumns
      ? table.data.Total.columns
      : ["Qualified"];

    const sortConfig = sortConfigs[tableId] || {
      colIndex: null,
      direction: "none",
    };

    let rows = [...table.data.Total.rows];

    // Fixed rows
    const fixedTop = rows.filter((r) => r.label === "Total");
    const fixedBottom = rows.filter((r) =>
      ["Mean", "Median", "StdDev"].includes(r.label)
    );
    let sortableRows = rows.filter(
      (r) => !["Total", "Mean", "Median", "StdDev"].includes(r.label)
    );

    if (sortConfig.colIndex !== null && sortConfig.direction !== "none") {
      sortableRows = sortableRows.sort((a, b) => {
        if (hasMultipleColumns) {
          const valA = a.cells?.[sortConfig.colIndex]?.count || 0;
          const valB = b.cells?.[sortConfig.colIndex]?.count || 0;
          return sortConfig.direction === "desc" ? valB - valA : valA - valB;
        }
        return sortConfig.direction === "desc"
          ? b.count - a.count
          : a.count - b.count;
      });
    }

    rows = [...fixedTop, ...sortableRows, ...fixedBottom];

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-xs">
          <thead className="bg-gray-100 sticky top-0 z-10">
            <tr>
              <th className="border px-3 py-2 text-left bg-gray-200">Scale</th>
              {columns.map((col, cIdx) => {
                const colSort = sortConfigs[tableId] || {};
                return (
                  <th
                    key={cIdx}
                    onClick={() => handleSort(tableId, cIdx)}
                    className={`border px-3 text-center cursor-pointer 
                    hover:bg-blue-50 transition-colors
                    ${
                      colSort.colIndex === cIdx && colSort.direction !== "none"
                        ? "text-blue-600 font-bold underline"
                        : "text-gray-800"
                    }`}
                  >
                    {col}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rIdx) => (
              <tr
                key={rIdx}
                className={`${
                  rIdx % 2 === 0 ? "bg-white" : "bg-gray-50"
                } hover:bg-yellow-50 transition-colors`}
              >
                <td className="border px-3  font-semibold text-gray-700">
                  {row.label}
                </td>
                {hasMultipleColumns ? (
                  row.cells.map((cell, idx) => (
                    <td
                      key={idx}
                      className="border px-3  text-center hover:bg-blue-100"
                    >
                      {cell.pct && table.type !== "NR" ? (
                        <div className="flex flex-col items-center text-gray-700">
                          <span>{cell.pct}</span>
                          <span className="text-gray-500">{cell.count}</span>
                        </div>
                      ) : (
                        <span>{cell.count}</span>
                      )}
                    </td>
                  ))
                ) : (
                  <td className="border px-3  text-center">
                    <div className="flex flex-col items-center text-gray-700">
                      <span>
                        {row.pct} {table.type === "NR" ? "" : "%"}
                      </span>
                      <span className="text-gray-500">{row.count}</span>
                    </div>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
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
                  onClick={() => handleExportExcel(table, idx)}
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
            {renderTable(table, idx)} {/* 👈 pass idx as tableId */}
          </div>
        ))}
      </div>
    </div>
  );
}

export default CrosstabPage;
